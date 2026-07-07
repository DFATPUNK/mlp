from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict, deque
from pathlib import Path

from .taco_intake import DOWNLOAD_PLAN_COLUMNS
from .utils import ensure_dir, write_json

DEFAULT_SEED = "phase_0_8_4"
DEFAULT_LIMITS = {
    "multiple_objects": 24,
    "ambiguous_scene": 14,
    "unsupported_material": 9,
}

REVIEW_BATCH_COLUMNS = [
    "batch_id",
    "selection_rank",
    "selection_bucket",
    "selection_reason",
    "plan_id",
    "source_dataset_id",
    "source_item_id",
    "relative_path",
    "source_url",
    "source_license",
    "source_license_reference",
    "source_attribution",
    "license_status",
    "resolution_rule",
    "annotation_count",
    "source_labels",
    "mapped_labels",
    "object_area_ratio",
    "proposed_classifier_role",
    "proposed_gate_value",
    "proposed_review_reason",
    "review_status",
    "review_decision",
    "review_notes",
]

BUCKET_ORDER = [
    "classifier_and_gate_candidate",
    "multiple_objects",
    "ambiguous_scene",
    "unsupported_material",
]
GATE_BUCKETS = ["multiple_objects", "ambiguous_scene", "unsupported_material"]
SELECTION_POLICY_VERSION = "taco_review_batch_v1"
OUTPUT_SCHEMA_VERSION = "taco_review_batch_schema_v1"


class TacoReviewBatchError(ValueError):
    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def _read_download_plan(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        found = reader.fieldnames or []
        if found != DOWNLOAD_PLAN_COLUMNS:
            missing = [column for column in DOWNLOAD_PLAN_COLUMNS if column not in found]
            unknown = [column for column in found if column not in DOWNLOAD_PLAN_COLUMNS]
            messages = [f"download plan: expected exact columns {DOWNLOAD_PLAN_COLUMNS}, found {found}"]
            if missing:
                messages.append(f"download plan: missing required columns {missing}")
            if unknown:
                messages.append(f"download plan: unknown columns {unknown}")
            raise TacoReviewBatchError(messages)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _write_csv(path: str | Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def _stable_digest(seed: str, source_item_id: str, plan_id: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{source_item_id}\x1f{plan_id}".encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _canonical_list(value: str) -> str:
    parts = sorted({part.strip() for part in value.split(";") if part.strip()})
    return ";".join(parts)


def _validate_duplicate_source_items(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["source_item_id"] for row in rows if row.get("source_item_id"))
    duplicates = sorted(source_item_id for source_item_id, count in counts.items() if count > 1)
    if duplicates:
        raise TacoReviewBatchError([f"download plan: duplicate source_item_id values are not allowed: {duplicates}"])


def _eligible(row: dict[str, str]) -> bool:
    return (
        row.get("source_dataset_id") == "taco"
        and row.get("license_status") == "eligible_for_review"
        and row.get("plan_status") == "eligible_for_manual_download"
        and row.get("source_url") != ""
        and row.get("source_license") != ""
        and row.get("source_license_reference") != ""
        and row.get("source_attribution") != ""
    )


def _bucket_for_row(row: dict[str, str]) -> str | None:
    if (
        row.get("proposed_classifier_role") == "classifier_and_gate_candidate"
        and row.get("proposed_gate_value") == "true"
        and row.get("proposed_review_reason") == "none"
    ):
        return "classifier_and_gate_candidate"
    if row.get("proposed_classifier_role") == "gate_only" and row.get("proposed_gate_value") == "false":
        reason = row.get("proposed_review_reason")
        if reason in GATE_BUCKETS:
            return reason
    return None


def _ranked_rows(rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _stable_digest(seed, row["source_item_id"], row["plan_id"]),
            row["source_item_id"],
            row["plan_id"],
        ),
    )


def _rows_for_mapped_group(rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    by_source_labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source_labels[_canonical_list(row["source_labels"])].append(row)

    queues = []
    for source_labels, group_rows in by_source_labels.items():
        ranked = _ranked_rows(group_rows, seed)
        queues.append((source_labels, _stable_digest(seed, ranked[0]["source_item_id"], ranked[0]["plan_id"]), deque(ranked)))
    queues.sort(key=lambda item: (item[1], item[0]))

    ordered = []
    while queues:
        next_queues = []
        for source_labels, group_rank, queue in queues:
            ordered.append(queue.popleft())
            if queue:
                next_queues.append((source_labels, group_rank, queue))
        queues = next_queues
    return ordered


def _select_gate_bucket(rows: list[dict[str, str]], *, seed: str, limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    by_mapped_labels: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mapped_labels[_canonical_list(row["mapped_labels"])].append(row)

    queues = []
    for mapped_labels, group_rows in by_mapped_labels.items():
        ordered = _rows_for_mapped_group(group_rows, seed)
        queues.append((mapped_labels, _stable_digest(seed, ordered[0]["source_item_id"], ordered[0]["plan_id"]), deque(ordered)))
    queues.sort(key=lambda item: (item[1], item[0]))

    selected = []
    while queues and len(selected) < limit:
        next_queues = []
        for mapped_labels, group_rank, queue in queues:
            if len(selected) >= limit:
                next_queues.append((mapped_labels, group_rank, queue))
                continue
            selected.append(queue.popleft())
            if queue:
                next_queues.append((mapped_labels, group_rank, queue))
        queues = next_queues
    return selected


def _selection_reason(bucket: str, limit: int | None) -> str:
    if bucket == "classifier_and_gate_candidate":
        return "all eligible classifier-and-gate candidates are retained"
    return f"deterministic diverse gate-only sample for {bucket} up to limit {limit}"


def _batch_row(row: dict[str, str], *, seed: str, limits: dict[str, int], bucket: str, selection_rank: int) -> dict[str, str]:
    limit_text = ";".join(f"{name}={limits[name]}" for name in sorted(limits))
    return {
        "batch_id": _stable_id("taco_batch", row["plan_id"], row["source_item_id"], seed, limit_text),
        "selection_rank": str(selection_rank),
        "selection_bucket": bucket,
        "selection_reason": _selection_reason(bucket, limits.get(bucket)),
        "plan_id": row["plan_id"],
        "source_dataset_id": row["source_dataset_id"],
        "source_item_id": row["source_item_id"],
        "relative_path": row["relative_path"],
        "source_url": row["source_url"],
        "source_license": row["source_license"],
        "source_license_reference": row["source_license_reference"],
        "source_attribution": row["source_attribution"],
        "license_status": row["license_status"],
        "resolution_rule": row["resolution_rule"],
        "annotation_count": row["annotation_count"],
        "source_labels": row["source_labels"],
        "mapped_labels": row["mapped_labels"],
        "object_area_ratio": row["object_area_ratio"],
        "proposed_classifier_role": row["proposed_classifier_role"],
        "proposed_gate_value": row["proposed_gate_value"],
        "proposed_review_reason": row["proposed_review_reason"],
        "review_status": "pending",
        "review_decision": "",
        "review_notes": "",
    }


def _diversity_summary(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    counts = Counter(
        (
            row["selection_bucket"],
            _canonical_list(row["mapped_labels"]),
            _canonical_list(row["source_labels"]),
        )
        for row in rows
    )
    return [
        {
            "selection_bucket": bucket,
            "mapped_labels": mapped_labels,
            "source_labels": source_labels,
            "count": count,
        }
        for (bucket, mapped_labels, source_labels), count in sorted(counts.items())
    ]


def build_taco_review_batch(
    *,
    download_plan: str | Path,
    output_dir: str | Path,
    seed: str = DEFAULT_SEED,
    multiple_objects_limit: int = DEFAULT_LIMITS["multiple_objects"],
    ambiguous_scene_limit: int = DEFAULT_LIMITS["ambiguous_scene"],
    unsupported_material_limit: int = DEFAULT_LIMITS["unsupported_material"],
) -> dict:
    limits = {
        "multiple_objects": multiple_objects_limit,
        "ambiguous_scene": ambiguous_scene_limit,
        "unsupported_material": unsupported_material_limit,
    }
    invalid_limits = {name: value for name, value in limits.items() if value < 0}
    if invalid_limits:
        raise TacoReviewBatchError([f"limits must be non-negative; got {invalid_limits}"])

    rows = _read_download_plan(download_plan)
    _validate_duplicate_source_items(rows)
    eligible_rows = [row for row in rows if _eligible(row)]

    rows_by_bucket: dict[str, list[dict[str, str]]] = {bucket: [] for bucket in BUCKET_ORDER}
    for row in eligible_rows:
        bucket = _bucket_for_row(row)
        if bucket is not None:
            rows_by_bucket[bucket].append(row)

    selected_by_bucket: dict[str, list[dict[str, str]]] = {}
    selected_by_bucket["classifier_and_gate_candidate"] = _ranked_rows(rows_by_bucket["classifier_and_gate_candidate"], seed)
    for bucket in GATE_BUCKETS:
        selected_by_bucket[bucket] = _select_gate_bucket(rows_by_bucket[bucket], seed=seed, limit=limits[bucket])

    batch_rows = []
    selected_source_item_ids: set[str] = set()
    selection_rank = 1
    for bucket in BUCKET_ORDER:
        for row in selected_by_bucket[bucket]:
            if row["source_item_id"] in selected_source_item_ids:
                continue
            selected_source_item_ids.add(row["source_item_id"])
            batch_rows.append(_batch_row(row, seed=seed, limits=limits, bucket=bucket, selection_rank=selection_rank))
            selection_rank += 1

    available_counts = {bucket: len(rows_by_bucket[bucket]) for bucket in BUCKET_ORDER}
    selected_counts = {bucket: len([row for row in batch_rows if row["selection_bucket"] == bucket]) for bucket in BUCKET_ORDER}
    shortages = {bucket: max(0, limits[bucket] - selected_counts[bucket]) for bucket in GATE_BUCKETS}

    report = {
        "input_plan_path": str(download_plan),
        "input_plan_rows": len(rows),
        "eligible_plan_rows": len(eligible_rows),
        "seed": seed,
        "requested_limits": limits,
        "available_counts_by_bucket": available_counts,
        "selected_counts_by_bucket": selected_counts,
        "shortages_by_bucket": shortages,
        "selected_total_rows": len(batch_rows),
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "diversity_summary": _diversity_summary(batch_rows),
    }

    output_dir = ensure_dir(output_dir)
    _write_csv(output_dir / "taco_review_batch.csv", REVIEW_BATCH_COLUMNS, batch_rows)
    write_json(output_dir / "taco_review_batch_report.json", report)
    return report
