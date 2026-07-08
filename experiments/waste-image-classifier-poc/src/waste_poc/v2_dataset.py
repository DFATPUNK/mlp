from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath

from .feedback import CLASSIFIER_CANDIDATE_COLUMNS, EXTERNAL_DIAGNOSTIC_V1, GATE_CANDIDATE_COLUMNS, REVIEW_REASONS, SOURCE_SPLIT_BY_SOURCE
from .utils import CLASS_NAMES, ensure_dir, sha256_file, write_json, write_text

TRASHNET_COLUMNS = [
    "image_id",
    "relative_path",
    "source_class",
    "label",
    "split",
    "sha256",
    "width",
    "height",
    "format",
    "source_commit",
    "is_valid_image",
]

PUBLIC_CLASSIFIER_COLUMNS = [
    "public_row_id",
    "source_dataset_id",
    "source_item_id",
    "relative_path",
    "source_url",
    "source_license",
    "source_license_reference",
    "source_attribution",
    "license_status",
    "source_annotation_id",
    "object_area_ratio",
    "source_label",
    "mapped_label",
    "mapping_rule_id",
    "annotation_type",
    "object_count",
    "approved_for_classifier_training",
    "annotation_notes",
]

PUBLIC_GATE_COLUMNS = [
    "public_row_id",
    "source_dataset_id",
    "source_item_id",
    "relative_path",
    "source_url",
    "source_license",
    "source_license_reference",
    "source_attribution",
    "license_status",
    "source_annotation_id",
    "object_area_ratio",
    "auto_route_eligible",
    "review_reason",
    "annotation_type",
    "object_count",
    "approved_for_gate_training",
    "annotation_notes",
]

LABEL_MAPPING_COLUMNS = [
    "mapping_rule_id",
    "source_dataset_id",
    "source_label",
    "mapped_label",
    "mapping_status",
    "rationale",
]

CLASSIFICATION_OUTPUT_COLUMNS = [
    "dataset_row_id",
    "image_id",
    "relative_path",
    "label",
    "split",
    "source_kind",
    "source_dataset_id",
    "source_split",
    "source_item_id",
    "source_url",
    "source_license",
    "source_license_reference",
    "source_attribution",
    "license_status",
    "source_annotation_id",
    "object_area_ratio",
    "parent_feedback_id",
    "original_label",
    "mapping_rule_id",
    "source_commit",
    "sha256",
]

GATE_OUTPUT_COLUMNS = [
    "gate_row_id",
    "image_id",
    "relative_path",
    "auto_route_eligible",
    "review_reason",
    "split",
    "source_kind",
    "source_dataset_id",
    "source_split",
    "source_item_id",
    "source_url",
    "source_license",
    "source_license_reference",
    "source_attribution",
    "license_status",
    "source_annotation_id",
    "object_area_ratio",
    "parent_feedback_id",
    "annotation_type",
    "object_count",
    "sha256",
]

TRASHNET_QUARANTINE_COLUMNS = [
    "quarantine_id",
    "sha256",
    "quarantine_reason",
    "source_dataset_id",
    "image_id",
    "relative_path",
    "label",
    "split",
    "source_commit",
    "source_manifest_sha256",
]

ANNOTATION_TYPES = {"single_object", "cropped_object", "scene"}
MAPPING_STATUSES = {"approved", "excluded", "needs_review"}
LICENSE_STATUSES = {"eligible_for_review", "approved", "blocked"}
PUBLIC_LICENCE_PROVENANCE_COLUMNS = ["source_url", "source_license", "source_license_reference", "source_attribution"]
SPLITS = {"train", "validation", "test"}
ENRICHMENT_SPLIT = "train"
TRASHNET_SOURCE_DATASET_ID = "trashnet"
TRASHNET_LABEL_CONFLICT_REASON = "conflicting_labels_same_sha256"
V2_LEAKAGE_NOTICE = """# V2 External Diagnostic Promotion Notice

Rows from `external_diagnostic_v1` were explicitly included in the V2 assembly outputs.

`external_diagnostic_v1` is permanently retired from future comparative or final evaluation. Any future model selection or final evaluation will require a new untouched external holdout.
"""


class V2DatasetError(ValueError):
    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def _read_csv_exact(path: str | Path, expected_columns: list[str], name: str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            found = reader.fieldnames or []
            missing = [column for column in expected_columns if column not in found]
            unknown = [column for column in found if column not in expected_columns]
            messages = [f"{name}: expected exact columns {expected_columns}, found {found}"]
            if missing:
                messages.append(f"{name}: missing columns {missing}")
            if unknown:
                messages.append(f"{name}: unknown columns {unknown}")
            raise V2DatasetError(messages)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _read_optional_csv(path: str | Path | None, expected_columns: list[str], name: str) -> list[dict[str, str]]:
    if path is None:
        return []
    return _read_csv_exact(path, expected_columns, name)


def _write_csv(path: str | Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts not in {(), (".",)}


def _bool_value(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _valid_ratio(value: str) -> bool:
    if value == "":
        return True
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= parsed <= 1.0


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _sha256_if_available(image_root: str | Path | None, relative_path: str) -> str:
    if image_root is None:
        return ""
    if not _safe_relative_path(relative_path):
        return ""
    path = Path(image_root) / relative_path
    return sha256_file(path) if path.is_file() else ""


def _row_label(kind: str, index: int, row: dict[str, str], id_column: str) -> str:
    return f"{kind} row {index} {id_column}={row.get(id_column, '<blank>')}"


def _count_duplicates(rows: list[dict[str, str]], key: str) -> Counter[str]:
    return Counter(row.get(key, "") for row in rows)


def _validate_common_relative_path(messages: list[str], label: str, row: dict[str, str]) -> None:
    if not _safe_relative_path(row.get("relative_path", "")):
        messages.append(f"{label}: relative_path must be relative and must not use parent traversal")


def _validate_public_licence_fields(messages: list[str], label: str, row: dict[str, str], approved_for_training: bool) -> None:
    if row.get("license_status") not in LICENSE_STATUSES:
        messages.append(f"{label}: license_status must be one of {sorted(LICENSE_STATUSES)}")
    if not _valid_ratio(row.get("object_area_ratio", "")):
        messages.append(f"{label}: object_area_ratio must be blank or in [0, 1]")
    if not approved_for_training:
        return
    if row.get("license_status") != "approved":
        messages.append(f"{label}: public training candidates require license_status=approved")
    missing = [column for column in PUBLIC_LICENCE_PROVENANCE_COLUMNS if row.get(column, "") == ""]
    if missing:
        messages.append(f"{label}: approved public training candidates require nonblank licence provenance fields {missing}")


def _normalized_relative_path(value: str) -> str:
    if not _safe_relative_path(value):
        return value
    return PurePosixPath(value).as_posix()


def _validate_feedback_provenance(messages: list[str], label: str, row: dict[str, str]) -> None:
    source = row.get("source", "")
    source_split = row.get("source_split", "")
    expected_split = SOURCE_SPLIT_BY_SOURCE.get(source)
    if expected_split is None:
        messages.append(f"{label}: source must be one of {sorted(SOURCE_SPLIT_BY_SOURCE)}")
        return
    if source_split != expected_split:
        messages.append(f"{label}: source={source} requires source_split={expected_split}; got source_split={source_split or '<blank>'}")


def _validate_trashnet_rows(rows: list[dict[str, str]], messages: list[str]) -> None:
    image_ids = _count_duplicates(rows, "image_id")
    for index, row in enumerate(rows, start=2):
        label = _row_label("trashnet", index, row, "image_id")
        for column in TRASHNET_COLUMNS:
            if row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        if image_ids[row.get("image_id", "")] > 1:
            messages.append(f"{label}: duplicate image_id {row.get('image_id')!r}")
        if row.get("label") not in CLASS_NAMES:
            messages.append(f"{label}: label must be one of {CLASS_NAMES}")
        if row.get("split") not in SPLITS:
            messages.append(f"{label}: split must be one of {sorted(SPLITS)}")
        _validate_common_relative_path(messages, label, row)


def _validate_feedback_classifier_rows(rows: list[dict[str, str]], messages: list[str]) -> None:
    feedback_ids = _count_duplicates(rows, "feedback_id")
    image_ids = _count_duplicates(rows, "image_id")
    for index, row in enumerate(rows, start=2):
        label = _row_label("classifier feedback", index, row, "feedback_id")
        for column in CLASSIFIER_CANDIDATE_COLUMNS:
            if row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        if feedback_ids[row.get("feedback_id", "")] > 1:
            messages.append(f"{label}: duplicate feedback_id {row.get('feedback_id')!r}")
        if image_ids[row.get("image_id", "")] > 1:
            messages.append(f"{label}: duplicate image_id {row.get('image_id')!r}")
        if row.get("confirmed_label") not in CLASS_NAMES:
            messages.append(f"{label}: confirmed_label must be one of {CLASS_NAMES}")
        _validate_feedback_provenance(messages, label, row)
        if row.get("source_split") in {"validation", "test"}:
            messages.append(f"{label}: feedback candidates must not claim validation or test source_split")
        _validate_common_relative_path(messages, label, row)


def _validate_feedback_gate_rows(rows: list[dict[str, str]], messages: list[str]) -> None:
    feedback_ids = _count_duplicates(rows, "feedback_id")
    image_ids = _count_duplicates(rows, "image_id")
    for index, row in enumerate(rows, start=2):
        label = _row_label("gate feedback", index, row, "feedback_id")
        for column in GATE_CANDIDATE_COLUMNS:
            if row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        if feedback_ids[row.get("feedback_id", "")] > 1:
            messages.append(f"{label}: duplicate feedback_id {row.get('feedback_id')!r}")
        if image_ids[row.get("image_id", "")] > 1:
            messages.append(f"{label}: duplicate image_id {row.get('image_id')!r}")
        if _bool_value(row.get("auto_route_eligible", "")) is None:
            messages.append(f"{label}: auto_route_eligible must be lowercase true or false")
        if row.get("review_reason") not in REVIEW_REASONS:
            messages.append(f"{label}: review_reason must be one of {sorted(REVIEW_REASONS)}")
        _validate_feedback_provenance(messages, label, row)
        if row.get("source_split") in {"validation", "test"}:
            messages.append(f"{label}: feedback candidates must not claim validation or test source_split")
        _validate_common_relative_path(messages, label, row)


def _validate_label_mappings(rows: list[dict[str, str]], messages: list[str]) -> dict[str, dict[str, str]]:
    rule_ids = _count_duplicates(rows, "mapping_rule_id")
    mappings: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        label = _row_label("label mapping", index, row, "mapping_rule_id")
        for column in LABEL_MAPPING_COLUMNS:
            if column != "mapped_label" and row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        rule_id = row.get("mapping_rule_id", "")
        if rule_ids[rule_id] > 1:
            messages.append(f"{label}: duplicate mapping_rule_id {rule_id!r}")
        status = row.get("mapping_status")
        mapped_label = row.get("mapped_label", "")
        if status not in MAPPING_STATUSES:
            messages.append(f"{label}: mapping_status must be one of {sorted(MAPPING_STATUSES)}")
        if status == "approved" and mapped_label not in CLASS_NAMES:
            messages.append(f"{label}: approved mappings require mapped_label to be one of {CLASS_NAMES}")
        if status in {"excluded", "needs_review"} and mapped_label and mapped_label not in CLASS_NAMES:
            messages.append(f"{label}: mapped_label must be blank or one of {CLASS_NAMES}")
        mappings[rule_id] = row
    return mappings


def _validate_public_classifier_rows(rows: list[dict[str, str]], mappings: dict[str, dict[str, str]], messages: list[str]) -> None:
    row_ids = _count_duplicates(rows, "public_row_id")
    source_items = Counter((row.get("source_dataset_id", ""), row.get("source_item_id", "")) for row in rows)
    for index, row in enumerate(rows, start=2):
        label = _row_label("public classifier", index, row, "public_row_id")
        for column in PUBLIC_CLASSIFIER_COLUMNS:
            if column not in {"mapped_label", "annotation_notes"} and row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        if row_ids[row.get("public_row_id", "")] > 1:
            messages.append(f"{label}: duplicate public_row_id {row.get('public_row_id')!r}")
        if source_items[(row.get("source_dataset_id", ""), row.get("source_item_id", ""))] > 1:
            messages.append(f"{label}: duplicate source identity {(row.get('source_dataset_id'), row.get('source_item_id'))!r}")
        _validate_common_relative_path(messages, label, row)
        if row.get("annotation_type") not in ANNOTATION_TYPES:
            messages.append(f"{label}: annotation_type must be one of {sorted(ANNOTATION_TYPES)}")
        object_count = _positive_int(row.get("object_count", ""))
        if object_count is None:
            messages.append(f"{label}: object_count must be a positive integer")
        approved = _bool_value(row.get("approved_for_classifier_training", ""))
        if approved is None:
            messages.append(f"{label}: approved_for_classifier_training must be lowercase true or false")
        _validate_public_licence_fields(messages, label, row, approved is True)
        if not approved:
            continue
        if row.get("annotation_type") not in {"single_object", "cropped_object"}:
            messages.append(f"{label}: approved classifier candidates must be single_object or cropped_object")
        if object_count != 1:
            messages.append(f"{label}: approved classifier candidates require object_count=1")
        if row.get("mapped_label") not in CLASS_NAMES:
            messages.append(f"{label}: mapped_label must be one of {CLASS_NAMES}")
        rule = mappings.get(row.get("mapping_rule_id", ""))
        if rule is None:
            messages.append(f"{label}: mapping_rule_id {row.get('mapping_rule_id')!r} is absent from the label mapping file")
            continue
        if rule.get("mapping_status") != "approved":
            messages.append(f"{label}: mapping_rule_id {row.get('mapping_rule_id')!r} is not approved")
        if rule.get("source_dataset_id") != row.get("source_dataset_id"):
            messages.append(f"{label}: mapping_rule_id {row.get('mapping_rule_id')!r} belongs to source_dataset_id={rule.get('source_dataset_id')}")
        if rule.get("source_label") != row.get("source_label"):
            messages.append(f"{label}: mapping_rule_id {row.get('mapping_rule_id')!r} maps source_label={rule.get('source_label')!r}")
        if rule.get("mapped_label") != row.get("mapped_label"):
            messages.append(f"{label}: mapped_label must match approved mapping_rule_id {row.get('mapping_rule_id')!r}")


def _validate_public_gate_rows(rows: list[dict[str, str]], messages: list[str]) -> None:
    row_ids = _count_duplicates(rows, "public_row_id")
    source_items = Counter((row.get("source_dataset_id", ""), row.get("source_item_id", "")) for row in rows)
    for index, row in enumerate(rows, start=2):
        label = _row_label("public gate", index, row, "public_row_id")
        for column in PUBLIC_GATE_COLUMNS:
            if column != "annotation_notes" and row.get(column, "") == "":
                messages.append(f"{label}: {column} is required")
        if row_ids[row.get("public_row_id", "")] > 1:
            messages.append(f"{label}: duplicate public_row_id {row.get('public_row_id')!r}")
        if source_items[(row.get("source_dataset_id", ""), row.get("source_item_id", ""))] > 1:
            messages.append(f"{label}: duplicate source identity {(row.get('source_dataset_id'), row.get('source_item_id'))!r}")
        _validate_common_relative_path(messages, label, row)
        if row.get("annotation_type") not in ANNOTATION_TYPES:
            messages.append(f"{label}: annotation_type must be one of {sorted(ANNOTATION_TYPES)}")
        object_count = _positive_int(row.get("object_count", ""))
        if object_count is None:
            messages.append(f"{label}: object_count must be a positive integer")
        eligible = _bool_value(row.get("auto_route_eligible", ""))
        if eligible is None:
            messages.append(f"{label}: auto_route_eligible must be lowercase true or false")
        approved = _bool_value(row.get("approved_for_gate_training", ""))
        if approved is None:
            messages.append(f"{label}: approved_for_gate_training must be lowercase true or false")
        _validate_public_licence_fields(messages, label, row, approved is True)
        if row.get("review_reason") not in REVIEW_REASONS:
            messages.append(f"{label}: review_reason must be one of {sorted(REVIEW_REASONS)}")
        if approved and eligible and (object_count != 1 or row.get("annotation_type") not in {"single_object", "cropped_object"}):
            messages.append(f"{label}: auto_route_eligible=true requires object_count=1 and single_object or cropped_object annotation")


def _external_diagnostic_feedback_ids(classifier_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]) -> set[str]:
    return {
        row["feedback_id"]
        for row in [*classifier_rows, *gate_rows]
        if row.get("source_split") == EXTERNAL_DIAGNOSTIC_V1
    }


def _trashnet_label_conflict_shas(rows: list[dict[str, str]]) -> set[str]:
    labels_by_sha: dict[str, set[str]] = {}
    for row in rows:
        digest = row.get("sha256", "")
        if not digest:
            continue
        labels_by_sha.setdefault(digest, set()).add(row.get("label", ""))
    return {digest for digest, labels in labels_by_sha.items() if len(labels) > 1}


def _trashnet_quarantine_row(row: dict[str, str], source_manifest_sha256: str) -> dict[str, str]:
    return {
        "quarantine_id": _stable_id("v2quarantine", row["sha256"], row["image_id"], row["relative_path"], row["label"], row["split"]),
        "sha256": row["sha256"],
        "quarantine_reason": TRASHNET_LABEL_CONFLICT_REASON,
        "source_dataset_id": TRASHNET_SOURCE_DATASET_ID,
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
        "label": row["label"],
        "split": row["split"],
        "source_commit": row.get("source_commit", ""),
        "source_manifest_sha256": source_manifest_sha256,
    }


def _sort_trashnet_quarantine_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            row["sha256"],
            split_order.get(row["split"], 99),
            row["label"],
            row["image_id"],
            row["relative_path"],
            row["quarantine_id"],
        ),
    )


def _curate_trashnet_label_conflicts(rows: list[dict[str, str]], source_manifest_sha256: str) -> tuple[list[dict[str, str]], list[dict[str, str]], set[str]]:
    conflict_shas = _trashnet_label_conflict_shas(rows)
    included_rows = [row for row in rows if row.get("sha256", "") not in conflict_shas]
    quarantine_rows = [_trashnet_quarantine_row(row, source_manifest_sha256) for row in rows if row.get("sha256", "") in conflict_shas]
    return included_rows, _sort_trashnet_quarantine_rows(quarantine_rows), conflict_shas


def _classification_identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["source_kind"], row["source_dataset_id"], row["source_item_id"] or row["image_id"])


def _gate_identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["source_kind"], row["source_dataset_id"], row["source_item_id"] or row["image_id"])


def _validate_output_rows(rows: list[dict[str, str]], columns: list[str], id_column: str, messages: list[str]) -> None:
    ids = Counter(row.get(id_column, "") for row in rows)
    for index, row in enumerate(rows, start=1):
        keys = list(row.keys())
        if keys != columns:
            messages.append(f"output {id_column} row {index}: expected schema {columns}, found {keys}")
        if ids[row.get(id_column, "")] > 1:
            messages.append(f"output {id_column} row {index}: duplicate {id_column} {row.get(id_column)!r}")


def _validate_classification_conflicts(rows: list[dict[str, str]], messages: list[str]) -> None:
    by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        identity = _classification_identity(row)
        previous = by_identity.get(identity)
        if previous is None:
            by_identity[identity] = row
            continue
        if previous["label"] != row["label"]:
            messages.append(f"classification output identity {identity!r} has conflicting labels {previous['label']!r} and {row['label']!r}")
        else:
            messages.append(f"classification output identity {identity!r} is duplicated")


def _row_identity_message(row: dict[str, str]) -> str:
    stable_id = row.get("dataset_row_id") or row.get("gate_row_id") or "<missing row id>"
    return (
        f"{stable_id} source_kind={row.get('source_kind')} source_dataset_id={row.get('source_dataset_id')} "
        f"source_item_id={row.get('source_item_id')} image_id={row.get('image_id')} relative_path={row.get('relative_path')}"
    )


def _validate_unique_non_empty_classification_sha(rows: list[dict[str, str]], messages: list[str]) -> None:
    by_sha: dict[str, dict[str, str]] = {}
    for row in rows:
        digest = row.get("sha256", "")
        if not digest:
            continue
        previous = by_sha.get(digest)
        if previous is None:
            by_sha[digest] = row
            continue
        message = (
            f"duplicate non-empty sha256 {digest}: {_row_identity_message(previous)} label={previous['label']} "
            f"and {_row_identity_message(row)} label={row['label']}"
        )
        if previous["label"] != row["label"]:
            message += f"; conflicting labels {previous['label']} and {row['label']}"
        messages.append(message)


def _validate_no_quarantined_sha_reentry(rows: list[dict[str, str]], quarantine_rows: list[dict[str, str]], messages: list[str]) -> None:
    labels_by_sha: dict[str, set[str]] = {}
    for row in quarantine_rows:
        labels_by_sha.setdefault(row["sha256"], set()).add(row["label"])
    for row in rows:
        digest = row.get("sha256", "")
        if row.get("source_kind") == "trashnet" or not digest or digest not in labels_by_sha:
            continue
        labels = sorted(labels_by_sha[digest])
        messages.append(
            f"{_row_identity_message(row)} has sha256={digest}, which is quarantined due to conflicting TrashNet labels {labels}"
        )


def _immutable_identity_message(enrichment_row: dict[str, str], immutable_row: dict[str, str], reason: str) -> str:
    return (
        f"{_row_identity_message(enrichment_row)} reuses TrashNet validation/test {reason} from "
        f"{_row_identity_message(immutable_row)}; validation/test splits are immutable"
    )


def _validate_enrichment_does_not_reuse_validation_test_identities(
    classification_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    immutable_trashnet_rows: list[dict[str, str]],
    messages: list[str],
) -> None:
    immutable_rows = [row for row in immutable_trashnet_rows if row["split"] in {"validation", "test"}]
    immutable_by_sha = {row["sha256"]: row for row in immutable_rows if row.get("sha256")}
    immutable_by_image_id = {row["image_id"]: row for row in immutable_rows if row.get("image_id")}
    immutable_by_context_path = {
        (row["source_dataset_id"], _normalized_relative_path(row["relative_path"])): row
        for row in immutable_rows
        if row.get("source_dataset_id") and row.get("relative_path")
    }

    enrichment_rows = [row for row in classification_rows if row["source_kind"] != "trashnet"]
    enrichment_rows.extend(row for row in gate_rows if row["source_kind"] != "trashnet")
    for row in enrichment_rows:
        digest = row.get("sha256", "")
        if digest and digest in immutable_by_sha:
            messages.append(_immutable_identity_message(row, immutable_by_sha[digest], f"sha256={digest}"))
            continue
        candidate_ids = [row.get("image_id", ""), row.get("source_item_id", "")]
        matching_image_id = next((candidate_id for candidate_id in candidate_ids if candidate_id and candidate_id in immutable_by_image_id), None)
        if matching_image_id:
            messages.append(_immutable_identity_message(row, immutable_by_image_id[matching_image_id], f"image_id={matching_image_id}"))
            continue
        context_path = (row.get("source_dataset_id", ""), _normalized_relative_path(row.get("relative_path", "")))
        if context_path in immutable_by_context_path:
            messages.append(_immutable_identity_message(row, immutable_by_context_path[context_path], f"relative_path={context_path[1]} in source_dataset_id={context_path[0]}"))


def _validate_gate_duplicates(rows: list[dict[str, str]], messages: list[str]) -> None:
    identities = Counter(_gate_identity(row) for row in rows)
    for identity, count in identities.items():
        if count > 1:
            messages.append(f"gate output identity {identity!r} is duplicated")


def _trashnet_classification_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "dataset_row_id": _stable_id("v2cls", "trashnet", row["image_id"], row["relative_path"]),
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
        "label": row["label"],
        "split": row["split"],
        "source_kind": "trashnet",
        "source_dataset_id": "trashnet",
        "source_split": row["split"],
        "source_item_id": row["image_id"],
        "source_url": "",
        "source_license": "",
        "source_license_reference": "",
        "source_attribution": "",
        "license_status": "",
        "source_annotation_id": "",
        "object_area_ratio": "",
        "parent_feedback_id": "",
        "original_label": row.get("source_class", ""),
        "mapping_rule_id": "",
        "source_commit": row.get("source_commit", ""),
        "sha256": row.get("sha256", ""),
    }


def _feedback_classification_row(row: dict[str, str], feedback_image_root: str | Path | None) -> dict[str, str]:
    return {
        "dataset_row_id": _stable_id("v2cls", "feedback", row["feedback_id"]),
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
        "label": row["confirmed_label"],
        "split": ENRICHMENT_SPLIT,
        "source_kind": "feedback",
        "source_dataset_id": row["source"],
        "source_split": row["source_split"],
        "source_item_id": row["image_id"],
        "source_url": "",
        "source_license": "",
        "source_license_reference": "",
        "source_attribution": "",
        "license_status": "",
        "source_annotation_id": "",
        "object_area_ratio": "",
        "parent_feedback_id": row["feedback_id"],
        "original_label": "",
        "mapping_rule_id": "",
        "source_commit": "",
        "sha256": _sha256_if_available(feedback_image_root, row["relative_path"]),
    }


def _public_image_id(row: dict[str, str]) -> str:
    return _stable_id("publicimg", row["source_dataset_id"], row["source_item_id"])


def _public_classification_row(row: dict[str, str], public_image_root: str | Path | None) -> dict[str, str]:
    return {
        "dataset_row_id": _stable_id("v2cls", "public", row["public_row_id"], row["source_dataset_id"], row["source_item_id"]),
        "image_id": _public_image_id(row),
        "relative_path": row["relative_path"],
        "label": row["mapped_label"],
        "split": ENRICHMENT_SPLIT,
        "source_kind": "public_dataset",
        "source_dataset_id": row["source_dataset_id"],
        "source_split": "public_dataset",
        "source_item_id": row["source_item_id"],
        "source_url": row["source_url"],
        "source_license": row["source_license"],
        "source_license_reference": row["source_license_reference"],
        "source_attribution": row["source_attribution"],
        "license_status": row["license_status"],
        "source_annotation_id": row["source_annotation_id"],
        "object_area_ratio": row["object_area_ratio"],
        "parent_feedback_id": "",
        "original_label": row["source_label"],
        "mapping_rule_id": row["mapping_rule_id"],
        "source_commit": "",
        "sha256": _sha256_if_available(public_image_root, row["relative_path"]),
    }


def _feedback_gate_row(row: dict[str, str], feedback_image_root: str | Path | None) -> dict[str, str]:
    return {
        "gate_row_id": _stable_id("v2gate", "feedback", row["feedback_id"]),
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
        "auto_route_eligible": row["auto_route_eligible"],
        "review_reason": row["review_reason"],
        "split": ENRICHMENT_SPLIT,
        "source_kind": "feedback",
        "source_dataset_id": row["source"],
        "source_split": row["source_split"],
        "source_item_id": row["image_id"],
        "source_url": "",
        "source_license": "",
        "source_license_reference": "",
        "source_attribution": "",
        "license_status": "",
        "source_annotation_id": "",
        "object_area_ratio": "",
        "parent_feedback_id": row["feedback_id"],
        "annotation_type": "",
        "object_count": "",
        "sha256": _sha256_if_available(feedback_image_root, row["relative_path"]),
    }


def _public_gate_row(row: dict[str, str], public_image_root: str | Path | None) -> dict[str, str]:
    return {
        "gate_row_id": _stable_id("v2gate", "public", row["public_row_id"], row["source_dataset_id"], row["source_item_id"]),
        "image_id": _public_image_id(row),
        "relative_path": row["relative_path"],
        "auto_route_eligible": row["auto_route_eligible"],
        "review_reason": row["review_reason"],
        "split": ENRICHMENT_SPLIT,
        "source_kind": "public_dataset",
        "source_dataset_id": row["source_dataset_id"],
        "source_split": "public_dataset",
        "source_item_id": row["source_item_id"],
        "source_url": row["source_url"],
        "source_license": row["source_license"],
        "source_license_reference": row["source_license_reference"],
        "source_attribution": row["source_attribution"],
        "license_status": row["license_status"],
        "source_annotation_id": row["source_annotation_id"],
        "object_area_ratio": row["object_area_ratio"],
        "parent_feedback_id": "",
        "annotation_type": row["annotation_type"],
        "object_count": row["object_count"],
        "sha256": _sha256_if_available(public_image_root, row["relative_path"]),
    }


def _sort_classification_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    source_order = {"trashnet": 0, "feedback": 1, "public_dataset": 2}
    split_order = {"train": 0, "validation": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            source_order.get(row["source_kind"], 99),
            split_order.get(row["split"], 99),
            row["source_dataset_id"],
            row["source_item_id"],
            row["dataset_row_id"],
        ),
    )


def _sort_gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    source_order = {"feedback": 0, "public_dataset": 1}
    return sorted(
        rows,
        key=lambda row: (
            source_order.get(row["source_kind"], 99),
            row["source_dataset_id"],
            row["source_item_id"],
            row["gate_row_id"],
        ),
    )


def build_v2_dataset(
    *,
    trashnet_manifest: str | Path,
    classifier_feedback_manifest: str | Path,
    gate_feedback_manifest: str | Path,
    output_dir: str | Path,
    public_classifier_manifest: str | Path | None = None,
    public_gate_manifest: str | Path | None = None,
    label_mapping: str | Path | None = None,
    allow_promoted_external_diagnostic: bool = False,
    feedback_image_root: str | Path | None = None,
    public_image_root: str | Path | None = None,
) -> dict:
    messages: list[str] = []
    source_manifest_sha256 = sha256_file(trashnet_manifest)
    trashnet_rows = _read_csv_exact(trashnet_manifest, TRASHNET_COLUMNS, "trashnet manifest")
    classifier_feedback_rows = _read_csv_exact(classifier_feedback_manifest, CLASSIFIER_CANDIDATE_COLUMNS, "classifier feedback manifest")
    gate_feedback_rows = _read_csv_exact(gate_feedback_manifest, GATE_CANDIDATE_COLUMNS, "gate feedback manifest")
    public_classifier_rows = _read_optional_csv(public_classifier_manifest, PUBLIC_CLASSIFIER_COLUMNS, "public classifier manifest")
    public_gate_rows = _read_optional_csv(public_gate_manifest, PUBLIC_GATE_COLUMNS, "public gate manifest")
    label_mapping_rows = _read_optional_csv(label_mapping, LABEL_MAPPING_COLUMNS, "label mapping")

    _validate_trashnet_rows(trashnet_rows, messages)
    _validate_feedback_classifier_rows(classifier_feedback_rows, messages)
    _validate_feedback_gate_rows(gate_feedback_rows, messages)
    mappings = _validate_label_mappings(label_mapping_rows, messages)
    _validate_public_classifier_rows(public_classifier_rows, mappings, messages)
    _validate_public_gate_rows(public_gate_rows, messages)

    promoted_feedback_ids = _external_diagnostic_feedback_ids(classifier_feedback_rows, gate_feedback_rows)
    if promoted_feedback_ids and not allow_promoted_external_diagnostic:
        messages.extend(
            [
                "external_diagnostic_v1 rows would be included in V2 outputs.",
                "Use --allow-promoted-external-diagnostic only when intentionally promoting inspected diagnostic images.",
            ]
        )

    if messages:
        raise V2DatasetError(messages)

    curated_trashnet_rows, quarantine_rows, conflict_shas = _curate_trashnet_label_conflicts(trashnet_rows, source_manifest_sha256)

    classification_rows = [_trashnet_classification_row(row) for row in curated_trashnet_rows]
    classification_rows.extend(_feedback_classification_row(row, feedback_image_root) for row in classifier_feedback_rows)
    classification_rows.extend(
        _public_classification_row(row, public_image_root)
        for row in public_classifier_rows
        if row["approved_for_classifier_training"] == "true"
    )
    gate_rows = [_feedback_gate_row(row, feedback_image_root) for row in gate_feedback_rows]
    gate_rows.extend(_public_gate_row(row, public_image_root) for row in public_gate_rows if row["approved_for_gate_training"] == "true")

    output_messages: list[str] = []
    _validate_output_rows(classification_rows, CLASSIFICATION_OUTPUT_COLUMNS, "dataset_row_id", output_messages)
    _validate_output_rows(gate_rows, GATE_OUTPUT_COLUMNS, "gate_row_id", output_messages)
    _validate_output_rows(quarantine_rows, TRASHNET_QUARANTINE_COLUMNS, "quarantine_id", output_messages)
    _validate_classification_conflicts(classification_rows, output_messages)
    _validate_no_quarantined_sha_reentry(classification_rows, quarantine_rows, output_messages)
    _validate_unique_non_empty_classification_sha(classification_rows, output_messages)
    immutable_trashnet_rows = [_trashnet_classification_row(row) for row in trashnet_rows if row["split"] in {"validation", "test"}]
    _validate_enrichment_does_not_reuse_validation_test_identities(classification_rows, gate_rows, immutable_trashnet_rows, output_messages)
    _validate_gate_duplicates(gate_rows, output_messages)
    if output_messages:
        raise V2DatasetError(output_messages)

    classification_rows = _sort_classification_rows(classification_rows)
    gate_rows = _sort_gate_rows(gate_rows)

    output_dir = ensure_dir(output_dir)
    _write_csv(output_dir / "v2_classification_manifest.csv", CLASSIFICATION_OUTPUT_COLUMNS, classification_rows)
    _write_csv(output_dir / "v2_gate_manifest.csv", GATE_OUTPUT_COLUMNS, gate_rows)
    _write_csv(output_dir / "v2_trashnet_quarantine_report.csv", TRASHNET_QUARANTINE_COLUMNS, quarantine_rows)

    leakage_notice_written = False
    if promoted_feedback_ids and allow_promoted_external_diagnostic:
        write_text(output_dir / "v2_leakage_notice.md", V2_LEAKAGE_NOTICE)
        leakage_notice_written = True

    report = {
        "classification_rows": len(classification_rows),
        "gate_rows": len(gate_rows),
        "classification_split_counts": dict(Counter(row["split"] for row in classification_rows)),
        "classification_source_kind_counts": dict(Counter(row["source_kind"] for row in classification_rows)),
        "gate_split_counts": dict(Counter(row["split"] for row in gate_rows)),
        "gate_source_kind_counts": dict(Counter(row["source_kind"] for row in gate_rows)),
        "promoted_external_diagnostic_rows": len(promoted_feedback_ids),
        "allow_promoted_external_diagnostic": allow_promoted_external_diagnostic,
        "leakage_notice_written": leakage_notice_written,
        "public_classifier_input_rows": len(public_classifier_rows),
        "public_gate_input_rows": len(public_gate_rows),
        "trashnet_input_rows": len(trashnet_rows),
        "trashnet_included_rows": len(curated_trashnet_rows),
        "trashnet_quarantined_rows": len(quarantine_rows),
        "trashnet_label_conflict_sha_groups": len(conflict_shas),
        "trashnet_quarantine_reason_counts": dict(Counter(row["quarantine_reason"] for row in quarantine_rows)),
    }
    write_json(output_dir / "v2_dataset_report.json", report)
    return report
