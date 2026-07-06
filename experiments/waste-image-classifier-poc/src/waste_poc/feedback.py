from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path, PurePosixPath

from .utils import CLASS_NAMES, ensure_dir, write_json, write_text

FEEDBACK_COLUMNS = [
    "feedback_id",
    "image_id",
    "relative_path",
    "source",
    "source_split",
    "model_version",
    "predicted_label",
    "calibrated_confidence",
    "original_decision",
    "original_threshold",
    "human_outcome",
    "confirmed_label",
    "auto_route_eligible",
    "review_reason",
    "approved_for_classifier_training",
    "approved_for_gate_training",
    "annotation_notes",
]

CLASSIFIER_CANDIDATE_COLUMNS = [
    "feedback_id",
    "image_id",
    "relative_path",
    "confirmed_label",
    "source",
    "source_split",
    "model_version",
]

GATE_CANDIDATE_COLUMNS = [
    "feedback_id",
    "image_id",
    "relative_path",
    "auto_route_eligible",
    "review_reason",
    "source",
    "source_split",
    "model_version",
]

SOURCES = {"external_diagnostic", "workflow_feedback", "manual_capture", "public_dataset"}
SOURCE_SPLITS = {"external_diagnostic_v1", "workflow_feedback", "manual_capture", "public_dataset"}
SOURCE_SPLIT_BY_SOURCE = {
    "external_diagnostic": "external_diagnostic_v1",
    "workflow_feedback": "workflow_feedback",
    "manual_capture": "manual_capture",
    "public_dataset": "public_dataset",
}
ORIGINAL_DECISIONS = {"auto_route", "needs_review"}
HUMAN_OUTCOMES = {"confirmed", "corrected", "rejected", "unresolved"}
REVIEW_REASONS = {
    "none",
    "wrong_material",
    "multiple_objects",
    "ambiguous_scene",
    "unsupported_material",
    "non_waste",
    "low_quality",
    "other",
}
BOOLEAN_FIELDS = {"auto_route_eligible", "approved_for_classifier_training", "approved_for_gate_training"}
AUTO_ROUTE_INELIGIBLE_REASONS = {"multiple_objects", "ambiguous_scene", "unsupported_material", "non_waste", "low_quality"}
EXTERNAL_DIAGNOSTIC_V1 = "external_diagnostic_v1"
LEAKAGE_NOTICE = """# External Diagnostic Promotion Notice

Rows from `external_diagnostic_v1` were explicitly promoted into feedback-derived candidate datasets.

That diagnostic set can no longer be treated as a final or comparative benchmark for future model selection. Any future selection or final evaluation will require a new untouched external final set.
"""


class FeedbackValidationError(ValueError):
    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def _row_label(row_number: int, row: dict[str, str]) -> str:
    feedback_id = row.get("feedback_id") or "<blank feedback_id>"
    return f"row {row_number} feedback_id={feedback_id}"


def _is_true(value: str) -> bool:
    return value == "true"


def _valid_probability(value: str) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= parsed <= 1.0


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts not in {(), (".",)}


def read_feedback_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FEEDBACK_COLUMNS:
            found = reader.fieldnames or []
            missing = [column for column in FEEDBACK_COLUMNS if column not in found]
            unknown = [column for column in found if column not in FEEDBACK_COLUMNS]
            messages = [f"header: expected exact columns {FEEDBACK_COLUMNS}, found {found}"]
            if missing:
                messages.append(f"header: missing columns {missing}")
            if unknown:
                messages.append(f"header: unknown columns {unknown}")
            raise FeedbackValidationError(messages)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def validate_feedback_rows(rows: list[dict[str, str]]) -> list[str]:
    messages: list[str] = []
    feedback_ids: Counter[str] = Counter(row.get("feedback_id", "") for row in rows)
    image_ids: Counter[str] = Counter(row.get("image_id", "") for row in rows)

    for index, row in enumerate(rows, start=2):
        label = _row_label(index, row)
        for field in FEEDBACK_COLUMNS:
            if field not in {"confirmed_label", "annotation_notes"} and row.get(field, "") == "":
                messages.append(f"{label}: {field} is required")

        if feedback_ids[row.get("feedback_id", "")] > 1:
            messages.append(f"{label}: duplicate feedback_id {row.get('feedback_id')!r}")
        if image_ids[row.get("image_id", "")] > 1:
            messages.append(f"{label}: duplicate image_id {row.get('image_id')!r}")

        predicted_label = row.get("predicted_label", "")
        confirmed_label = row.get("confirmed_label", "")
        human_outcome = row.get("human_outcome", "")
        review_reason = row.get("review_reason", "")

        if predicted_label not in CLASS_NAMES:
            messages.append(f"{label}: predicted_label must be one of {CLASS_NAMES}")
        if confirmed_label and confirmed_label not in CLASS_NAMES:
            messages.append(f"{label}: confirmed_label must be blank or one of {CLASS_NAMES}")
        source = row.get("source")
        source_split = row.get("source_split")
        if source not in SOURCES:
            messages.append(f"{label}: source must be one of {sorted(SOURCES)}")
        if source_split not in SOURCE_SPLITS:
            messages.append(f"{label}: source_split must be one of {sorted(SOURCE_SPLITS)}")
        if source in SOURCE_SPLIT_BY_SOURCE and source_split in SOURCE_SPLITS and source_split != SOURCE_SPLIT_BY_SOURCE[source]:
            messages.append(f"{label}: source={source} requires source_split={SOURCE_SPLIT_BY_SOURCE[source]}")
        if row.get("original_decision") not in ORIGINAL_DECISIONS:
            messages.append(f"{label}: original_decision must be one of {sorted(ORIGINAL_DECISIONS)}")
        if human_outcome not in HUMAN_OUTCOMES:
            messages.append(f"{label}: human_outcome must be one of {sorted(HUMAN_OUTCOMES)}")
        if review_reason not in REVIEW_REASONS:
            messages.append(f"{label}: review_reason must be one of {sorted(REVIEW_REASONS)}")
        for field in BOOLEAN_FIELDS:
            if row.get(field) not in {"true", "false"}:
                messages.append(f"{label}: {field} must be lowercase true or false")

        if not _valid_probability(row.get("calibrated_confidence", "")):
            messages.append(f"{label}: calibrated_confidence must be in [0, 1]")
        if not _valid_probability(row.get("original_threshold", "")):
            messages.append(f"{label}: original_threshold must be in [0, 1]")
        if not _safe_relative_path(row.get("relative_path", "")):
            messages.append(f"{label}: relative_path must be relative and must not use parent traversal")

        if human_outcome == "confirmed" and confirmed_label != predicted_label:
            messages.append(f"{label}: confirmed rows must have confirmed_label equal to predicted_label")
        if human_outcome == "corrected" and confirmed_label not in CLASS_NAMES:
            messages.append(f"{label}: corrected rows require a valid confirmed_label")
        if human_outcome == "rejected" and confirmed_label:
            messages.append(f"{label}: rejected rows must leave confirmed_label blank")
        if human_outcome == "rejected" and _is_true(row.get("auto_route_eligible", "")):
            messages.append(f"{label}: rejected rows must have auto_route_eligible=false")
        if human_outcome == "unresolved" and (_is_true(row.get("approved_for_classifier_training", "")) or _is_true(row.get("approved_for_gate_training", ""))):
            messages.append(f"{label}: unresolved rows cannot be approved for training datasets")
        if _is_true(row.get("auto_route_eligible", "")) and review_reason in AUTO_ROUTE_INELIGIBLE_REASONS:
            messages.append(f"{label}: auto_route_eligible=true conflicts with review_reason={review_reason}")
        if _is_true(row.get("approved_for_classifier_training", "")) and not (human_outcome in {"confirmed", "corrected"} and confirmed_label in CLASS_NAMES):
            messages.append(f"{label}: approved_for_classifier_training=true requires confirmed or corrected outcome with valid confirmed_label")
        if _is_true(row.get("approved_for_gate_training", "")) and human_outcome == "unresolved":
            messages.append(f"{label}: approved_for_gate_training=true requires a reviewed row")

    return messages


def validate_feedback_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_feedback_manifest(path)
    messages = validate_feedback_rows(rows)
    if messages:
        raise FeedbackValidationError(messages)
    return rows


def classifier_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        if _is_true(row["approved_for_classifier_training"]) and row["human_outcome"] in {"confirmed", "corrected"} and row["confirmed_label"] in CLASS_NAMES:
            candidates.append({column: row[column] for column in CLASSIFIER_CANDIDATE_COLUMNS})
    return candidates


def gate_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        if _is_true(row["approved_for_gate_training"]) and row["human_outcome"] != "unresolved":
            candidates.append({column: row[column] for column in GATE_CANDIDATE_COLUMNS})
    return candidates


def write_csv(path: str | Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_feedback_candidates(manifest_path: str | Path, output_dir: str | Path, allow_promoted_external_diagnostic: bool = False) -> dict:
    rows = validate_feedback_manifest(manifest_path)
    classification_rows = classifier_candidate_rows(rows)
    gate_rows = gate_candidate_rows(rows)
    promoted_feedback_ids = {row["feedback_id"] for row in [*classification_rows, *gate_rows] if row["source_split"] == EXTERNAL_DIAGNOSTIC_V1}
    if promoted_feedback_ids and not allow_promoted_external_diagnostic:
        raise FeedbackValidationError(
            [
                "external_diagnostic_v1 rows would be included in candidate outputs.",
                "Use --allow-promoted-external-diagnostic only when intentionally promoting inspected diagnostic images.",
            ]
        )

    output_dir = ensure_dir(output_dir)
    write_csv(output_dir / "classification_feedback_manifest.csv", CLASSIFIER_CANDIDATE_COLUMNS, classification_rows)
    write_csv(output_dir / "review_gate_feedback_manifest.csv", GATE_CANDIDATE_COLUMNS, gate_rows)
    leakage_notice_written = False
    if promoted_feedback_ids and allow_promoted_external_diagnostic:
        write_text(output_dir / "leakage_notice.md", LEAKAGE_NOTICE)
        leakage_notice_written = True

    summary = {
        "feedback_rows": len(rows),
        "classification_candidate_rows": len(classification_rows),
        "review_gate_candidate_rows": len(gate_rows),
        "promoted_external_diagnostic_rows": len(promoted_feedback_ids),
        "allow_promoted_external_diagnostic": allow_promoted_external_diagnostic,
        "leakage_notice_written": leakage_notice_written,
        "source_split_counts": dict(Counter(row["source_split"] for row in rows)),
    }
    write_json(output_dir / "candidate_summary.json", summary)
    return summary
