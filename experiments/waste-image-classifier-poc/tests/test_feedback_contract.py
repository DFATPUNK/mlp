from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from waste_poc.feedback import (
    CLASSIFIER_CANDIDATE_COLUMNS,
    GATE_CANDIDATE_COLUMNS,
    FeedbackValidationError,
    build_feedback_candidates,
    read_feedback_manifest,
    validate_feedback_manifest,
)


HEADER = [
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


def base_row(**overrides):
    row = {
        "feedback_id": "fb_001",
        "image_id": "img_001",
        "relative_path": "images/item.jpg",
        "source": "workflow_feedback",
        "source_split": "workflow_feedback",
        "model_version": "poc_model",
        "predicted_label": "plastic",
        "calibrated_confidence": "0.91",
        "original_decision": "auto_route",
        "original_threshold": "0.90",
        "human_outcome": "confirmed",
        "confirmed_label": "plastic",
        "auto_route_eligible": "true",
        "review_reason": "none",
        "approved_for_classifier_training": "true",
        "approved_for_gate_training": "true",
        "annotation_notes": "",
    }
    row.update(overrides)
    return row


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FeedbackContractTests(unittest.TestCase):
    def validate_rows(self, rows: list[dict[str, str]]):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.csv"
            write_manifest(path, rows)
            return validate_feedback_manifest(path)

    def test_valid_confirmed_row_passes(self):
        rows = self.validate_rows([base_row()])
        self.assertEqual(rows[0]["human_outcome"], "confirmed")

    def test_valid_corrected_row_passes(self):
        rows = self.validate_rows(
            [
                base_row(
                    human_outcome="corrected",
                    predicted_label="glass",
                    confirmed_label="plastic",
                    review_reason="wrong_material",
                )
            ]
        )
        self.assertEqual(rows[0]["confirmed_label"], "plastic")

    def test_invalid_corrected_row_without_confirmed_label_fails(self):
        with self.assertRaisesRegex(FeedbackValidationError, "corrected rows require"):
            self.validate_rows([base_row(human_outcome="corrected", confirmed_label="", review_reason="wrong_material")])

    def test_duplicate_image_id_fails(self):
        with self.assertRaisesRegex(FeedbackValidationError, "duplicate image_id"):
            self.validate_rows([base_row(feedback_id="fb_001"), base_row(feedback_id="fb_002")])

    def test_unsafe_relative_path_fails(self):
        for path in ["/absolute/image.jpg", "../outside.jpg", "images/../outside.jpg"]:
            with self.subTest(path=path):
                with self.assertRaisesRegex(FeedbackValidationError, "relative_path"):
                    self.validate_rows([base_row(relative_path=path)])

    def test_invalid_probability_and_threshold_fail(self):
        with self.assertRaisesRegex(FeedbackValidationError, "calibrated_confidence"):
            self.validate_rows([base_row(calibrated_confidence="1.2")])
        with self.assertRaisesRegex(FeedbackValidationError, "original_threshold"):
            self.validate_rows([base_row(original_threshold="-0.1")])

    def test_unresolved_row_approved_for_training_fails(self):
        with self.assertRaisesRegex(FeedbackValidationError, "unresolved rows cannot be approved"):
            self.validate_rows(
                [
                    base_row(
                        human_outcome="unresolved",
                        confirmed_label="",
                        approved_for_classifier_training="false",
                        approved_for_gate_training="true",
                        auto_route_eligible="false",
                        review_reason="other",
                    )
                ]
            )

    def test_source_and_source_split_mismatch_fails(self):
        with self.assertRaisesRegex(FeedbackValidationError, "source=workflow_feedback requires source_split=workflow_feedback"):
            self.validate_rows([base_row(source="workflow_feedback", source_split="manual_capture")])

    def test_rejected_row_must_not_be_auto_route_eligible(self):
        with self.assertRaisesRegex(FeedbackValidationError, "rejected rows must have auto_route_eligible=false"):
            self.validate_rows(
                [
                    base_row(
                        human_outcome="rejected",
                        confirmed_label="",
                        auto_route_eligible="true",
                        review_reason="other",
                        approved_for_classifier_training="false",
                        approved_for_gate_training="true",
                    )
                ]
            )

    def test_builder_rejects_external_diagnostic_candidates_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "feedback.csv"
            write_manifest(
                manifest,
                [
                    base_row(
                        source="external_diagnostic",
                        source_split="external_diagnostic_v1",
                    )
                ],
            )
            with self.assertRaisesRegex(FeedbackValidationError, "external_diagnostic_v1"):
                build_feedback_candidates(manifest, Path(tmp) / "out")

    def test_builder_includes_external_diagnostic_only_with_opt_in_and_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "feedback.csv"
            output_dir = root / "out"
            write_manifest(
                manifest,
                [
                    base_row(
                        source="external_diagnostic",
                        source_split="external_diagnostic_v1",
                    )
                ],
            )
            summary = build_feedback_candidates(manifest, output_dir, allow_promoted_external_diagnostic=True)
            classification = read_csv(output_dir / "classification_feedback_manifest.csv")
            notice_exists = (output_dir / "leakage_notice.md").exists()

        self.assertEqual(summary["promoted_external_diagnostic_rows"], 1)
        self.assertEqual(classification[0]["source_split"], "external_diagnostic_v1")
        self.assertTrue(notice_exists)

    def test_builder_outputs_only_intended_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "feedback.csv"
            output_dir = root / "out"
            write_manifest(
                manifest,
                [
                    base_row(feedback_id="fb_confirmed", image_id="img_confirmed"),
                    base_row(
                        feedback_id="fb_rejected",
                        image_id="img_rejected",
                        human_outcome="rejected",
                        confirmed_label="",
                        auto_route_eligible="false",
                        review_reason="multiple_objects",
                        approved_for_classifier_training="false",
                        approved_for_gate_training="true",
                    ),
                    base_row(
                        feedback_id="fb_unapproved",
                        image_id="img_unapproved",
                        approved_for_classifier_training="false",
                        approved_for_gate_training="false",
                    ),
                ],
            )
            build_feedback_candidates(manifest, output_dir)
            classification = read_csv(output_dir / "classification_feedback_manifest.csv")
            gate = read_csv(output_dir / "review_gate_feedback_manifest.csv")

        self.assertEqual(list(classification[0]), CLASSIFIER_CANDIDATE_COLUMNS)
        self.assertEqual([row["feedback_id"] for row in classification], ["fb_confirmed"])
        self.assertEqual(list(gate[0]), GATE_CANDIDATE_COLUMNS)
        self.assertEqual([row["feedback_id"] for row in gate], ["fb_confirmed", "fb_rejected"])

    def test_template_manifest_has_exact_contract_header(self):
        template = Path(__file__).resolve().parents[1] / "data" / "feedback" / "feedback_manifest.template.csv"
        rows = read_feedback_manifest(template)
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
