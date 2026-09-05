from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from waste_poc.gate_experiment import (
    GATE_MANIFEST_COLUMNS,
    GateExperimentError,
    build_gate_manifest,
    combine_policy_rows,
)
from waste_poc.utils import CLASS_NAMES
from waste_poc.v2_dataset import CLASSIFICATION_OUTPUT_COLUMNS, GATE_OUTPUT_COLUMNS


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def classification_row(**overrides) -> dict[str, str]:
    row = {column: "" for column in CLASSIFICATION_OUTPUT_COLUMNS}
    row.update(
        {
            "dataset_row_id": "v2cls_trash_1",
            "image_id": "trash_1",
            "relative_path": "cardboard/trash_1.jpg",
            "label": "cardboard",
            "split": "train",
            "source_kind": "trashnet",
            "source_dataset_id": "trashnet",
            "source_split": "train",
            "source_item_id": "trash_1",
        }
    )
    row.update(overrides)
    return row


def gate_row(**overrides) -> dict[str, str]:
    row = {column: "" for column in GATE_OUTPUT_COLUMNS}
    row.update(
        {
            "gate_row_id": "v2gate_feedback_1",
            "image_id": "feedback_1",
            "relative_path": "feedback_1.jpg",
            "auto_route_eligible": "false",
            "review_reason": "multiple_objects",
            "split": "train",
            "source_kind": "feedback",
            "source_dataset_id": "workflow_feedback",
            "source_split": "workflow_feedback",
            "source_item_id": "feedback_1",
        }
    )
    row.update(overrides)
    return row


class GateManifestTests(unittest.TestCase):
    def make_inputs(self, tmp: Path, gate_rows: list[dict[str, str]] | None = None) -> dict[str, Path]:
        trashnet_source = tmp / "trashnet-source"
        trashnet_root = trashnet_source / "data" / "dataset-resized"
        for label in CLASS_NAMES:
            (trashnet_root / label).mkdir(parents=True, exist_ok=True)
        (trashnet_root / "cardboard" / "trash_1.jpg").write_bytes(b"trashnet")

        feedback_root = tmp / "feedback-images"
        feedback_root.mkdir()
        (feedback_root / "feedback_1.jpg").write_bytes(b"feedback")

        public_root = tmp / "public-images"
        public_root.mkdir()
        (public_root / "taco_1.jpg").write_bytes(b"public")

        rows = gate_rows
        if rows is None:
            rows = [
                gate_row(),
                gate_row(
                    gate_row_id="v2gate_public_1",
                    image_id="taco_1",
                    relative_path="taco_1.jpg",
                    source_kind="public_dataset",
                    source_dataset_id="taco",
                    source_split="public_dataset",
                    source_item_id="taco_1",
                    review_reason="ambiguous_scene",
                ),
            ]
        return {
            "classification": write_csv(
                tmp / "v2_classification_manifest.csv",
                CLASSIFICATION_OUTPUT_COLUMNS,
                [classification_row()],
            ),
            "gate": write_csv(tmp / "v2_gate_manifest.csv", GATE_OUTPUT_COLUMNS, rows),
            "trashnet": trashnet_source,
            "feedback": feedback_root,
            "public": public_root,
            "output": tmp / "output",
        }

    def build(self, tmp: Path, paths: dict[str, Path], source_filter: str) -> dict:
        return build_gate_manifest(
            v2_classification_manifest=paths["classification"],
            v2_gate_manifest=paths["gate"],
            output_dir=paths["output"],
            negative_source_filter=source_filter,
            trashnet_image_root=paths["trashnet"],
            feedback_image_root=paths["feedback"],
            public_image_root=paths["public"],
            poc_root=tmp,
        )

    def test_feedback_filter_builds_positive_and_feedback_negative_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = self.make_inputs(tmp)
            report = self.build(tmp, paths, "feedback")
            rows = read_csv(paths["output"] / "gate_manifest.csv")

        self.assertEqual(report["positive_rows"], 1)
        self.assertEqual(report["negative_rows"], 1)
        self.assertEqual([row["auto_route_eligible"] for row in rows], ["true", "false"])
        self.assertEqual([row["source_kind"] for row in rows], ["trashnet", "feedback"])
        self.assertEqual(list(rows[0]), GATE_MANIFEST_COLUMNS)

    def test_all_filter_includes_feedback_and_public_negative_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = self.make_inputs(tmp)
            report = self.build(tmp, paths, "all")
            rows = read_csv(paths["output"] / "gate_manifest.csv")

        self.assertEqual(report["positive_rows"], 1)
        self.assertEqual(report["negative_rows"], 2)
        self.assertEqual({row["source_kind"] for row in rows}, {"trashnet", "feedback", "public_dataset"})
        self.assertTrue(all(row["auto_route_eligible"] == "false" for row in rows if row["source_kind"] != "trashnet"))

    def test_missing_images_raise_clear_error_with_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = self.make_inputs(tmp)
            (paths["feedback"] / "feedback_1.jpg").unlink()

            with self.assertRaisesRegex(GateExperimentError, "Gate manifest images are missing") as raised:
                self.build(tmp, paths, "feedback")

        self.assertIn("feedback_1.jpg", str(raised.exception))

    def test_duplicate_gate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = self.make_inputs(tmp, gate_rows=[gate_row(), gate_row()])

            with self.assertRaisesRegex(GateExperimentError, "Duplicate gate_example_id"):
                self.build(tmp, paths, "feedback")


class CombinedPolicyTests(unittest.TestCase):
    def baseline_row(self, image_id: str, action: str) -> dict[str, str]:
        return {
            "image_id": image_id,
            "relative_path": f"{image_id}.jpg",
            "expected_label": "",
            "expected_routing": "needs_review",
            "predicted_label": "plastic",
            "recommended_action": action,
            "route": "plastic" if action == "auto_route" else "needs_review",
        }

    def gate_prediction(self, image_id: str, eligible: str) -> dict[str, str]:
        return {
            "image_id": image_id,
            "relative_path": f"{image_id}.jpg",
            "predicted_auto_route_eligible": eligible,
            "gate_probability_auto_route_eligible": "0.9" if eligible == "true" else "0.1",
            "gate_decision": "allow_classifier" if eligible == "true" else "needs_review",
        }

    def test_gate_false_overrides_classifier_auto_route(self):
        combined, report = combine_policy_rows(
            [self.baseline_row("image_1", "auto_route")],
            [self.gate_prediction("image_1", "false")],
        )

        self.assertEqual(combined[0]["combined_policy_route"], "needs_review")
        self.assertEqual(report["baseline_unsafe_auto_route_count"], 1)
        self.assertEqual(report["combined_unsafe_auto_route_count"], 0)
        self.assertEqual(report["unsafe_auto_route_delta"], -1)

    def test_gate_true_preserves_classifier_route(self):
        baseline = [
            self.baseline_row("auto", "auto_route"),
            self.baseline_row("review", "needs_review"),
        ]
        gate = [
            self.gate_prediction("auto", "true"),
            self.gate_prediction("review", "true"),
        ]

        combined, _ = combine_policy_rows(baseline, gate)

        self.assertEqual([row["combined_policy_route"] for row in combined], ["auto_route", "needs_review"])


if __name__ == "__main__":
    unittest.main()
