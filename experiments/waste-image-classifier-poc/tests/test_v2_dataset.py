from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from waste_poc.feedback import CLASSIFIER_CANDIDATE_COLUMNS, GATE_CANDIDATE_COLUMNS
from waste_poc.v2_dataset import (
    CLASSIFICATION_OUTPUT_COLUMNS,
    GATE_OUTPUT_COLUMNS,
    LABEL_MAPPING_COLUMNS,
    PUBLIC_CLASSIFIER_COLUMNS,
    PUBLIC_GATE_COLUMNS,
    TRASHNET_COLUMNS,
    V2DatasetError,
    build_v2_dataset,
)


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


def header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def trashnet_row(image_id: str, label: str, split: str, idx: int) -> dict[str, str]:
    return {
        "image_id": image_id,
        "relative_path": f"{label}/{image_id}.jpg",
        "source_class": label,
        "label": label,
        "split": split,
        "sha256": f"sha-{idx}",
        "width": "512",
        "height": "384",
        "format": "JPG",
        "source_commit": "trashnet-commit",
        "is_valid_image": "True",
    }


def base_trashnet_rows() -> list[dict[str, str]]:
    return [
        trashnet_row("trash_train", "plastic", "train", 1),
        trashnet_row("trash_val", "glass", "validation", 2),
        trashnet_row("trash_test", "paper", "test", 3),
    ]


def feedback_classifier_row(**overrides) -> dict[str, str]:
    row = {
        "feedback_id": "fb_cls_1",
        "image_id": "fb_image_1",
        "relative_path": "feedback/fb_image_1.jpg",
        "confirmed_label": "plastic",
        "source": "workflow_feedback",
        "source_split": "workflow_feedback",
        "model_version": "poc_v1",
    }
    row.update(overrides)
    return row


def feedback_gate_row(**overrides) -> dict[str, str]:
    row = {
        "feedback_id": "fb_gate_1",
        "image_id": "fb_gate_image_1",
        "relative_path": "feedback/fb_gate_image_1.jpg",
        "auto_route_eligible": "false",
        "review_reason": "multiple_objects",
        "source": "workflow_feedback",
        "source_split": "workflow_feedback",
        "model_version": "poc_v1",
    }
    row.update(overrides)
    return row


def public_classifier_row(**overrides) -> dict[str, str]:
    row = {
        "public_row_id": "pub_cls_1",
        "source_dataset_id": "fictional_public_v1",
        "source_item_id": "item_1",
        "relative_path": "ingested/fictional_public_v1/item_1.jpg",
        "source_label": "plastic bottle",
        "mapped_label": "plastic",
        "mapping_rule_id": "map_plastic_bottle",
        "annotation_type": "single_object",
        "object_count": "1",
        "approved_for_classifier_training": "true",
        "annotation_notes": "fictional",
    }
    row.update(overrides)
    return row


def public_gate_row(**overrides) -> dict[str, str]:
    row = {
        "public_row_id": "pub_gate_1",
        "source_dataset_id": "fictional_public_v1",
        "source_item_id": "gate_item_1",
        "relative_path": "ingested/fictional_public_v1/gate_item_1.jpg",
        "auto_route_eligible": "false",
        "review_reason": "multiple_objects",
        "annotation_type": "scene",
        "object_count": "3",
        "approved_for_gate_training": "true",
        "annotation_notes": "fictional",
    }
    row.update(overrides)
    return row


def mapping_row(**overrides) -> dict[str, str]:
    row = {
        "mapping_rule_id": "map_plastic_bottle",
        "source_dataset_id": "fictional_public_v1",
        "source_label": "plastic bottle",
        "mapped_label": "plastic",
        "mapping_status": "approved",
        "rationale": "fictional explicit mapping",
    }
    row.update(overrides)
    return row


class V2DatasetTests(unittest.TestCase):
    def make_inputs(
        self,
        tmp: Path,
        *,
        trashnet_rows: list[dict[str, str]] | None = None,
        classifier_feedback_rows: list[dict[str, str]] | None = None,
        gate_feedback_rows: list[dict[str, str]] | None = None,
        public_classifier_rows: list[dict[str, str]] | None = None,
        public_gate_rows: list[dict[str, str]] | None = None,
        mapping_rows: list[dict[str, str]] | None = None,
    ) -> dict[str, Path]:
        return {
            "trashnet": write_csv(tmp / "trashnet_manifest.csv", TRASHNET_COLUMNS, trashnet_rows or base_trashnet_rows()),
            "classifier_feedback": write_csv(tmp / "classification_feedback_manifest.csv", CLASSIFIER_CANDIDATE_COLUMNS, classifier_feedback_rows or []),
            "gate_feedback": write_csv(tmp / "review_gate_feedback_manifest.csv", GATE_CANDIDATE_COLUMNS, gate_feedback_rows or []),
            "public_classifier": write_csv(tmp / "public_classifier_candidates.csv", PUBLIC_CLASSIFIER_COLUMNS, public_classifier_rows or []),
            "public_gate": write_csv(tmp / "public_gate_candidates.csv", PUBLIC_GATE_COLUMNS, public_gate_rows or []),
            "mapping": write_csv(tmp / "label_mapping.csv", LABEL_MAPPING_COLUMNS, mapping_rows or []),
            "output": tmp / "v2",
        }

    def build(self, paths: dict[str, Path], **overrides) -> dict:
        kwargs = {
            "trashnet_manifest": paths["trashnet"],
            "classifier_feedback_manifest": paths["classifier_feedback"],
            "gate_feedback_manifest": paths["gate_feedback"],
            "output_dir": paths["output"],
        }
        kwargs.update(overrides)
        return build_v2_dataset(**kwargs)

    def test_trashnet_rows_preserve_original_splits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(Path(tmp_dir))
            self.build(paths)

            rows = read_csv(paths["output"] / "v2_classification_manifest.csv")

        splits_by_image = {row["image_id"]: row["split"] for row in rows}
        self.assertEqual(splits_by_image["trash_train"], "train")
        self.assertEqual(splits_by_image["trash_val"], "validation")
        self.assertEqual(splits_by_image["trash_test"], "test")

    def test_feedback_classifier_rows_enter_classification_output_as_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(Path(tmp_dir), classifier_feedback_rows=[feedback_classifier_row()])
            self.build(paths)

            rows = read_csv(paths["output"] / "v2_classification_manifest.csv")

        feedback_rows = [row for row in rows if row["source_kind"] == "feedback"]
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["split"], "train")
        self.assertEqual(feedback_rows[0]["label"], "plastic")
        self.assertEqual(feedback_rows[0]["parent_feedback_id"], "fb_cls_1")

    def test_feedback_gate_rows_enter_gate_output_as_train(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(Path(tmp_dir), gate_feedback_rows=[feedback_gate_row()])
            self.build(paths)

            rows = read_csv(paths["output"] / "v2_gate_manifest.csv")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["auto_route_eligible"], "false")
        self.assertEqual(rows[0]["parent_feedback_id"], "fb_gate_1")

    def test_external_diagnostic_feedback_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                classifier_feedback_rows=[feedback_classifier_row(source="external_diagnostic", source_split="external_diagnostic_v1")],
            )

            with self.assertRaisesRegex(V2DatasetError, "external_diagnostic_v1"):
                self.build(paths)

    def test_external_diagnostic_feedback_requires_opt_in_and_writes_notice(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                classifier_feedback_rows=[feedback_classifier_row(source="external_diagnostic", source_split="external_diagnostic_v1")],
            )
            report = self.build(paths, allow_promoted_external_diagnostic=True)

            rows = read_csv(paths["output"] / "v2_classification_manifest.csv")
            notice_exists = (paths["output"] / "v2_leakage_notice.md").exists()

        feedback_rows = [row for row in rows if row["source_kind"] == "feedback"]
        self.assertEqual(feedback_rows[0]["source_split"], "external_diagnostic_v1")
        self.assertEqual(report["promoted_external_diagnostic_rows"], 1)
        self.assertTrue(notice_exists)

    def test_public_classifier_with_approved_mapping_and_one_object_is_included(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                public_classifier_rows=[public_classifier_row()],
                mapping_rows=[mapping_row()],
            )
            self.build(paths, public_classifier_manifest=paths["public_classifier"], label_mapping=paths["mapping"])

            rows = read_csv(paths["output"] / "v2_classification_manifest.csv")

        public_rows = [row for row in rows if row["source_kind"] == "public_dataset"]
        self.assertEqual(len(public_rows), 1)
        self.assertEqual(public_rows[0]["label"], "plastic")
        self.assertEqual(public_rows[0]["original_label"], "plastic bottle")
        self.assertEqual(public_rows[0]["mapping_rule_id"], "map_plastic_bottle")

    def test_public_classifier_with_multiple_objects_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                public_classifier_rows=[public_classifier_row(object_count="2")],
                mapping_rows=[mapping_row()],
            )

            with self.assertRaisesRegex(V2DatasetError, "object_count=1"):
                self.build(paths, public_classifier_manifest=paths["public_classifier"], label_mapping=paths["mapping"])

    def test_public_classifier_with_excluded_or_missing_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                public_classifier_rows=[
                    public_classifier_row(public_row_id="pub_cls_1", source_item_id="item_1", mapping_rule_id="map_excluded"),
                    public_classifier_row(public_row_id="pub_cls_2", source_item_id="item_2", mapping_rule_id="map_missing"),
                ],
                mapping_rows=[mapping_row(mapping_rule_id="map_excluded", mapping_status="excluded", mapped_label="")],
            )

            with self.assertRaisesRegex(V2DatasetError, "not approved|absent"):
                self.build(paths, public_classifier_manifest=paths["public_classifier"], label_mapping=paths["mapping"])

    def test_public_gate_multi_object_scene_can_be_included_when_not_auto_route_eligible(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(Path(tmp_dir), public_gate_rows=[public_gate_row()])
            self.build(paths, public_gate_manifest=paths["public_gate"])

            rows = read_csv(paths["output"] / "v2_gate_manifest.csv")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_kind"], "public_dataset")
        self.assertEqual(rows[0]["auto_route_eligible"], "false")
        self.assertEqual(rows[0]["annotation_type"], "scene")
        self.assertEqual(rows[0]["object_count"], "3")

    def test_duplicate_public_source_identity_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                public_classifier_rows=[
                    public_classifier_row(public_row_id="pub_cls_1", source_item_id="same_item"),
                    public_classifier_row(public_row_id="pub_cls_2", source_item_id="same_item", mapped_label="glass", mapping_rule_id="map_glass"),
                ],
                mapping_rows=[
                    mapping_row(),
                    mapping_row(mapping_rule_id="map_glass", source_label="plastic bottle", mapped_label="glass"),
                ],
            )

            with self.assertRaisesRegex(V2DatasetError, "duplicate source identity"):
                self.build(paths, public_classifier_manifest=paths["public_classifier"], label_mapping=paths["mapping"])

    def test_outputs_have_exact_schemas_and_deterministic_ordering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.make_inputs(
                Path(tmp_dir),
                trashnet_rows=list(reversed(base_trashnet_rows())),
                classifier_feedback_rows=[feedback_classifier_row()],
                gate_feedback_rows=[feedback_gate_row()],
                public_classifier_rows=[public_classifier_row()],
                public_gate_rows=[public_gate_row()],
                mapping_rows=[mapping_row()],
            )
            self.build(
                paths,
                public_classifier_manifest=paths["public_classifier"],
                public_gate_manifest=paths["public_gate"],
                label_mapping=paths["mapping"],
            )

            classification_header = header(paths["output"] / "v2_classification_manifest.csv")
            gate_header = header(paths["output"] / "v2_gate_manifest.csv")
            classification_rows = read_csv(paths["output"] / "v2_classification_manifest.csv")
            gate_rows = read_csv(paths["output"] / "v2_gate_manifest.csv")

        self.assertEqual(classification_header, CLASSIFICATION_OUTPUT_COLUMNS)
        self.assertEqual(gate_header, GATE_OUTPUT_COLUMNS)
        self.assertEqual([row["source_kind"] for row in classification_rows], ["trashnet", "trashnet", "trashnet", "feedback", "public_dataset"])
        self.assertEqual([row["split"] for row in classification_rows[:3]], ["train", "validation", "test"])
        self.assertEqual([row["source_kind"] for row in gate_rows], ["feedback", "public_dataset"])


if __name__ == "__main__":
    unittest.main()
