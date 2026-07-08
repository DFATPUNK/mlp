from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from waste_poc.taco_intake import TACO_LICENSE_REFERENCE, TACO_MISSING_LICENSE_RESOLUTION_RULE, TACO_RESOLVED_LICENSE, prepare_taco_intake, resolve_image_license
from waste_poc.v2_dataset import LABEL_MAPPING_COLUMNS, PUBLIC_CLASSIFIER_COLUMNS, PUBLIC_GATE_COLUMNS


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


def write_annotations(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def mapping_row(**overrides) -> dict[str, str]:
    row = {
        "mapping_rule_id": "map_plastic",
        "source_dataset_id": "taco",
        "source_label": "plastic bottle",
        "mapped_label": "plastic",
        "mapping_status": "approved",
        "rationale": "synthetic mapping for tests",
    }
    row.update(overrides)
    return row


def image_row(image_id: int, file_name: str, **overrides) -> dict:
    row = {
        "id": image_id,
        "file_name": file_name,
        "width": 100,
        "height": 100,
        "license": 1,
        "flickr_url": f"https://example.test/taco/{image_id}",
        "author": f"Author {image_id}",
    }
    row.update(overrides)
    return row


def annotation_row(annotation_id: int, image_id: int, category_id: int, bbox: list[int] | None = None) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox or [0, 0, 50, 50],
    }


def base_payload(*, images: list[dict], annotations: list[dict], categories: list[dict] | None = None, licenses: list[dict] | None = None) -> dict:
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [
            {"id": 1, "name": "plastic bottle"},
            {"id": 2, "name": "metal can"},
            {"id": 3, "name": "banana peel"},
            {"id": 4, "name": "unmapped category"},
        ],
        "licenses": licenses or [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
    }


class TacoIntakeTests(unittest.TestCase):
    def run_intake(
        self,
        tmp: Path,
        *,
        payload: dict,
        mapping_rows: list[dict[str, str]],
        image_files: list[str],
        min_object_area_ratio: float = 0.20,
    ) -> tuple[dict, Path]:
        image_root = tmp / "images"
        for relative_path in image_files:
            image_path = image_root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"synthetic image bytes")
        annotations_path = write_annotations(tmp / "annotations.json", payload)
        mapping_path = write_csv(tmp / "label_mapping.csv", LABEL_MAPPING_COLUMNS, mapping_rows)
        output_dir = tmp / "out"
        report = prepare_taco_intake(
            annotations=annotations_path,
            image_root=image_root,
            label_mapping=mapping_path,
            output_dir=output_dir,
            min_object_area_ratio=min_object_area_ratio,
        )
        return report, output_dir

    def run_plan_only(
        self,
        tmp: Path,
        *,
        payload: dict,
        mapping_rows: list[dict[str, str]],
        min_object_area_ratio: float = 0.20,
    ) -> tuple[dict, Path]:
        annotations_path = write_annotations(tmp / "annotations.json", payload)
        mapping_path = write_csv(tmp / "label_mapping.csv", LABEL_MAPPING_COLUMNS, mapping_rows)
        output_dir = tmp / "out"
        report = prepare_taco_intake(
            annotations=annotations_path,
            image_root=None,
            label_mapping=mapping_path,
            output_dir=output_dir,
            min_object_area_ratio=min_object_area_ratio,
            plan_only=True,
        )
        return report, output_dir

    def test_category_inventory_is_deterministic_and_includes_mapped_and_unmapped_categories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "one.jpg")],
                annotations=[annotation_row(10, 1, 2)],
                categories=[
                    {"id": 2, "name": "metal can"},
                    {"id": 1, "name": "plastic bottle"},
                    {"id": 3, "name": "unmapped category"},
                ],
            )
            _, output_dir = self.run_intake(
                tmp,
                payload=payload,
                mapping_rows=[mapping_row(), mapping_row(mapping_rule_id="map_metal", source_label="metal can", mapped_label="metal")],
                image_files=["one.jpg"],
            )

            inventory_rows = read_csv(output_dir / "taco_category_inventory.csv")

        self.assertEqual([row["source_label"] for row in inventory_rows], ["metal can", "plastic bottle", "unmapped category"])
        self.assertEqual(inventory_rows[0]["mapped_label"], "metal")
        self.assertEqual(inventory_rows[2]["mapping_status"], "")

    def test_single_object_with_explicit_license_creates_unapproved_classifier_and_gate_drafts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(images=[image_row(1, "one.jpg")], annotations=[annotation_row(10, 1, 1)])
            report, output_dir = self.run_intake(tmp, payload=payload, mapping_rows=[mapping_row()], image_files=["one.jpg"])

            classifier_rows = read_csv(output_dir / "taco_classifier_candidates.draft.csv")
            gate_rows = read_csv(output_dir / "taco_gate_candidates.draft.csv")

        self.assertEqual(report["classifier_draft_rows"], 1)
        self.assertEqual(report["gate_draft_rows"], 1)
        self.assertEqual(list(classifier_rows[0].keys()), PUBLIC_CLASSIFIER_COLUMNS)
        self.assertEqual(list(gate_rows[0].keys()), PUBLIC_GATE_COLUMNS)
        self.assertEqual(classifier_rows[0]["approved_for_classifier_training"], "false")
        self.assertEqual(classifier_rows[0]["license_status"], "eligible_for_review")
        self.assertEqual(gate_rows[0]["auto_route_eligible"], "true")
        self.assertEqual(gate_rows[0]["review_reason"], "none")
        self.assertEqual(gate_rows[0]["approved_for_gate_training"], "false")

    def test_blank_taco_license_with_source_url_resolves_to_cc_by_for_review_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "blank.jpg", license=None)],
                annotations=[annotation_row(10, 1, 1)],
                licenses=[],
            )
            report, output_dir = self.run_plan_only(tmp, payload=payload, mapping_rows=[mapping_row()])

            ledger_rows = read_csv(output_dir / "taco_license_resolution.csv")

        self.assertEqual(report["images_with_eligible_licence"], 1)
        self.assertEqual(ledger_rows[0]["raw_license"], "")
        self.assertEqual(ledger_rows[0]["resolved_license"], TACO_RESOLVED_LICENSE)
        self.assertEqual(ledger_rows[0]["source_license_reference"], TACO_LICENSE_REFERENCE)
        self.assertEqual(ledger_rows[0]["license_status"], "eligible_for_review")
        self.assertEqual(ledger_rows[0]["resolution_rule"], TACO_MISSING_LICENSE_RESOLUTION_RULE)

    def test_blank_taco_license_without_source_url_remains_blocked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "blank_no_url.jpg", license=None, flickr_url="")],
                annotations=[annotation_row(10, 1, 1)],
                licenses=[],
            )
            _, output_dir = self.run_plan_only(tmp, payload=payload, mapping_rows=[mapping_row()])

            ledger_rows = read_csv(output_dir / "taco_license_resolution.csv")

        self.assertEqual(ledger_rows[0]["license_status"], "blocked")
        self.assertEqual(ledger_rows[0]["resolved_license"], "")
        self.assertIn("missing original source URL", ledger_rows[0]["resolution_notes"])

    def test_explicit_cc_and_odbl_license_values_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[
                    image_row(1, "cc.jpg", license="CC"),
                    image_row(2, "odbl.jpg", license="ODBL (c) OpenLitterMap & Contributors"),
                ],
                annotations=[annotation_row(10, 1, 1), annotation_row(11, 2, 1)],
                licenses=[],
            )
            _, output_dir = self.run_plan_only(tmp, payload=payload, mapping_rows=[mapping_row()])

            ledger_rows = read_csv(output_dir / "taco_license_resolution.csv")

        statuses = {row["source_item_id"]: row for row in ledger_rows}
        self.assertEqual(statuses["1"]["raw_license"], "CC")
        self.assertEqual(statuses["1"]["license_status"], "blocked")
        self.assertIn("not specific enough", statuses["1"]["resolution_notes"])
        self.assertEqual(statuses["2"]["raw_license"], "ODBL (c) OpenLitterMap & Contributors")
        self.assertEqual(statuses["2"]["license_status"], "blocked")
        self.assertIn("ODBL", statuses["2"]["resolution_notes"])

    def test_blank_license_resolution_rule_does_not_apply_to_non_taco_dataset(self):
        record = resolve_image_license(image_row(1, "other.jpg", license=None), {}, source_dataset_id="other_public")

        self.assertEqual(record["license_status"], "blocked")
        self.assertEqual(record["resolved_license"], "")
        self.assertEqual(record["resolution_rule"], "")

    def test_multi_object_image_creates_only_gate_draft_with_multiple_objects(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "multi.jpg")],
                annotations=[annotation_row(10, 1, 1), annotation_row(11, 1, 2)],
            )
            _, output_dir = self.run_intake(
                tmp,
                payload=payload,
                mapping_rows=[mapping_row(), mapping_row(mapping_rule_id="map_metal", source_label="metal can", mapped_label="metal")],
                image_files=["multi.jpg"],
            )

            classifier_rows = read_csv(output_dir / "taco_classifier_candidates.draft.csv")
            gate_rows = read_csv(output_dir / "taco_gate_candidates.draft.csv")

        self.assertEqual(classifier_rows, [])
        self.assertEqual(len(gate_rows), 1)
        self.assertEqual(gate_rows[0]["auto_route_eligible"], "false")
        self.assertEqual(gate_rows[0]["review_reason"], "multiple_objects")
        self.assertEqual(gate_rows[0]["object_count"], "2")

    def test_plan_only_writes_inventory_ledger_and_plan_without_candidate_files_or_image_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "planned.jpg", license=None)],
                annotations=[annotation_row(10, 1, 1)],
                licenses=[],
            )
            report, output_dir = self.run_plan_only(tmp, payload=payload, mapping_rows=[mapping_row()])
            inventory_exists = (output_dir / "taco_category_inventory.csv").is_file()
            ledger_exists = (output_dir / "taco_license_resolution.csv").is_file()
            plan_exists = (output_dir / "taco_download_plan.csv").is_file()
            classifier_exists = (output_dir / "taco_classifier_candidates.draft.csv").exists()
            gate_exists = (output_dir / "taco_gate_candidates.draft.csv").exists()

        self.assertEqual(report["download_plan_rows"], 1)
        self.assertTrue(inventory_exists)
        self.assertTrue(ledger_exists)
        self.assertTrue(plan_exists)
        self.assertFalse(classifier_exists)
        self.assertFalse(gate_exists)

    def test_plan_only_roles_are_generated_for_mapped_reviewable_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[
                    image_row(1, "single.jpg", license=None),
                    image_row(2, "multi.jpg", license=None),
                    image_row(3, "excluded.jpg", license=None),
                    image_row(4, "unmapped.jpg", license=None),
                    image_row(5, "small.jpg", license=None),
                ],
                annotations=[
                    annotation_row(10, 1, 1, [0, 0, 60, 60]),
                    annotation_row(20, 2, 1, [0, 0, 50, 50]),
                    annotation_row(21, 2, 2, [10, 10, 40, 40]),
                    annotation_row(30, 3, 3, [0, 0, 50, 50]),
                    annotation_row(40, 4, 4, [0, 0, 50, 50]),
                    annotation_row(50, 5, 1, [0, 0, 5, 5]),
                ],
                licenses=[],
            )
            _, output_dir = self.run_plan_only(
                tmp,
                payload=payload,
                mapping_rows=[
                    mapping_row(),
                    mapping_row(mapping_rule_id="map_metal", source_label="metal can", mapped_label="metal"),
                    mapping_row(mapping_rule_id="map_banana", source_label="banana peel", mapped_label="", mapping_status="excluded"),
                ],
            )

            plan_rows = read_csv(output_dir / "taco_download_plan.csv")

        rows_by_item = {row["source_item_id"]: row for row in plan_rows}
        self.assertEqual(set(rows_by_item), {"1", "2", "3", "5"})
        self.assertEqual(rows_by_item["1"]["proposed_classifier_role"], "classifier_and_gate_candidate")
        self.assertEqual(rows_by_item["1"]["proposed_gate_value"], "true")
        self.assertEqual(rows_by_item["1"]["proposed_review_reason"], "none")
        self.assertEqual(rows_by_item["2"]["proposed_classifier_role"], "gate_only")
        self.assertEqual(rows_by_item["2"]["proposed_gate_value"], "false")
        self.assertEqual(rows_by_item["2"]["proposed_review_reason"], "multiple_objects")
        self.assertEqual(rows_by_item["3"]["proposed_classifier_role"], "gate_only")
        self.assertEqual(rows_by_item["3"]["proposed_review_reason"], "unsupported_material")
        self.assertEqual(rows_by_item["5"]["proposed_classifier_role"], "gate_only")
        self.assertEqual(rows_by_item["5"]["proposed_review_reason"], "ambiguous_scene")
        self.assertTrue(all(row["plan_status"] == "eligible_for_manual_download" for row in plan_rows))

    def test_excluded_mapping_creates_gate_false_unsupported_material_not_classifier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(images=[image_row(1, "banana.jpg")], annotations=[annotation_row(10, 1, 3)])
            _, output_dir = self.run_intake(
                tmp,
                payload=payload,
                mapping_rows=[mapping_row(mapping_rule_id="map_banana", source_label="banana peel", mapped_label="", mapping_status="excluded")],
                image_files=["banana.jpg"],
            )

            classifier_rows = read_csv(output_dir / "taco_classifier_candidates.draft.csv")
            gate_rows = read_csv(output_dir / "taco_gate_candidates.draft.csv")

        self.assertEqual(classifier_rows, [])
        self.assertEqual(gate_rows[0]["auto_route_eligible"], "false")
        self.assertEqual(gate_rows[0]["review_reason"], "unsupported_material")

    def test_small_mapped_object_creates_gate_false_ambiguous_scene_not_classifier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(images=[image_row(1, "small.jpg")], annotations=[annotation_row(10, 1, 1, [0, 0, 5, 5])])
            _, output_dir = self.run_intake(tmp, payload=payload, mapping_rows=[mapping_row()], image_files=["small.jpg"])

            classifier_rows = read_csv(output_dir / "taco_classifier_candidates.draft.csv")
            gate_rows = read_csv(output_dir / "taco_gate_candidates.draft.csv")

        self.assertEqual(classifier_rows, [])
        self.assertEqual(gate_rows[0]["auto_route_eligible"], "false")
        self.assertEqual(gate_rows[0]["review_reason"], "ambiguous_scene")

    def test_missing_or_unresolved_license_metadata_is_blocked_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(images=[image_row(1, "blocked.jpg", license=99)], annotations=[annotation_row(10, 1, 1)])
            report, output_dir = self.run_intake(tmp, payload=payload, mapping_rows=[mapping_row()], image_files=["blocked.jpg"])

            excluded_rows = read_csv(output_dir / "taco_excluded_rows.csv")

        self.assertEqual(report["images_blocked_by_licence"], 1)
        self.assertEqual(report["classifier_draft_rows"], 0)
        self.assertEqual(report["gate_draft_rows"], 0)
        self.assertEqual(excluded_rows[0]["reason"], "blocked_license")
        self.assertEqual(excluded_rows[0]["license_status"], "blocked")

    def test_nc_or_nd_license_strings_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "nc.jpg")],
                annotations=[annotation_row(10, 1, 1)],
                licenses=[{"id": 1, "name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"}],
            )
            _, output_dir = self.run_intake(tmp, payload=payload, mapping_rows=[mapping_row()], image_files=["nc.jpg"])

            excluded_rows = read_csv(output_dir / "taco_excluded_rows.csv")
            classifier_rows = read_csv(output_dir / "taco_classifier_candidates.draft.csv")

        self.assertEqual(excluded_rows[0]["reason"], "blocked_license")
        self.assertEqual(classifier_rows, [])

    def test_missing_local_image_is_excluded_with_clear_reason(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(images=[image_row(1, "missing.jpg")], annotations=[annotation_row(10, 1, 1)])
            _, output_dir = self.run_intake(tmp, payload=payload, mapping_rows=[mapping_row()], image_files=[])

            excluded_rows = read_csv(output_dir / "taco_excluded_rows.csv")
            gate_rows = read_csv(output_dir / "taco_gate_candidates.draft.csv")

        self.assertEqual(excluded_rows[0]["reason"], "missing_local_image")
        self.assertEqual(gate_rows, [])

    def test_gallery_html_escapes_unsafe_annotation_text(self):
        unsafe_label = "plastic <script>alert(1)</script> bottle"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            payload = base_payload(
                images=[image_row(1, "unsafe.jpg")],
                annotations=[annotation_row(10, 1, 1)],
                categories=[{"id": 1, "name": unsafe_label}],
            )
            _, output_dir = self.run_intake(
                tmp,
                payload=payload,
                mapping_rows=[mapping_row(source_label=unsafe_label)],
                image_files=["unsafe.jpg"],
            )

            gallery_html = (output_dir / "taco_review_gallery.html").read_text(encoding="utf-8")

        self.assertIn("plastic &lt;script&gt;alert(1)&lt;/script&gt; bottle", gallery_html)
        self.assertNotIn("<script>alert(1)</script>", gallery_html)


if __name__ == "__main__":
    unittest.main()
