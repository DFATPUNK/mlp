from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from waste_poc.taco_intake import DOWNLOAD_PLAN_COLUMNS
from waste_poc.taco_review_batch import REVIEW_BATCH_COLUMNS, TacoReviewBatchError, build_taco_review_batch
from waste_poc.utils import read_json


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


def plan_row(source_item_id: str, *, bucket: str = "classifier", mapped_labels: str = "plastic", source_labels: str = "plastic bottle", **overrides) -> dict[str, str]:
    role = "classifier_and_gate_candidate"
    gate_value = "true"
    review_reason = "none"
    if bucket in {"multiple_objects", "ambiguous_scene", "unsupported_material"}:
        role = "gate_only"
        gate_value = "false"
        review_reason = bucket
    row = {
        "plan_id": f"plan_{source_item_id}",
        "source_dataset_id": "taco",
        "source_item_id": source_item_id,
        "relative_path": f"batch/{source_item_id}.jpg",
        "source_url": f"https://example.test/taco/{source_item_id}",
        "source_license": "CC BY 4.0",
        "source_license_reference": "https://tacodataset.org/",
        "source_attribution": f"TACO test attribution {source_item_id}",
        "license_status": "eligible_for_review",
        "resolution_rule": "taco_missing_license_default_cc_by_4_0",
        "annotation_count": "1" if bucket != "multiple_objects" else "2",
        "source_labels": source_labels,
        "mapped_labels": mapped_labels,
        "object_area_ratio": "0.500000",
        "proposed_classifier_role": role,
        "proposed_gate_value": gate_value,
        "proposed_review_reason": review_reason,
        "plan_status": "eligible_for_manual_download",
        "plan_notes": "synthetic review plan row",
    }
    row.update(overrides)
    return row


class TacoReviewBatchTests(unittest.TestCase):
    def build(self, tmp: Path, rows: list[dict[str, str]], **kwargs) -> tuple[dict, Path]:
        plan_path = write_csv(tmp / "taco_download_plan.csv", DOWNLOAD_PLAN_COLUMNS, rows)
        output_dir = tmp / "batch"
        report = build_taco_review_batch(download_plan=plan_path, output_dir=output_dir, **kwargs)
        return report, output_dir

    def test_missing_required_input_columns_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            path = write_csv(tmp / "bad.csv", DOWNLOAD_PLAN_COLUMNS[:-1], [])

            with self.assertRaisesRegex(TacoReviewBatchError, "missing required columns"):
                build_taco_review_batch(download_plan=path, output_dir=tmp / "out")

    def test_non_eligible_license_status_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report, output_dir = self.build(Path(tmp_dir), [plan_row("blocked", license_status="blocked")])

            rows = read_csv(output_dir / "taco_review_batch.csv")

        self.assertEqual(report["eligible_plan_rows"], 0)
        self.assertEqual(rows, [])

    def test_missing_provenance_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report, output_dir = self.build(Path(tmp_dir), [plan_row("missing_attr", source_attribution="")])

            rows = read_csv(output_dir / "taco_review_batch.csv")

        self.assertEqual(report["eligible_plan_rows"], 0)
        self.assertEqual(rows, [])

    def test_duplicate_source_item_id_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            plan_path = write_csv(tmp / "plan.csv", DOWNLOAD_PLAN_COLUMNS, [plan_row("same"), plan_row("same", plan_id="plan_same_2")])

            with self.assertRaisesRegex(TacoReviewBatchError, "duplicate source_item_id"):
                build_taco_review_batch(download_plan=plan_path, output_dir=tmp / "out")

    def test_all_classifier_and_gate_candidates_are_selected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [plan_row(f"cls_{index}") for index in range(5)]
            report, output_dir = self.build(Path(tmp_dir), rows)

            batch_rows = read_csv(output_dir / "taco_review_batch.csv")

        self.assertEqual(report["selected_counts_by_bucket"]["classifier_and_gate_candidate"], 5)
        self.assertEqual({row["source_item_id"] for row in batch_rows}, {f"cls_{index}" for index in range(5)})

    def test_all_output_rows_share_one_batch_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [plan_row("cls_1")]
            rows.extend(plan_row(f"multi_{index}", bucket="multiple_objects") for index in range(4))
            report, output_dir = self.build(Path(tmp_dir), rows)

            batch_rows = read_csv(output_dir / "taco_review_batch.csv")
            report_json = read_json(output_dir / "taco_review_batch_report.json")

        batch_ids = {row["batch_id"] for row in batch_rows}
        self.assertEqual(batch_ids, {report["batch_id"]})
        self.assertEqual(report_json["batch_id"], report["batch_id"])

    def test_batch_id_changes_when_seed_or_limits_change(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rows = [plan_row("cls_1")]
            rows.extend(plan_row(f"multi_{index}", bucket="multiple_objects") for index in range(8))
            plan_path = write_csv(tmp / "plan.csv", DOWNLOAD_PLAN_COLUMNS, rows)

            report_default = build_taco_review_batch(download_plan=plan_path, output_dir=tmp / "default", seed="seed_a", multiple_objects_limit=4)
            report_seed = build_taco_review_batch(download_plan=plan_path, output_dir=tmp / "seed", seed="seed_b", multiple_objects_limit=4)
            report_limit = build_taco_review_batch(download_plan=plan_path, output_dir=tmp / "limit", seed="seed_a", multiple_objects_limit=5)

        self.assertNotEqual(report_default["batch_id"], report_seed["batch_id"])
        self.assertNotEqual(report_default["batch_id"], report_limit["batch_id"])

    def test_default_gate_buckets_respect_configured_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [plan_row(f"cls_{index}") for index in range(3)]
            rows.extend(plan_row(f"multi_{index}", bucket="multiple_objects", mapped_labels=f"group_{index % 5}") for index in range(30))
            rows.extend(plan_row(f"amb_{index}", bucket="ambiguous_scene", mapped_labels=f"group_{index % 4}") for index in range(20))
            rows.extend(plan_row(f"unsup_{index}", bucket="unsupported_material", mapped_labels=f"group_{index % 3}") for index in range(12))
            report, _ = self.build(Path(tmp_dir), rows)

        self.assertEqual(report["selected_counts_by_bucket"]["classifier_and_gate_candidate"], 3)
        self.assertEqual(report["selected_counts_by_bucket"]["multiple_objects"], 24)
        self.assertEqual(report["selected_counts_by_bucket"]["ambiguous_scene"], 14)
        self.assertEqual(report["selected_counts_by_bucket"]["unsupported_material"], 9)
        self.assertEqual(report["selected_total_rows"], 50)

    def test_short_bucket_reports_shortage_and_does_not_borrow_from_other_bucket(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [plan_row(f"multi_{index}", bucket="multiple_objects") for index in range(10)]
            rows.append(plan_row("amb_1", bucket="ambiguous_scene"))
            report, _ = self.build(Path(tmp_dir), rows, multiple_objects_limit=4, ambiguous_scene_limit=5, unsupported_material_limit=2)

        self.assertEqual(report["selected_counts_by_bucket"]["multiple_objects"], 4)
        self.assertEqual(report["selected_counts_by_bucket"]["ambiguous_scene"], 1)
        self.assertEqual(report["selected_counts_by_bucket"]["unsupported_material"], 0)
        self.assertEqual(report["shortages_by_bucket"]["ambiguous_scene"], 4)
        self.assertEqual(report["selected_total_rows"], 5)

    def test_selected_source_item_ids_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [plan_row(f"cls_{index}") for index in range(2)]
            rows.extend(plan_row(f"multi_{index}", bucket="multiple_objects") for index in range(5))
            _, output_dir = self.build(Path(tmp_dir), rows)

            batch_rows = read_csv(output_dir / "taco_review_batch.csv")

        source_item_ids = [row["source_item_id"] for row in batch_rows]
        self.assertEqual(len(source_item_ids), len(set(source_item_ids)))

    def test_identical_inputs_generate_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rows = [plan_row(f"multi_{index}", bucket="multiple_objects", mapped_labels=f"group_{index % 3}") for index in range(12)]
            plan_path = write_csv(tmp / "plan.csv", DOWNLOAD_PLAN_COLUMNS, rows)
            out_a = tmp / "a"
            out_b = tmp / "b"

            build_taco_review_batch(download_plan=plan_path, output_dir=out_a, multiple_objects_limit=5)
            build_taco_review_batch(download_plan=plan_path, output_dir=out_b, multiple_objects_limit=5)

            csv_a = (out_a / "taco_review_batch.csv").read_bytes()
            csv_b = (out_b / "taco_review_batch.csv").read_bytes()
            json_a = (out_a / "taco_review_batch_report.json").read_bytes()
            json_b = (out_b / "taco_review_batch_report.json").read_bytes()
            report_a = read_json(out_a / "taco_review_batch_report.json")
            report_b = read_json(out_b / "taco_review_batch_report.json")

        self.assertEqual(csv_a, csv_b)
        self.assertEqual(json_a, json_b)
        self.assertEqual(report_a["batch_id"], report_b["batch_id"])

    def test_changing_seed_can_alter_gate_selection_but_keeps_classifier_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rows = [plan_row("classifier_keep")]
            rows.extend(plan_row(f"multi_{index}", bucket="multiple_objects", mapped_labels=f"group_{index % 5}") for index in range(30))
            plan_path = write_csv(tmp / "plan.csv", DOWNLOAD_PLAN_COLUMNS, rows)
            out_a = tmp / "seed_a"
            out_b = tmp / "seed_b"

            build_taco_review_batch(download_plan=plan_path, output_dir=out_a, seed="seed_a", multiple_objects_limit=6)
            build_taco_review_batch(download_plan=plan_path, output_dir=out_b, seed="seed_b", multiple_objects_limit=6)
            rows_a = read_csv(out_a / "taco_review_batch.csv")
            rows_b = read_csv(out_b / "taco_review_batch.csv")

        self.assertIn("classifier_keep", {row["source_item_id"] for row in rows_a})
        self.assertIn("classifier_keep", {row["source_item_id"] for row in rows_b})
        gate_a = [row["source_item_id"] for row in rows_a if row["selection_bucket"] == "multiple_objects"]
        gate_b = [row["source_item_id"] for row in rows_b if row["selection_bucket"] == "multiple_objects"]
        self.assertNotEqual(gate_a, gate_b)

    def test_sampling_includes_diverse_mapped_label_groups_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                plan_row("multi_plastic", bucket="multiple_objects", mapped_labels="plastic"),
                plan_row("multi_glass", bucket="multiple_objects", mapped_labels="glass"),
                plan_row("multi_metal", bucket="multiple_objects", mapped_labels="metal"),
                plan_row("multi_paper", bucket="multiple_objects", mapped_labels="paper"),
            ]
            _, output_dir = self.build(Path(tmp_dir), rows, multiple_objects_limit=3)

            batch_rows = read_csv(output_dir / "taco_review_batch.csv")

        self.assertEqual(len({row["mapped_labels"] for row in batch_rows}), 3)

    def test_review_csv_schema_and_pending_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, output_dir = self.build(Path(tmp_dir), [plan_row("cls_1")])

            batch_header = header(output_dir / "taco_review_batch.csv")
            batch_rows = read_csv(output_dir / "taco_review_batch.csv")

        self.assertEqual(batch_header, REVIEW_BATCH_COLUMNS)
        self.assertEqual(batch_rows[0]["review_status"], "pending")
        self.assertEqual(batch_rows[0]["review_decision"], "")
        self.assertEqual(batch_rows[0]["review_notes"], "")
        self.assertNotIn("approved_for_classifier_training", batch_header)
        self.assertNotIn("approved_for_gate_training", batch_header)

    def test_empty_qualifying_input_writes_outputs_and_zero_count_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report, output_dir = self.build(Path(tmp_dir), [plan_row("not_taco", source_dataset_id="other")])

            batch_rows = read_csv(output_dir / "taco_review_batch.csv")
            report_json = read_json(output_dir / "taco_review_batch_report.json")

        self.assertEqual(batch_rows, [])
        self.assertEqual(report["selected_total_rows"], 0)
        self.assertEqual(report_json["selected_total_rows"], 0)
        self.assertEqual(report_json["available_counts_by_bucket"]["multiple_objects"], 0)


if __name__ == "__main__":
    unittest.main()
