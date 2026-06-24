import tempfile
import unittest
from pathlib import Path

from waste_poc.external_io import prepare_external_output_dir
from waste_poc.external_metrics import enrich_external_row, summarize_external_rows


class ExternalMetricsTests(unittest.TestCase):
    def test_wrong_auto_route_is_unsafe_not_success(self):
        row = enrich_external_row({"expected_label": "plastic", "expected_routing": "auto_route_if_confident", "predicted_label": "metal", "recommended_action": "auto_route", "route": "metal"})
        summary = summarize_external_rows([row])
        self.assertFalse(row["auto_route_is_correct"])
        self.assertTrue(row["is_unsafe_auto_route"])
        self.assertEqual(summary["correct_auto_route_count"], 0)
        self.assertEqual(summary["incorrect_auto_route_count"], 1)
        self.assertEqual(summary["policy_safe_outcome_rate"], 0.0)

    def test_safe_abstention_and_expected_review(self):
        rows = [
            enrich_external_row({"expected_label": "glass", "expected_routing": "auto_route_if_confident", "predicted_label": "glass", "recommended_action": "needs_review", "route": "needs_review"}),
            enrich_external_row({"expected_label": "", "expected_routing": "needs_review", "predicted_label": "trash", "recommended_action": "needs_review", "route": "needs_review"}),
        ]
        summary = summarize_external_rows(rows)
        self.assertTrue(rows[0]["is_safe_abstention"])
        self.assertTrue(rows[1]["expected_review_correctly_routed"])
        self.assertIsNone(rows[1]["known_label_top1_correct"])
        self.assertEqual(summary["safe_abstention_count"], 1)
        self.assertEqual(summary["correctly_reviewed_count"], 1)

    def test_expected_review_auto_route_is_unsafe(self):
        row = enrich_external_row({"expected_label": "", "expected_routing": "needs_review", "predicted_label": "paper", "recommended_action": "auto_route", "route": "paper"})
        summary = summarize_external_rows([row])
        self.assertTrue(row["is_unsafe_auto_route"])
        self.assertEqual(summary["unsafe_auto_route_on_expected_review_count"], 1)

    def test_existing_output_dir_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "external_eval"
            out.mkdir()
            (out / "external_summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare_external_output_dir(str(out), "checkpoint.pt", overwrite=False)
            self.assertEqual(prepare_external_output_dir(str(out), "checkpoint.pt", overwrite=True), out)


if __name__ == "__main__":
    unittest.main()

class ExternalEvaluationSessionTests(unittest.TestCase):
    def test_evaluate_external_reuses_one_session_for_multiple_images(self):
        import sys
        import tempfile
        from pathlib import Path
        from unittest import mock

        from scripts import evaluate_external

        class FakeSession:
            init_count = 0
            predict_count = 0
            threshold_policy = {"selected_threshold": 0.5}
            selected_device_name = "cpu"

            @classmethod
            def from_files(cls, *_args, **_kwargs):
                cls.init_count += 1
                return cls()

            def predict(self, _image_path):
                type(self).predict_count += 1
                return {
                    "predicted_label": "plastic",
                    "raw_top_confidence": 0.9,
                    "calibrated_top_confidence": 0.8,
                    "probabilities": {"plastic": 0.8},
                    "logits": [1.0],
                    "recommended_action": "auto_route",
                    "route": "plastic",
                    "reason": "test",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "external_manifest.csv"
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_text("fake", encoding="utf-8")
            (images / "b.jpg").write_text("fake", encoding="utf-8")
            manifest.write_text(
                "image_id,relative_path,expected_label,scenario,expected_routing,notes\n"
                "a,a.jpg,plastic,in_scope_clean,auto_route_if_confident,\n"
                "b,b.jpg,plastic,in_scope_clean,auto_route_if_confident,\n",
                encoding="utf-8",
            )
            checkpoint = root / "best_model.pt"
            temperature = root / "temperature.json"
            policy = root / "policy.json"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            temperature.write_text('{"temperature": 1.0}', encoding="utf-8")
            policy.write_text('{"selected_threshold": 0.5}', encoding="utf-8")
            output_dir = root / "out"
            argv = [
                "evaluate_external.py",
                "--checkpoint",
                str(checkpoint),
                "--temperature",
                str(temperature),
                "--threshold-policy",
                str(policy),
                "--external-manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(evaluate_external, "ROOT", Path("/")), mock.patch.object(evaluate_external, "ImageInferenceSession", FakeSession), mock.patch.object(evaluate_external, "save_gallery"), mock.patch.object(evaluate_external, "plot_confusion_matrix"):
                self.assertEqual(evaluate_external.main(), 0)
            self.assertEqual(FakeSession.init_count, 1)
            self.assertEqual(FakeSession.predict_count, 2)
