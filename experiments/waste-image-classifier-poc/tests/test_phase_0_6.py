from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from waste_poc.clip_candidate import MODEL_FAMILY as CLIP_FAMILY, build_clip_frozen_head, load_clip_vision_model, trainable_parameter_names
from waste_poc.model import EFFICIENTNET_FAMILY, build_model_from_checkpoint
from waste_poc.model_selection import select_best_candidate_by_validation_metric
from waste_poc.utils import read_json, write_json

POC_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = POC_ROOT / "scripts" / name
    module_name = f"_test_{name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLoadedModel:
    def __init__(self):
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


class Phase06Tests(unittest.TestCase):
    def test_editable_install_imports_package(self):
        module = importlib.import_module("waste_poc")
        self.assertTrue(Path(module.__file__).as_posix().endswith("waste_poc/__init__.py"))

    def test_default_clip_loader_requests_safetensors(self):
        from_pretrained = mock.Mock(return_value=object())
        fake_transformers = SimpleNamespace(CLIPVisionModel=SimpleNamespace(from_pretrained=from_pretrained))

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            loaded = load_clip_vision_model("fake-clip")

        self.assertIs(loaded, from_pretrained.return_value)
        from_pretrained.assert_called_once_with("fake-clip", use_safetensors=True)

    def test_clip_encoder_is_frozen_and_only_head_receives_gradients(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"torch is not installed: {exc}")

        class FakeVisionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.encoder_weight = torch.nn.Parameter(torch.eye(4))

            def forward(self, pixel_values):
                return SimpleNamespace(pooler_output=pixel_values @ self.encoder_weight)

        model = build_clip_frozen_head(6, "fake-clip", vision_model_factory=lambda _model_id: FakeVisionModel())
        self.assertEqual(trainable_parameter_names(model), ["classifier.weight", "classifier.bias"])
        self.assertFalse(model.vision_model.encoder_weight.requires_grad)

        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        self.assertIsNone(model.vision_model.encoder_weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(model.classifier.bias.grad)

    def test_checkpoint_metadata_dispatches_to_clip_loader(self):
        fake_model = FakeLoadedModel()
        with mock.patch("waste_poc.model.build_clip_frozen_head", return_value=fake_model) as builder:
            loaded = build_model_from_checkpoint(
                {
                    "model_family": CLIP_FAMILY,
                    "hf_model_id": "fake-clip",
                    "class_names": ["cardboard", "glass"],
                    "model_state_dict": {"vision_model.encoder_weight": object(), "classifier.weight": object()},
                }
            )
        self.assertIs(loaded, fake_model)
        builder.assert_called_once_with(2, "fake-clip")

    def test_checkpoint_metadata_dispatches_to_efficientnet_loader(self):
        fake_model = FakeLoadedModel()
        with mock.patch("waste_poc.model.create_efficientnet_b0", return_value=(fake_model, None)) as builder:
            loaded = build_model_from_checkpoint(
                {
                    "model_family": EFFICIENTNET_FAMILY,
                    "class_names": ["cardboard", "glass"],
                    "model_state_dict": {"features.0.weight": object(), "classifier.1.weight": object()},
                }
            )
        self.assertIs(loaded, fake_model)
        builder.assert_called_once_with(["cardboard", "glass"], pretrained_weights="NONE", mode="fine_tune")

    def test_incompatible_checkpoint_family_combination_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "looks like EfficientNet-B0"):
            build_model_from_checkpoint(
                {
                    "model_family": CLIP_FAMILY,
                    "class_names": ["trash"],
                    "model_state_dict": {"features.0.weight": object(), "classifier.1.weight": object()},
                }
            )
        with self.assertRaisesRegex(ValueError, "looks like CLIP"):
            build_model_from_checkpoint(
                {
                    "model_family": EFFICIENTNET_FAMILY,
                    "class_names": ["trash"],
                    "model_state_dict": {"vision_model.encoder_weight": object(), "classifier.weight": object()},
                }
            )

    def test_candidate_selection_uses_best_validation_metric_not_latest_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strong_best = root / "strong_best"
            strong_final = root / "strong_final"
            strong_best.mkdir()
            strong_final.mkdir()
            write_json(strong_best / "best_metrics.json", {"best_epoch": 2, "best_validation_macro_f1": 0.91, "selected_hyperparameters": {"learning_rate": 0.001}, "checkpoint_path": str(strong_best / "best_model.pt")})
            write_json(strong_best / "latest_metrics.json", {"epoch": 5, "validation": {"macro_f1": 0.60}})
            write_json(strong_final / "best_metrics.json", {"best_epoch": 5, "best_validation_macro_f1": 0.82, "selected_hyperparameters": {"learning_rate": 0.0003}, "checkpoint_path": str(strong_final / "best_model.pt")})
            write_json(strong_final / "latest_metrics.json", {"epoch": 5, "validation": {"macro_f1": 0.99}})

            selected_dir, selected_metrics = select_best_candidate_by_validation_metric([strong_final, strong_best])

        self.assertEqual(selected_dir, strong_best)
        self.assertEqual(selected_metrics["best_validation_macro_f1"], 0.91)

    def test_clip_orchestration_skips_fine_tune_and_external_by_default(self):
        run_poc = load_script("run_poc.py")
        commands = []

        run_poc.main(["--config", "configs/clip_vit_b32_frozen_head.yaml", "--skip-download", "--skip-manifest", "--run-name", "clip_smoke", "--device", "cpu"], runner=commands.append)

        flattened = [" ".join(command) for command in commands]
        self.assertFalse(any("fine_tune" in command for command in flattened))
        self.assertFalse(any("scripts/evaluate_external.py" in command for command in flattened))
        self.assertTrue(any("scripts/evaluate.py" in command for command in flattened))

    def test_external_evaluation_runs_only_with_include_external(self):
        run_poc = load_script("run_poc.py")
        commands = []

        run_poc.main(["--config", "configs/clip_vit_b32_frozen_head.yaml", "--skip-download", "--skip-manifest", "--run-name", "clip_smoke", "--device", "cpu", "--include-external"], runner=commands.append)

        external = [command for command in commands if "scripts/evaluate_external.py" in command]
        self.assertEqual(len(external), 1)
        self.assertIn("artifacts/runs/clip_smoke/external_diagnostic_v1", external[0])

    def test_external_evaluation_initializes_inference_session_once(self):
        evaluate_external = load_script("evaluate_external.py")

        class FakeSession:
            threshold_policy = {"selected_threshold": 0.7}
            selected_device_name = "cpu"

            def __init__(self):
                self.predict_calls = []

            def predict(self, image_path):
                self.predict_calls.append(Path(image_path).name)
                return {
                    "predicted_label": "plastic",
                    "raw_top_confidence": 0.8,
                    "calibrated_top_confidence": 0.75,
                    "recommended_action": "auto_route",
                    "route": "plastic",
                    "probabilities": {"plastic": 0.75},
                    "logits": [1.0],
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            for name in ["one.jpg", "two.jpg"]:
                (image_root / name).write_bytes(b"fake image bytes")
            manifest = root / "external_manifest.csv"
            manifest.write_text(
                "image_id,relative_path,expected_label,scenario,expected_routing\n"
                "one,one.jpg,plastic,in_scope_clean,auto_route_if_confident\n"
                "two,two.jpg,plastic,in_scope_clean,auto_route_if_confident\n",
                encoding="utf-8",
            )
            output_dir = root / "out"
            fake_session = FakeSession()
            argv = [
                "evaluate_external.py",
                "--checkpoint",
                str(root / "best_model.pt"),
                "--temperature",
                str(root / "temperature_scaling.json"),
                "--threshold-policy",
                str(root / "threshold_policy.json"),
                "--external-manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
            (root / "best_model.pt").write_bytes(b"checkpoint")
            write_json(root / "temperature_scaling.json", {"temperature": 1.0})
            write_json(root / "threshold_policy.json", {"selected_threshold": 0.7})

            with mock.patch.object(sys, "argv", argv), mock.patch.object(evaluate_external.ImageInferenceSession, "from_files", return_value=fake_session) as from_files, mock.patch.object(evaluate_external, "plot_confusion_matrix"), mock.patch.object(evaluate_external, "save_gallery"):
                self.assertEqual(evaluate_external.main(), 0)

        from_files.assert_called_once()
        self.assertEqual(fake_session.predict_calls, ["one.jpg", "two.jpg"])

    def test_compare_models_uses_explicit_external_dirs_and_warns_on_split_mismatch(self):
        compare_models = load_script("compare_models.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            efficientnet_run = root / "eff"
            clip_run = root / "clip"
            efficientnet_external = root / "eff_diag"
            clip_external = root / "clip_diag"
            output_dir = root / "comparison"
            for path in [efficientnet_run, clip_run, efficientnet_external, clip_external]:
                path.mkdir()
            write_json(efficientnet_run / "metrics_validation.json", {"macro_f1": 0.7})
            write_json(efficientnet_run / "metrics_test.json", {"macro_f1": 0.6})
            write_json(efficientnet_run / "model_metadata.json", {"split_manifest_sha256": "hash-a"})
            write_json(clip_run / "metrics_validation.json", {"macro_f1": 0.8})
            write_json(clip_run / "metrics_test.json", {"macro_f1": 0.65})
            write_json(clip_run / "model_metadata.json", {"split_manifest_sha256": "hash-b"})
            write_json(efficientnet_external / "external_summary.json", {"policy_safe_outcome_rate": 0.5})
            write_json(clip_external / "external_summary.json", {"policy_safe_outcome_rate": 0.9})

            argv = [
                "compare_models.py",
                "--efficientnet-run",
                str(efficientnet_run),
                "--clip-run",
                str(clip_run),
                "--efficientnet-external-dir",
                str(efficientnet_external),
                "--clip-external-dir",
                str(clip_external),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(compare_models.main(), 0)
            report = read_json(output_dir / "efficientnet_vs_clip_frozen_head.json")

        self.assertIn("not used for training", report["external_diagnostic_warning"])
        self.assertIsNotNone(report["split_manifest_warning"])
        self.assertEqual(report["candidates"]["clip_vit_b32_frozen_head"]["external_diagnostic_v1"]["policy_safe_outcome_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
