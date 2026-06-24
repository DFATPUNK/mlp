import unittest

from waste_poc.orchestration import build_pipeline_commands, should_run_fine_tune


class RunPocTests(unittest.TestCase):
    def test_clip_skips_invalid_fine_tune_even_without_flag(self):
        config = {"model": {"family": "clip_vit_b32_frozen_head"}, "training": {"fine_tune_epochs": 0}}
        self.assertFalse(should_run_fine_tune(config, skip_fine_tune=False))
        commands, final_run = build_pipeline_commands("configs/clip_vit_b32_frozen_head.yaml", config, "clip_run", "auto", skip_fine_tune=False)
        joined = [" ".join(command) for command in commands]
        self.assertEqual(final_run, "clip_run")
        self.assertEqual(sum("--mode fine_tune" in command for command in joined), 0)
        self.assertTrue(any("scripts/evaluate.py" in command for command in joined))

    def test_efficientnet_preserves_fine_tune_when_configured(self):
        config = {"model": {"architecture": "efficientnet_b0"}, "training": {"fine_tune_epochs": 8}}
        self.assertTrue(should_run_fine_tune(config, skip_fine_tune=False))
        commands, final_run = build_pipeline_commands("configs/efficientnet_b0_baseline.yaml", config, "eff_run", "cpu", skip_fine_tune=False)
        self.assertEqual(final_run, "eff_run_fine_tune")
        self.assertTrue(any("fine_tune" in command for command in [" ".join(item) for item in commands]))


if __name__ == "__main__":
    unittest.main()
