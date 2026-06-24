from __future__ import annotations

import sys

from .model import infer_model_family


def should_run_fine_tune(config: dict, skip_fine_tune: bool) -> bool:
    if skip_fine_tune:
        return False
    family = infer_model_family(config)
    fine_tune_epochs = int(config.get("training", {}).get("fine_tune_epochs", 0) or 0)
    if family == "clip_vit_b32_frozen_head":
        return False
    return fine_tune_epochs > 0


def build_pipeline_commands(config_path: str, config: dict, run_name: str | None, device: str, skip_fine_tune: bool) -> tuple[list[list[str]], str]:
    family = infer_model_family(config)
    base_run = run_name or ("poc_clip_vit_b32_frozen_head" if family == "clip_vit_b32_frozen_head" else "poc_frozen_backbone")
    commands = [[sys.executable, "scripts/train.py", "--config", config_path, "--output-dir", base_run, "--mode", "frozen_backbone", "--device", device]]
    final_run = base_run
    if should_run_fine_tune(config, skip_fine_tune):
        final_run = f"{base_run}_fine_tune"
        commands.append([sys.executable, "scripts/train.py", "--config", config_path, "--output-dir", final_run, "--mode", "fine_tune", "--resume-checkpoint", f"artifacts/runs/{base_run}/best_model.pt", "--device", device])
    commands.append([sys.executable, "scripts/evaluate.py", "--checkpoint", f"artifacts/runs/{final_run}/best_model.pt", "--device", device])
    return commands, final_run
