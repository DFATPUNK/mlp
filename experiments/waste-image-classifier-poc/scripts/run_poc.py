#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_poc.config import load_config
from waste_poc.model import infer_model_family


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete TrashNet image-classifier POC candidate pipeline.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-fine-tune", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    family = infer_model_family(config)
    if not args.skip_download:
        run([sys.executable, "scripts/download_trashnet.py"])
    if not args.skip_manifest:
        run([sys.executable, "scripts/build_manifest.py", "--config", args.config])
    commands, final_run = build_pipeline_commands(args.config, config, args.run_name, args.device, args.skip_fine_tune)
    for command in commands:
        run(command)
    if not args.skip_external:
        run([sys.executable, "scripts/evaluate_external.py", "--checkpoint", f"artifacts/runs/{final_run}/best_model.pt", "--temperature", f"artifacts/runs/{final_run}/temperature_scaling.json", "--threshold-policy", f"artifacts/runs/{final_run}/threshold_policy.json", "--output-dir", f"artifacts/runs/{final_run}/external_evaluation", "--device", args.device])
    print("\nFinal summary")
    print(f"Run: {final_run}")
    print(f"Model family: {family}")
    print(f"Artifacts: artifacts/runs/{final_run}/")
    print("Open metrics_validation.json, metrics_test.json, threshold_policy.json, test_policy_report.json, and model_card.md for the final results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
