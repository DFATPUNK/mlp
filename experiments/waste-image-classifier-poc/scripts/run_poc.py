#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.clip_candidate import MODEL_FAMILY as CLIP_FAMILY
from waste_poc.config import load_config
from waste_poc.model import infer_model_family


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def should_run_fine_tune(config: dict, skip_fine_tune: bool) -> bool:
    if skip_fine_tune:
        return False
    if infer_model_family(config) == CLIP_FAMILY:
        return False
    return int(config.get("training", {}).get("fine_tune_epochs", 0)) > 0


def main(argv: list[str] | None = None, runner=run) -> int:
    parser = argparse.ArgumentParser(description="Run the complete TrashNet waste image classifier POC pipeline.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-fine-tune", action="store_true")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args(argv)
    config = load_config(ROOT / args.config)
    family = infer_model_family(config)
    if not args.skip_download:
        runner([sys.executable, "scripts/download_trashnet.py"])
    if not args.skip_manifest:
        runner([sys.executable, "scripts/build_manifest.py", "--config", args.config])
    frozen_run = args.run_name or "poc_frozen_backbone"
    runner([sys.executable, "scripts/train.py", "--config", args.config, "--output-dir", frozen_run, "--mode", "frozen_backbone", "--device", args.device])
    final_run = frozen_run
    if should_run_fine_tune(config, args.skip_fine_tune):
        final_run = f"{frozen_run}_fine_tune"
        runner([sys.executable, "scripts/train.py", "--config", args.config, "--output-dir", final_run, "--mode", "fine_tune", "--resume-checkpoint", f"artifacts/runs/{frozen_run}/best_model.pt", "--device", args.device])
    checkpoint = f"artifacts/runs/{final_run}/best_model.pt"
    runner([sys.executable, "scripts/evaluate.py", "--checkpoint", checkpoint, "--device", args.device])
    if args.include_external:
        runner([sys.executable, "scripts/evaluate_external.py", "--checkpoint", checkpoint, "--temperature", f"artifacts/runs/{final_run}/temperature_scaling.json", "--threshold-policy", f"artifacts/runs/{final_run}/threshold_policy.json", "--output-dir", f"artifacts/runs/{final_run}/external_diagnostic_v1", "--device", args.device])
    print("\nFinal summary")
    print(f"Run: {final_run}")
    print(f"Model family: {family}")
    print(f"Artifacts: artifacts/runs/{final_run}/")
    print("Open metrics_validation.json, metrics_test.json, threshold_policy.json, test_policy_report.json, and model_card.md for the final results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
