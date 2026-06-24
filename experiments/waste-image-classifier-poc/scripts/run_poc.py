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
from waste_poc.orchestration import build_pipeline_commands, should_run_fine_tune


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete TrashNet image-classifier POC candidate pipeline.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-fine-tune", action="store_true")
    parser.add_argument("--include-external", action="store_true", help="Opt in to external_diagnostic_v1 evaluation after internal evaluation.")
    parser.add_argument("--skip-external", action="store_true", help="Deprecated compatibility flag; external evaluation is skipped by default.")
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
    if args.include_external and not args.skip_external:
        run([sys.executable, "scripts/evaluate_external.py", "--checkpoint", f"artifacts/runs/{final_run}/best_model.pt", "--temperature", f"artifacts/runs/{final_run}/temperature_scaling.json", "--threshold-policy", f"artifacts/runs/{final_run}/threshold_policy.json", "--output-dir", f"artifacts/runs/{final_run}/external_diagnostic_v1", "--device", args.device])
    print("\nFinal summary")
    print(f"Run: {final_run}")
    print(f"Model family: {family}")
    print(f"Artifacts: artifacts/runs/{final_run}/")
    print("Open metrics_validation.json, metrics_test.json, threshold_policy.json, test_policy_report.json, and model_card.md for the final results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
