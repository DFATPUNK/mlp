#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_poc.utils import read_json, write_json, write_text

WARNING = "This external diagnostic set was not used for training, calibration, threshold selection, or model selection. It has already been inspected and must not be treated as a blind final benchmark."


def load_optional(path: Path):
    return read_json(path) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EfficientNet-B0 and CLIP frozen-head POC candidates without declaring an external winner.")
    parser.add_argument("--efficientnet-run", required=True)
    parser.add_argument("--clip-run", required=True)
    parser.add_argument("--output-dir", default="artifacts/model_comparisons")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"efficientnet_b0": ROOT / args.efficientnet_run, "clip_vit_b32_frozen_head": ROOT / args.clip_run}
    report = {"external_diagnostic_warning": WARNING, "candidates": {}}
    for name, run_dir in runs.items():
        report["candidates"][name] = {
            "validation": load_optional(run_dir / "metrics_validation.json"),
            "test": load_optional(run_dir / "metrics_test.json"),
            "external_diagnostic_v1": load_optional(run_dir / "external_evaluation" / "external_summary.json"),
            "metadata": load_optional(run_dir / "model_metadata.json"),
        }
    write_json(output_dir / "efficientnet_vs_clip_frozen_head.json", report)
    md = ["# EfficientNet-B0 vs CLIP frozen-head", "", WARNING, ""]
    for name, data in report["candidates"].items():
        md.append(f"## {name}")
        for section in ["validation", "test", "external_diagnostic_v1"]:
            metrics = data.get(section) or {}
            md.append(f"- {section}: macro_f1={metrics.get('macro_f1')} accuracy={metrics.get('accuracy')} policy_safe_outcome_rate={metrics.get('policy_safe_outcome_rate')}")
        md.append("")
    md.append("Do not automatically declare a production winner from `external_diagnostic_v1`; collect a new unseen external set before final model selection.")
    write_text(output_dir / "efficientnet_vs_clip_frozen_head.md", "\n".join(md) + "\n")
    print(f"Wrote comparison report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
