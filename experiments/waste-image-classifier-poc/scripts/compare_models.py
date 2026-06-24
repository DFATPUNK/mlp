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
    return read_json(path) if path and path.exists() else None


def candidate_report(run_dir: Path, external_dir: Path | None) -> dict:
    metadata = load_optional(run_dir / "model_metadata.json") or {}
    return {
        "validation": load_optional(run_dir / "metrics_validation.json"),
        "test": load_optional(run_dir / "metrics_test.json"),
        "external_diagnostic_v1": load_optional(external_dir / "external_summary.json") if external_dir else None,
        "external_dir": str(external_dir) if external_dir else None,
        "metadata": metadata,
        "split_manifest_sha256": metadata.get("split_manifest_sha256") or metadata.get("manifest_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EfficientNet-B0 and CLIP frozen-head POC candidates without declaring an external winner.")
    parser.add_argument("--efficientnet-run", required=True)
    parser.add_argument("--clip-run", required=True)
    parser.add_argument("--efficientnet-external-dir", default=None)
    parser.add_argument("--clip-external-dir", default=None)
    parser.add_argument("--output-dir", default="artifacts/model_comparisons")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    efficientnet_run = ROOT / args.efficientnet_run
    clip_run = ROOT / args.clip_run
    efficientnet_external = ROOT / args.efficientnet_external_dir if args.efficientnet_external_dir else None
    clip_external = ROOT / args.clip_external_dir if args.clip_external_dir else None
    report = {
        "external_diagnostic_warning": WARNING,
        "candidates": {
            "efficientnet_b0": candidate_report(efficientnet_run, efficientnet_external),
            "clip_vit_b32_frozen_head": candidate_report(clip_run, clip_external),
        },
    }
    eff_hash = report["candidates"]["efficientnet_b0"].get("split_manifest_sha256")
    clip_hash = report["candidates"]["clip_vit_b32_frozen_head"].get("split_manifest_sha256")
    report["same_split_manifest_sha256"] = bool(eff_hash and clip_hash and eff_hash == clip_hash)
    report["split_manifest_warning"] = None if report["same_split_manifest_sha256"] else "EfficientNet and CLIP runs do not report the same split manifest SHA256; compare metrics cautiously."
    write_json(output_dir / "efficientnet_vs_clip_frozen_head.json", report)
    md = ["# EfficientNet-B0 vs CLIP frozen-head", "", WARNING, ""]
    if report["split_manifest_warning"]:
        md.extend([f"**Warning:** {report['split_manifest_warning']}", ""])
    for name, data in report["candidates"].items():
        md.append(f"## {name}")
        md.append(f"- split manifest SHA256: `{data.get('split_manifest_sha256')}`")
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
