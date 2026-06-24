#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.utils import read_json, write_json, write_text

WARNING = "This external diagnostic set was not used for training, calibration, threshold selection, or model selection. It has already been inspected and must not be treated as a blind final benchmark."


def load_optional(path: Path):
    return read_json(path) if path.exists() else None


def root_path(path: str | None) -> Path | None:
    if path is None:
        return None
    return ROOT / Path(path)


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
    runs = {"efficientnet_b0": ROOT / args.efficientnet_run, "clip_vit_b32_frozen_head": ROOT / args.clip_run}
    external_dirs = {
        "efficientnet_b0": root_path(args.efficientnet_external_dir),
        "clip_vit_b32_frozen_head": root_path(args.clip_external_dir),
    }
    report = {"external_diagnostic_warning": WARNING, "candidates": {}}
    for name, run_dir in runs.items():
        metadata = load_optional(run_dir / "model_metadata.json") or {}
        external_dir = external_dirs[name]
        report["candidates"][name] = {
            "validation": load_optional(run_dir / "metrics_validation.json"),
            "test": load_optional(run_dir / "metrics_test.json"),
            "external_diagnostic_v1": load_optional(external_dir / "external_summary.json") if external_dir else None,
            "external_diagnostic_dir": str(external_dir) if external_dir else None,
            "metadata": metadata,
            "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        }
    hashes = {name: data.get("split_manifest_sha256") for name, data in report["candidates"].items()}
    known_hashes = {value for value in hashes.values() if value}
    split_warning = len(known_hashes) > 1
    report["split_manifest_sha256"] = hashes
    report["split_manifest_warning"] = "Split manifest SHA256 values differ; internal metrics are not directly comparable without accounting for the split change." if split_warning else None
    write_json(output_dir / "efficientnet_vs_clip_frozen_head.json", report)
    md = ["# EfficientNet-B0 vs CLIP frozen-head", "", WARNING, ""]
    md.append("## Split Manifest SHA256")
    for name, digest in hashes.items():
        md.append(f"- {name}: `{digest}`")
    if split_warning:
        md.extend(["", "**Warning:** Split manifest SHA256 values differ; internal metrics are not directly comparable without accounting for the split change."])
    md.append("")
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
