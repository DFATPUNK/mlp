#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.gate_experiment import build_gate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local Phase 0.9 binary gate training manifest from V2 manifests.")
    parser.add_argument("--v2-classification-manifest", required=True)
    parser.add_argument("--v2-gate-manifest", required=True)
    parser.add_argument("--negative-source-filter", required=True, choices=["feedback", "all"])
    parser.add_argument("--trashnet-image-root", default=None)
    parser.add_argument("--trashnet-metadata", default="data/raw/trashnet_source_metadata.json")
    parser.add_argument("--feedback-image-root", default=None)
    parser.add_argument("--public-image-root", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_gate_manifest(
        v2_classification_manifest=ROOT / args.v2_classification_manifest,
        v2_gate_manifest=ROOT / args.v2_gate_manifest,
        output_dir=ROOT / args.output_dir,
        negative_source_filter=args.negative_source_filter,
        trashnet_image_root=args.trashnet_image_root,
        feedback_image_root=args.feedback_image_root,
        public_image_root=args.public_image_root,
        poc_root=ROOT,
        trashnet_metadata=args.trashnet_metadata,
    )
    print(f"Wrote {report['rows']} gate examples to {report['gate_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
