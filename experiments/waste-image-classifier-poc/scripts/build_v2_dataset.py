#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from waste_poc.v2_dataset import V2DatasetError, build_v2_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble leakage-safe V2 dataset manifests without training a model.")
    parser.add_argument("--trashnet-manifest", required=True)
    parser.add_argument("--classifier-feedback-manifest", required=True)
    parser.add_argument("--gate-feedback-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-classifier-manifest")
    parser.add_argument("--public-gate-manifest")
    parser.add_argument("--label-mapping")
    parser.add_argument("--allow-promoted-external-diagnostic", action="store_true")
    parser.add_argument("--feedback-image-root", help="Optional local root for computing feedback image SHA-256 values when files exist.")
    parser.add_argument("--public-image-root", help="Optional local root for computing public image SHA-256 values when files exist.")
    args = parser.parse_args()

    try:
        report = build_v2_dataset(
            trashnet_manifest=args.trashnet_manifest,
            classifier_feedback_manifest=args.classifier_feedback_manifest,
            gate_feedback_manifest=args.gate_feedback_manifest,
            output_dir=args.output_dir,
            public_classifier_manifest=args.public_classifier_manifest,
            public_gate_manifest=args.public_gate_manifest,
            label_mapping=args.label_mapping,
            allow_promoted_external_diagnostic=args.allow_promoted_external_diagnostic,
            feedback_image_root=args.feedback_image_root,
            public_image_root=args.public_image_root,
        )
    except V2DatasetError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return 1

    print(f"V2 classification rows: {report['classification_rows']}")
    print(f"V2 gate rows: {report['gate_rows']}")
    print(f"V2 outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
