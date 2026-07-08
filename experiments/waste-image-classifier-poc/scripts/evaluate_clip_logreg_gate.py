#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.gate_experiment import _bool_text, evaluate_gate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the experimental CLIP plus logistic-regression binary gate.")
    parser.add_argument("--gate-model", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--expected-auto-route-eligible", required=True, choices=["true", "false"])
    parser.add_argument("--evaluation-label", default="honest_mini_holdout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    report = evaluate_gate_model(
        gate_model=ROOT / args.gate_model,
        external_manifest=ROOT / args.external_manifest,
        image_root=ROOT / args.image_root,
        output_dir=ROOT / args.output_dir,
        expected_auto_route_eligible=_bool_text(args.expected_auto_route_eligible, label="--expected-auto-route-eligible"),
        evaluation_label=args.evaluation_label,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    print(f"Gate negative recall: {report['negative_recall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
