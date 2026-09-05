#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.clip_candidate import DEFAULT_HF_MODEL_ID
from waste_poc.gate_experiment import train_gate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the experimental frozen CLIP plus logistic-regression binary review gate.")
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-positive-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    report = train_gate_model(
        gate_manifest=ROOT / args.gate_manifest,
        output_dir=ROOT / args.output_dir,
        max_positive_examples=args.max_positive_examples,
        seed=args.seed,
        hf_model_id=args.hf_model_id,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    print(f"Saved gate model with {report['rows']} examples to {ROOT / args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
