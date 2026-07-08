#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.gate_experiment import evaluate_combined_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the V1 classifier policy with the experimental binary-gate override.")
    parser.add_argument("--v1-external-predictions", required=True)
    parser.add_argument("--gate-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = evaluate_combined_policy(
        v1_external_predictions=ROOT / args.v1_external_predictions,
        gate_predictions=ROOT / args.gate_predictions,
        output_dir=ROOT / args.output_dir,
    )
    print(
        "Unsafe auto-routes: "
        f"{report['baseline_unsafe_auto_route_count']} before, "
        f"{report['combined_unsafe_auto_route_count']} after"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
