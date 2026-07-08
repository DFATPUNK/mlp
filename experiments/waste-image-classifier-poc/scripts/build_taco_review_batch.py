#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from waste_poc.taco_review_batch import DEFAULT_LIMITS, DEFAULT_SEED, TacoReviewBatchError, build_taco_review_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic local TACO review batch from a Phase 0.8.3 download plan.")
    parser.add_argument("--download-plan", required=True, help="Local taco_download_plan.csv from plan-only TACO intake.")
    parser.add_argument("--output-dir", required=True, help="Ignored local directory for taco_review_batch.csv and report JSON.")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--multiple-objects-limit", type=int, default=DEFAULT_LIMITS["multiple_objects"])
    parser.add_argument("--ambiguous-scene-limit", type=int, default=DEFAULT_LIMITS["ambiguous_scene"])
    parser.add_argument("--unsupported-material-limit", type=int, default=DEFAULT_LIMITS["unsupported_material"])
    args = parser.parse_args()

    try:
        report = build_taco_review_batch(
            download_plan=args.download_plan,
            output_dir=args.output_dir,
            seed=args.seed,
            multiple_objects_limit=args.multiple_objects_limit,
            ambiguous_scene_limit=args.ambiguous_scene_limit,
            unsupported_material_limit=args.unsupported_material_limit,
        )
    except TacoReviewBatchError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return 1

    print(f"Input plan rows: {report['input_plan_rows']}")
    print(f"Eligible plan rows: {report['eligible_plan_rows']}")
    print(f"Selected review rows: {report['selected_total_rows']}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
