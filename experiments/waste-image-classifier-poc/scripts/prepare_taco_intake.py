#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from waste_poc.taco_intake import TacoIntakeError, prepare_taco_intake


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local, licence-aware TACO review candidates without approving them for V2 training.")
    parser.add_argument("--annotations", required=True, help="Local TACO COCO-style annotation JSON.")
    parser.add_argument("--image-root", help="Local root containing TACO image files referenced by file_name. Required unless --plan-only is used.")
    parser.add_argument("--label-mapping", required=True, help="Explicit TACO label mapping CSV.")
    parser.add_argument("--output-dir", required=True, help="Ignored local directory for inventory, draft candidates, report, and gallery.")
    parser.add_argument("--min-object-area-ratio", type=float, default=0.20)
    parser.add_argument("--plan-only", action="store_true", help="Write category inventory, licence ledger, and manual download plan without requiring local images.")
    args = parser.parse_args()
    if not args.plan_only and not args.image_root:
        parser.error("--image-root is required unless --plan-only is used")

    try:
        report = prepare_taco_intake(
            annotations=args.annotations,
            image_root=args.image_root,
            label_mapping=args.label_mapping,
            output_dir=args.output_dir,
            min_object_area_ratio=args.min_object_area_ratio,
            plan_only=args.plan_only,
        )
    except TacoIntakeError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return 1

    print(f"TACO images inspected: {report['total_images']}")
    print(f"Classifier draft rows: {report['classifier_draft_rows']}")
    print(f"Gate draft rows: {report['gate_draft_rows']}")
    print(f"Excluded rows: {report['excluded_rows']}")
    print(f"Download plan rows: {report['download_plan_rows']}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
