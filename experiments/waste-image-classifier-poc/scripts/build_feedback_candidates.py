#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from waste_poc.feedback import FeedbackValidationError, build_feedback_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Build feedback-derived candidate dataset manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-promoted-external-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        summary = build_feedback_candidates(args.manifest, args.output_dir, args.allow_promoted_external_diagnostic)
    except FeedbackValidationError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return 1
    print(f"Classification candidate rows: {summary['classification_candidate_rows']}")
    print(f"Review-gate candidate rows: {summary['review_gate_candidate_rows']}")
    print(f"Candidate outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
