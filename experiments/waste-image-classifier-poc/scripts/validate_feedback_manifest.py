#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from waste_poc.feedback import FeedbackValidationError, validate_feedback_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a waste-image feedback manifest CSV.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        rows = validate_feedback_manifest(args.manifest)
    except FeedbackValidationError as exc:
        for message in exc.messages:
            print(message, file=sys.stderr)
        return 1
    print(f"Feedback manifest valid: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
