#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_poc.inference import ImageInferenceSession


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-image prediction and apply the needs_review routing policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    session = ImageInferenceSession.from_files(args.checkpoint, args.temperature, args.threshold_policy, args.device)
    result = session.predict(args.image)
    print(f"Prediction: {result['predicted_label']}")
    print(f"Raw confidence: {result['raw_top_confidence']:.2f}")
    print(f"Calibrated confidence: {result['calibrated_top_confidence']:.2f}")
    print(f"Routing policy threshold: {session.threshold_policy.get('selected_threshold')}")
    print(f"Recommended action: {result['recommended_action']}")
    print(f"Route: {result['route']}")
    if result["recommended_action"] == "needs_review":
        print(f"Reason: {result['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
