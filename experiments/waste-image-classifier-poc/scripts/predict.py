#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_poc.inference import predict_image
from waste_poc.thresholding import apply_policy
from waste_poc.utils import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-image prediction and apply the needs_review routing policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    temperature = read_json(args.temperature)["temperature"]
    policy = read_json(args.threshold_policy)
    result = predict_image(args.checkpoint, args.image, temperature=temperature, device_name=args.device)
    routing = apply_policy(result["predicted_label"], result["calibrated_top_confidence"], policy)
    print(f"Prediction: {result['predicted_label']}")
    print(f"Raw confidence: {result['raw_top_confidence']:.2f}")
    print(f"Calibrated confidence: {result['calibrated_top_confidence']:.2f}")
    print(f"Routing policy threshold: {policy.get('selected_threshold')}")
    print(f"Recommended action: {routing['recommended_action']}")
    print(f"Route: {routing['route']}")
    if routing["recommended_action"] == "needs_review":
        print(f"Reason: {routing['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
