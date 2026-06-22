#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from waste_poc.inference import predict_image
from waste_poc.reporting import plot_confusion_matrix, save_gallery
from waste_poc.thresholding import apply_policy
from waste_poc.utils import CLASS_NAMES, EXTERNAL_ROUTING, EXTERNAL_SCENARIOS, read_json, validate_external_expected_label, write_json, write_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate manually supplied external challenge images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--external-manifest", default="data/external_images/external_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    manifest_path = ROOT / args.external_manifest
    image_root = manifest_path.parent / "images"
    if not manifest_path.exists() or not image_root.exists():
        print("No external image manifest/images found. Add 20 images to data/external_images/images/ and fill data/external_images/external_manifest.csv.")
        return 0
    frame = pd.read_csv(manifest_path).fillna("")
    frame = frame[frame["relative_path"].astype(str).str.len() > 0]
    if frame.empty:
        print("External manifest has no image rows yet. Add manually curated challenge images and rerun evaluate_external.py.")
        return 0
    temperature = read_json(args.temperature)["temperature"]
    policy = read_json(args.threshold_policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in frame.to_dict("records"):
        validate_external_expected_label(row.get("expected_label"))
        if row.get("scenario") not in EXTERNAL_SCENARIOS:
            raise ValueError(f"Unsupported scenario {row.get('scenario')!r}")
        if row.get("expected_routing") not in EXTERNAL_ROUTING:
            raise ValueError(f"Unsupported expected_routing {row.get('expected_routing')!r}")
        image_path = image_root / row["relative_path"]
        if not image_path.exists():
            rows.append({**row, "error": "missing image file"})
            continue
        result = predict_image(args.checkpoint, image_path, temperature, args.device)
        routing = apply_policy(result["predicted_label"], result["calibrated_top_confidence"], policy)
        expected_label = row.get("expected_label") or ""
        known_correct = (result["predicted_label"] == expected_label) if expected_label else None
        routing_matches = routing["recommended_action"] == "needs_review" if row["expected_routing"] == "needs_review" else routing["recommended_action"] == "auto_route"
        rows.append({**row, **result, "threshold": policy.get("selected_threshold"), "routing_decision": routing["route"], "known_in_scope_prediction_correct": known_correct, "routing_decision_matched_expected": routing_matches})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "external_predictions.csv", index=False)
    summary = {
        "external_images_evaluated": int(len(out)),
        "known_label_accuracy": float(out[out["expected_label"] != ""]["known_in_scope_prediction_correct"].mean()) if (out["expected_label"] != "").any() else None,
        "routing_match_rate": float(out["routing_decision_matched_expected"].mean()) if "routing_decision_matched_expected" in out else None,
        "notes": "This 20-image set is a qualitative challenge set, not a statistically representative benchmark.",
    }
    write_json(output_dir / "external_summary.json", summary)
    known = out[out["expected_label"].isin(CLASS_NAMES)]
    if not known.empty:
        import numpy as np
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix([CLASS_NAMES.index(x) for x in known["expected_label"]], [CLASS_NAMES.index(x) for x in known["predicted_label"]], labels=list(range(len(CLASS_NAMES))))
        plot_confusion_matrix(matrix, CLASS_NAMES, output_dir / "external_confusion_matrix.png")
    save_gallery(rows, image_root, output_dir / "external_prediction_gallery.png")
    save_gallery(rows, image_root, output_dir / "external_routing_gallery.png")
    failures = out[(out.get("known_in_scope_prediction_correct") == False) | (out.get("routing_decision_matched_expected") == False)]
    write_text(output_dir / "external_failure_notes.md", failures.to_markdown(index=False) if not failures.empty else "# External failure notes\n\nNo failures recorded in the supplied external set.\n")
    print(f"External images evaluated: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
