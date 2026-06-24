#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prepare_external_output_dir = None
enrich_external_row = summarize_external_rows = write_failure_notes = None
PREPROCESSING_VERSION = None
ImageInferenceSession = None
dataframe_to_markdown_safe = plot_confusion_matrix = save_gallery = None
CLASS_NAMES = EXTERNAL_ROUTING = EXTERNAL_SCENARIOS = None
file_sha256_text = read_json = utc_now_iso = validate_external_expected_label = write_json = None



def _ensure_runtime_imports():
    global prepare_external_output_dir, enrich_external_row, summarize_external_rows, write_failure_notes
    global PREPROCESSING_VERSION, ImageInferenceSession, dataframe_to_markdown_safe, plot_confusion_matrix, save_gallery
    global CLASS_NAMES, EXTERNAL_ROUTING, EXTERNAL_SCENARIOS, file_sha256_text, read_json, utc_now_iso, validate_external_expected_label, write_json
    if prepare_external_output_dir is None:
        from waste_poc.external_io import prepare_external_output_dir as _prepare_external_output_dir
        prepare_external_output_dir = _prepare_external_output_dir
    if enrich_external_row is None:
        from waste_poc.external_metrics import enrich_external_row as _enrich_external_row, summarize_external_rows as _summarize_external_rows, write_failure_notes as _write_failure_notes
        enrich_external_row = _enrich_external_row
        summarize_external_rows = _summarize_external_rows
        write_failure_notes = _write_failure_notes
    if PREPROCESSING_VERSION is None:
        from waste_poc.images import PREPROCESSING_VERSION as _PREPROCESSING_VERSION
        PREPROCESSING_VERSION = _PREPROCESSING_VERSION
    if ImageInferenceSession is None:
        from waste_poc.inference import ImageInferenceSession as _ImageInferenceSession
        ImageInferenceSession = _ImageInferenceSession
    if dataframe_to_markdown_safe is None:
        from waste_poc.reporting import dataframe_to_markdown_safe as _dataframe_to_markdown_safe, plot_confusion_matrix as _plot_confusion_matrix, save_gallery as _save_gallery
        dataframe_to_markdown_safe = _dataframe_to_markdown_safe
        plot_confusion_matrix = _plot_confusion_matrix
        save_gallery = _save_gallery
    if CLASS_NAMES is None:
        from waste_poc.utils import CLASS_NAMES as _CLASS_NAMES, EXTERNAL_ROUTING as _EXTERNAL_ROUTING, EXTERNAL_SCENARIOS as _EXTERNAL_SCENARIOS, file_sha256_text as _file_sha256_text, read_json as _read_json, utc_now_iso as _utc_now_iso, validate_external_expected_label as _validate_external_expected_label, write_json as _write_json
        CLASS_NAMES = _CLASS_NAMES
        EXTERNAL_ROUTING = _EXTERNAL_ROUTING
        EXTERNAL_SCENARIOS = _EXTERNAL_SCENARIOS
        file_sha256_text = _file_sha256_text
        read_json = _read_json
        utc_now_iso = _utc_now_iso
        validate_external_expected_label = _validate_external_expected_label
        write_json = _write_json

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate manually supplied external challenge images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--external-manifest", default="data/external_images/external_manifest.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    import pandas as pd
    _ensure_runtime_imports()

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
    output_dir = prepare_external_output_dir(args.output_dir, args.checkpoint, args.overwrite)
    session = ImageInferenceSession.from_files(args.checkpoint, args.temperature, args.threshold_policy, args.device)
    policy = session.threshold_policy or read_json(args.threshold_policy)
    rows = []
    for row in frame.to_dict("records"):
        validate_external_expected_label(row.get("expected_label"))
        if row.get("scenario") not in EXTERNAL_SCENARIOS:
            raise ValueError(f"Unsupported scenario {row.get('scenario')!r}")
        if row.get("expected_routing") not in EXTERNAL_ROUTING:
            raise ValueError(f"Unsupported expected_routing {row.get('expected_routing')!r}")
        image_path = image_root / row["relative_path"]
        if not image_path.exists():
            rows.append(enrich_external_row({**row, "error": "missing image file", "recommended_action": "needs_review", "route": "needs_review"}))
            continue
        result = session.predict(image_path)
        rows.append(enrich_external_row({**row, **result, "threshold": policy.get("selected_threshold"), "routing_decision": result["route"]}))
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "external_predictions.csv", index=False)
    summary = summarize_external_rows(rows)
    write_json(output_dir / "external_summary.json", summary)
    metadata = {
        "checkpoint_sha256": file_sha256_text(args.checkpoint),
        "checkpoint_path": str(Path(args.checkpoint)),
        "temperature_sha256": file_sha256_text(args.temperature),
        "threshold_policy_sha256": file_sha256_text(args.threshold_policy),
        "external_manifest_sha256": file_sha256_text(manifest_path),
        "source_code_preprocessing_version": PREPROCESSING_VERSION,
        "exif_orientation_applied": True,
        "selected_device": session.selected_device_name,
        "timestamp": utc_now_iso(),
        "run_identifier": Path(args.checkpoint).resolve().parent.name,
    }
    write_json(output_dir / "external_evaluation_metadata.json", metadata)
    known = out[out["expected_label"].isin(CLASS_NAMES)]
    if not known.empty:
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix([CLASS_NAMES.index(x) for x in known["expected_label"]], [CLASS_NAMES.index(x) for x in known["predicted_label"]], labels=list(range(len(CLASS_NAMES))))
        plot_confusion_matrix(matrix, CLASS_NAMES, output_dir / "external_confusion_matrix.png")
    prediction_rows = [{**row, "gallery_badge": row.get("gallery_badge")} for row in rows]
    routing_rows = [{**row, "gallery_badge": row.get("policy_badge")} for row in rows]
    save_gallery(prediction_rows, image_root, output_dir / "external_prediction_gallery.png")
    save_gallery(routing_rows, image_root, output_dir / "external_routing_gallery.png")
    definitions = pd.DataFrame([{"field": key, "definition": value} for key, value in summary["field_definitions"].items()])
    write_failure_notes(output_dir / "external_failure_notes.md", rows, dataframe_to_markdown_safe(definitions))
    print(f"External images evaluated: {len(out)}")
    print(f"External evaluation outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
