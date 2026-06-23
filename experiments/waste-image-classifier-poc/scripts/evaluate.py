#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_poc.calibration import apply_temperature_np, fit_temperature_scaling, softmax_np
from waste_poc.data import TrashNetManifestDataset, build_transforms
from waste_poc.device import resolve_num_workers
from waste_poc.metrics import classification_metrics, save_classification_report, save_metric_json
from waste_poc.model import build_model_from_checkpoint, load_checkpoint, selected_device
from waste_poc.reporting import plot_confidence_histogram, plot_confusion_matrix, plot_reliability, write_model_card
from waste_poc.thresholding import evaluate_policy_on_split, select_threshold_policy
from waste_poc.utils import read_json, write_json


def collect_logits(checkpoint: dict, checkpoint_path: Path, split: str, device_name: str | None = None):
    import torch

    device = selected_device(device_name)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    source_metadata = read_json(ROOT / "data" / "raw" / "trashnet_source_metadata.json") if (ROOT / "data" / "raw" / "trashnet_source_metadata.json").exists() else {}
    image_root = ROOT / source_metadata.get("source_directory_detected", "data/raw/trashnet-source")
    dataset = TrashNetManifestDataset(ROOT / checkpoint["config"]["dataset"]["manifest_path"], image_root, split, build_transforms(split, checkpoint.get("image_size", 224)), checkpoint["class_names"])
    worker_count = resolve_num_workers(checkpoint["config"]["training"].get("num_workers", "auto"))
    loader = DataLoader(dataset, batch_size=checkpoint["config"]["training"]["batch_size"], shuffle=False, num_workers=worker_count)
    logits, labels = [], []
    with torch.no_grad():
        for images, target, _, _ in tqdm(loader, leave=False):
            output = model(images.to(device)).detach().cpu().numpy()
            logits.append(output)
            labels.extend(target.numpy().tolist())
    return np.vstack(logits), np.asarray(labels)


def write_split_outputs(output_dir: Path, split: str, labels, probabilities, class_names):
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions, probabilities, class_names)
    save_metric_json(output_dir / f"metrics_{split}.json", metrics)
    save_classification_report(output_dir / f"classification_report_{split}.csv", labels, predictions, class_names)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / f"confusion_matrix_{split}.png")
    plot_confusion_matrix(metrics["confusion_matrix_normalized"], class_names, output_dir / f"confusion_matrix_{split}_normalized.png", normalized=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate validation/test splits, calibrate on validation, and select needs_review policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = checkpoint["class_names"]
    val_logits, val_labels = collect_logits(checkpoint, checkpoint_path, "validation", args.device)
    test_logits, test_labels = collect_logits(checkpoint, checkpoint_path, "test", args.device)
    val_probs_before = softmax_np(val_logits)
    val_pred_before = val_probs_before.argmax(axis=1)
    plot_reliability(val_probs_before.max(axis=1), val_pred_before == val_labels, output_dir / "reliability_before_calibration.png", title="Before temperature scaling")
    temperature = fit_temperature_scaling(val_logits, val_labels)
    write_json(output_dir / "temperature_scaling.json", {"method": "temperature_scaling", "temperature": temperature, "fit_split": "validation"})
    val_probs = apply_temperature_np(val_logits, temperature)
    test_probs = apply_temperature_np(test_logits, temperature)
    val_pred = val_probs.argmax(axis=1)
    plot_reliability(val_probs.max(axis=1), val_pred == val_labels, output_dir / "reliability_after_calibration.png", title="After temperature scaling")
    plot_confidence_histogram(val_probs.max(axis=1), output_dir / "confidence_distribution_validation.png")
    val_metrics = write_split_outputs(output_dir, "validation", val_labels, val_probs, class_names)
    test_metrics = write_split_outputs(output_dir, "test", test_labels, test_probs, class_names)
    routing_cfg = checkpoint["config"].get("routing_policy", {})
    policy = select_threshold_policy(val_labels, val_pred, val_probs.max(axis=1), class_names, routing_cfg.get("target_auto_route_precision", 0.95), routing_cfg.get("minimum_auto_route_coverage", 0.10), routing_cfg.get("threshold_candidates", 101))
    write_json(output_dir / "threshold_policy.json", policy)
    test_policy = evaluate_policy_on_split(test_labels, test_probs.argmax(axis=1), test_probs.max(axis=1), class_names, policy)
    write_json(output_dir / "test_policy_report.json", {"selection_split": "validation", "evaluation_split": "test", **test_policy})
    write_model_card(
        output_dir / "model_card.md",
        {
            "class_names": class_names,
            "repository_url": "https://github.com/garythung/trashnet.git",
            "source_commit": checkpoint.get("source_commit"),
            "training_mode": checkpoint.get("training_mode"),
            "training_config_json": json.dumps(checkpoint.get("config", {}), indent=2),
            "validation_metrics_json": json.dumps(val_metrics, indent=2),
            "test_metrics_json": json.dumps(test_metrics, indent=2),
            "threshold_policy_json": json.dumps(policy, indent=2),
        },
    )
    print(f"Validation macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Selected routing threshold: {policy.get('selected_threshold')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
