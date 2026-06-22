#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_poc.config import apply_overrides, load_config
from waste_poc.data import TrashNetManifestDataset, build_transforms, compute_class_weights
from waste_poc.model import checkpoint_payload, create_efficientnet_b0, save_checkpoint, selected_device
from waste_poc.utils import CLASS_NAMES, file_sha256_text, read_json, set_seed, write_json


def run_epoch(model, loader, criterion, optimizer, device, use_amp: bool, training: bool):
    import torch

    model.train(training)
    losses, labels, preds = [], [], []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for images, target, _, _ in tqdm(loader, leave=False):
        images, target = images.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, target)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        labels.extend(target.detach().cpu().tolist())
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {"loss": sum(losses) / max(len(losses), 1), "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def train_phase(config: dict, run_dir: Path, mode: str, resume_checkpoint: Path | None = None) -> Path:
    import torch

    device = selected_device(config.get("device"))
    use_amp = device.type == "cuda"
    class_names = config["dataset"]["class_names"]
    manifest_path = ROOT / config["dataset"]["manifest_path"]
    metadata_path = ROOT / "data" / "raw" / "trashnet_source_metadata.json"
    source_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    image_root = ROOT / source_metadata.get("source_directory_detected", "data/raw/trashnet-source")
    train_ds = TrashNetManifestDataset(manifest_path, image_root, "train", build_transforms("train", config["dataset"]["image_size"]), class_names)
    val_ds = TrashNetManifestDataset(manifest_path, image_root, "validation", build_transforms("validation", config["dataset"]["image_size"]), class_names)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])
    model, _ = create_efficientnet_b0(class_names, config["model"].get("pretrained_weights", "DEFAULT"), mode)
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    weights = compute_class_weights(manifest_path, class_names).to(device) if config["training"].get("use_class_weights", True) else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    lr = config["training"]["learning_rate_fine_tune" if mode == "fine_tune" else "learning_rate_frozen"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=config["training"]["weight_decay"])
    epochs = config["training"]["fine_tune_epochs" if mode == "fine_tune" else "frozen_epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_f1, stale_epochs = -1.0, 0
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, use_amp, training=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, use_amp, training=False)
        metrics = {"train": train_metrics, "validation": val_metrics, "epoch": epoch}
        payload = checkpoint_payload(model, optimizer, epoch=epoch, config=config, class_names=class_names, manifest_hash=file_sha256_text(manifest_path), source_commit=source_metadata.get("resolved_commit_sha", "unknown"), run_id=run_dir.name, training_mode=mode, metrics=metrics)
        save_checkpoint(run_dir / "latest_checkpoint.pt", payload)
        write_json(run_dir / "latest_metrics.json", metrics)
        print(f"Epoch {epoch}: val macro F1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            save_checkpoint(best_path, payload)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping triggered")
                break
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 transfer-learning baseline on the TrashNet manifest.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["frozen_backbone", "fine_tune"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    config = apply_overrides(config, {"seed": args.seed, "device": args.device, "model.mode": args.mode, "training.batch_size": args.batch_size})
    if args.epochs is not None:
        config["training"]["frozen_epochs" if config["model"]["mode"] == "frozen_backbone" else "fine_tune_epochs"] = args.epochs
    set_seed(config.get("seed", 42))
    mode = config["model"].get("mode", "frozen_backbone")
    run_id = args.output_dir or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{mode}"
    run_dir = ROOT / "artifacts" / "runs" / run_id if not Path(run_id).is_absolute() else Path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    best = train_phase(config, run_dir, mode, Path(args.resume_checkpoint) if args.resume_checkpoint else None)
    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
