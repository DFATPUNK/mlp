from __future__ import annotations

from pathlib import Path

from .clip_candidate import MODEL_FAMILY as CLIP_FAMILY, build_clip_frozen_head, clip_metadata
from .device import resolve_device
from .utils import CLASS_NAMES, utc_now_iso

EFFICIENTNET_FAMILY = "efficientnet_b0"


def selected_device(requested: str | None = "auto"):
    return resolve_device(requested)


def infer_model_family(config_or_checkpoint: dict) -> str:
    model_cfg = config_or_checkpoint.get("model", {})
    return config_or_checkpoint.get("model_family") or model_cfg.get("family") or model_cfg.get("architecture") or config_or_checkpoint.get("architecture", EFFICIENTNET_FAMILY)


def create_efficientnet_b0(class_names: list[str] | None = None, pretrained_weights: str = "DEFAULT", mode: str = "frozen_backbone"):
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    class_names = class_names or CLASS_NAMES
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained_weights == "DEFAULT" else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    if mode == "frozen_backbone":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier")
    elif mode == "fine_tune":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("mode must be frozen_backbone or fine_tune")
    return model, weights


def create_model_from_config(config: dict, class_names: list[str], mode: str = "frozen_backbone"):
    family = infer_model_family(config)
    if family == CLIP_FAMILY:
        return build_clip_frozen_head(len(class_names), config.get("model", {}).get("hf_model_id")), None
    if family in {EFFICIENTNET_FAMILY, "efficientnet_b0"}:
        return create_efficientnet_b0(class_names, config.get("model", {}).get("pretrained_weights", "DEFAULT"), mode)
    raise ValueError(f"Unsupported model family: {family}")


def checkpoint_payload(model, optimizer, *, epoch: int, config: dict, class_names: list[str], manifest_hash: str, source_commit: str, run_id: str, training_mode: str, metrics: dict, selected_hyperparameters: dict | None = None, device_used: str | None = None) -> dict:
    family = infer_model_family(config)
    base_metadata = clip_metadata(config, selected_hyperparameters) if family == CLIP_FAMILY else {
        "model_family": EFFICIENTNET_FAMILY,
        "hf_model_id": None,
        "frozen_encoder": training_mode == "frozen_backbone",
        "head_architecture": "torchvision_efficientnet_classifier",
        "preprocessing_identifier": "torchvision EfficientNet_B0_Weights.DEFAULT",
        "training_code_version": "efficientnet_b0_transfer_v1",
        "selected_hyperparameters": selected_hyperparameters or {},
    }
    return {
        "architecture": base_metadata["model_family"],
        "model_family": base_metadata["model_family"],
        "hf_model_id": base_metadata["hf_model_id"],
        "head_architecture": base_metadata["head_architecture"],
        "preprocessing_identifier": base_metadata["preprocessing_identifier"],
        "training_code_version": base_metadata["training_code_version"],
        "selected_hyperparameters": base_metadata["selected_hyperparameters"],
        "device_used_for_training": device_used,
        "timestamp": utc_now_iso(),
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "label_to_index": {label: index for index, label in enumerate(class_names)},
        "index_to_label": {index: label for index, label in enumerate(class_names)},
        "image_size": config.get("dataset", {}).get("image_size", 224),
        "normalization": base_metadata["preprocessing_identifier"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
        "manifest_hash": manifest_hash,
        "split_manifest_sha256": manifest_hash,
        "source_commit": source_commit,
        "run_id": run_id,
        "training_mode": training_mode,
        "metrics": metrics,
    }


def metadata_from_checkpoint_payload(payload: dict, checkpoint_sha256: str | None = None) -> dict:
    keys = [
        "model_family",
        "hf_model_id",
        "head_architecture",
        "preprocessing_identifier",
        "training_code_version",
        "selected_hyperparameters",
        "device_used_for_training",
        "timestamp",
        "class_names",
        "label_to_index",
        "image_size",
        "manifest_hash",
        "split_manifest_sha256",
        "source_commit",
        "run_id",
        "training_mode",
        "metrics",
    ]
    metadata = {key: payload.get(key) for key in keys}
    metadata["checkpoint_sha256"] = checkpoint_sha256
    return metadata


def save_checkpoint(path: str | Path, payload: dict) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    import torch

    return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(checkpoint: dict):
    family = checkpoint.get("model_family") or checkpoint.get("architecture", EFFICIENTNET_FAMILY)
    if family == CLIP_FAMILY:
        model = build_clip_frozen_head(len(checkpoint["class_names"]), checkpoint.get("hf_model_id"))
    elif family in {EFFICIENTNET_FAMILY, "efficientnet_b0"}:
        model, _ = create_efficientnet_b0(checkpoint["class_names"], pretrained_weights="NONE", mode="fine_tune")
    else:
        raise ValueError(f"Unsupported checkpoint model_family {family!r}; cannot load safely")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
