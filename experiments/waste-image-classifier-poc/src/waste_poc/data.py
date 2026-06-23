from __future__ import annotations

from pathlib import Path

import pandas as pd
from .images import load_rgb_image
from .utils import CLASS_NAMES, require_columns


class TrashNetManifestDataset:
    def __init__(self, manifest_path: str | Path, image_root: str | Path, split: str, transform=None, class_names: list[str] | None = None):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.split = split
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.label_to_index = {label: index for index, label in enumerate(self.class_names)}
        frame = pd.read_csv(self.manifest_path)
        require_columns(frame.columns, ["image_id", "relative_path", "label", "split"])
        self.frame = frame[(frame["split"] == split) & (frame["label"].isin(self.class_names))].reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = load_rgb_image(self.image_root / row.relative_path)
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_index[row.label], row.image_id, row.relative_path


def build_transforms(split: str, image_size: int = 224):
    from torchvision import transforms
    from torchvision.models import EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def compute_class_weights(manifest_path: str | Path, class_names: list[str] | None = None):
    import torch

    class_names = class_names or CLASS_NAMES
    frame = pd.read_csv(manifest_path)
    train = frame[frame["split"] == "train"]
    counts = train["label"].value_counts().reindex(class_names).fillna(0).to_numpy(dtype=float)
    counts = counts.clip(min=1.0)
    weights = counts.sum() / (len(class_names) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_transform_from_metadata(metadata: dict, split: str):
    from .clip_candidate import MODEL_FAMILY as CLIP_FAMILY, build_clip_transform

    family = metadata.get("model_family") or metadata.get("architecture") or metadata.get("model", {}).get("family")
    if family == CLIP_FAMILY:
        return build_clip_transform(metadata.get("hf_model_id") or metadata.get("model", {}).get("hf_model_id"))
    return build_transforms(split, metadata.get("image_size") or metadata.get("dataset", {}).get("image_size", 224))
