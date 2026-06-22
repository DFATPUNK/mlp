from __future__ import annotations

import csv
import json
import math
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


from .utils import CLASS_NAMES, ensure_dir, sha256_file, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_trashnet_image_root(source_dir: str | Path, expected_classes: Iterable[str] = CLASS_NAMES) -> Path:
    source_dir = Path(source_dir)
    expected = set(expected_classes)
    candidates: list[Path] = []
    for path in [source_dir, *source_dir.rglob("*")]:
        if not path.is_dir():
            continue
        child_names = {child.name for child in path.iterdir() if child.is_dir()}
        if expected.issubset(child_names):
            image_count = sum(
                1
                for label in expected
                for image_path in (path / label).rglob("*")
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_count > 0:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not find a TrashNet image root containing class folders {sorted(expected)}")
    return min(candidates, key=lambda candidate: len(candidate.parts))


def validate_image(path: str | Path) -> dict:
    path = Path(path)
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format or path.suffix.lstrip(".").upper()
        return {"is_valid_image": True, "width": width, "height": height, "format": fmt, "reason": None}
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {"is_valid_image": False, "width": None, "height": None, "format": None, "reason": str(exc)}


def collect_image_rows(image_root: str | Path, source_commit: str, expected_classes: Iterable[str] = CLASS_NAMES) -> tuple[list[dict], list[dict]]:
    image_root = Path(image_root)
    valid_rows: list[dict] = []
    invalid_rows: list[dict] = []
    for label in expected_classes:
        class_dir = image_root / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder missing: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = image_path.relative_to(image_root).as_posix()
            validation = validate_image(image_path)
            if not validation["is_valid_image"]:
                invalid_rows.append({"relative_path": relative_path, "source_class": label, "label": label, **validation})
                continue
            digest = sha256_file(image_path)
            valid_rows.append(
                {
                    "image_id": f"{label}_{digest[:12]}_{uuid.uuid5(uuid.NAMESPACE_URL, relative_path).hex[:8]}",
                    "relative_path": relative_path,
                    "source_class": label,
                    "label": label,
                    "sha256": digest,
                    "width": validation["width"],
                    "height": validation["height"],
                    "format": validation["format"],
                    "source_commit": source_commit,
                    "is_valid_image": True,
                }
            )
    if not valid_rows:
        raise ValueError(f"No valid images found under {image_root}")
    return valid_rows, invalid_rows


def _largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(math.floor(value)) for split, value in raw.items()}
    for split, _ in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True)[: total - sum(counts.values())]:
        counts[split] += 1
    return counts


def assign_grouped_stratified_splits(rows: list[dict], ratios: dict[str, float], seed: int = 42) -> list[dict]:
    import random

    required = {"train", "validation", "test"}
    if set(ratios) != required:
        raise ValueError(f"Ratios must contain exactly {sorted(required)}")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    rng = random.Random(seed)
    groups_by_label: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups_by_label[row["label"]][row["sha256"]].append(row)

    assigned: list[dict] = []
    for label, groups in sorted(groups_by_label.items()):
        group_items = list(groups.values())
        rng.shuffle(group_items)
        target_counts = _largest_remainder_counts(sum(len(group) for group in group_items), ratios)
        current_counts = {"train": 0, "validation": 0, "test": 0}
        for group in sorted(group_items, key=lambda item: len(item), reverse=True):
            def deficit(split: str) -> tuple[float, float]:
                target = max(target_counts[split], 1)
                return ((target_counts[split] - current_counts[split]) / target, target_counts[split] - current_counts[split])

            split = max(["train", "validation", "test"], key=deficit)
            for row in group:
                assigned.append({**row, "split": split})
            current_counts[split] += len(group)
    return sorted(assigned, key=lambda row: row["image_id"])


def dataset_report(rows: list[dict], invalid_rows: list[dict], source_metadata: dict) -> dict:
    duplicate_groups = defaultdict(list)
    for row in rows:
        duplicate_groups[row["sha256"]].append(row["image_id"])
    duplicate_groups = {digest: ids for digest, ids in duplicate_groups.items() if len(ids) > 1}
    dims = [(int(row["width"]), int(row["height"])) for row in rows]
    return {
        "valid_images": len(rows),
        "invalid_images": len(invalid_rows),
        "invalid_files": invalid_rows,
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_count": sum(len(ids) for ids in duplicate_groups.values()),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "class_split_counts": {label: dict(Counter(row["split"] for row in rows if row["label"] == label)) for label in CLASS_NAMES},
        "image_dimensions_summary": {
            "min_width": min((width for width, _ in dims), default=None),
            "max_width": max((width for width, _ in dims), default=None),
            "min_height": min((height for _, height in dims), default=None),
            "max_height": max((height for _, height in dims), default=None),
        },
        "source_metadata": source_metadata,
    }


def write_manifest_outputs(rows: list[dict], invalid_rows: list[dict], source_metadata: dict, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "trashnet_manifest.csv"
    fieldnames = ["image_id", "relative_path", "source_class", "label", "split", "sha256", "width", "height", "format", "source_commit", "is_valid_image"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])
    write_json(output_dir / "trashnet_manifest.json", {"rows": rows, "schema": fieldnames})
    write_json(output_dir / "trashnet_dataset_report.json", dataset_report(rows, invalid_rows, source_metadata))
