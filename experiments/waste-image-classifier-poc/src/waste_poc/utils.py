from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NEEDS_REVIEW = "needs_review"
EXTERNAL_SCENARIOS = {
    "in_scope_clean",
    "in_scope_hard",
    "multi_item",
    "non_waste",
    "ambiguous",
    "low_light",
    "blurred",
}
EXTERNAL_ROUTING = {"auto_route_if_confident", "needs_review"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_text(path: str | Path) -> str:
    return sha256_file(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def validate_training_labels(labels: Iterable[str]) -> None:
    invalid = sorted({label for label in labels if label not in CLASS_NAMES})
    if NEEDS_REVIEW in invalid:
        raise ValueError("needs_review is a routing policy outcome, not a training class")
    if invalid:
        raise ValueError(f"Unsupported training labels: {invalid}. Expected only {CLASS_NAMES}")


def validate_external_expected_label(label: str | None) -> None:
    if label is None or str(label).strip() == "":
        return
    if label not in CLASS_NAMES:
        raise ValueError(f"Unsupported external expected_label {label!r}; leave blank for out-of-scope images")


def require_columns(columns: Sequence[str], required: Sequence[str]) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
