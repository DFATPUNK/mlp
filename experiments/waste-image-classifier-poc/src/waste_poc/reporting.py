from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from .utils import ensure_dir, write_text


def plot_confusion_matrix(matrix, class_names: list[str], output_path: str | Path, normalized: bool = False) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.asarray(matrix), annot=True, fmt=".2f" if normalized else "d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized confusion matrix" if normalized else "Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confidence_histogram(confidences, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=20, range=(0, 1), color="#2563eb", alpha=0.85)
    plt.xlabel("Top-class confidence")
    plt.ylabel("Images")
    plt.title("Confidence distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_reliability(confidences, correctness, output_path: str | Path, bins: int = 15, title: str = "Reliability diagram") -> None:
    ensure_dir(Path(output_path).parent)
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)
    edges = np.linspace(0, 1, bins + 1)
    centers, accs = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            centers.append((lower + upper) / 2)
            accs.append(float(correctness[mask].mean()))
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.bar(centers, accs, width=1 / bins, alpha=0.75, edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_model_card(path: str | Path, context: dict) -> None:
    text = f"""# Model Card: Single-item waste image routing demonstration

## Purpose
Single-item waste image routing demonstration.

## Intended use
Classify one primary waste item in a whole image into one of: {', '.join(context.get('class_names', []))}. Low-confidence or ambiguous images should route to human review.

## Non-intended use
This is not object detection, does not identify multiple items individually, and must not be used as the sole basis for industrial waste-sorting decisions.

## Source dataset
TrashNet from `{context.get('repository_url', 'unknown')}` at commit `{context.get('source_commit', 'unknown')}`.

## Split methodology
Deterministic stratified 70/15/15 manifest split with exact duplicate SHA256 groups kept in one split.

## Model architecture
EfficientNet-B0 transfer learning, mode: `{context.get('training_mode', 'unknown')}`.

## Training configuration
```json
{context.get('training_config_json', '{}')}
```

## Internal validation results
```json
{context.get('validation_metrics_json', '{}')}
```

## Internal test results
```json
{context.get('test_metrics_json', '{}')}
```

## Calibration and needs-review policy
```json
{context.get('threshold_policy_json', '{}')}
```

## External challenge results
{context.get('external_summary', 'External challenge images were not evaluated yet.')}

## Known failure modes
- Controlled TrashNet backgrounds may not represent real workflow photos.
- Multi-item, low-light, blurred, or ambiguous images should be routed to needs_review.
- Confidence is a routing signal, not a guarantee of correctness.

## Ethical and operational limits
This proof of concept requires human review for uncertain predictions and should be validated with representative operational images before any production integration.
"""
    write_text(path, text)


def save_gallery(rows: list[dict], image_root: str | Path, output_path: str | Path, limit: int = 20) -> None:
    ensure_dir(Path(output_path).parent)
    rows = rows[:limit]
    if not rows:
        return
    cols = 4
    rows_count = int(np.ceil(len(rows) / cols))
    plt.figure(figsize=(cols * 4, rows_count * 4))
    for idx, row in enumerate(rows, start=1):
        ax = plt.subplot(rows_count, cols, idx)
        image_path = Path(image_root) / row["relative_path"]
        try:
            ax.imshow(Image.open(image_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "unreadable", ha="center")
        ax.axis("off")
        ax.set_title(f"{row.get('image_id')}\npred={row.get('predicted_label')} conf={row.get('calibrated_top_confidence')}\nroute={row.get('routing_decision')}", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
