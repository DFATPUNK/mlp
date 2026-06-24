from __future__ import annotations

from pathlib import Path

from .utils import read_json


def read_best_metrics(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "best_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing durable candidate metric artifact: {path}")
    metrics = read_json(path)
    if "best_validation_macro_f1" not in metrics:
        raise ValueError(f"{path} must contain best_validation_macro_f1")
    if "checkpoint_path" not in metrics:
        raise ValueError(f"{path} must contain checkpoint_path")
    return metrics


def select_best_candidate_by_validation_metric(candidate_dirs: list[str | Path]) -> tuple[Path, dict]:
    if not candidate_dirs:
        raise ValueError("At least one candidate directory is required")
    ranked = []
    for candidate_dir in candidate_dirs:
        path = Path(candidate_dir)
        metrics = read_best_metrics(path)
        ranked.append((float(metrics["best_validation_macro_f1"]), path, metrics))
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, selected_dir, selected_metrics = ranked[0]
    return selected_dir, selected_metrics
