from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss, precision_recall_fscore_support

from .calibration import expected_calibration_error
from .utils import ensure_dir, write_json


def classification_metrics(y_true, y_pred, probabilities, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    top_conf = np.asarray(probabilities).max(axis=1)
    correctness = y_true == y_pred
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            class_name: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, class_name in enumerate(class_names)
        },
        "negative_log_likelihood": float(log_loss(y_true, probabilities, labels=list(range(len(class_names))))),
        "brier_score_top_class": float(np.mean((top_conf - correctness.astype(float)) ** 2)),
        "expected_calibration_error": float(expected_calibration_error(top_conf, correctness)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "confusion_matrix_normalized": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true").tolist(),
    }


def save_classification_report(path: str | Path, y_true, y_pred, class_names: list[str]) -> None:
    ensure_dir(Path(path).parent)
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(path)


def save_metric_json(path: str | Path, metrics: dict) -> None:
    write_json(path, metrics)
