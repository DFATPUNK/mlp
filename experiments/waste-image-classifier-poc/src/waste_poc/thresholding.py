from __future__ import annotations

from collections import Counter
from typing import Sequence



def evaluate_thresholds(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float],
    class_names: Sequence[str],
    thresholds: Sequence[float],
) -> list[dict]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    confidences = [float(value) for value in confidences]
    if not (len(y_true) == len(y_pred) == len(confidences)):
        raise ValueError("y_true, y_pred, and confidences must have identical lengths")
    rows: list[dict] = []
    for threshold in thresholds:
        accepted = [confidence >= threshold for confidence in confidences]
        accepted_count = sum(1 for value in accepted if value)
        needs_review_count = len(accepted) - accepted_count
        if accepted_count:
            accepted_true = [truth for truth, keep in zip(y_true, accepted) if keep]
            accepted_pred = [pred for pred, keep in zip(y_pred, accepted) if keep]
            accepted_accuracy = sum(1 for truth, pred in zip(accepted_true, accepted_pred) if truth == pred) / accepted_count
            labels_present = sorted(set(accepted_true) | set(accepted_pred))
            per_label_f1 = []
            for label in labels_present:
                tp = sum(1 for truth, pred in zip(accepted_true, accepted_pred) if truth == label and pred == label)
                fp = sum(1 for truth, pred in zip(accepted_true, accepted_pred) if truth != label and pred == label)
                fn = sum(1 for truth, pred in zip(accepted_true, accepted_pred) if truth == label and pred != label)
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                per_label_f1.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
            accepted_macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else None
            class_distribution = Counter(class_names[int(label)] for label in accepted_pred)
        else:
            accepted_accuracy = None
            accepted_macro_f1 = None
            class_distribution = Counter()
        rows.append(
            {
                "threshold": float(threshold),
                "accepted_count": accepted_count,
                "needs_review_count": needs_review_count,
                "coverage": float(accepted_count / len(y_true)) if len(y_true) else 0.0,
                "accepted_accuracy": accepted_accuracy,
                "accepted_macro_f1": accepted_macro_f1,
                "accepted_class_distribution": dict(class_distribution),
            }
        )
    return rows


def select_threshold_policy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float],
    class_names: Sequence[str],
    target_auto_route_precision: float = 0.95,
    minimum_auto_route_coverage: float = 0.10,
    threshold_candidates: int = 101,
) -> dict:
    if threshold_candidates < 2:
        thresholds = [0.0]
    else:
        thresholds = [index / (threshold_candidates - 1) for index in range(threshold_candidates)]
    evaluated = evaluate_thresholds(y_true, y_pred, confidences, class_names, thresholds)
    all_classes = set(class_names)
    viable = [
        row
        for row in evaluated
        if row["accepted_accuracy"] is not None
        and row["accepted_accuracy"] >= target_auto_route_precision
        and row["coverage"] >= minimum_auto_route_coverage
    ]
    with_all_classes = [row for row in viable if all_classes.issubset(set(row["accepted_class_distribution"]))]
    candidates = with_all_classes or viable
    if not candidates:
        return {
            "policy_version": 1,
            "confidence_type": "temperature_scaled_top_class_probability",
            "selected_threshold": None,
            "target_auto_route_precision": target_auto_route_precision,
            "validation_auto_route_precision": None,
            "validation_coverage": 0.0,
            "needs_review_rate": 1.0,
            "auto_route_enabled": False,
            "selection_split": "validation",
            "threshold_grid": evaluated,
            "notes": [
                "No validation threshold met the requested precision and coverage constraints.",
                "All external images should be sent to needs_review until more representative data is available.",
                "Threshold selection used validation data only.",
            ],
        }
    selected = min(candidates, key=lambda row: row["threshold"])
    return {
        "policy_version": 1,
        "confidence_type": "temperature_scaled_top_class_probability",
        "selected_threshold": selected["threshold"],
        "target_auto_route_precision": target_auto_route_precision,
        "validation_auto_route_precision": selected["accepted_accuracy"],
        "validation_coverage": selected["coverage"],
        "needs_review_rate": 1.0 - selected["coverage"],
        "auto_route_enabled": True,
        "selection_split": "validation",
        "accepted_class_distribution": selected["accepted_class_distribution"],
        "threshold_grid": evaluated,
        "notes": [
            "Threshold chosen without using test data.",
            "Confidence is a routing signal, not a guarantee of correctness.",
        ],
    }


def apply_policy(predicted_label: str, calibrated_confidence: float, policy: dict) -> dict:
    if not policy.get("auto_route_enabled"):
        return {"recommended_action": "needs_review", "route": "needs_review", "reason": "auto-routing is disabled by the selected validation policy"}
    threshold = policy.get("selected_threshold")
    if threshold is not None and calibrated_confidence >= threshold:
        return {"recommended_action": "auto_route", "route": predicted_label, "reason": "calibrated confidence meets the selected validation threshold"}
    return {"recommended_action": "needs_review", "route": "needs_review", "reason": "model confidence is below the safe auto-routing threshold"}


def evaluate_policy_on_split(y_true, y_pred, confidences, class_names, policy: dict) -> dict:
    threshold = policy.get("selected_threshold")
    enabled = bool(policy.get("auto_route_enabled")) and threshold is not None
    if not enabled:
        return {"auto_route_enabled": False, "coverage": 0.0, "needs_review_rate": 1.0, "auto_route_precision": None}
    rows = evaluate_thresholds(y_true, y_pred, confidences, class_names, [threshold])[0]
    return {
        "auto_route_enabled": True,
        "threshold": threshold,
        "coverage": rows["coverage"],
        "needs_review_rate": 1.0 - rows["coverage"],
        "auto_route_precision": rows["accepted_accuracy"],
        "accepted_class_distribution": rows["accepted_class_distribution"],
    }
