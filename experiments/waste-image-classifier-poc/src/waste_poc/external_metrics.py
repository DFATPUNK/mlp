from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir


def enrich_external_row(row: dict) -> dict:
    expected_label = (row.get("expected_label") or "").strip()
    expected_routing = row.get("expected_routing")
    predicted = row.get("predicted_label")
    action = row.get("recommended_action")
    is_auto = action == "auto_route"
    is_review = action == "needs_review"
    known_correct = (predicted == expected_label) if expected_label else None
    auto_route_is_correct = (is_auto and known_correct is True) if expected_label else None
    is_safe_abstention = bool(expected_label and is_review)
    expected_review_correct = bool(expected_routing == "needs_review" and is_review)
    is_unsafe_auto = bool(is_auto and ((expected_label and predicted != expected_label) or expected_routing == "needs_review"))
    policy_safe = bool((expected_label and ((is_auto and predicted == expected_label) or is_review)) or (expected_routing == "needs_review" and is_review))
    prediction_badge = "Unknown expected class" if not expected_label else ("Correct prediction" if known_correct else "Incorrect prediction")
    if is_auto and auto_route_is_correct:
        policy_badge = "Correct auto-route"
    elif is_safe_abstention:
        policy_badge = "Safe abstention"
    elif expected_review_correct:
        policy_badge = "Correct review routing"
    elif expected_routing == "needs_review" and is_auto:
        policy_badge = "Unsafe review miss"
    elif is_unsafe_auto:
        policy_badge = "Unsafe auto-route"
    else:
        policy_badge = "Needs review"
    return {
        **row,
        "known_label_top1_correct": known_correct,
        "is_auto_routed": is_auto,
        "is_needs_review": is_review,
        "auto_route_is_correct": auto_route_is_correct,
        "is_safe_abstention": is_safe_abstention,
        "expected_review_correctly_routed": expected_review_correct,
        "is_unsafe_auto_route": is_unsafe_auto,
        "policy_outcome_safe": policy_safe,
        "gallery_badge": prediction_badge,
        "policy_badge": policy_badge,
    }


def _rate(numerator: int, denominator: int):
    return None if denominator == 0 else numerator / denominator


def summarize_external_rows(rows: list[dict]) -> dict:
    total = len(rows)
    known = [row for row in rows if row.get("expected_label")]
    auto = [row for row in rows if row.get("is_auto_routed")]
    review = [row for row in rows if row.get("is_needs_review")]
    auto_known = [row for row in auto if row.get("expected_label")]
    expected_review = [row for row in rows if row.get("expected_routing") == "needs_review"]
    correct_auto = [row for row in auto_known if row.get("auto_route_is_correct") is True]
    incorrect_auto = [row for row in auto_known if row.get("auto_route_is_correct") is False]
    safe_abstentions = [row for row in known if row.get("is_safe_abstention")]
    correctly_reviewed = [row for row in expected_review if row.get("expected_review_correctly_routed")]
    unsafe_expected_review = [row for row in expected_review if row.get("is_auto_routed")]
    unsafe_auto = [row for row in rows if row.get("is_unsafe_auto_route")]
    policy_safe = [row for row in rows if row.get("policy_outcome_safe")]
    return {
        "external_images_evaluated": total,
        "known_label_count": len(known),
        "known_label_top1_accuracy": _rate(sum(1 for row in known if row.get("known_label_top1_correct") is True), len(known)),
        "auto_route_count": len(auto),
        "auto_route_coverage": _rate(len(auto), total),
        "needs_review_count": len(review),
        "needs_review_rate": _rate(len(review), total),
        "correct_auto_route_count": len(correct_auto),
        "incorrect_auto_route_count": len(incorrect_auto),
        "auto_route_precision_known_labels": _rate(len(correct_auto), len(auto_known)),
        "safe_abstention_count": len(safe_abstentions),
        "safe_abstention_rate_known_labels": _rate(len(safe_abstentions), len(known)),
        "expected_review_count": len(expected_review),
        "correctly_reviewed_count": len(correctly_reviewed),
        "review_recall_for_expected_review": _rate(len(correctly_reviewed), len(expected_review)),
        "unsafe_auto_route_on_expected_review_count": len(unsafe_expected_review),
        "unsafe_auto_route_count": len(unsafe_auto),
        "policy_safe_outcome_rate": _rate(len(policy_safe), total),
        "field_definitions": {
            "known_label_top1_correct": "Predicted label equals expected_label; blank when expected_label is blank.",
            "is_auto_routed": "The calibrated confidence policy routed directly to a material class.",
            "is_needs_review": "The calibrated confidence policy abstained for human review.",
            "auto_route_is_correct": "Auto-routed known-label image was routed to the expected class; blank without expected_label.",
            "is_safe_abstention": "Known-label image was conservatively sent to needs_review.",
            "expected_review_correctly_routed": "A row expected to need review was sent to needs_review.",
            "is_unsafe_auto_route": "The policy auto-routed to the wrong known class or auto-routed an expected-review image.",
            "policy_outcome_safe": "Correct auto-route or conservative needs_review for the row contract.",
        },
        "notes": [
            "This external set is qualitative and not statistically representative.",
            "External images were not used for training, calibration, threshold selection, or model selection.",
        ],
    }


def write_failure_notes(path: str | Path, rows: list[dict], markdown_table: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    unsafe = [row for row in rows if row.get("is_unsafe_auto_route")]
    safe = [row for row in rows if row.get("is_safe_abstention") and row.get("known_label_top1_correct") is False]
    review_miss = [row for row in rows if row.get("expected_routing") == "needs_review" and row.get("is_auto_routed")]
    lines = ["# External failure notes", ""]
    lines.append("## Priority 1: Unsafe automatic routes")
    if unsafe:
        for row in unsafe:
            lines.append(f"- `{row.get('image_id')}` expected `{row.get('expected_label') or 'review'}`, predicted `{row.get('predicted_label')}` at calibrated confidence `{row.get('calibrated_top_confidence')}` in scenario `{row.get('scenario')}`. Unsafe because an automatic route would bypass review.")
    else:
        lines.append("- None recorded.")
    lines.append("\n## Priority 2: Known-label misclassifications sent to needs_review")
    if safe:
        for row in safe:
            lines.append(f"- `{row.get('image_id')}` expected `{row.get('expected_label')}`, predicted `{row.get('predicted_label')}`, but was safely abstained to needs_review.")
    else:
        lines.append("- None recorded.")
    lines.append("\n## Priority 3: Expected-review images automatically routed")
    if review_miss:
        for row in review_miss:
            lines.append(f"- `{row.get('image_id')}` scenario `{row.get('scenario')}` was expected to need review but auto-routed to `{row.get('route')}`.")
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Row-level output field definitions", markdown_table])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
