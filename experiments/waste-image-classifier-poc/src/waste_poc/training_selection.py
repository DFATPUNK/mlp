from __future__ import annotations


def select_best_candidate_from_summaries(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.get("best_validation_macro_f1", -1.0))
