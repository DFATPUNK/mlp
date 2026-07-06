from __future__ import annotations

import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_temperature_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax_np(np.asarray(logits, dtype=float) / max(float(temperature), 1e-6))


def expected_calibration_error(confidences, correctness, n_bins: int = 15) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))
    return ece


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    import torch

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature.clamp_min(1e-3), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())
