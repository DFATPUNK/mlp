from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import apply_temperature_np, softmax_np
from .data import build_transforms
from .model import build_model_from_checkpoint, load_checkpoint, selected_device
from .thresholding import apply_policy


def predict_image(checkpoint_path: str | Path, image_path: str | Path, temperature: float | None = None, device_name: str | None = None) -> dict:
    import torch

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    device = selected_device(device_name)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    transform = build_transforms("test", checkpoint.get("image_size", 224))
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image).detach().cpu().numpy()
    raw_probs = softmax_np(logits)[0]
    calibrated_probs = apply_temperature_np(logits, temperature or 1.0)[0]
    pred_index = int(np.argmax(calibrated_probs))
    return {
        "predicted_label": class_names[pred_index],
        "raw_top_confidence": float(raw_probs[pred_index]),
        "calibrated_top_confidence": float(calibrated_probs[pred_index]),
        "probabilities": {class_names[index]: float(value) for index, value in enumerate(calibrated_probs)},
        "logits": logits[0].tolist(),
    }
