from __future__ import annotations

from pathlib import Path

from .calibration import apply_temperature_np, softmax_np
from .data import build_transforms
from .device import mps_operation_error_hint, resolve_device
from .images import load_rgb_image
from .model import build_model_from_checkpoint, load_checkpoint
from .thresholding import apply_policy
from .utils import read_json


class ImageInferenceSession:
    def __init__(self, checkpoint_path: str | Path, temperature: float | None = None, threshold_policy: dict | None = None, device_name: str | None = "auto"):
        import torch

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.class_names = self.checkpoint["class_names"]
        self.temperature = 1.0 if temperature is None else float(temperature)
        self.threshold_policy = threshold_policy
        self.device = resolve_device(device_name)
        self.model = build_model_from_checkpoint(self.checkpoint).to(self.device)
        self.model.eval()
        self.transform = build_transforms("test", self.checkpoint.get("image_size", 224))
        self.torch = torch

    @classmethod
    def from_files(cls, checkpoint_path: str | Path, temperature_path: str | Path | None = None, threshold_policy_path: str | Path | None = None, device_name: str | None = "auto"):
        temperature = read_json(temperature_path)["temperature"] if temperature_path else None
        policy = read_json(threshold_policy_path) if threshold_policy_path else None
        return cls(checkpoint_path, temperature, policy, device_name)

    @property
    def selected_device_name(self) -> str:
        return str(self.device)

    def predict(self, image_path: str | Path) -> dict:
        import numpy as np

        image = self.transform(load_rgb_image(image_path)).unsqueeze(0).to(self.device)
        try:
            with self.torch.no_grad():
                logits = self.model(image).detach().cpu().numpy()
        except RuntimeError as exc:
            if str(self.device) == "mps":
                raise mps_operation_error_hint(exc) from exc
            raise
        raw_probs = softmax_np(logits)[0]
        calibrated_probs = apply_temperature_np(logits, self.temperature)[0]
        pred_index = int(np.argmax(calibrated_probs))
        result = {
            "predicted_label": self.class_names[pred_index],
            "raw_top_confidence": float(raw_probs[pred_index]),
            "calibrated_top_confidence": float(calibrated_probs[pred_index]),
            "probabilities": {self.class_names[index]: float(value) for index, value in enumerate(calibrated_probs)},
            "logits": logits[0].tolist(),
        }
        if self.threshold_policy:
            result.update(apply_policy(result["predicted_label"], result["calibrated_top_confidence"], self.threshold_policy))
        return result


def predict_image(checkpoint_path: str | Path, image_path: str | Path, temperature: float | None = None, device_name: str | None = "auto") -> dict:
    return ImageInferenceSession(checkpoint_path, temperature=temperature, device_name=device_name).predict(image_path)
