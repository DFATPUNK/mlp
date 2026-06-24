from __future__ import annotations

from typing import Callable

MODEL_FAMILY = "clip_vit_b32_frozen_head"
DEFAULT_HF_MODEL_ID = "openai/clip-vit-base-patch32"
TRAINING_CODE_VERSION = "clip_frozen_head_v1"


def freeze_module_parameters(module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def trainable_parameter_names(module) -> list[str]:
    return [name for name, parameter in module.named_parameters() if getattr(parameter, "requires_grad", False)]


def assert_only_head_trainable(module, head_prefix: str = "classifier") -> None:
    unexpected = [name for name in trainable_parameter_names(module) if not name.startswith(head_prefix)]
    if unexpected:
        raise ValueError(f"Frozen CLIP candidate has trainable encoder parameters: {unexpected}")


def build_clip_frozen_head(num_classes: int, hf_model_id: str = DEFAULT_HF_MODEL_ID, vision_model_factory: Callable[[str], object] | None = None):
    import torch

    hf_model_id = hf_model_id or DEFAULT_HF_MODEL_ID
    if vision_model_factory is None:
        from transformers import CLIPVisionModel

        vision_model_factory = CLIPVisionModel.from_pretrained

    class ClipFrozenHeadClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_family = MODEL_FAMILY
            self.hf_model_id = hf_model_id
            self.vision_model = vision_model_factory(hf_model_id)
            freeze_module_parameters(self.vision_model)
            hidden_size = int(self.vision_model.config.hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, pixel_values):
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
                pooled = outputs.pooler_output
            return self.classifier(pooled)

    model = ClipFrozenHeadClassifier()
    assert_only_head_trainable(model)
    return model


def build_clip_transform(hf_model_id: str = DEFAULT_HF_MODEL_ID):
    from transformers import CLIPImageProcessor

    hf_model_id = hf_model_id or DEFAULT_HF_MODEL_ID
    processor = CLIPImageProcessor.from_pretrained(hf_model_id)

    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


def clip_metadata(config: dict, selected_hyperparameters: dict | None = None) -> dict:
    model_cfg = config.get("model", {})
    hf_model_id = model_cfg.get("hf_model_id", DEFAULT_HF_MODEL_ID)
    return {
        "model_family": MODEL_FAMILY,
        "hf_model_id": hf_model_id,
        "huggingface_model_id": hf_model_id,
        "frozen_encoder": True,
        "head_architecture": "linear",
        "preprocessing_identifier": f"CLIPImageProcessor::{hf_model_id}",
        "training_code_version": TRAINING_CODE_VERSION,
        "selected_hyperparameters": selected_hyperparameters or {},
    }
