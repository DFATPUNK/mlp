import unittest

from waste_poc.clip_candidate import MODEL_FAMILY, assert_only_head_trainable, clip_metadata, freeze_module_parameters, trainable_parameter_names
from waste_poc.model import build_model_from_checkpoint


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class FakeModule:
    def __init__(self):
        self.encoder_weight = FakeParameter(True)
        self.classifier_weight = FakeParameter(True)

    def parameters(self):
        return [self.encoder_weight, self.classifier_weight]

    def named_parameters(self):
        return [("vision_model.weight", self.encoder_weight), ("classifier.weight", self.classifier_weight)]


class ClipCandidateTests(unittest.TestCase):
    def test_freeze_helper_marks_encoder_parameters_not_trainable(self):
        module = FakeModule()
        freeze_module_parameters(module)
        self.assertFalse(module.encoder_weight.requires_grad)
        self.assertFalse(module.classifier_weight.requires_grad)

    def test_only_head_trainable_contract(self):
        module = FakeModule()
        module.encoder_weight.requires_grad = False
        self.assertEqual(trainable_parameter_names(module), ["classifier.weight"])
        assert_only_head_trainable(module)
        module.encoder_weight.requires_grad = True
        with self.assertRaisesRegex(ValueError, "trainable encoder"):
            assert_only_head_trainable(module)

    def test_clip_metadata_identifies_family_and_preprocessor(self):
        metadata = clip_metadata({"model": {"hf_model_id": "openai/clip-vit-base-patch32"}}, {"learning_rate": 0.001})
        self.assertEqual(metadata["model_family"], MODEL_FAMILY)
        self.assertIn("CLIPImageProcessor", metadata["preprocessing_identifier"])
        self.assertEqual(metadata["selected_hyperparameters"]["learning_rate"], 0.001)

    def test_unknown_checkpoint_family_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint model_family"):
            build_model_from_checkpoint({"model_family": "unknown_family", "class_names": ["trash"], "model_state_dict": {}})


if __name__ == "__main__":
    unittest.main()
