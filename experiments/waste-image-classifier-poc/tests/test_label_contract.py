import unittest

from waste_poc.utils import CLASS_NAMES, validate_external_expected_label, validate_training_labels


class LabelContractTests(unittest.TestCase):
    def test_only_six_training_labels_are_accepted(self):
        validate_training_labels(CLASS_NAMES)
        with self.assertRaises(ValueError):
            validate_training_labels([*CLASS_NAMES, "compost"])

    def test_needs_review_is_not_training_class(self):
        with self.assertRaisesRegex(ValueError, "needs_review"):
            validate_training_labels(["cardboard", "needs_review"])

    def test_external_manifest_may_have_blank_expected_label(self):
        validate_external_expected_label("")
        validate_external_expected_label(None)
        validate_external_expected_label("trash")
        with self.assertRaises(ValueError):
            validate_external_expected_label("needs_review")


if __name__ == "__main__":
    unittest.main()
