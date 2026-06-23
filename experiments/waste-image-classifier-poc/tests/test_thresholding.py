import unittest

from waste_poc.thresholding import select_threshold_policy
from waste_poc.utils import CLASS_NAMES


class ThresholdingTests(unittest.TestCase):
    def test_selected_threshold_meets_precision_target(self):
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 4, 5, 1, 1, 2, 4]
        confidences = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.30, 0.92, 0.91, 0.20]
        policy = select_threshold_policy(y_true, y_pred, confidences, CLASS_NAMES, target_auto_route_precision=0.95, minimum_auto_route_coverage=0.10, threshold_candidates=101)
        self.assertTrue(policy["auto_route_enabled"])
        self.assertGreaterEqual(policy["validation_auto_route_precision"], 0.95)
        self.assertEqual(policy["selection_split"], "validation")

    def test_no_valid_threshold_disables_auto_routing(self):
        policy = select_threshold_policy([0, 1, 2], [1, 2, 0], [0.99, 0.98, 0.97], CLASS_NAMES, target_auto_route_precision=0.95, minimum_auto_route_coverage=0.10)
        self.assertFalse(policy["auto_route_enabled"])
        self.assertIsNone(policy["selected_threshold"])

    def test_no_test_labels_parameter_exists(self):
        with self.assertRaises(TypeError):
            select_threshold_policy([0], [0], [0.9], CLASS_NAMES, test_labels=[0])


if __name__ == "__main__":
    unittest.main()
