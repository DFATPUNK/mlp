import unittest

try:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import main
    MISSING_DEPENDENCY = None
except ModuleNotFoundError as exc:
    MISSING_DEPENDENCY = exc


@unittest.skipIf(MISSING_DEPENDENCY is not None, "ML training dependencies are not installed")
class PredictionExplanationTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({"age": [20, 22, 24, 45, 48, 52, 55, 58], "hours": [20, 25, 30, 40, 45, 50, 55, 60]})
        self.y = ["low", "low", "low", "high", "high", "high", "high", "high"]
        self.pipeline = Pipeline([
            ("preprocess", ColumnTransformer([("num", StandardScaler(), ["age", "hours"])])),
            ("model", RandomForestClassifier(n_estimators=7, max_depth=3, random_state=7)),
        ])
        self.pipeline.fit(self.x, self.y)
        self.frame = pd.DataFrame([{"age": 54, "hours": 52}])
        self.trained_content = {
            "recommended_model_name": "Random Forest",
            "preprocessing": {"numeric_columns": ["age", "hours"], "categorical_columns": [], "boolean_columns": []},
        }
        self.prediction = main.prediction_payload_for_pipeline("tabular_classification", self.pipeline, self.frame)

    def test_random_forest_tree_labels_align_to_forest_label_space(self):
        explanation = main.build_random_forest_classifier_explanation(self.pipeline, self.trained_content, self.frame, self.prediction)
        forest_labels = {str(label) for label in self.pipeline.named_steps["model"].classes_}
        tree_predictions = {str(tree["tree_prediction"]) for tree in explanation["forest_vote"]["trees"]}
        self.assertTrue(tree_predictions <= forest_labels)
        self.assertEqual(
            explanation["forest_vote"]["tree_count"],
            explanation["forest_vote"]["agreement_count"] + explanation["forest_vote"]["disagreement_count"],
        )
        self.assertGreater(explanation["forest_vote"]["agreement_count"], 0)

    def test_tree_detail_uses_requested_tree_index(self):
        detail = main.build_tree_detail("tabular_classification", self.pipeline, self.trained_content, self.frame, self.prediction, 3)
        self.assertEqual(detail["tree_index"], 3)
        self.assertIn(str(detail["tree_prediction"]), {str(label) for label in self.pipeline.named_steps["model"].classes_})
        self.assertIn("decision_path", detail)


if __name__ == "__main__":
    unittest.main()
