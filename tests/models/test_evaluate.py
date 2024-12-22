import unittest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.models.evaluate import evaluate_model
from src.models.predict import predict_model


class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        """Create mock data and model for testing."""
        self.X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]}
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0])
        self.X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [5, 4, 3]})
        self.y_test = pd.Series([0, 1, 0])

        self.model_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        self.model = GradientBoostingClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    def test_evaluate_model(self):
        """Test the evaluate_model function with a valid model."""
        y_pred, y_proba = predict_model(self.model, self.X_test)
        metrics = evaluate_model(self.y_test, y_pred, y_proba)

        self.assertIn("accuracy", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("roc_auc_score", metrics)
        self.assertIn("log_loss", metrics)

        self.assertTrue(isinstance(metrics["accuracy"], float))
        self.assertTrue(isinstance(metrics["classification_report"], dict))


if __name__ == "__main__":
    unittest.main()
