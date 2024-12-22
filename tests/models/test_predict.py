import unittest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.models.predict import predict_model


class TestPredictModel(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]}
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0])
        self.X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [5, 4, 3]})
        self.y_test = pd.Series([0, 1, 0])

        self.model_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        self.model = GradientBoostingClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    def test_predict_model(self):
        y_pred, y_proba = predict_model(self.model, self.X_test)

        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])
        self.assertEqual(y_proba.shape[0], self.X_test.shape[0])

    def test_predict_model_invalid_input(self):
        invalid_model = object()
        y_pred, y_proba = predict_model(invalid_model, self.X_test)
        self.assertIsNone(y_pred)
        self.assertIsNone(y_proba)

        y_pred, y_proba = predict_model(self.model, "invalid_input")
        self.assertIsNone(y_pred)
        self.assertIsNone(y_proba)


if __name__ == "__main__":
    unittest.main()
