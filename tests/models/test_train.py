import unittest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.models.train import train_model


class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]}
        )
        self.y_train = pd.Series([0, 1, 0, 1, 0])
        self.model_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train, self.model_params)
        self.assertIsInstance(model, GradientBoostingClassifier)

    def test_train_model_invalid_input(self):
        model = train_model("invalid_X_train", self.y_train, self.model_params)
        self.assertIsNone(model)

        model = train_model(self.X_train, "invalid_y_train", self.model_params)
        self.assertIsNone(model)

        model_params_invalid = {
            "n_estimators": "invalid",
            "learning_rate": 0.1,
            "max_depth": 3,
        }
        model = train_model(self.X_train, self.y_train, model_params_invalid)
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()
