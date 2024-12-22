import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def train_model(X_train, y_train, model_params):
    try:
        if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
            raise ValueError("X_train must be a pandas DataFrame or numpy ndarray.")

        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise ValueError("y_train must be a pandas Series or numpy ndarray.")

        if not isinstance(model_params, dict):
            raise ValueError(
                "model_params must be a dictionary containing the model hyperparameters."
            )

        gbc = GradientBoostingClassifier(**model_params)

        gbc.fit(X_train, y_train)

        return gbc

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except TypeError as te:
        print(f"TypeError: {te}")
        return None
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None
