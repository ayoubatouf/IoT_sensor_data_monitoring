import numpy as np
import pandas as pd


def predict_model(model, X_test):
    try:
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise ValueError(
                "The provided model is not a valid trained model, it must have 'predict' and 'predict_proba' methods."
            )

        if not isinstance(X_test, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError(
                "X_test must be a pandas DataFrame, Series, or numpy array."
            )

        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            if X_test.shape[1] != expected_features:
                raise ValueError(
                    f"X_test has {X_test.shape[1]} features, but the model expects {expected_features} features."
                )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_proba

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None
    except Exception as e:
        print(f"An error occurred while making predictions: {e}")
        return None, None
