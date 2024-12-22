import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV
from config.data import SEARCH_SPACE_JSON_PATH
from config.paths import PROCESSED_TRAIN_DATA_PATH
from src.utils.file_io import load_from_json


def tune_gbc_with_bayes(X_train, y_train, param_space, n_iter=150, cv=4, random_state=42):
    try:
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            raise ValueError(
                "X_train must be a pandas DataFrame and y_train must be a pandas Series."
            )

        if not isinstance(param_space, dict):
            raise ValueError(
                "param_space must be a dictionary representing the search space."
            )

        gbc = GradientBoostingClassifier(random_state=random_state)

        opt = BayesSearchCV(
            estimator=gbc,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        opt.fit(X_train, y_train)

        return opt.best_params_

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred while tuning the GradientBoostingClassifier: {e}")
        return None


if __name__ == "__main__":

    try:
        data = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)

        if "fail" not in data.columns:
            raise ValueError(
                f"The expected target column 'fail' is not found in the dataset at {PROCESSED_TRAIN_DATA_PATH}."
            )

        X_train = data.drop(columns=["fail"])
        y_train = data["fail"]

        try:
            param_space = load_from_json(SEARCH_SPACE_JSON_PATH)
        except FileNotFoundError:
            print(
                f"Error: The parameter search space file was not found at {SEARCH_SPACE_JSON_PATH}."
            )
            param_space = None
        except Exception as e:
            print(f"Error while loading parameter search space: {e}")
            param_space = None

        if param_space:
            best_params = tune_gbc_with_bayes(X_train, y_train, param_space)

            if best_params:
                print("Best parameters found:", best_params)
            else:
                print("Hyperparameter tuning failed.")
        else:
            print(
                "Skipping hyperparameter tuning due to parameter space loading error."
            )

    except FileNotFoundError:
        print(f"Error: The data file was not found at {PROCESSED_TRAIN_DATA_PATH}.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
