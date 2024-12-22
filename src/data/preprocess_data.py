import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def remove_duplicates(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        return data.drop_duplicates()

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while removing duplicates: {e}")


def count_outliers(data, column):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' is not numeric.")

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return len(outliers)

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return 0
    except Exception as e:
        print(f"An error occurred while counting outliers in column '{column}': {e}")
        return 0


def list_outliers(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                num_outliers = count_outliers(data, column)
                print(f"Number of outliers in {column}: {num_outliers}")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while listing outliers: {e}")


def split_data(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if "fail" not in data.columns:
            raise ValueError("The column 'fail' is not present in the data.")

        X = data.drop(columns=["fail"])
        y = data["fail"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while splitting data: {e}")
        return None, None, None, None


def scale_data(X_train, X_test):
    try:
        if not isinstance(X_train, pd.DataFrame) or not isinstance(
            X_test, pd.DataFrame
        ):
            raise ValueError("Both X_train and X_test must be pandas DataFrames.")

        scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while scaling data: {e}")
        return None, None, None
