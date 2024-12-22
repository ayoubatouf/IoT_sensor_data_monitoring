import pandas as pd


def show_data_stats(data):
    try:

        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        print(data.head())
        print(data.shape)
        print(data.info())
        print(data.describe())

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred while showing data statistics: {e}")


def validate_data(data, required_columns):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if not isinstance(required_columns, list):
            raise ValueError(
                "The required_columns parameter must be a list of column names."
            )

        missing_columns = [col for col in required_columns if col not in data.columns]
        non_numeric_columns = [
            col
            for col in required_columns
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col])
        ]
        missing_values_columns = [
            col
            for col in required_columns
            if col in data.columns and data[col].isnull().any()
        ]

        if missing_columns or non_numeric_columns or missing_values_columns:
            reasons = {
                "missing_columns": missing_columns,
                "non_numeric_columns": non_numeric_columns,
                "missing_values_columns": missing_values_columns,
            }
            return False, reasons

        return True, {}

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return False, {}
    except Exception as e:
        print(f"An error occurred while validating data: {e}")
        return False, {}
