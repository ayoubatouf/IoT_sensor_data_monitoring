import pandas as pd


def keep_columns(data, columns_to_keep):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")

        if not isinstance(columns_to_keep, list):
            raise ValueError(
                "The columns_to_keep parameter must be a list of column names."
            )

        missing_columns = [col for col in columns_to_keep if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns are missing in the data: {', '.join(missing_columns)}"
            )

        return data[columns_to_keep]

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred while keeping columns: {e}")
        return None
