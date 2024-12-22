import pandas as pd


def create_test_data():
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, 6],
            "col2": [2, 3, 4, 5, 6, 7],
            "fail": [0, 1, 0, 1, 0, 1],
        }
    )


def get_required_columns():
    return ["col1", "col2", "fail"]
