import unittest
import pandas as pd
from src.data.loading_data import show_data_stats, validate_data
from src.data.preprocess_data import (
    count_outliers,
    list_outliers,
    remove_duplicates,
    scale_data,
    split_data,
)
from test_data_setup import create_test_data, get_required_columns


class TestDataProcessingFunctions(unittest.TestCase):

    def setUp(self):
        self.data = create_test_data()
        self.required_columns = get_required_columns()

    def test_show_data_stats(self):
        try:
            show_data_stats(self.data)
        except Exception as e:
            self.fail(f"show_data_stats raised an exception: {e}")

    def test_validate_data_valid(self):
        valid, reasons = validate_data(self.data, self.required_columns)
        self.assertTrue(valid)
        self.assertEqual(reasons, {})

    def test_validate_data_missing_column(self):
        invalid_columns = ["col1", "col3", "fail"]
        valid, reasons = validate_data(self.data, invalid_columns)
        self.assertFalse(valid)
        self.assertIn("missing_columns", reasons)
        self.assertIn("col3", reasons["missing_columns"])

    def test_validate_data_non_numeric_column(self):
        self.data["col3"] = ["a", "b", "c", "d", "e", "f"]
        invalid_columns = ["col1", "col3", "fail"]
        valid, reasons = validate_data(self.data, invalid_columns)
        self.assertFalse(valid)
        self.assertIn("non_numeric_columns", reasons)
        self.assertIn("col3", reasons["non_numeric_columns"])

    def test_remove_duplicates(self):
        data_with_duplicates = pd.concat(
            [self.data, self.data.iloc[0:1]], ignore_index=True
        )
        cleaned_data = remove_duplicates(data_with_duplicates)
        self.assertEqual(cleaned_data.shape[0], self.data.shape[0])

    def test_count_outliers(self):
        outlier_count = count_outliers(self.data, "col1")
        self.assertEqual(outlier_count, 0)

    def test_list_outliers(self):
        try:
            list_outliers(self.data)
        except Exception as e:
            self.fail(f"list_outliers raised an exception: {e}")

    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.data)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_split_data_missing_column(self):
        data_missing_column = self.data.drop(columns=["fail"])
        X_train, X_test, y_train, y_test = split_data(data_missing_column)
        self.assertIsNone(X_train)
        self.assertIsNone(X_test)
        self.assertIsNone(y_train)
        self.assertIsNone(y_test)

    def test_scale_data(self):
        X_train, X_test, y_train, y_test = split_data(self.data)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)

    def test_scale_data_invalid_type(self):
        X_train_invalid = "invalid_data"
        X_test_invalid = "invalid_data"
        X_train_scaled, X_test_scaled, scaler = scale_data(
            X_train_invalid, X_test_invalid
        )
        self.assertIsNone(X_train_scaled)
        self.assertIsNone(X_test_scaled)


if __name__ == "__main__":
    unittest.main()
