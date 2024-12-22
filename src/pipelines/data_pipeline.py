from src.data.loading_data import validate_data
from src.data.preprocess_data import remove_duplicates, scale_data, split_data
from src.features.build_features import keep_columns


def run_data_pipeline(data, columns_to_keep, required_columns):

    if validate_data(data, required_columns):

        data = remove_duplicates(data)

        data = keep_columns(data, columns_to_keep)

        X_train, X_test, y_train, y_test = split_data(data)

        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
