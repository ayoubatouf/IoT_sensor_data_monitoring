import joblib
import pandas as pd
from config.config import *
from src.pipelines.data_pipeline import run_data_pipeline
from src.utils.file_io import load_from_json
from src.utils.logger import setup_logger

logger = setup_logger("data")


def run_data_script(raw_data_path, output_train_path, output_test_path, scaler_path):
    try:
        logger.info("Starting data script execution")

        logger.info(f"Reading raw data from: {raw_data_path}")
        raw_data = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded, shape: {raw_data.shape}")

        logger.info(f"Loading features from: {FEATURES_JSON_PATH}")
        features_params = load_from_json(FEATURES_JSON_PATH)
        columns_to_keep = features_params.get("columns_to_keep", [])
        logger.info(f"Columns to keep: {columns_to_keep}")

        logger.info(f"Loading required columns from: {REQUIRED_COLUMNS_JSON_PATH}")
        required_columns_params = load_from_json(REQUIRED_COLUMNS_JSON_PATH)
        required_columns = required_columns_params.get("required_columns", [])
        logger.info(f"Required columns: {required_columns}")

        logger.info("Running data pipeline...")
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = run_data_pipeline(
            raw_data, columns_to_keep, required_columns
        )
        logger.info("Data pipeline executed successfully")

        feature_columns = [col for col in columns_to_keep if col != "fail"]
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

        y_train = pd.DataFrame(y_train).reset_index(drop=True)
        y_test = pd.DataFrame(y_test).reset_index(drop=True)

        train_data = pd.concat(
            [X_train_scaled, y_train.rename(columns={0: "fail"})], axis=1
        )
        test_data = pd.concat(
            [X_test_scaled, y_test.rename(columns={0: "fail"})], axis=1
        )

        logger.info(f"Saving training data to: {output_train_path}")
        train_data.to_csv(output_train_path, index=False)

        logger.info(f"Saving test data to: {output_test_path}")
        test_data.to_csv(output_test_path, index=False)

        logger.info(f"Saving scaler to: {scaler_path}")
        joblib.dump(scaler, scaler_path)

        logger.info("Data script executed successfully")

    except Exception as e:
        logger.error(f"An error occurred in the data script: {e}")
        raise


if __name__ == "__main__":

    raw_data_path = RAW_DATA_PATH
    output_train_path = PROCESSED_TRAIN_DATA_PATH
    output_test_path = PROCESSED_TEST_DATA_PATH
    scaler_path = SCALER_PATH

    logger.info("Script execution started")
    run_data_script(raw_data_path, output_train_path, output_test_path, scaler_path)
    logger.info("Script execution completed")
