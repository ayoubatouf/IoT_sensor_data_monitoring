import logging
import joblib
import pandas as pd
from config.config import *
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.file_io import load_from_json
from src.utils.metrics import save_metrics
from src.utils.logger import setup_logger

logger = setup_logger("training")


def run_training_script(
    train_data_path, test_data_path, model_params, model_save_path, metrics_path
):

    try:
        logger.info("Starting training script execution")

        logger.info(f"Reading training data from: {train_data_path}")
        train_data = pd.read_csv(train_data_path)
        logger.info(f"Training data loaded, shape: {train_data.shape}")

        logger.info(f"Reading testing data from: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Testing data loaded, shape: {test_data.shape}")

        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]

        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        logger.info("Running training pipeline...")
        model, predictions, metrics = run_training_pipeline(
            X_train, y_train, X_test, y_test, model_params
        )
        logger.info("Training pipeline executed successfully")

        logger.info(f"Saving trained model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        logger.info("Model saved successfully")

        logger.info(f"Saving evaluation metrics to: {metrics_path}")
        save_metrics(metrics, predictions, y_test, metrics_path)
        logger.info("Metrics saved successfully")

        logger.info("Training script execution completed successfully")

    except Exception as e:
        logger.error(f"An error occurred in the training script: {e}")
        raise


if __name__ == "__main__":

    train_data_path = PROCESSED_TRAIN_DATA_PATH
    test_data_path = PROCESSED_TEST_DATA_PATH
    model_params = load_from_json(MODEL_PARAMS_JSON_PATH)
    model_save_path = MODEL_PATH
    metrics_path = EVALUATION_PATH

    logger.info("Script execution started")
    run_training_script(
        train_data_path, test_data_path, model_params, model_save_path, metrics_path
    )
    logger.info("Script execution completed")
