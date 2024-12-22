import logging
import joblib
import pandas as pd
from config.config import *
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.file_io import load_from_json
from src.utils.logger import setup_logger

logger = setup_logger("inference")


def run_inference_script(inference_data_path, model_path, scaler_path, output_path):
    try:
        logger.info("Starting inference script execution")

        logger.info(f"Reading inference data from: {inference_data_path}")
        inference_data = pd.read_csv(inference_data_path)
        logger.info(f"Inference data loaded, shape: {inference_data.shape}")

        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")

        logger.info(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully")

        logger.info(f"Loading required columns from: {REQUIRED_COLUMNS_JSON_PATH}")
        required_columns_params = load_from_json(REQUIRED_COLUMNS_JSON_PATH)
        required_columns = required_columns_params.get("required_columns", [])
        logger.info(f"Required columns: {required_columns}")

        logger.info(f"Loading features from: {FEATURES_JSON_PATH}")
        features_params = load_from_json(FEATURES_JSON_PATH)
        columns_to_keep = features_params.get("columns_to_keep", [])
        columns_to_keep.remove("fail")
        logger.info(f"Columns to keep: {columns_to_keep}")

        logger.info("Running inference pipeline...")
        predictions = run_inference_pipeline(
            model, scaler, inference_data, required_columns, columns_to_keep
        )
        logger.info("Inference pipeline executed successfully")

        inference_results = pd.DataFrame(
            {"y_pred": predictions["y_pred"], "y_proba": predictions["y_proba"]}
        )
        result_data = pd.concat([inference_data, inference_results], axis=1)

        logger.info(f"Saving combined results to: {output_path}")
        result_data.to_csv(output_path, index=False)
        logger.info(f"Combined data (input + inference results) saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred in the inference script: {e}")
        raise


if __name__ == "__main__":

    inference_data_path = INFERECE_INPUT_PATH
    model_path = MODEL_PATH
    scaler_path = SCALER_PATH
    output_path = INFERECE_OUTPUT_PATH

    logger.info("Script execution started")
    run_inference_script(inference_data_path, model_path, scaler_path, output_path)
    logger.info("Script execution completed")
