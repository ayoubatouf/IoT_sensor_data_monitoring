from config.config import *
from scripts.data_script import run_data_script
from scripts.inference_script import run_inference_script
from scripts.train_script import run_training_script
from src.utils.file_io import load_from_json
from src.utils.logger import setup_logger

logger = setup_logger("entire pipeline")


def run_all_scripts():
    try:
        logger.info("Starting data script execution")
        run_data_script(
            RAW_DATA_PATH,
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
            SCALER_PATH,
        )
        logger.info("Data script executed successfully")

        logger.info("Starting training script execution")
        run_training_script(
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
            load_from_json(MODEL_PARAMS_JSON_PATH),
            MODEL_PATH,
            EVALUATION_PATH,
        )
        logger.info("Training script executed successfully")

        logger.info("Starting inference script execution")
        run_inference_script(
            INFERECE_INPUT_PATH, MODEL_PATH, SCALER_PATH, INFERECE_OUTPUT_PATH
        )
        logger.info("Inference script executed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    logger.info("Script execution started")
    run_all_scripts()
    logger.info("Script execution completed")
