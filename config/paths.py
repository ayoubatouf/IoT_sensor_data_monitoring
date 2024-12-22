from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_data_raw.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "pretrained_models" / "trained_model.joblib"
SCALER_PATH = PROJECT_ROOT / "results" / "scaler.pkl"
PROCESSED_TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
PROCESSED_TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
EVALUATION_PATH = PROJECT_ROOT / "results" / "evaluation_results"
INFERECE_INPUT_PATH = PROJECT_ROOT / "data" / "inference" / "inference_input.csv"
INFERECE_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "inference_results" / "inference_results.csv"
)
EXPERIMENTS_LOG_PATH = PROJECT_ROOT / "logs" / "pipeline_execution.log"
PRODUCTION_LOG_PATH = PROJECT_ROOT / "logs" / "production_execution.csv"
REPORTS_PATH = PROJECT_ROOT / "reports" / "figures"
