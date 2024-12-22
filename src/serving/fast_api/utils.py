import os
from typing import Dict, List
import joblib
import pandas as pd
import psutil
from config.config import *
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.file_io import load_from_json


def get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        required_columns = load_from_json(REQUIRED_COLUMNS_JSON_PATH).get(
            "required_columns", []
        )
        columns_to_keep = load_from_json(FEATURES_JSON_PATH).get("columns_to_keep", [])
        if "fail" in columns_to_keep:
            columns_to_keep.remove("fail")
        return model, scaler, required_columns, columns_to_keep
    except Exception as e:
        raise RuntimeError(f"Error loading resources: {str(e)}")


def log_results(
    inference_df: pd.DataFrame,
    predictions: Dict,
    inference_time: float,
    memory_used: float,
    model_name: str,
    model_params: str,
    output_csv_path: str,
):
    try:
        inference_results = pd.DataFrame(
            {"y_pred": predictions["y_pred"], "y_proba": predictions["y_proba"]}
        )

        result_data = pd.concat([inference_df, inference_results], axis=1)
        result_data["inference_time"] = inference_time
        result_data["memory_usage_mb"] = memory_used
        result_data["model_name"] = model_name
        result_data["model_params"] = model_params

        file_exists = os.path.exists(output_csv_path)
        result_data.to_csv(
            output_csv_path, mode="a", header=not file_exists, index=False
        )
    except Exception as e:
        raise RuntimeError(f"Error logging results: {str(e)}")


def run_inference(
    model,
    scaler,
    inference_df: pd.DataFrame,
    required_columns: List[str],
    columns_to_keep: List[str],
) -> Dict[str, List]:
    try:
        predictions = run_inference_pipeline(
            model, scaler, inference_df, required_columns, columns_to_keep
        )
        return predictions
    except Exception as e:
        raise RuntimeError(f"Error during inference: {str(e)}")
