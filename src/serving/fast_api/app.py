import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from config.paths import PRODUCTION_LOG_PATH
from src.serving.fast_api.utils import (
    get_memory_usage,
    load_resources,
    log_results,
    run_inference,
)


app = FastAPI()


class InferenceData(BaseModel):
    data: List[Dict[str, float]]


@app.post("/predict/")
async def predict(inference_data: InferenceData):
    try:
        start_time = time.time()
        initial_memory = get_memory_usage()

        inference_df = pd.DataFrame(inference_data.data)

        model, scaler, required_columns, columns_to_keep = load_resources()

        predictions = run_inference(
            model, scaler, inference_df, required_columns, columns_to_keep
        )

        end_time = time.time()
        final_memory = get_memory_usage()

        inference_time = end_time - start_time
        memory_used = final_memory - initial_memory
        model_params = model.get_params()
        model_name = type(model).__name__

        log_results(
            inference_df,
            predictions,
            inference_time,
            memory_used,
            model_name,
            str(model_params),
            PRODUCTION_LOG_PATH,
        )

        return {
            "predictions": pd.DataFrame(predictions)[["y_pred", "y_proba"]].to_dict(
                orient="records"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )
