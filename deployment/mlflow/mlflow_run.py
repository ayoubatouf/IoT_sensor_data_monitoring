import mlflow
import pandas as pd
from config.data import MODEL_PARAMS_JSON_PATH
from config.paths import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.file_io import load_from_json
import json


def log_mlflow_pipeline(X_train, y_train, X_test, y_test, model_params):

    with mlflow.start_run():
        model, predictions, metrics = run_training_pipeline(
            X_train, y_train, X_test, y_test, model_params
        )

        mlflow.log_params(model_params)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc_score", metrics["roc_auc_score"])
        mlflow.log_metric("log_loss", metrics["log_loss"])

        for artifact_name, artifact_content in {
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }.items():
            artifact_path = f"{artifact_name}.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact_content, f)
            mlflow.log_artifact(artifact_path)

        mlflow.sklearn.log_model(model, "trained_model")

        predictions_path = "predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(
                predictions,
                f,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
            )
        mlflow.log_artifact(predictions_path)

        print(f"Run logged in MLflow with ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    params = load_from_json(MODEL_PARAMS_JSON_PATH)

    training_data = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
    test_data = pd.read_csv(PROCESSED_TEST_DATA_PATH)

    X_train = training_data.drop(columns=["fail"])
    y_train = training_data["fail"]
    X_test = test_data.drop(columns=["fail"])
    y_test = test_data["fail"]

    log_mlflow_pipeline(X_train, y_train, X_test, y_test, params)
