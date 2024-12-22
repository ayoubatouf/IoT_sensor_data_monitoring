from src.data.loading_data import validate_data
from src.data.preprocess_data import remove_duplicates
from src.features.build_features import keep_columns
from src.models.predict import predict_model


def run_inference_pipeline(model, scaler, data, required_columns, columns_to_keep):

    if validate_data(data, required_columns):
        data = remove_duplicates(data)

        data = keep_columns(data, columns_to_keep)

        data = scaler.transform(data)

        y_pred, y_proba = predict_model(model, data)

        predictions = {"y_pred": y_pred, "y_proba": y_proba}

        return predictions
