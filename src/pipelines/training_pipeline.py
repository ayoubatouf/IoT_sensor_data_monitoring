from src.models.evaluate import evaluate_model
from src.models.predict import predict_model
from src.models.train import train_model


def run_training_pipeline(X_train, y_train, X_test, y_test, model_params):

    model = train_model(X_train, y_train, model_params)

    y_pred, y_proba = predict_model(model, X_test)

    metrics = evaluate_model(y_test, y_pred, y_proba)

    predictions = {"y_pred": y_pred, "y_proba": y_proba}

    return model, predictions, metrics
