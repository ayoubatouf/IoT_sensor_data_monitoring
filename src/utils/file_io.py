import json

from config.paths import EVALUATION_PATH


def load_from_json(file_path):
    with open(file_path, "r") as json_file:
        params = json.load(json_file)
    return params


def save_metrics_to_file(metrics, metrics_path):

    metrics_file_path = metrics_path / "evaluation_metrics.txt"
    with open(metrics_file_path, "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"\nClassification Report:\n{metrics['classification_report']}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
        f.write(f"\nROC AUC Score: {metrics['roc_auc_score']}\n")
        f.write(f"Log Loss: {metrics['log_loss']}\n")
    print(f"Metrics saved to: {metrics_file_path}")
