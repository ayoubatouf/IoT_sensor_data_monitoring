import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from src.utils.file_io import save_metrics_to_file


def save_confusion_matrix(cm, metrics_path):

    confusion_matrix_save_path = metrics_path / "confusion_matrix.png"
    cm_df = pd.DataFrame(
        cm,
        index=["True Negative", "True Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_save_path)
    plt.close()


def save_roc_curve(y_proba, y_test, metrics_path):

    roc_curve_save_path = metrics_path / "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_save_path)
    plt.close()


def save_metrics(metrics, predictions, y_test, metrics_path):

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    save_metrics_to_file(metrics, metrics_path)
    save_confusion_matrix(metrics["confusion_matrix"], metrics_path)
    save_roc_curve(predictions["y_proba"], y_test, metrics_path)
