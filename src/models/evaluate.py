from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    accuracy_score,
)


def evaluate_model(y_test, y_pred, y_proba):

    accuracy = accuracy_score(y_test, y_pred)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_proba)

    log_loss_val = log_loss(y_test, y_proba)

    metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "roc_auc_score": roc_auc,
        "log_loss": log_loss_val,
    }

    return metrics
