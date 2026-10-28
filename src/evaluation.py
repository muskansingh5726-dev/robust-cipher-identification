
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, labels=None):
    """Calculate consistent metrics for a classifier's predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }
