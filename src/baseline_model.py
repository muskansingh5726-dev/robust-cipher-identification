
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

from src.data_splitting import split_by_source
from src.feature_extraction import extract_features


def build_feature_table(data: pd.DataFrame):
    """Convert ciphertext rows into statistical features."""
    required = {"ciphertext", "label", "source_id"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Dataset is empty.")

    X = pd.DataFrame(
        [extract_features(text) for text in data["ciphertext"]]
    )
    y = data["label"].astype(str)

    return X, y


def train_and_evaluate(
    data: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """Train a statistical-feature Random Forest on shared data splits."""
    if data.empty:
        raise ValueError("Dataset is empty.")

    train, validation, test = split_by_source(
        data,
        random_state=random_state,
    )

    # Build features independently for each partition.
    X_train, y_train = build_feature_table(train)
    X_test, y_test = build_feature_table(test)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(data["label"].astype(str).unique())

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(
            precision_score(
                y_test, predictions,
                labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, predictions,
                labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test, predictions,
                labels=labels, average="macro", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ),
        "class_labels": labels,
        "model": model,
        "test_source_ids": test["source_id"].tolist(),
        "test_ciphertexts": test["ciphertext"].tolist(),
        "test_labels": y_test.tolist(),
        "test_predictions": predictions.tolist(),
    }
