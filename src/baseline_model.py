
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from src.feature_extraction import extract_features


def build_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Convert ciphertext rows into a numeric feature table."""
    required = {"ciphertext", "label", "source_id"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Dataset is empty.")

    feature_rows = [
        extract_features(text)
        for text in data["ciphertext"]
    ]

    X = pd.DataFrame(feature_rows, index=data.index)
    y = data["label"].astype(str)
    groups = data["source_id"]

    return X, y, groups


def train_and_evaluate(data: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Train and evaluate a baseline using source-grouped holdout data.

    Rows derived from the same source_id are kept in the same split.
    """
    X, y, groups = build_feature_table(data)

    if y.nunique() < 2:
        raise ValueError("At least two cipher classes are required.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training split contains fewer than two classes. "
            "Check source-group distribution or adjust the split."
        )

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

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=sorted(y.unique())
        ),
        "class_labels": sorted(y.unique()),
        "model": model,
        "test_indices": list(X_test.index),
    }

    return results
