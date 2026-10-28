import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from src.data_splitting import split_by_source
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from src.feature_extraction import extract_features
from src.data_loader import REQUIRED_COLUMNS


def _metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, predictions, average="macro", zero_division=0
            )
        ),
    }


def _check_split_classes(y_train, y_val, y_test):
    classes = set(y_train.unique())

    if len(classes) < 2:
        raise ValueError(
            "Training split has fewer than two classes. "
            "Use more diverse source plaintexts or revise the split."
        )

    if set(y_val.unique()) != classes or set(y_test.unique()) != classes:
        raise ValueError(
            "Train, validation, and test splits must contain every class. "
            "Review source-group coverage and dataset size."
        )


def train_and_evaluate_fusion(data, random_state=42):
    """
    Train statistical and character-level models, tune fusion on
    validation data, and evaluate the selected fusion on the test set.

    Expected columns: ciphertext, label, source_id.
    Each source_id is kept within one split.
    """
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = data.reset_index(drop=True).copy()

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required dataset fields cannot be missing.")

    data["ciphertext"] = data["ciphertext"].astype(str)
    data["label"] = data["label"].astype(str)
    data["source_id"] = data["source_id"].astype(str)

    if (data["ciphertext"].str.strip() == "").any():
        raise ValueError("Ciphertext cannot be empty.")

    if data["label"].nunique() < 2:
        raise ValueError("At least two cipher classes are required.")

    y = data["label"]
    groups = data["source_id"]

       # Use the shared, reproducible source-grouped split.
    train, val, test = split_by_source(
        data,
        random_state=random_state,
    )

    _check_split_classes(
        train["label"],
        val["label"],
        test["label"],
    )

    _check_split_classes(
        train["label"], val["label"], test["label"]
    )

    # View 1: engineered statistical features.
    X_train_stats = pd.DataFrame(
        [extract_features(t) for t in train["ciphertext"]]
    )
    X_val_stats = pd.DataFrame(
        [extract_features(t) for t in val["ciphertext"]]
    )
    X_test_stats = pd.DataFrame(
        [extract_features(t) for t in test["ciphertext"]]
    )

    statistical_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    statistical_model.fit(X_train_stats, train["label"])

    # View 2: character n-gram TF-IDF features.
    # Fit the vocabulary on training ciphertexts only.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=100_000,
    )

    X_train_char = vectorizer.fit_transform(train["ciphertext"])
    X_val_char = vectorizer.transform(val["ciphertext"])
    X_test_char = vectorizer.transform(test["ciphertext"])

    character_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    character_model.fit(X_train_char, train["label"])

    # Ensure probability columns use the same class order.
    classes = statistical_model.classes_
    if not np.array_equal(classes, character_model.classes_):
        raise ValueError("The two model branches have different classes.")

    val_stats = statistical_model.predict_proba(X_val_stats)
    val_char = character_model.predict_proba(X_val_char)

    # Alpha is the statistical branch weight.
    # Select it using validation macro F1, not test results.
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 11):
        probabilities = (
            alpha * val_stats + (1.0 - alpha) * val_char
        )
        predictions = classes[np.argmax(probabilities, axis=1)]
        score = f1_score(
            val["label"],
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        candidates.append((float(score), float(alpha)))

    # Deterministic tie-break: prefer equal weights when scores tie.
    best_score = max(score for score, _ in candidates)
    best_alpha = min(
        alpha
        for score, alpha in candidates
        if np.isclose(score, best_score)
    )

    # Evaluate the chosen fusion on the untouched test set.
    test_stats = statistical_model.predict_proba(X_test_stats)
    test_char = character_model.predict_proba(X_test_char)

    test_probabilities = (
        best_alpha * test_stats
        + (1.0 - best_alpha) * test_char
    )
    test_predictions = classes[np.argmax(test_probabilities, axis=1)]

    # Report both individual branches on the same test set.
    stats_predictions = classes[np.argmax(test_stats, axis=1)]
    char_predictions = classes[np.argmax(test_char, axis=1)]

    return {
        "fusion_weight_statistical": best_alpha,
        "validation_macro_f1": best_score,
        "statistical_metrics": _metrics(
            test["label"], stats_predictions
        ),
        "character_metrics": _metrics(
            test["label"], char_predictions
        ),
        "fusion_metrics": _metrics(
            test["label"], test_predictions
        ),
        "class_labels": classes.tolist(),
        "train_source_ids": train["source_id"].tolist(),
        "validation_source_ids": val["source_id"].tolist(),
        "test_source_ids": test["source_id"].tolist(),
        "statistical_model": statistical_model,
        "character_vectorizer": vectorizer,
        "character_model": character_model,
        "test_ciphertexts": test["ciphertext"].tolist(),
        "test_labels": test["label"].tolist(),
        "statistical_predictions": stats_predictions.tolist(),
        "character_predictions": char_predictions.tolist(),
        "fusion_predictions": test_predictions.tolist(),
    }
