
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

from src.data_loader import REQUIRED_COLUMNS
from src.data_splitting import split_by_source
from src.feature_extraction import extract_features


def _metrics(y_true, predictions):
    """Calculate classification metrics."""
    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _check_split_classes(y_train, y_val, y_test):
    """Ensure every class occurs in every split."""
    classes = set(y_train.unique())

    if len(classes) < 2:
        raise ValueError(
            "Training split has fewer than two classes. "
            "Use more diverse source plaintexts or revise the split."
        )

    if (
        set(y_val.unique()) != classes
        or set(y_test.unique()) != classes
    ):
        raise ValueError(
            "Train, validation, and test splits must contain every "
            "class. Review source-group coverage and dataset size."
        )



def _select_fusion_weight(candidates):
    """
    Select the best validation score.
    On ties, prefer alpha closest to 0.5, then the lower alpha.
    """
    if not candidates:
        raise ValueError("At least one fusion candidate is required.")

    best_score = max(score for score, _ in candidates)

    tied_alphas = [
        alpha
        for score, alpha in candidates
        if np.isclose(score, best_score)
    ]

    closest_distance = min(
        abs(alpha - 0.5) for alpha in tied_alphas
    )

    closest_alphas = [
        alpha
        for alpha in tied_alphas
        if np.isclose(abs(alpha - 0.5), closest_distance)
    ]

    best_alpha = min(closest_alphas)

    return float(best_score), float(best_alpha)



def train_and_evaluate_fusion(data, random_state=42):
    """
    Train statistical and character-level models, tune fusion using
    validation macro F1, and evaluate on the held-out test set.

    Required columns: ciphertext, label, source_id.
    Each source_id must remain within a single split.
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

    if (data["label"].str.strip() == "").any():
        raise ValueError("Cipher labels cannot be empty.")

    if (data["source_id"].str.strip() == "").any():
        raise ValueError("Source IDs cannot be empty.")

    if data["label"].nunique() < 2:
        raise ValueError("At least two cipher classes are required.")

    # Use the shared source-aware split.
    train, val, test = split_by_source(
        data,
        random_state=random_state,
    )

    _check_split_classes(
        train["label"],
        val["label"],
        test["label"],
    )

    # View 1: engineered statistical features.
    X_train_stats = pd.DataFrame(
        [extract_features(text) for text in train["ciphertext"]]
    )
    X_val_stats = pd.DataFrame(
        [extract_features(text) for text in val["ciphertext"]]
    )
    X_test_stats = pd.DataFrame(
        [extract_features(text) for text in test["ciphertext"]]
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

    statistical_model.fit(
        X_train_stats,
        train["label"],
    )

    # View 2: character n-gram TF-IDF.
    # Fit the vocabulary using training ciphertexts only.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=100_000,
    )

    X_train_char = vectorizer.fit_transform(
        train["ciphertext"]
    )
    X_val_char = vectorizer.transform(
        val["ciphertext"]
    )
    X_test_char = vectorizer.transform(
        test["ciphertext"]
    )

    character_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )

    character_model.fit(
        X_train_char,
        train["label"],
    )

    # Both branches must have identical probability-column ordering.
    classes = statistical_model.classes_

    if not np.array_equal(
        classes,
        character_model.classes_,
    ):
        raise ValueError(
            "The two model branches have different classes."
        )

    # Tune the statistical branch weight on validation data only.
    val_stats = statistical_model.predict_proba(
        X_val_stats
    )
    val_char = character_model.predict_proba(
        X_val_char
    )

    candidates = []

    for alpha in np.linspace(0.0, 1.0, 11):
        probabilities = (
            alpha * val_stats
            + (1.0 - alpha) * val_char
        )

        predictions = classes[
            np.argmax(probabilities, axis=1)
        ]

        score = f1_score(
            val["label"],
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        candidates.append(
            (float(score), float(alpha))
        )

    best_score, best_alpha = _select_fusion_weight(
        candidates
    )

    # Evaluate the selected model on the held-out test set.
    test_stats = statistical_model.predict_proba(
        X_test_stats
    )
    test_char = character_model.predict_proba(
        X_test_char
    )

    test_probabilities = (
        best_alpha * test_stats
        + (1.0 - best_alpha) * test_char
    )

    test_predictions = classes[
        np.argmax(test_probabilities, axis=1)
    ]

    # Individual branches use the same test examples.
    stats_predictions = classes[
        np.argmax(test_stats, axis=1)
    ]
    char_predictions = classes[
        np.argmax(test_char, axis=1)
    ]

    return {
        "fusion_weight_statistical": best_alpha,
        "validation_macro_f1": best_score,
        "statistical_metrics": _metrics(
            test["label"],
            stats_predictions,
        ),
        "character_metrics": _metrics(
            test["label"],
            char_predictions,
        ),
        "fusion_metrics": _metrics(
            test["label"],
            test_predictions,
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
