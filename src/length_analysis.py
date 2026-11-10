
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_LENGTH_BINS = {
    "short": (1, 50),
    "medium": (51, 150),
    "long": (151, float("inf")),
}


def evaluate_by_ciphertext_length(
    ciphertexts,
    y_true,
    y_pred,
    length_bins=None,
    labels=None,
):
    """
    Evaluate predictions separately by alphabetic ciphertext length.

    Default groups:
      short: 1-50 letters
      medium: 51-150 letters
      long: 151+ letters

    Use the same held-out predictions for every group.
    These boundaries are configurable and should be documented
    before interpreting research results.
    """
    if not (len(ciphertexts) == len(y_true) == len(y_pred)):
        raise ValueError(
            "ciphertexts, y_true, and y_pred must have equal lengths."
        )

    if len(ciphertexts) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if length_bins is None:
        length_bins = DEFAULT_LENGTH_BINS

    if not length_bins:
        raise ValueError("At least one length bin is required.")

    if labels is None:
        labels = sorted(set(y_true))

    lengths = [
        sum(character.isascii() and character.isalpha() for character in text)
        for text in ciphertexts
    ]

    if any(length == 0 for length in lengths):
        raise ValueError(
            "Every ciphertext must contain at least one ASCII letter."
        )

    rows = []

    for group_name, (minimum, maximum) in length_bins.items():
        indices = [
            i for i, length in enumerate(lengths)
            if minimum <= length <= maximum
        ]

        if not indices:
            rows.append({
                "length_group": group_name,
                "min_length": minimum,
                "max_length": maximum,
                "sample_count": 0,
                "accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
            })
            continue

        actual = [y_true[i] for i in indices]
        predicted = [y_pred[i] for i in indices]

        rows.append({
            "length_group": group_name,
            "min_length": minimum,
            "max_length": maximum,
            "sample_count": len(indices),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_precision": float(
                precision_score(
                    actual, predicted, labels=labels,
                    average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    actual, predicted, labels=labels,
                    average="macro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    actual, predicted, labels=labels,
                    average="macro", zero_division=0
                )
            ),
        })

    return pd.DataFrame(rows)
