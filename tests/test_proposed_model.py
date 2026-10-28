
import string

import pandas as pd
import pytest

from src.proposed_model import (
    _select_fusion_weight,
    train_and_evaluate_fusion,
)


ALPHABET = string.ascii_uppercase


def caesar(text, shift=3):
    return "".join(
        ALPHABET[(ALPHABET.index(char) + shift) % 26]
        if char in ALPHABET else char
        for char in text
    )


def atbash(text):
    return "".join(
        ALPHABET[25 - ALPHABET.index(char)]
        if char in ALPHABET else char
        for char in text
    )


def make_smoke_test_data():
    """Synthetic pipeline test data, not research results."""
    rows = []

    for i in range(30):
        suffix = ""
        number = i

        while True:
            suffix = ALPHABET[number % 26] + suffix
            number = number // 26 - 1

            if number < 0:
                break

        plaintext = (
            f"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
            f"WITH EXTRA WORDS FOR SAMPLE {suffix}"
        )

        variants = {
            "caesar": caesar(plaintext),
            "atbash": atbash(plaintext),
            "reverse": plaintext[::-1],
        }

        for label, ciphertext in variants.items():
            rows.append({
                "ciphertext": ciphertext,
                "label": label,
                "source_id": f"source_{i}",
            })

    return pd.DataFrame(rows)


def test_fusion_pipeline_runs_and_prevents_source_leakage():
    data = make_smoke_test_data()

    results = train_and_evaluate_fusion(
        data,
        random_state=42,
    )

    train_sources = set(results["train_source_ids"])
    validation_sources = set(results["validation_source_ids"])
    test_sources = set(results["test_source_ids"])

    assert train_sources.isdisjoint(validation_sources)
    assert train_sources.isdisjoint(test_sources)
    assert validation_sources.isdisjoint(test_sources)

    assert 0.0 <= results["fusion_weight_statistical"] <= 1.0

    expected_metrics = {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }

    for branch in (
        "statistical_metrics",
        "character_metrics",
        "fusion_metrics",
    ):
        assert set(results[branch]) == expected_metrics

        assert all(
            0.0 <= score <= 1.0
            for score in results[branch].values()
        )


def test_fusion_weight_tie_prefers_equal_weights():
    candidates = [
        (0.80, 0.0),
        (0.90, 0.3),
        (0.90, 0.5),
        (0.90, 0.7),
        (0.80, 1.0),
    ]

    score, alpha = _select_fusion_weight(candidates)

    assert score == pytest.approx(0.90)
    assert alpha == pytest.approx(0.5)


def test_fusion_weight_tie_is_deterministic():
    candidates = [
        (0.90, 0.3),
        (0.90, 0.7),
    ]

    score, alpha = _select_fusion_weight(candidates)

    assert score == pytest.approx(0.90)
    assert alpha == pytest.approx(0.3)


def test_fusion_weight_rejects_empty_candidates():
    with pytest.raises(
        ValueError,
        match="At least one fusion candidate",
    ):
        _select_fusion_weight([])
