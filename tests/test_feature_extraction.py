
import pytest

from src.feature_extraction import (
    extract_features,
    normalize_ciphertext,
)


def test_normalize_ciphertext():
    assert normalize_ciphertext("Hello, World! 123") == "HELLOWORLD"


def test_feature_count_and_letter_frequencies():
    features = extract_features("ABCA")

    assert len(features) == 33
    assert features["frequency_A"] == pytest.approx(0.5)
    assert features["frequency_B"] == pytest.approx(0.25)
    assert features["frequency_C"] == pytest.approx(0.25)


def test_features_are_finite():
    features = extract_features("THISISATEST")

    assert all(
        value == value and abs(value) != float("inf")
        for value in features.values()
    )


def test_empty_ciphertext_is_rejected():
    with pytest.raises(ValueError):
        extract_features("123 !")


def test_single_character():
    features = extract_features("A")

    assert features["index_of_coincidence"] == 0.0
    assert features["shannon_entropy"] == 0.0
