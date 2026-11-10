
import pytest

from src.length_analysis import evaluate_by_ciphertext_length


def test_length_analysis_groups_predictions():
    ciphertexts = [
        "A" * 20,
        "B" * 40,
        "C" * 80,
        "D" * 120,
        "E" * 200,
        "F" * 250,
    ]
    actual = ["caesar", "atbash", "caesar", "atbash", "caesar", "atbash"]
    predicted = actual.copy()

    result = evaluate_by_ciphertext_length(
        ciphertexts, actual, predicted
    )

    assert result["sample_count"].tolist() == [2, 2, 2]
    assert result["accuracy"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert result["macro_f1"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_length_analysis_handles_empty_bins():
    result = evaluate_by_ciphertext_length(
        ["A" * 20],
        ["caesar"],
        ["caesar"],
    )

    assert result.loc[0, "sample_count"] == 1
    assert result.loc[1, "sample_count"] == 0
    assert result.loc[2, "sample_count"] == 0


def test_length_analysis_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_by_ciphertext_length(
            ["ABC", "DEF"],
            ["caesar"],
            ["caesar"],
        )


def test_length_analysis_rejects_nonletter_ciphertext():
    with pytest.raises(ValueError, match="at least one ASCII letter"):
        evaluate_by_ciphertext_length(
            ["123 !"],
            ["caesar"],
            ["caesar"],
        )
