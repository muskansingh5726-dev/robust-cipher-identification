
import pytest

from src.evaluation import evaluate_predictions


def test_evaluation_returns_expected_metrics():
    actual = ["caesar", "caesar", "atbash", "atbash"]
    predicted = ["caesar", "caesar", "atbash", "caesar"]

    results = evaluate_predictions(actual, predicted)

    assert results["accuracy"] == pytest.approx(0.75)
    assert 0 <= results["macro_precision"] <= 1
    assert 0 <= results["macro_recall"] <= 1
    assert 0 <= results["macro_f1"] <= 1
    assert len(results["confusion_matrix"]) == 2


def test_evaluation_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty test set"):
        evaluate_predictions([], [])


def test_evaluation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions(["caesar"], [])
