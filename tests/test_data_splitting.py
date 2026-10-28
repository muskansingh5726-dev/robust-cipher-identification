
import pandas as pd
import pytest

from src.data_splitting import split_by_source


def make_data():
    rows = []

    for source_num in range(30):
        for label in ("caesar", "atbash", "vigenere"):
            rows.append({
                "ciphertext": f"SAMPLE{source_num}{label.upper()}",
                "label": label,
                "source_id": f"source_{source_num}",
            })

    return pd.DataFrame(rows)


def test_split_keeps_sources_separate_and_classes_present():
    train, validation, test = split_by_source(make_data())

    train_ids = set(train["source_id"])
    val_ids = set(validation["source_id"])
    test_ids = set(test["source_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    expected = {"caesar", "atbash", "vigenere"}

    assert set(train["label"]) == expected
    assert set(validation["label"]) == expected
    assert set(test["label"]) == expected


def test_split_is_reproducible():
    first = split_by_source(make_data(), random_state=42)
    second = split_by_source(make_data(), random_state=42)

    for a, b in zip(first, second):
        assert a["source_id"].tolist() == b["source_id"].tolist()


def test_split_rejects_missing_columns():
    data = pd.DataFrame({
        "ciphertext": ["ABC"],
        "label": ["caesar"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        split_by_source(data)
