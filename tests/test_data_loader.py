
import pandas as pd
import pytest

from src.data_loader import load_dataset


def test_load_dataset_adds_text_length(tmp_path):
    path = tmp_path / "sample.csv"

    pd.DataFrame({
        "ciphertext": ["KHOOR", "ZRUOG"],
        "label": ["caesar", "caesar"],
        "source_id": ["source_1", "source_2"],
    }).to_csv(path, index=False)

    data = load_dataset(path)

    assert len(data) == 2
    assert data["text_length"].tolist() == [5, 5]
    assert data["ciphertext"].tolist() == ["KHOOR", "ZRUOG"]


def test_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"

    pd.DataFrame({
        "ciphertext": ["KHOOR"],
        "label": ["caesar"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"

    pd.DataFrame(
        columns=["ciphertext", "label", "source_id"]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="no samples"):
        load_dataset(path)
