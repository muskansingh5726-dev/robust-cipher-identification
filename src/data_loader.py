
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"ciphertext", "label", "source_id"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate its required fields."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Dataset contains no samples.")

    for column in REQUIRED_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

        if data[column].isna().any() or data[column].eq("").any():
            raise ValueError(
                f"Column '{column}' contains missing or empty values."
            )

    # Keep only ciphertext rows containing English letters.
    data["ciphertext"] = data["ciphertext"].str.upper()

    if not data["ciphertext"].str.contains(
        r"[A-Z]", regex=True
    ).all():
        raise ValueError(
            "Every ciphertext must contain at least one English letter."
        )

    data["text_length"] = (
        data["ciphertext"].str.count(r"[A-Z]")
    )

    return data
