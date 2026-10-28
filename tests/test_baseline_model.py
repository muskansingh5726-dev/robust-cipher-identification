
import pandas as pd

from src.baseline_model import train_and_evaluate


def make_baseline_test_data():
    """Synthetic data for pipeline testing, not research evaluation."""
    rows = []

    for i in range(40):
        plaintext = (
            f"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
            f"EXAMPLE NUMBER {i:03d}"
        )

        # Simple deterministic transformations for a smoke test.
        caesar = "".join(
            chr((ord(c) - ord("A") + 3) % 26 + ord("A"))
            if "A" <= c <= "Z" else c
            for c in plaintext
        )

        atbash = "".join(
            chr(ord("Z") - (ord(c) - ord("A")))
            if "A" <= c <= "Z" else c
            for c in plaintext
        )

        rows.extend([
            {
                "ciphertext": caesar,
                "label": "caesar",
                "source_id": f"source_{i}",
            },
            {
                "ciphertext": atbash,
                "label": "atbash",
                "source_id": f"source_{i}",
            },
        ])

    return pd.DataFrame(rows)


def test_baseline_trains_and_returns_metrics():
    data = make_baseline_test_data()

    results = train_and_evaluate(data, random_state=42)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["macro_precision"] <= 1
    assert 0 <= results["macro_recall"] <= 1
    assert 0 <= results["macro_f1"] <= 1
    assert set(results["class_labels"]) == {"caesar", "atbash"}
    assert len(results["test_source_ids"]) > 0
