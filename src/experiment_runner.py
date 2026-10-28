
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

from src.baseline_model import train_and_evaluate
from src.data_loader import load_dataset
from src.length_analysis import evaluate_by_ciphertext_length
from src.proposed_model import train_and_evaluate_fusion


METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def file_sha256(path):
    """Return a SHA-256 checksum for the dataset file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    data_path="data/ciphertexts.csv",
    output_dir="results",
    random_state=42,
):
    """Compare models on identical source-separated test examples."""
    data_file = Path(data_path)
    data = load_dataset(data_file)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Samples: {len(data)}")
    print(f"Classes: {data['label'].nunique()}")
    print(f"Distinct sources: {data['source_id'].nunique()}")

    print("\nTraining statistical Random Forest baseline...")
    baseline = train_and_evaluate(data, random_state=random_state)

    print("Training character-level and fusion models...")
    fusion = train_and_evaluate_fusion(
        data,
        random_state=random_state,
    )

    # Ensure all models use identical source partitions.
    for split in ("train", "validation", "test"):
        key = f"{split}_source_ids"
        if set(baseline[key]) != set(fusion[key]):
            raise RuntimeError(
                f"Baseline and fusion models used different "
                f"{split} sources."
            )

    # Ensure predictions correspond to identical test examples.
    test_texts = baseline["test_ciphertexts"]
    test_labels = baseline["test_labels"]

    if test_texts != fusion["test_ciphertexts"]:
        raise RuntimeError("Test ciphertexts do not match.")

    if test_labels != fusion["test_labels"]:
        raise RuntimeError("Test labels do not match.")

    labels = list(baseline["class_labels"])
    if labels != list(fusion["class_labels"]):
        raise RuntimeError("Model class-label ordering does not match.")

    predictions_by_model = {
        "statistical_random_forest": baseline["test_predictions"],
        "character_tfidf_logistic_regression": (
            fusion["character_predictions"]
        ),
        "statistical_character_fusion": fusion["fusion_predictions"],
    }

    for name, predictions in predictions_by_model.items():
        if len(predictions) != len(test_labels):
            raise RuntimeError(
                f"{name} returned a different number of predictions."
            )

    rows = [
        {
            "model": "statistical_random_forest",
            **{metric: baseline[metric] for metric in METRICS},
        },
        {
            "model": "character_tfidf_logistic_regression",
            **fusion["character_metrics"],
        },
        {
            "model": "statistical_character_fusion",
            **fusion["fusion_metrics"],
        },
    ]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_path / "model_comparison.csv",
        index=False,
    )

    # Calculate confusion matrices using one fixed label order.
    confusion_matrices = {}
    for name, predictions in predictions_by_model.items():
        matrix = confusion_matrix(
            test_labels,
            predictions,
            labels=labels,
        )
        confusion_matrices[name] = matrix.tolist()

    # Evaluate every model on the same test ciphertexts.
    length_results = []

    for model_name, predictions in predictions_by_model.items():
        length_table = evaluate_by_ciphertext_length(
            ciphertexts=test_texts,
            y_true=test_labels,
            y_pred=predictions,
            labels=labels,
        )
        length_table.insert(0, "model", model_name)
        length_results.append(length_table)

    length_comparison = pd.concat(
        length_results,
        ignore_index=True,
    )
    length_comparison.to_csv(
        output_path / "length_comparison.csv",
        index=False,
    )

    report = {
        "dataset_path": str(data_file),
        "dataset_sha256": file_sha256(data_file),
        "random_state": random_state,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "sample_count": len(data),
        "class_count": int(data["label"].nunique()),
        "source_count": int(data["source_id"].nunique()),
        "train_source_count": len(
            set(baseline["train_source_ids"])
        ),
        "validation_source_count": len(
            set(baseline["validation_source_ids"])
        ),
        "test_source_count": len(
            set(baseline["test_source_ids"])
        ),
        "fusion_statistical_weight": fusion[
            "fusion_weight_statistical"
        ],
        "fusion_validation_macro_f1": fusion[
            "validation_macro_f1"
        ],
        "results": rows,
        "confusion_matrices": confusion_matrices,
        "length_results": length_comparison.to_dict(
            orient="records"
        ),
        "class_labels": labels,
    }

    with (output_path / "experiment_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== Overall comparison ===")
    print(comparison.to_string(index=False))

    print("\n=== Ciphertext-length comparison ===")
    print(length_comparison.to_string(index=False))

    print(f"\nResults saved in: {output_path.resolve()}")

    return comparison, length_comparison


if __name__ == "__main__":
    run_experiment()
