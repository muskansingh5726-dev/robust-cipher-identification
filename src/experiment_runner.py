
import json
from pathlib import Path

import pandas as pd

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


def run_experiment(
    data_path="data/ciphertexts.csv",
    output_dir="results",
    random_state=42,
):
    """Compare models and analyze held-out predictions by text length."""
    data = load_dataset(data_path)
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

    baseline_sources = set(baseline["test_source_ids"])
    fusion_sources = set(fusion["test_source_ids"])

    if baseline_sources != fusion_sources:
        raise RuntimeError(
            "Baseline and fusion models used different test sources."
        )

    baseline_texts = baseline["test_ciphertexts"]
    fusion_texts = fusion["test_ciphertexts"]

    if baseline_texts != fusion_texts:
        raise RuntimeError(
            "Baseline and fusion models used different test examples."
        )

    if baseline["test_labels"] != fusion["test_labels"]:
        raise RuntimeError(
            "Baseline and fusion models used different test labels."
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

    # Analyze every model on the same held-out examples.
    length_results = []

    prediction_sets = {
        "statistical_random_forest": baseline["test_predictions"],
        "character_tfidf_logistic_regression": (
            fusion["character_predictions"]
        ),
        "statistical_character_fusion": fusion["fusion_predictions"],
    }

    for model_name, predictions in prediction_sets.items():
        length_table = evaluate_by_ciphertext_length(
            ciphertexts=baseline_texts,
            y_true=baseline["test_labels"],
            y_pred=predictions,
            labels=baseline["class_labels"],
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
        "dataset": str(data_path),
        "random_state": random_state,
        "sample_count": len(data),
        "class_count": int(data["label"].nunique()),
        "source_count": int(data["source_id"].nunique()),
        "test_source_count": len(baseline_sources),
        "fusion_statistical_weight": fusion[
            "fusion_weight_statistical"
        ],
        "fusion_validation_macro_f1": fusion[
            "validation_macro_f1"
        ],
        "results": rows,
        "length_results": length_comparison.to_dict(orient="records"),
        "class_labels": baseline["class_labels"],
        "baseline_confusion_matrix": baseline[
            "confusion_matrix"
        ].tolist(),
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
