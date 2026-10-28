
import argparse
import json
from pathlib import Path

from src.data_loader import load_dataset
from src.baseline_model import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the classical-cipher baseline."
    )
    parser.add_argument(
        "--data",
        default="data/ciphertexts.csv",
        help="Path to the labeled ciphertext CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the holdout split and model.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data)

    print(f"Samples loaded: {len(data)}")
    print(f"Cipher classes: {data['label'].nunique()}")
    print(f"Unique plaintext sources: {data['source_id'].nunique()}")

    results = train_and_evaluate(data, random_state=args.seed)

    print("\n=== Baseline evaluation ===")
    for metric in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        print(f"{metric}: {results[metric]:.4f}")

    print("\n=== Per-class report ===")
    print(results["classification_report"])

    print("\n=== Confusion matrix ===")
    print("Class order:", results["class_labels"])
    print(results["confusion_matrix"])

    # Save metrics and split information, not the fitted model.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "dataset": str(args.data),
        "random_seed": args.seed,
        "sample_count": len(data),
        "class_count": int(data["label"].nunique()),
        "source_count": int(data["source_id"].nunique()),
        "metrics": {
            metric: float(results[metric])
            for metric in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            )
        },
        "classification_report": results["classification_report"],
        "class_labels": results["class_labels"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    output_path = output_dir / "baseline_metrics.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
