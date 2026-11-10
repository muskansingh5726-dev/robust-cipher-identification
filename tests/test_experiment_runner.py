
import pandas as pd
import pytest

from src import experiment_runner


def make_model_results():
    labels = ["caesar", "vigenere"]
    y_true = ["caesar", "vigenere", "caesar", "vigenere"]
    texts = ["ABCDEF", "GHIJKL", "MNOPQR", "STUVWX"]

    baseline = {
        "accuracy": 0.75,
        "macro_precision": 0.75,
        "macro_recall": 0.75,
        "macro_f1": 0.75,
        "class_labels": labels,
        "confusion_matrix": [[2, 0], [1, 1]],
        "train_source_ids": ["s1", "s2"],
        "validation_source_ids": ["s3"],
        "test_source_ids": ["s4", "s5"],
        "test_ciphertexts": texts,
        "test_labels": y_true,
        "test_predictions": ["caesar", "vigenere", "vigenere", "vigenere"],
    }

    fusion = {
        "class_labels": labels,
        "train_source_ids": ["s1", "s2"],
        "validation_source_ids": ["s3"],
        "test_source_ids": ["s4", "s5"],
        "test_ciphertexts": texts,
        "test_labels": y_true,
        "character_predictions": ["caesar", "vigenere", "caesar", "vigenere"],
        "fusion_predictions": ["caesar", "vigenere", "caesar", "vigenere"],
        "character_metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
        },
        "fusion_metrics": {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
        },
        "fusion_weight_statistical": 0.5,
        "validation_macro_f1": 0.9,
    }

    return baseline, fusion


def patch_runner(monkeypatch, tmp_path, baseline, fusion):
    data_path = tmp_path / "ciphertexts.csv"
    data_path.write_text(
        "ciphertext,label,source_id\nABCDEF,caesar,s1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        experiment_runner,
        "load_dataset",
        lambda path: pd.DataFrame(
            {
                "ciphertext": ["ABCDEF"],
                "label": ["caesar"],
                "source_id": ["s1"],
            }
        ),
    )
    monkeypatch.setattr(
        experiment_runner,
        "train_and_evaluate",
        lambda data, random_state: baseline,
    )
    monkeypatch.setattr(
        experiment_runner,
        "train_and_evaluate_fusion",
        lambda data, random_state: fusion,
    )
    monkeypatch.setattr(
        experiment_runner,
        "evaluate_by_ciphertext_length",
        lambda **kwargs: pd.DataFrame(
            [{"length_bin": "short", "accuracy": 1.0}]
        ),
    )

    return data_path


def test_runner_writes_result_files(monkeypatch, tmp_path):
    baseline, fusion = make_model_results()
    data_path = patch_runner(
        monkeypatch, tmp_path, baseline, fusion
    )

    output_dir = tmp_path / "results"
    experiment_runner.run_experiment(
        data_path=data_path,
        output_dir=output_dir,
        random_state=42,
    )

    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "length_comparison.csv").exists()
    assert (output_dir / "experiment_report.json").exists()


def test_runner_rejects_different_test_sources(monkeypatch, tmp_path):
    baseline, fusion = make_model_results()
    fusion["test_source_ids"] = ["different-source"]

    data_path = patch_runner(
        monkeypatch, tmp_path, baseline, fusion
    )

    with pytest.raises(RuntimeError, match="test sources"):
        experiment_runner.run_experiment(
            data_path=data_path,
            output_dir=tmp_path / "results",
        )


def test_runner_rejects_different_test_examples(monkeypatch, tmp_path):
    baseline, fusion = make_model_results()
    fusion["test_ciphertexts"] = ["DIFFERENT"]

    data_path = patch_runner(
        monkeypatch, tmp_path, baseline, fusion
    )

    with pytest.raises(RuntimeError, match="Test ciphertexts"):
        experiment_runner.run_experiment(
            data_path=data_path,
            output_dir=tmp_path / "results",
        )
