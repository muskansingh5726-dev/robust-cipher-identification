import json
import random

import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.dataset_pipeline import (
    CIPHER_NAMES,
    ENCRYPTORS,
    assign_source_partitions,
    build_dataset,
    encrypt_atbash,
    stable_source_id,
)
from src.data_splitting import split_by_source


def write_sources(directory, count=12):
    directory.mkdir()
    files = []
    words = [
        "amber", "birch", "cobalt", "dahlia", "elm", "flint",
        "garden", "harbor", "indigo", "juniper", "kelp", "lantern",
    ]
    for index in range(count):
        path = directory / f"source_{index:02d}.txt"
        path.write_text(
            f"The {words[index]} passage contains enough distinct English text "
            "for reproducible classical cipher generation and validation.",
            encoding="utf-8",
        )
        files.append(path.name)
    return files


def test_source_id_is_stable_and_atbash_is_defined():
    assert stable_source_id("HELLOWORLD") == stable_source_id("HELLOWORLD")
    ciphertext, key = encrypt_atbash("ABCXYZ", random.Random(1))
    assert ciphertext == "ZYXCBA"
    assert key["mapping"] == "A<->Z"


def test_all_supported_encryptors_preserve_letters_and_length():
    plaintext = "THISISALONGERPLAINTEXTFORTESTING"
    for cipher in CIPHER_NAMES:
        ciphertext, key = ENCRYPTORS[cipher](plaintext, random.Random(17))
        assert len(ciphertext) == len(plaintext)
        assert ciphertext.isascii() and ciphertext.isalpha()
        assert key
    beaufort = ENCRYPTORS["beaufort"](plaintext, random.Random(17))[1]
    assert beaufort["variant"] == "standard_beaufort_C_equals_K_minus_P_mod_26"


def test_dataset_generation_is_reproducible_and_records_provenance(tmp_path):
    source_dir = tmp_path / "sources"
    filenames = write_sources(source_dir)
    metadata_path = tmp_path / "sources.csv"
    pd.DataFrame(
        {
            "filename": filenames,
            "title": [f"Document {index}" for index in range(len(filenames))],
            "author": ["Test author"] * len(filenames),
            "source_url": ["https://example.test/source"] * len(filenames),
            "license": ["CC0-1.0"] * len(filenames),
            "retrieved_at": ["2026-01-01"] * len(filenames),
        }
    ).to_csv(metadata_path, index=False)
    first, first_manifest = build_dataset(
        source_dir, tmp_path / "one" / "ciphertexts.csv", 23, metadata_path
    )
    second, second_manifest = build_dataset(
        source_dir, tmp_path / "two" / "ciphertexts.csv", 23, metadata_path
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_manifest["dataset_sha256"] == second_manifest["dataset_sha256"]
    assert len(first) == len(filenames) * len(CIPHER_NAMES)
    assert {"source_sha256", "source_url", "license", "partition"} <= set(first.columns)
    assert "plaintext" not in first.columns
    assert first_manifest["ciphers"]["beaufort"]["variant"].startswith("standard Beaufort")

    loaded = load_dataset(tmp_path / "one" / "ciphertexts.csv")
    assert len(loaded) == len(first)
    saved_manifest = json.loads(
        (tmp_path / "one" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["quality_checks"]["all_sources_isolated"] is True


def test_source_partitions_are_reproducible_and_complete():
    sources = pd.DataFrame(
        {"source_id": [f"src_{index}" for index in range(20)]}
    )
    first = assign_source_partitions(sources, seed=9)
    second = assign_source_partitions(sources, seed=9)
    assert first["partition"].tolist() == second["partition"].tolist()
    assert set(first["partition"]) == {"train", "validation", "test"}


def test_duplicate_normalized_sources_are_rejected(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    text = "The same plaintext source has enough letters for validation."
    (source_dir / "first.txt").write_text(text, encoding="utf-8")
    (source_dir / "second.txt").write_text(text.lower(), encoding="utf-8")
    for index in range(4):
        (source_dir / f"other_{index}.txt").write_text(
            f"Different source number {index} has sufficient unique words for testing.",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Duplicate normalized source"):
        build_dataset(source_dir, tmp_path / "ciphertexts.csv")


def test_splitter_honors_preassigned_source_partitions():
    rows = []
    for source, partition in (
        ("s1", "train"),
        ("s2", "validation"),
        ("s3", "test"),
    ):
        rows.append(
            {
                "ciphertext": "ABCDEF",
                "label": "caesar",
                "source_id": source,
                "partition": partition,
            }
        )
    train, validation, test = split_by_source(pd.DataFrame(rows))
    assert train["source_id"].tolist() == ["s1"]
    assert validation["source_id"].tolist() == ["s2"]
    assert test["source_id"].tolist() == ["s3"]


def test_splitter_rejects_source_assigned_to_multiple_partitions():
    data = pd.DataFrame(
        [
            {"ciphertext": "ABC", "label": "caesar", "source_id": "s1", "partition": "train"},
            {"ciphertext": "DEF", "label": "atbash", "source_id": "s1", "partition": "test"},
            {"ciphertext": "GHI", "label": "caesar", "source_id": "s2", "partition": "validation"},
            {"ciphertext": "JKL", "label": "caesar", "source_id": "s3", "partition": "test"},
        ]
    )
    with pytest.raises(ValueError, match="assigned to more than one partition"):
        split_by_source(data)
