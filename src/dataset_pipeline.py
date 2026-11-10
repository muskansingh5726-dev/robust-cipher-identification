"""Build a reproducible synthetic classical-cipher dataset from local texts.

Plaintext is normalized to A-Z before encryption and is never written to output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PIPELINE_VERSION = "1.0"
CIPHER_NAMES = (
    "caesar",
    "atbash",
    "affine",
    "monoalphabetic_substitution",
    "vigenere",
    "beaufort",
    "rail_fence",
    "columnar_transposition",
)
CIPHER_KEY_RULES = {
    "caesar": "uniform shift from 1 through 25",
    "atbash": "fixed A-Z reversal",
    "affine": "a chosen uniformly from values coprime to 26; b from 0 through 25",
    "monoalphabetic_substitution": "uniform random permutation of A-Z",
    "vigenere": "8 uniformly sampled A-Z key characters; repeated",
    "beaufort": "8 uniformly sampled A-Z key characters; repeated",
    "rail_fence": "rails chosen from 2 through min(6, plaintext length - 1)",
    "columnar_transposition": (
        "width chosen from 2 through min(8, plaintext length - 1); "
        "uniform random column permutation; no padding"
    ),
}
PARTITIONS = ("train", "validation", "test")
PROVENANCE_COLUMNS = (
    "filename",
    "title",
    "author",
    "source_url",
    "license",
    "retrieved_at",
)


def normalize_plaintext(text: str) -> str:
    """Return only uppercase ASCII letters, matching model preprocessing."""
    return re.sub(r"[^A-Z]", "", text.upper())


def stable_source_id(normalized_text: str) -> str:
    """Identify a source by normalized content, independent of its file path."""
    return "src_" + hashlib.sha256(normalized_text.encode("ascii")).hexdigest()


def _derive_rng(seed: int, source_id: str, cipher: str) -> random.Random:
    token = f"{seed}\0{source_id}\0{cipher}\0{PIPELINE_VERSION}".encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return random.Random(derived)


def _shift(text: str, offsets: list[int]) -> str:
    return "".join(ALPHABET[(ord(char) - 65 + offsets[i]) % 26] for i, char in enumerate(text))


def encrypt_caesar(text: str, rng: random.Random) -> tuple[str, dict]:
    shift = rng.randint(1, 25)
    return _shift(text, [shift] * len(text)), {"shift": shift}


def encrypt_atbash(text: str, rng: random.Random) -> tuple[str, dict]:
    del rng
    return "".join(ALPHABET[25 - (ord(char) - 65)] for char in text), {"mapping": "A<->Z"}


def encrypt_affine(text: str, rng: random.Random) -> tuple[str, dict]:
    a = rng.choice([value for value in range(1, 26) if math.gcd(value, 26) == 1])
    b = rng.randrange(26)
    return "".join(ALPHABET[(a * (ord(char) - 65) + b) % 26] for char in text), {"a": a, "b": b}


def encrypt_monoalphabetic(text: str, rng: random.Random) -> tuple[str, dict]:
    key = list(ALPHABET)
    rng.shuffle(key)
    mapping = dict(zip(ALPHABET, key))
    return "".join(mapping[char] for char in text), {"alphabet": "".join(key)}


def _polyalphabetic(text: str, key: str, beaufort: bool = False) -> str:
    values = []
    for index, char in enumerate(text):
        plain = ord(char) - 65
        key_value = ord(key[index % len(key)]) - 65
        values.append((key_value - plain if beaufort else plain + key_value) % 26)
    return "".join(ALPHABET[value] for value in values)


def encrypt_vigenere(text: str, rng: random.Random) -> tuple[str, dict]:
    key = "".join(rng.choice(ALPHABET) for _ in range(8))
    return _polyalphabetic(text, key), {"key": key}


def encrypt_beaufort(text: str, rng: random.Random) -> tuple[str, dict]:
    """Standard Beaufort variant: C = (K - P) mod 26 (reciprocal form)."""
    key = "".join(rng.choice(ALPHABET) for _ in range(8))
    return _polyalphabetic(text, key, beaufort=True), {
        "key": key,
        "variant": "standard_beaufort_C_equals_K_minus_P_mod_26",
    }


def _rail_pattern(length: int, rails: int) -> list[int]:
    pattern = []
    rail, direction = 0, 1
    for _ in range(length):
        pattern.append(rail)
        if rail == 0:
            direction = 1
        elif rail == rails - 1:
            direction = -1
        rail += direction
    return pattern


def encrypt_rail_fence(text: str, rng: random.Random) -> tuple[str, dict]:
    rails = rng.randint(2, min(6, max(2, len(text) - 1)))
    pattern = _rail_pattern(len(text), rails)
    output = "".join(text[i] for rail in range(rails) for i, row in enumerate(pattern) if row == rail)
    return output, {"rails": rails, "read_order": "top_to_bottom"}


def encrypt_columnar(text: str, rng: random.Random) -> tuple[str, dict]:
    width = rng.randint(2, min(8, max(2, len(text) - 1)))
    order = list(range(width))
    rng.shuffle(order)
    rank_by_column = {column: rank for rank, column in enumerate(order)}
    ciphertext = "".join(
        text[column::width]
        for column in sorted(range(width), key=rank_by_column.__getitem__)
    )
    return ciphertext, {
        "column_order_zero_based": order,
        "padding": "none",
        "read_order": "columns_in_key_order",
    }


ENCRYPTORS: dict[str, Callable[[str, random.Random], tuple[str, dict]]] = {
    "caesar": encrypt_caesar,
    "atbash": encrypt_atbash,
    "affine": encrypt_affine,
    "monoalphabetic_substitution": encrypt_monoalphabetic,
    "vigenere": encrypt_vigenere,
    "beaufort": encrypt_beaufort,
    "rail_fence": encrypt_rail_fence,
    "columnar_transposition": encrypt_columnar,
}


def assign_source_partitions(
    sources: pd.DataFrame, seed: int, minimum_sources: int = 5
) -> pd.DataFrame:
    """Assign deterministic 60/20/20 partitions before cipher generation."""
    if sources.empty:
        raise ValueError("No source documents were provided.")
    if sources["source_id"].duplicated().any():
        raise ValueError("Duplicate normalized plaintext sources were found.")
    if len(sources) < minimum_sources:
        raise ValueError(f"At least {minimum_sources} distinct sources are required.")
    groups = sources["source_id"].astype(str)
    train_val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(
            sources, groups=groups
        )
    )
    train_val = sources.iloc[train_val_idx]
    train_idx, validation_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train_val, groups=train_val["source_id"].astype(str)
        )
    )
    result = sources.copy()
    result["partition"] = ""
    result.loc[train_val.index[train_idx], "partition"] = "train"
    result.loc[train_val.index[validation_idx], "partition"] = "validation"
    result.loc[sources.index[test_idx], "partition"] = "test"
    if result["partition"].eq("").any():
        raise RuntimeError("Some sources were not assigned a partition.")
    return result


def validate_dataset(data: pd.DataFrame, sources: pd.DataFrame) -> dict:
    """Check schema, labels, source isolation, duplicates, and row completeness."""
    required = {"ciphertext", "label", "source_id", "partition", "sample_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Generated data is missing columns: {sorted(missing)}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("Generated data is empty or has missing required fields.")
    if data["ciphertext"].astype(str).str.fullmatch(r"[A-Z]+").eq(False).any():
        raise ValueError("Ciphertexts must contain only A-Z letters.")
    if data["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs were generated.")
    if not set(data["label"]).issubset(CIPHER_NAMES):
        raise ValueError("Unexpected cipher label found.")
    if set(data["partition"]) != set(PARTITIONS):
        raise ValueError("All train, validation, and test partitions must be present.")
    per_source = data.groupby("source_id")["partition"].nunique()
    if per_source.ne(1).any():
        raise ValueError("A source appears in more than one partition.")
    if set(data["source_id"]) != set(sources["source_id"]):
        raise ValueError("Generated rows do not cover the source manifest.")
    expected_partitions = sources.set_index("source_id")["partition"].astype(str)
    observed_partitions = (
        data.drop_duplicates("source_id")
        .set_index("source_id")["partition"]
        .astype(str)
    )
    if not expected_partitions.sort_index().equals(
        observed_partitions.reindex(expected_partitions.index).sort_index()
    ):
        raise ValueError("Generated rows do not match source-manifest partitions.")
    expected = len(sources) * len(CIPHER_NAMES)
    if len(data) != expected:
        raise ValueError(f"Expected {expected} variants but generated {len(data)}.")
    counts = data.groupby(["partition", "label"]).size().unstack(fill_value=0)
    if (counts == 0).any().any():
        raise ValueError("Every cipher label must occur in every partition.")
    duplicate_ciphertexts = int(data["ciphertext"].duplicated(keep=False).sum())
    provenance_incomplete = int(
        sources["title"].astype(str).str.strip().eq("").sum()
        + sources["license"].astype(str).str.strip().eq("").sum()
    )
    return {
        "sample_count": int(len(data)),
        "source_count": int(len(sources)),
        "class_counts": {str(key): int(value) for key, value in data["label"].value_counts().items()},
        "partition_class_counts": {
            str(partition): {
                str(label): int(value)
                for label, value in row.items()
            }
            for partition, row in counts.iterrows()
        },
        "partition_source_counts": {
            str(key): int(value)
            for key, value in sources["partition"].value_counts().reindex(PARTITIONS, fill_value=0).items()
        },
        "ciphertext_length_letters": {
            "min": int(data["text_length"].min()),
            "median": float(data["text_length"].median()),
            "max": int(data["text_length"].max()),
        },
        "exact_duplicate_ciphertext_rows": duplicate_ciphertexts,
        "sources_missing_title_or_license": provenance_incomplete,
        "all_sources_isolated": True,
    }


def _read_sources(source_dir: Path, metadata_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(path for path in source_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"No .txt source documents found in {source_dir}.")
    metadata = pd.DataFrame(columns=PROVENANCE_COLUMNS)
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, dtype="string").fillna("")
        missing = set(PROVENANCE_COLUMNS) - set(metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
        if metadata["filename"].duplicated().any():
            raise ValueError("Metadata filenames must be unique.")
        metadata = metadata.set_index("filename")
    rows, plaintexts, seen_ids = [], [], set()
    for path in files:
        relative = path.relative_to(source_dir).as_posix()
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8-sig")
        normalized = normalize_plaintext(raw_text)
        if len(normalized) < 20:
            raise ValueError(f"Source has fewer than 20 A-Z letters: {relative}")
        source_id = stable_source_id(normalized)
        if source_id in seen_ids:
            raise ValueError(f"Duplicate normalized source content: {relative}")
        seen_ids.add(source_id)
        details = metadata.loc[relative].to_dict() if relative in metadata.index else {}
        rows.append({
            "source_id": source_id,
            "filename": relative,
            "title": details.get("title", ""),
            "author": details.get("author", ""),
            "source_url": details.get("source_url", ""),
            "license": details.get("license", ""),
            "retrieved_at": details.get("retrieved_at", ""),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "normalized_character_count": len(normalized),
        })
        plaintexts.append(normalized)
    if metadata_path is not None:
        unknown = set(metadata.index) - {row["filename"] for row in rows}
        if unknown:
            raise ValueError(f"Metadata lists files not found under source directory: {sorted(unknown)}")
    return pd.DataFrame(rows), plaintexts


def build_dataset(
    source_dir: str | Path,
    output_csv: str | Path,
    seed: int = 42,
    metadata_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create dataset CSV and manifests from a directory of UTF-8 text files."""
    source_dir = Path(source_dir)
    output_csv = Path(output_csv)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    sources, plaintexts = _read_sources(
        source_dir, Path(metadata_csv) if metadata_csv else None
    )
    # This assignment happens before the loop that creates any ciphertext.
    sources = assign_source_partitions(sources, seed)
    rows = []
    for source, plaintext in zip(sources.to_dict("records"), plaintexts):
        for label, encryptor in ENCRYPTORS.items():
            ciphertext, key_spec = encryptor(
                plaintext, _derive_rng(seed, source["source_id"], label)
            )
            sample_id = hashlib.sha256(
                f"{source['source_id']}\0{label}\0{json.dumps(key_spec, sort_keys=True)}".encode()
            ).hexdigest()
            rows.append({
                "sample_id": sample_id,
                "ciphertext": ciphertext,
                "label": label,
                "source_id": source["source_id"],
                "partition": source["partition"],
                "text_length": len(ciphertext),
                "key_spec": json.dumps(key_spec, sort_keys=True),
                **{column: source[column] for column in PROVENANCE_COLUMNS},
                "source_sha256": source["source_sha256"],
            })
    data = pd.DataFrame(rows)
    summary = validate_dataset(data, sources)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)
    source_manifest_path = output_csv.with_name("source_manifest.csv")
    sources.to_csv(source_manifest_path, index=False)
    digest = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "input_directory": str(source_dir),
        "input_metadata_csv": str(metadata_csv) if metadata_csv else None,
        "normalization": "uppercase; retain ASCII A-Z only; no plaintext is emitted",
        "source_id": "src_ plus SHA-256 of normalized A-Z plaintext",
        "partitioning": {
            "method": "GroupShuffleSplit on unique source IDs before encryption",
            "fractions_approximate": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "seed": seed,
        },
        "ciphers": {
            **{
                name: {
                    "key_generation": rule,
                    "randomness": (
                        "deterministic PRNG derived from seed, source_id, "
                        "cipher name, and pipeline version"
                    ),
                }
                for name, rule in CIPHER_KEY_RULES.items()
            },
            "beaufort": {
                "variant": "standard Beaufort, C = (K - P) mod 26",
                "key_generation": CIPHER_KEY_RULES["beaufort"],
            },
        },
        "quality_checks": summary,
        "source_manifest": source_manifest_path.name,
        "dataset_sha256": digest,
        "warning": (
            "All ciphertext is synthetic. Results measure performance on this "
            "generation setup and do not establish generalization to historical ciphertext."
        ),
    }
    if summary["sources_missing_title_or_license"]:
        manifest["provenance_warning"] = (
            "Some sources lack a title or license in the metadata. Complete and "
            "review source provenance before treating this as research data."
        )
    manifest_path = output_csv.with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", required=True, help="Directory of UTF-8 .txt source documents.")
    parser.add_argument("--metadata-csv", help="Optional CSV with filename,title,author,source_url,license,retrieved_at.")
    parser.add_argument("--output", default="data/ciphertexts.csv", help="Output canonical dataset CSV.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data, manifest = build_dataset(args.sources_dir, args.output, args.seed, args.metadata_csv)
    print(f"Generated {len(data)} rows from {manifest['quality_checks']['source_count']} sources.")
    print(f"Dataset: {Path(args.output).resolve()}")
    print(f"Manifest: {Path(args.output).with_name('dataset_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
