# Dataset pipeline

## Project
Robust Multi-View Classical Cipher Identification

## Purpose
This directory stores datasets and metadata used to train and evaluate classical-cipher identification models.

## Required Data Format
The prepared dataset should contain:

- `ciphertext`: encrypted text
- `label`: cipher type
- `source_id`: identifier of the original plaintext or source document
- `text_length`: number of alphabetic characters in the ciphertext

## Data Quality Requirements
- Cipher labels must be verified.
- Training and test partitions must not share the same source plaintext.
- Ciphertext lengths and class distributions must be documented.
- Duplicate and near-duplicate plaintext sources should be checked.
- Evaluation data must not be used to tune the model.

## Reproducibility
Record the dataset source, license, preprocessing procedure, cipher-generation settings, and random seeds.

## Storage
Do not commit large datasets, downloaded books, or generated model files to Git without checking repository size and licensing requirements first.

## Current Status
The pipeline described below generates reproducible synthetic data. It does
not replace a separately sourced historical test corpus.

## Generate a dataset

Put UTF-8 .txt source documents in data/sources/ and run from the project root:

    python -m src.dataset_pipeline --sources-dir data/sources --output data/ciphertexts.csv --seed 42

At least five distinct documents are required. Each must contain at least 20
ASCII letters after normalization. Normalization uppercases text and removes
every character outside A-Z. Plaintext is processed in memory and never
written to the generated CSV.

For document provenance, provide a metadata CSV with one row per filename,
relative to the source directory. Required headers are:

    filename,title,author,source_url,license,retrieved_at

The repository includes data/source_metadata.template.csv as a header-only
starter file. Fill in actual source details; do not leave sample or invented
citations in a research dataset.

Example:

    python -m src.dataset_pipeline --sources-dir data/sources --metadata-csv data/source_metadata.csv --output data/ciphertexts.csv --seed 42

The generator records each source file's SHA-256, a stable source ID based on
normalized plaintext, and supplied citation/license fields. Duplicate
normalized plaintexts are rejected. Keep source documents and generated data
out of Git unless their licenses and repository size have been reviewed.
The source folder and generated CSV/manifest files are ignored by Git.

## Outputs and reproducibility

- ciphertexts.csv: one row per source and cipher, with ciphertext, label,
  source_id, partition, key specification, and provenance fields.
- source_manifest.csv: source hashes, stable IDs, citation fields, letter
  counts, and the preassigned partition. It contains no plaintext.
- dataset_manifest.json: seed, generator version, algorithms, partition
  method, quality summary, and dataset SHA-256.

The source partition is assigned before cipher variants are generated.
src.data_splitting.split_by_source honors those assignments. The target
proportions are approximately 60% train, 20% validation, and 20% test; small
corpora cannot match these proportions exactly.

Each cipher receives one deterministic key per source, derived from the
configured seed, source ID, and cipher name. The eight labels are Caesar,
Atbash, affine, monoalphabetic substitution, Vigenere, Beaufort, rail fence,
and columnar transposition. Beaufort uses the standard reciprocal form
C = (K - P) mod 26. Letters are represented as A=0 through Z=25.
Vigenere uses C = (P + K) mod 26 with the same letter representation.

Caesar shifts range from 1 to 25. Affine keys use a value coprime to 26 and an
offset from 0 to 25. Substitution uses a random permutation. Vigenere and
Beaufort keys contain eight sampled letters and repeat. Rail fence uses 2 to
6 rails, bounded by plaintext length. Columnar transposition uses a random
column permutation, widths from 2 to 8 bounded by plaintext length, and no
padding. The realized key specification is stored on each row.

Quality checks reject malformed ciphertext, repeated sample IDs, duplicate
normalized sources, incomplete source coverage, missing classes in a
partition, and source leakage. Exact duplicate ciphertexts are counted in the
manifest for review. Near-duplicate prose still requires a separate source
review. The manifest includes class counts per partition, letter-length
statistics, and a warning when source titles or licenses are missing.

## Interpretation

This pipeline creates synthetic ciphertext from supplied text sources.
Results describe performance under the documented source texts and
implementations; they do not establish generalization to historical
ciphertext. Historical material may differ in language, spelling, encipherment
practice, damage, format, and key choice. Do not report synthetic data as a
historical test set.
