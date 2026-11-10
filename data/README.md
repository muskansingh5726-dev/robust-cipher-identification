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

At least five distinct documents are required. With the default sample
lengths, each must contain at least 200 ASCII letters after normalization.
Normalization uppercases text and removes every character outside A-Z.
Standard Project Gutenberg start/end boilerplate is stripped when both
markers are present. Plaintext is processed in memory and never written to
the generated CSV.

For document provenance, provide a metadata CSV with one row per filename,
relative to the source directory. Required headers are:

    filename,title,author,source_url,license,retrieved_at

Optional provenance headers are download_url, publication_year,
author_death_year, and terms_url.

The repository includes data/source_metadata.template.csv as a header-only
starter file. Fill in actual source details; do not leave sample or invented
citations in a research dataset.

For the curated Project Gutenberg corpus, download the listed texts with:

    powershell -ExecutionPolicy Bypass -File scripts/download_gutenberg_sources.ps1

The script saves texts locally under data/sources. Those source files remain
outside Git; the metadata CSV records their landing pages, download URLs,
retrieval date, and license note.

Example:

    python -m src.dataset_pipeline --sources-dir data/sources --metadata-csv data/source_metadata.csv --output data/ciphertexts.csv --seed 42

The generator records each source file's SHA-256, a stable source ID based on
normalized plaintext, and supplied citation/license fields. Duplicate
normalized plaintexts are rejected. Keep source documents and generated data
out of Git unless their licenses and repository size have been reviewed.
The source folder and generated CSV/manifest files are ignored by Git.

## Outputs and reproducibility

- ciphertexts.csv: one row per source window and cipher, with ciphertext,
  label, stable source_id, partition, key specification, and provenance
  fields. The default corpus has 30 windows per source and 8 cipher variants
  per window.
- source_manifest.csv: source hashes, stable IDs, citation fields, letter
  counts, and the preassigned partition. It contains no plaintext.
- dataset_manifest.json: seed, generator version, algorithms, partition
  method, quality summary, and dataset SHA-256.

The source partition is assigned before text windows or cipher variants are
generated. Each book is a single source group, so windows from the same book
cannot cross partitions. The default creates ten evenly spaced windows of
40, 100, and 200 letters per source. These target the short, medium, and long
analysis bins. src.data_splitting.split_by_source honors the saved
assignments. Source proportions are approximately 60% train, 20% validation,
and 20% test; small corpora cannot match these proportions exactly.

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
partition, source leakage, inconsistent window counts, and incomplete cipher
variants. Exact duplicate ciphertexts are counted in the manifest for review.
Near-duplicate prose still requires a separate source review. The manifest
includes class and length-group counts per partition, letter-length
statistics, and a warning when source titles or licenses are missing.

## Corpus rights and scope

The included text catalog uses Project Gutenberg item pages. Project Gutenberg
says most of its books are unrestricted under U.S. copyright, but asks readers
outside the U.S. to check local law. The metadata records the item page and
license note; it does not claim blanket worldwide clearance. Confirm local
reuse rights before redistributing source files.

## Interpretation

This pipeline creates synthetic ciphertext from supplied text sources.
Results describe performance under the documented source texts and
implementations; they do not establish generalization to historical
ciphertext. Historical material may differ in language, spelling, encipherment
practice, damage, format, and key choice. Do not report synthetic data as a
historical test set.
