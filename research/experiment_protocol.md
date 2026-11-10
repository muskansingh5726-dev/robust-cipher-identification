# Experimental Protocol: Robust Classical Cipher Identification

## 1. Objective

Compare a statistical-feature baseline with a proposed multi-view model for identifying classical cipher types from ciphertext.

The proposed model combines engineered statistical features with character-level TF-IDF features. The experiment will test whether combining these views improves classification performance and how performance changes with ciphertext length.

## 2. Research Questions

1. Does feature fusion improve macro F1 compared with the statistical-feature baseline?
2. How does performance differ between statistical features, character-level TF-IDF, and fusion?
3. How does ciphertext length affect classification performance?
4. Which cipher classes are most frequently confused?

## 3. Dataset Requirements

The dataset must contain:

- `ciphertext`: the encrypted text sample.
- `label`: the cipher type.
- `source_id`: an identifier grouping samples derived from the same underlying plaintext source.

Before training, document the dataset's origin, license, class distribution, source grouping, generation process, and any preprocessing.

Do not treat synthetic test fixtures as research data. Do not proceed to final evaluation until the real dataset has been inspected.

## 4. Data Splitting and Leakage Prevention

- Split by `source_id`, not by individual ciphertext rows.
- Use approximately 60% of sources for training, 20% for validation, and 20% for testing.
- Ensure source IDs do not overlap across splits.
- Fit the TF-IDF vocabulary and all learned preprocessing on training data only.
- Use validation data to select the fusion weight.
- Keep test data untouched until model choices are finalized.
- Confirm that all classes have sufficient representation in each split; document any classes that cannot be split reliably.

## 5. Models

### Baseline

Random Forest trained on engineered statistical features.

### Comparison model

Logistic Regression trained on character-level TF-IDF features.

### Proposed model

Probability-level fusion of the statistical-feature model and the character-level TF-IDF model. Select the fusion weight using validation macro F1 only.

## 6. Evaluation Metrics

Report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall, and F1
- Confusion matrix
- Performance grouped by ciphertext length

Use the same held-out test samples for every model.

## 7. Ciphertext-Length Analysis

Use predefined bins:

- Short: 1–50 alphabetic characters
- Medium: 51–150 alphabetic characters
- Long: more than 150 alphabetic characters

Report the number of test samples in each bin alongside its metrics. Treat results from very small bins as uncertain.

## 8. Reproducibility

Record:

- Dataset version or checksum
- Python and dependency versions
- Random seed
- Source-level split details
- Model parameters
- Selected fusion weight
- Metrics and confusion matrices
- Any errors, exclusions, or preprocessing decisions

Repeat experiments with multiple random seeds where dataset size permits. Report the mean and variability rather than relying on one favorable split.

## 9. Reporting Rules

Do not claim that the proposed model outperforms the baseline until the real experiment has completed successfully.

Do not compare results directly with published accuracy unless dataset, cipher classes, sample construction, and evaluation protocol are sufficiently comparable.

Treat the proposed research contribution as provisional until the related-work review establishes a defensible research gap.

## 10. Dataset Generation and Scope

src.dataset_pipeline creates one deterministic ciphertext variant per source
for each of eight supported classical ciphers. A source_id is derived from the
SHA-256 hash of normalized A-Z plaintext. The source manifest records the
original file hash and citation/license fields when provided.

Source partitions are assigned before encryption and saved in the partition
column. The model splitter honors those assignments, preventing variants of
one plaintext source from crossing train, validation, and test. The generator
also records keys and its seed for reproducibility.

Generated examples are synthetic. They measure performance under the
documented generation setup but do not demonstrate generalization to historical
ciphertext. Any historical evaluation requires a separate, independently
sourced and documented test corpus.

## 11. Current Status

- Software unit tests: 34 passing in the latest run.
- Dataset validation: completed for a 32-book Project Gutenberg source catalog;
  generated ciphertext is synthetic.
- Three source-level split-seed pilot runs are recorded in
  `research/real_corpus_pilot.md`.
- Fusion had the highest test macro F1 on all three synthetic splits; this is
  provisional evidence limited to this corpus and cipher-generation setup.
- Historical-ciphertext performance and research novelty remain
  unestablished. An independently sourced, labeled historical test corpus and
  further related-work review are still needed.
