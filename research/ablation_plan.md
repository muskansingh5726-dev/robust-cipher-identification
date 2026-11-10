# Ablation Plan: Multi-View Classical Cipher Identification

## 1. Objective

Measure the contribution of each model branch and determine whether probability-level fusion provides a measurable benefit over individual models.

## 2. Model Variants

| ID | Model | Purpose |
|---|---|---|
| M1 | Statistical features + Random Forest | Statistical-feature baseline |
| M2 | Character n-gram TF-IDF + Logistic Regression | Character-pattern baseline |
| M3 | Statistical + character probability fusion | Proposed multi-view model |

All variants must use identical source-separated training, validation, and test partitions.

## 3. Primary Metric

Macro F1 is the primary metric because it gives each cipher class equal weight, regardless of class frequency.

Secondary metrics:
- Accuracy
- Macro precision
- Macro recall

Also report per-class precision, recall, and F1, confusion matrices, and performance by ciphertext length.

## 4. Fusion-Weight Selection

Select the statistical-branch weight using validation macro F1 only.

Search the predefined weights from 0.0 to 1.0 in increments of 0.1. Resolve ties deterministically by preferring the weight closest to 0.5, then the lower weight if distances are equal.

Do not select weights using test-set performance.

## 5. Controlled Comparison

Keep the following identical across model variants:
- Dataset version
- Source-level split assignments
- Test examples and labels
- Cipher-class definitions
- Text normalization
- Evaluation metrics

Fit TF-IDF using training data only.

## 6. Robustness Analysis

Where the dataset is large enough, repeat the experiment using multiple predefined random seeds. Report the mean and standard deviation of the metrics.

Analyze results by ciphertext-length group and report the sample count for each group. Interpret results from small groups cautiously.

## 7. Decision Criteria

Feature fusion will be considered beneficial only if it improves the predefined primary metric over the relevant individual-model baseline under the same evaluation protocol.

Report improvements, ties, and regressions honestly. Do not describe a small numerical difference as statistically significant without suitable uncertainty analysis.

## 8. Limitations

Synthetic data results describe performance under the dataset's generation rules. They do not automatically establish performance on historical ciphertext or comparability with published results.

The novelty of multi-view fusion remains unconfirmed until the related literature has been reviewed adequately.

## 9. Current Status

- Model implementations: available.
- Unit tests: 26 passing in the last reported run.
- Real dataset: pending confirmation.
- Real-data experiments: pending.
- Performance improvement: not yet established.
- Novelty: not yet established.
