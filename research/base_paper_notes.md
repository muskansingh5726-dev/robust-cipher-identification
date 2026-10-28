# Base Paper Notes

## Paper Title
A Massive Machine-Learning Approach for Classical Cipher Type Detection Using Feature Engineering

## Authors
Ernst Leierzopf, Nils Kopal, Bernhard Esslinger, Harald Lampesberger, and Eckehard Hermann.

## Publication
HistoCrypt 2021.

## Research Problem
Identify the type of classical cipher from ciphertext alone, without knowing the encryption algorithm in advance.

## Objective
Evaluate engineered ciphertext features and machine-learning classifiers for classifying classical cipher types.

## Methods Investigated
- Feedforward neural networks
- Random forests
- Naive Bayes networks
- Other machine-learning approaches discussed in the paper

## Reported Result
The paper reports 80.24% accuracy for its best-performing model. This is a result reported by the authors, not a result reproduced by our project.

## Original Implementation
https://github.com/ernstleierzopf/ncid

## Our Initial Baseline
Random Forest using statistical ciphertext features.

## Proposed Direction
Investigate whether combining statistical features with character-level TF-IDF features improves classification, particularly for shorter ciphertexts.

## Research Questions
1. Does feature fusion improve macro F1 compared with statistical features alone?
2. How does performance change as ciphertext length decreases?
3. Which cipher classes are most frequently confused?
4. Does the proposed approach improve results under identical data splits?

## Important Limitations
- The proposed architecture is a research hypothesis, not a proven novel contribution.
- The original paper's results must not be compared directly with ours unless the evaluation settings are sufficiently comparable.
- Dataset provenance, cipher-generation settings, source-grouped splits, and software versions must be documented.
- Results must be measured experimentally rather than estimated or invented.

## Current Status
Feature extraction and dataset-loader tests have passed. A research dataset and baseline evaluation are still pending.