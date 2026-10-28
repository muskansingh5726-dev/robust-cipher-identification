# Literature Review: Classical Cipher Identification

## 1. Research Problem

Classical cipher identification is the task of predicting the cipher family or type from ciphertext alone. It is a cryptanalysis task that supports subsequent cipher-specific analysis; it does not itself decrypt the message.

## 2. Base Paper

**Leierzopf et al. (2021)**
*A Massive Machine-Learning Approach for Classical Cipher Type Detection Using Feature Engineering.*

The paper evaluates engineered ciphertext features and machine-learning classifiers across 56 classical cipher types. It reports 80.24% accuracy for its best-performing model.

Source: https://ecp.ep.liu.se/index.php/histocrypt/article/view/164

## 3. Related Work

### Classical cipher identification with machine learning

Earlier work explored neural networks for identifying classical cipher types, including the HistoCrypt 2020 study by Nils Kopal.

Source: https://ep.liu.se/en/conference-article.aspx?Article_No=11&issue=171&series=ecp

### Machine learning for historical cipher identification

Dalton and Stamp studied machine-learning and deep-learning classifiers for selected World War II-era ciphers. Their experiments considered different plaintext/key-generation scenarios and analyzed the effect of ciphertext length.

Source: https://doi.org/10.1080/01611194.2024.2304888

### Binary representations and deep learning

A 2026 study investigated binary-encoded ciphertext representations and deep-learning architectures for classical cipher classification.

Source: https://doi.org/10.1016/j.procs.2026.05.133

## 4. Proposed Experimental Direction

Investigate a combination of:

1. Statistical ciphertext features, including letter frequencies, index of coincidence, and entropy.
2. Character-level TF-IDF features representing ciphertext character n-grams.
3. Probability-level fusion of statistical and character-level classifiers.

Evaluate all approaches using source-separated data splits and the same held-out test examples.

## 5. Research Questions

- Does feature fusion improve macro F1 compared with each individual branch?
- Does performance change as ciphertext length decreases?
- Which cipher classes are most frequently confused?
- Are the results consistent across multiple random seeds?
- How sensitive are results to class balance and data-generation settings?

## 6. Research Gap: Tentative

The proposed study will investigate whether feature fusion provides a measurable advantage under a controlled source-separated evaluation, particularly across ciphertext lengths.

This is a provisional research direction. A more complete review is needed before claiming that the combination or evaluation protocol is novel.

## 7. Evaluation Metrics

Report accuracy, macro precision, macro recall, macro F1, per-class results, and confusion matrices. Document dataset provenance, source-grouping rules, random seeds, and cipher-generation settings.

## 8. Current Limitations

- The research dataset and baseline reproduction are pending.
- No real experimental performance has been established.
- The proposed fusion model is not yet demonstrated to outperform the baseline.
- Novelty must be established through further literature review and experimental evidence.
