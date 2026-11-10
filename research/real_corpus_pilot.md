# Real-source synthetic-cipher pilot

## Scope

This pilot uses 32 digitized public-domain books listed in
`data/source_metadata.csv`. The source texts are genuine historical prose;
the ciphertext examples are generated synthetically by this project's eight
documented classical-cipher implementations. This is not a historical
ciphertext benchmark and cannot establish performance on authentic archival
ciphertexts.

The metadata records Project Gutenberg item pages, mirror download URLs,
retrieval date, titles, authors, and author death years. Project Gutenberg
notes that copyright status may differ outside the United States. Check local
rights before redistributing the source files. Downloaded books stay local
under ignored `data/sources/`; no book text is committed.

## Dataset construction

- 32 source books, each assigned wholly to one partition before windowing or
  encryption.
- Stable `source_id`: SHA-256 of normalized, uppercase A-Z document content.
- Per book: ten evenly spaced 40-letter, ten 100-letter, and ten 200-letter
  windows (960 windows total).
- Eight ciphers per window: Caesar, Atbash, affine, monoalphabetic
  substitution, Vigenere, standard reciprocal Beaufort (`C = (K - P) mod
  26`), rail fence, and columnar transposition.
- 7,680 rows total; 960 rows per cipher.
- Each seed produces 18 train, 7 validation, and 7 test source books. Each
  partition contains all classes and all length groups. No source crosses a
  partition.
- Project Gutenberg header/footer text is removed when both standard markers
  are present. Text is normalized to A-Z; the plaintext is never written to
  the generated CSV.

Dataset checksums:

- Seed 42: `ce21c9b5ba83a241370ffb5b104da1afc8e725e01a1f6585372194afa7f63bf3`
- Seed 43: `a033e5da09faea3b24f8861582e56150e12bd409da9f2ee7e555378c28a01bad`
- Seed 44: `7df8a7ae2139f5cd7da201fe9df1cab6bd673d65b4b7b5652e1710c1da3e2cd1`

## Experiment

The existing experiment runner compared the statistical Random Forest,
character TF-IDF logistic regression, and the existing statistical/character
fusion model. The pipeline preserved the source partitions stored in each
dataset. Fusion weight selection used validation macro F1; the test set was
used only for reported metrics. Three separate datasets were generated with
seeds 42, 43, and 44 to vary source-level partitions and cipher keys. All
three runs used 18/7/7 source books in train/validation/test, with 1,680 test
rows (210 per class).

### Test macro F1 (mean and sample standard deviation across three seeds)

| Model | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---:|---:|---:|---:|
| Statistical Random Forest | 0.6202 | 0.6057 | 0.5773 | 0.6011 ± 0.0218 |
| Character TF-IDF logistic regression | 0.5769 | 0.5440 | 0.5621 | 0.5610 ± 0.0165 |
| Statistical + character fusion | 0.6732 | 0.6650 | 0.6690 | 0.6691 ± 0.0041 |

Fusion had the highest test macro F1 on all three generated splits. The
average difference from the statistical baseline was 0.0680 macro-F1 points.
This is evidence only for this corpus and generator setup; it is not a
generalization claim and is not a significance test.

### Fusion test macro F1 by length

| Length group | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---:|---:|---:|---:|
| Short (40 letters) | 0.6231 | 0.6057 | 0.6089 | 0.6126 ± 0.0093 |
| Medium (100 letters) | 0.6784 | 0.6878 | 0.7085 | 0.6916 ± 0.0154 |
| Long (200 letters) | 0.7157 | 0.7007 | 0.6901 | 0.7022 ± 0.0129 |

The seed-42 confusion matrix showed strong Atbash identification (210/210)
and recurring confusion between Beaufort/Vigenere, affine/monoalphabetic
substitution, and columnar transposition/rail fence. Those patterns should
be checked across further seeds and with a historical corpus before being
treated as stable properties.

## Reproduction and limitations

From the repository root, download the catalog, then generate a corpus and
run one source-split experiment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_gutenberg_sources.ps1
python -m src.dataset_pipeline --sources-dir data/sources --metadata-csv data/source_metadata.csv --output data/ciphertexts.csv --seed 42
python -m src.experiment_runner
```

Repeat the generator and runner commands with seeds 43 and 44, writing each
dataset and experiment to its own directory, to reproduce the three-split
summary above. Generated corpora, manifests, and model outputs are ignored by
Git; only the source catalog and this report are tracked.

Limitations include a small set of 32 English-language books, synthetic
encryption with simplified key and preprocessing assumptions, and only seven
held-out source books per run. Ciphertext variants are derived from the same
source windows, so samples are not independent observations. The fixed
window lengths are controlled analysis conditions, not a model of natural
historical message lengths. A separately sourced and labeled historical
ciphertext corpus remains necessary to evaluate historical generalization.
