# Robust Classical Cipher Identification

This repository compares statistical and character-level models for
identifying classical cipher families. It includes a reproducible generator
for synthetic ciphertext and source-separated experiments.

See data/README.md for source preparation, provenance, generation, and quality
checks. The generated corpus is synthetic; results must not be presented as
evidence of performance on historical ciphertext.

Set up the Python environment from the project root:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Run the model experiment after creating the local dataset:

    python -m src.experiment_runner
