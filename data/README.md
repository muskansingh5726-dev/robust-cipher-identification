# Dataset Documentation

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
No research dataset has been added yet. Model evaluation must wait until the dataset and split strategy are established.