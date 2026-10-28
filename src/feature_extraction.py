
from collections import Counter
import math
import string

import numpy as np


ALPHABET = string.ascii_uppercase


def normalize_ciphertext(ciphertext: str) -> str:
    """Keep English letters and normalize them to uppercase."""
    if not isinstance(ciphertext, str):
        raise TypeError("Ciphertext must be a string.")

    return "".join(
        character for character in ciphertext.upper()
        if character in ALPHABET
    )


def extract_features(ciphertext: str) -> dict[str, float]:
    """
    Extract statistical features from classical-cipher ciphertext.

    Returns 33 features:
    - 7 summary statistics
    - 26 normalized letter frequencies
    """
    text = normalize_ciphertext(ciphertext)

    if not text:
        raise ValueError("Ciphertext must contain at least one English letter.")

    length = len(text)
    counts = Counter(text)
    probabilities = np.array(
        [counts.get(letter, 0) / length for letter in ALPHABET],
        dtype=float,
    )

    # Index of coincidence
    ioc = (
        sum(count * (count - 1) for count in counts.values())
        / (length * (length - 1))
        if length > 1
        else 0.0
    )

    # Shannon entropy, measured in bits per character
    entropy = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )

    # Repeated-character and unique-character proportions
    unique_ratio = len(counts) / length
    repeated_ratio = (length - len(counts)) / length

    # Number of distinct adjacent bigrams divided by all bigrams
    bigrams = [text[i:i + 2] for i in range(length - 1)]
    bigram_diversity = (
        len(set(bigrams)) / len(bigrams)
        if bigrams
        else 0.0
    )

    features = {
        "index_of_coincidence": float(ioc),
        "shannon_entropy": float(entropy),
        "mean_letter_frequency": float(probabilities.mean()),
        "std_letter_frequency": float(probabilities.std()),
        "repeated_character_ratio": float(repeated_ratio),
        "unique_character_ratio": float(unique_ratio),
        "bigram_diversity": float(bigram_diversity),
    }

    for letter, probability in zip(ALPHABET, probabilities):
        features[f"frequency_{letter}"] = float(probability)

    return features
