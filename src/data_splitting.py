
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_source(data, random_state=42):
    """
    Split rows into train, validation, and test partitions by source_id.

    Approximate proportions: 60% train, 20% validation, 20% test.
    The same source_id must never appear in multiple partitions.
    """
    required = {"ciphertext", "label", "source_id"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data[list(required)].isna().any().any():
        raise ValueError("Required fields cannot be missing.")

    # Dataset-generation pipelines may assign source partitions before
    # producing cipher variants. Preserve those assignments so training does
    # not silently re-split the generated examples.
    if "partition" in data.columns:
        partition = data["partition"].astype("string").str.strip().str.lower()
        allowed = {"train", "validation", "test"}
        if partition.isna().any() or not set(partition).issubset(allowed):
            raise ValueError(
                "Preassigned partitions must be train, validation, or test."
            )
        if set(partition) != allowed:
            raise ValueError(
                "Preassigned data must contain train, validation, and test."
            )
        source_partitions = pd.DataFrame({
            "source_id": data["source_id"].astype(str),
            "partition": partition,
        }).groupby("source_id")["partition"].nunique()
        if source_partitions.gt(1).any():
            raise ValueError(
                "A source_id is assigned to more than one partition."
            )
        train = data.loc[partition.eq("train")].copy()
        validation = data.loc[partition.eq("validation")].copy()
        test = data.loc[partition.eq("test")].copy()
        if any(frame.empty for frame in (train, validation, test)):
            raise ValueError("Preassigned partitions cannot be empty.")
        return train, validation, test

    source_counts = data["source_id"].nunique()

    if source_counts < 5:
        raise ValueError(
            "At least five distinct source IDs are required."
        )

    groups = data["source_id"].astype(str)

    # First, hold out approximately 20% of sources for testing.
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(
        outer_split.split(data, groups=groups)
    )

    train_val = data.iloc[train_val_idx].copy()
    test = data.iloc[test_idx].copy()

    # Of the remaining 80%, hold out 25% for validation.
    # This yields approximately 60/20/20 overall.
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=random_state,
    )

    inner_groups = train_val["source_id"].astype(str)

    train_idx, val_idx = next(
        inner_split.split(train_val, groups=inner_groups)
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    train_sources = set(train["source_id"].astype(str))
    val_sources = set(validation["source_id"].astype(str))
    test_sources = set(test["source_id"].astype(str))

    if train_sources & val_sources:
        raise RuntimeError("Source leakage between train and validation.")

    if train_sources & test_sources:
        raise RuntimeError("Source leakage between train and test.")

    if val_sources & test_sources:
        raise RuntimeError("Source leakage between validation and test.")

    return train, validation, test
