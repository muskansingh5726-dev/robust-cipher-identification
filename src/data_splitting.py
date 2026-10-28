
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_source(data: pd.DataFrame, random_state: int = 42):
    """
    Split data into approximately 60% train, 20% validation,
    and 20% test while keeping each source_id in one partition.

    All three partitions must contain every class.
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
        raise ValueError("Required fields cannot contain missing values.")

    data = data.reset_index(drop=True).copy()
    data["label"] = data["label"].astype(str)
    data["source_id"] = data["source_id"].astype(str)

    if data["source_id"].str.strip().eq("").any():
        raise ValueError("source_id cannot be empty.")

    if data["label"].nunique() < 2:
        raise ValueError("At least two classes are required.")

    if data["source_id"].nunique() < 5:
        raise ValueError(
            "At least five distinct source IDs are needed for this split."
        )

    # Reserve 20% of source groups for testing.
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(
        outer.split(data, data["label"], groups=data["source_id"])
    )

    train_val = data.iloc[train_val_idx].copy()
    test = data.iloc[test_idx].copy()

    # Reserve 25% of remaining source groups for validation.
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=random_state + 1,
    )

    train_idx, val_idx = next(
        inner.split(
            train_val,
            train_val["label"],
            groups=train_val["source_id"],
        )
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    expected_classes = set(data["label"].unique())

    for name, partition in (
        ("training", train),
        ("validation", validation),
        ("test", test),
    ):
        if set(partition["label"].unique()) != expected_classes:
            raise ValueError(
                f"The {name} split does not contain every class. "
                "Use more source groups or review the dataset."
            )

    train_sources = set(train["source_id"])
    val_sources = set(validation["source_id"])
    test_sources = set(test["source_id"])

    if (
        train_sources & val_sources
        or train_sources & test_sources
        or val_sources & test_sources
    ):
        raise RuntimeError("Source leakage detected between splits.")

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
