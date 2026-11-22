
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = Path("data/interim/cicids2017_clean.parquet")
OUTPUT_DIR = Path("data/processed/v3")
MANIFEST_FILE = OUTPUT_DIR / "split_manifest_v3.json"

RANDOM_STATE = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "calibration": 0.10,
    "val": 0.10,
    "test": 0.10,
}

SOURCE_COLUMN = "source_file"
ROW_COLUMN = "source_row"
FAMILY_COLUMN = "attack_family"
TARGET_COLUMN = "is_malicious"

REQUIRED_COLUMNS = {
    SOURCE_COLUMN,
    ROW_COLUMN,
    FAMILY_COLUMN,
    TARGET_COLUMN,
}


# ---------------------------------------------------------
# Allocation
# ---------------------------------------------------------

def allocate_counts(n):
    """
    Allocate rows across four partitions.

    Every stratum with at least four rows receives at least
    one row in each partition. Remaining rows follow the
    configured 70/10/10/10 proportions.
    """

    if n < 4:
        raise ValueError(
            f"Stratum has only {n} rows. "
            "At least four are required for all partitions."
        )

    names = list(SPLIT_RATIOS)
    weights = np.array(
        [SPLIT_RATIOS[name] for name in names],
        dtype=float,
    )

    remaining = n - len(names)
    quotas = remaining * weights

    counts = np.floor(quotas).astype(int) + 1

    remainder = n - int(counts.sum())

    fractional = quotas - np.floor(quotas)
    order = np.argsort(-fractional, kind="stable")

    for index in order[:remainder]:
        counts[index] += 1

    assert int(counts.sum()) == n
    assert np.all(counts >= 1)

    return dict(zip(names, counts.tolist()))


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def validate_input(df):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Input dataset is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    for column in REQUIRED_COLUMNS:
        if df[column].isna().any():
            raise ValueError(
                f"Missing values found in required metadata column: {column}"
            )

    labels = set(pd.to_numeric(df[TARGET_COLUMN], errors="raise").unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 and 1. Found: {labels}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    # Benign must be 0; every non-Benign family must be 1.
    benign_mask = (
        df[FAMILY_COLUMN].astype(str).str.strip().str.lower() == "benign"
    )

    inconsistent = (
        (benign_mask & (df[TARGET_COLUMN] != 0))
        | (~benign_mask & (df[TARGET_COLUMN] != 1))
    )

    if inconsistent.any():
        examples = df.loc[
            inconsistent,
            [FAMILY_COLUMN, TARGET_COLUMN],
        ].head(10)

        raise ValueError(
            "attack_family and is_malicious disagree. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    # The source-row identity must be unique.
    duplicate_mask = df.duplicated(
        subset=[SOURCE_COLUMN, ROW_COLUMN],
        keep=False,
    )

    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask,
            [SOURCE_COLUMN, ROW_COLUMN],
        ].head(10)

        raise ValueError(
            "Duplicate source_file/source_row identities detected:\n"
            f"{examples.to_string(index=False)}"
        )

    return df


# ---------------------------------------------------------
# Stratified partitioning
# ---------------------------------------------------------

def create_partitions(df):
    rng = np.random.default_rng(RANDOM_STATE)

    partition_indices = {
        name: [] for name in SPLIT_RATIOS
    }

    grouped = df.groupby(
        [SOURCE_COLUMN, FAMILY_COLUMN],
        sort=True,
        dropna=False,
    )

    print("\nAllocating rows by source capture and attack family...")

    for (source_file, family), group in grouped:
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        counts = allocate_counts(len(indices))

        cursor = 0

        for partition_name in SPLIT_RATIOS:
            count = counts[partition_name]

            selected = indices[cursor:cursor + count]

            partition_indices[partition_name].extend(
                selected.tolist()
            )

            cursor += count

        print(
            f"{source_file} | {family}: "
            f"{len(group):,} rows -> "
            + ", ".join(
                f"{name}={counts[name]:,}"
                for name in SPLIT_RATIOS
            )
        )

        if cursor != len(indices):
            raise RuntimeError(
                f"Allocation error for {source_file} / {family}"
            )

    partitions = {}

    for name, indices in partition_indices.items():
        # Keep original columns and values. No synthetic metadata
        # columns are added to the actual model datasets.
        part = df.loc[indices].copy()

        part = part.sort_values(
            [SOURCE_COLUMN, ROW_COLUMN],
            kind="stable",
        ).reset_index(drop=True)

        partitions[name] = part

    return partitions


# ---------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------

def row_identity_set(df):
    return set(
        zip(
            df[SOURCE_COLUMN].astype(str),
            df[ROW_COLUMN].astype(str),
        )
    )


def verify_partitions(original, partitions):
    print("\n========== INTEGRITY CHECKS ==========")

    original_ids = row_identity_set(original)

    if len(original_ids) != len(original):
        raise RuntimeError("Original dataset contains duplicate identities.")

    seen_ids = set()

    for name, part in partitions.items():
        ids = row_identity_set(part)

        if len(ids) != len(part):
            raise RuntimeError(
                f"Duplicate source-row identities inside {name}."
            )

        overlap = seen_ids.intersection(ids)

        if overlap:
            raise RuntimeError(
                f"Data leakage detected: {len(overlap)} rows overlap "
                f"with an earlier partition."
            )

        seen_ids.update(ids)

        if list(part.columns) != list(original.columns):
            raise RuntimeError(
                f"Column preservation check failed for {name}."
            )

        print(
            f"PASS: {name} has {len(part):,} unique rows "
            "and preserved columns."
        )

    if seen_ids != original_ids:
        missing = original_ids - seen_ids
        extra = seen_ids - original_ids

        raise RuntimeError(
            "Partition union does not match original dataset. "
            f"Missing={len(missing)}, extra={len(extra)}"
        )

    print("PASS: No overlap between partitions.")
    print("PASS: Every original row appears exactly once.")
    print("PASS: Source metadata and all original columns preserved.")


# ---------------------------------------------------------
# Reporting
# ---------------------------------------------------------

def partition_report(part):
    family_counts = (
        part[FAMILY_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    source_counts = (
        part[SOURCE_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "rows": int(len(part)),
        "benign_rows": int((part[TARGET_COLUMN] == 0).sum()),
        "malicious_rows": int((part[TARGET_COLUMN] == 1).sum()),
        "attack_families": {
            str(key): int(value)
            for key, value in family_counts.items()
        },
        "source_files": {
            str(key): int(value)
            for key, value in source_counts.items()
        },
    }


def print_report(partitions):
    print("\n========== PARTITION CLASS DISTRIBUTIONS ==========")

    for name, part in partitions.items():
        report = partition_report(part)

        print(f"\n{name.upper()}")
        print(f"Rows: {report['rows']:,}")
        print(f"Benign: {report['benign_rows']:,}")
        print(f"Malicious: {report['malicious_rows']:,}")

        print("Attack families:")
        for family, count in report["attack_families"].items():
            print(f"  {family}: {count:,}")

        print("Source captures:")
        for source, count in report["source_files"].items():
            print(f"  {source}: {count:,}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("========== NIDS V3 DATA SPLITTER ==========")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Ratios: {SPLIT_RATIOS}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nLoaded {len(df):,} rows.")
    print(f"Columns: {len(df.columns)}")

    df = validate_input(df)

    partitions = create_partitions(df)

    verify_partitions(df, partitions)

    print_report(partitions)

    # Write to a new v3 directory. Existing v1/v2 data is untouched.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    partition_manifest = {}

    for name, part in partitions.items():
        output_file = OUTPUT_DIR / f"{name}.parquet"

        part.to_parquet(
            output_file,
            index=False,
        )

        partition_manifest[name] = {
            "file": str(output_file),
            **partition_report(part),
        }

        print(f"\nSaved {output_file}")

    manifest = {
        "split_version": "3.0",
        "random_state": RANDOM_STATE,
        "strategy": (
            "Stratified row-level split within each source_file and "
            "attack_family group."
        ),
        "ratios": SPLIT_RATIOS,
        "input_file": str(INPUT_FILE),
        "input_rows": int(len(df)),
        "source_metadata_columns": [
            SOURCE_COLUMN,
            ROW_COLUMN,
        ],
        "stratification_columns": [
            SOURCE_COLUMN,
            FAMILY_COLUMN,
        ],
        "target_column": TARGET_COLUMN,
        "feature_count": int(
            len([
                c for c in df.columns
                if c not in {
                    SOURCE_COLUMN,
                    ROW_COLUMN,
                    FAMILY_COLUMN,
                    TARGET_COLUMN,
                }
            ])
        ),
        "integrity_checks": {
            "duplicate_source_rows": False,
            "partition_overlap": False,
            "missing_rows": False,
            "all_rows_assigned_once": True,
            "original_columns_preserved": True,
        },
        "limitations": [
            "This is a row-level split, not a session-level split.",
            "Source IPs, source ports and timestamps are unavailable "
            "for verifying session independence.",
            "Rows from the same capture and attack family can occur "
            "in different partitions.",
            "The test partition is held out from training, calibration "
            "and validation within this split, but it is not an "
            "independent capture-file evaluation.",
        ],
        "partitions": partition_manifest,
    }

    with MANIFEST_FILE.open("w") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nManifest saved: {MANIFEST_FILE}")

    print("\n========== V3 SPLIT COMPLETE ==========")
    print("All integrity checks passed.")
    print("Existing v1 and v2 files were not modified.")


if __name__ == "__main__":
    main()
