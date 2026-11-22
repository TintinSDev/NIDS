
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


CLEAN_PATH = Path("data/interim/cicids2017_clean.parquet")
OUT_DIR = Path("data/processed")

RANDOM_STATE = 42
CALIBRATION_SIZE = 0.20

DDOS_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
PORTSCAN_FILE = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
BRUTE_FORCE_FILE = "Tuesday-WorkingHours.pcap_ISCX.csv"

SPLIT_FILES = {
    "train": "train.parquet",
    "calibration": "calibration.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}


def distribution(df):
    return {
        str(key): int(value)
        for key, value in df["attack_family"].value_counts().items()
    }


def validate_source(df):
    required = {
        "source_file",
        "source_row",
        "attack_family",
        "is_malicious",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Required split metadata contains missing values.")

    if df.duplicated(["source_file", "source_row"]).any():
        raise ValueError("Duplicate source-file/source-row identifiers found.")


def validate_splits(splits):
    # Every partition must contain rows.
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"{name} split is empty.")

    # Source records must never appear in multiple partitions.
    seen = set()

    for name, frame in splits.items():
        keys = set(
            zip(frame["source_file"], frame["source_row"])
        )

        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(
                f"Source-row leakage detected in {name}: "
                f"{len(overlap)} overlapping records."
            )

        seen.update(keys)

    # Validate the intended capture separation.
    expected_sources = {
        "train": {DDOS_FILE},
        "calibration": {DDOS_FILE},
        "val": {PORTSCAN_FILE},
        "test": {BRUTE_FORCE_FILE},
    }

    for name, expected in expected_sources.items():
        actual = set(splits[name]["source_file"].unique())

        if actual != expected:
            raise ValueError(
                f"{name} source mismatch. "
                f"Expected {expected}, got {actual}"
            )

    # Training must contain both binary classes.
    if set(splits["train"]["is_malicious"].astype(int).unique()) != {0, 1}:
        raise ValueError("Training split must contain benign and malicious rows.")

    # Calibration and validation need both binary classes.
    for name in ("calibration", "val"):
        if set(splits[name]["is_malicious"].astype(int).unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both binary classes.")

    # The test partition must contain benign and malicious traffic.
    if set(splits["test"]["is_malicious"].astype(int).unique()) != {0, 1}:
        raise ValueError("Test split must contain both binary classes.")


def make_splits():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLEAN_PATH}. Run preprocessing first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CLEAN_PATH)
    validate_source(df)

    expected_files = {DDOS_FILE, PORTSCAN_FILE, BRUTE_FORCE_FILE}
    actual_files = set(df["source_file"].unique())

    if actual_files != expected_files:
        raise ValueError(
            "Unexpected source-file inventory.\n"
            f"Expected: {sorted(expected_files)}\n"
            f"Actual:   {sorted(actual_files)}"
        )

    ddos = df[df["source_file"] == DDOS_FILE].copy()
    portscan = df[df["source_file"] == PORTSCAN_FILE].copy()
    brute_force = df[df["source_file"] == BRUTE_FORCE_FILE].copy()

    # Stable ordering before deterministic stratified partitioning.
    ddos = ddos.sort_values("source_row").reset_index(drop=True)

    train_df, calibration_df = train_test_split(
        ddos,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=ddos["attack_family"],
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "calibration": calibration_df.reset_index(drop=True),
        "val": portscan.reset_index(drop=True),
        "test": brute_force.reset_index(drop=True),
    }

    validate_splits(splits)

    # Write each output to a temporary file before replacing its destination.
    for name, filename in SPLIT_FILES.items():
        destination = OUT_DIR / filename
        temporary = OUT_DIR / f"{filename}.tmp"

        splits[name].to_parquet(temporary, index=False)
        temporary.replace(destination)

    manifest = {
        "split_version": "2.0",
        "random_state": RANDOM_STATE,
        "calibration_fraction": CALIBRATION_SIZE,
        "strategy": (
            "Capture-file-separated evaluation. DDoS capture is "
            "stratified into train and calibration partitions; "
            "PortScan is validation; Tuesday Brute Force is final test."
        ),
        "limitations": [
            "Train and calibration originate from the same capture file.",
            "Source IPs, source ports, and timestamps are unavailable.",
            "This evaluates generalization to attack families absent from training.",
            "It does not establish performance on independent capture sessions.",
        ],
        "partitions": {},
    }

    for name, frame in splits.items():
        manifest["partitions"][name] = {
            "file": SPLIT_FILES[name],
            "rows": int(len(frame)),
            "source_files": sorted(frame["source_file"].unique().tolist()),
            "attack_families": distribution(frame),
            "benign_rows": int((frame["is_malicious"] == 0).sum()),
            "malicious_rows": int((frame["is_malicious"] == 1).sum()),
        }

    manifest_path = OUT_DIR / "split_manifest_v2.json"
    temporary_manifest = OUT_DIR / "split_manifest_v2.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    temporary_manifest.replace(manifest_path)

    print("\n========== SPLIT GENERATION COMPLETE ==========")

    for name, frame in splits.items():
        print(f"\n{name.upper()}: {len(frame):,} rows")
        print("Sources:", frame["source_file"].unique().tolist())
        print("Attack families:", distribution(frame))
        print(
            "Benign:",
            int((frame["is_malicious"] == 0).sum()),
            "| Malicious:",
            int((frame["is_malicious"] == 1).sum()),
        )

    print(f"\nManifest: {manifest_path}")
    print("All split validation checks passed.")


if __name__ == "__main__":
    make_splits()
