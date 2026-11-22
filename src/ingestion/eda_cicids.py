import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
SCHEMA_PATH = Path("src/features/features_v1.json")

CIC_TO_V1_MAP = {
    "flow duration": "flow_duration",
    "total fwd packets": "total_fwd_packets",
    "total backward packets": "total_bwd_packets",
    "total length of fwd packets": "total_length_fwd_packets",
    "total length of bwd packets": "total_length_bwd_packets",
    "fwd packet length max": "fwd_packet_length_max",
    "fwd packet length min": "fwd_packet_length_min",
    "fwd packet length mean": "fwd_packet_length_mean",
    "bwd packet length max": "bwd_packet_length_max",
    "bwd packet length min": "bwd_packet_length_min",
    "bwd packet length mean": "bwd_packet_length_mean",
    "flow bytes/s": "flow_bytes_per_sec",
    "flow packets/s": "flow_packets_per_sec",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat max": "flow_iat_max",
    "flow iat min": "flow_iat_min",
    "fwd iat total": "fwd_iat_total",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "fwd iat max": "fwd_iat_max",
    "fwd iat min": "fwd_iat_min",
    "bwd iat total": "bwd_iat_total",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "bwd iat max": "bwd_iat_max",
    "bwd iat min": "bwd_iat_min",
    "fwd psh flags": "fwd_psh_flags",
    "bwd psh flags": "bwd_psh_flags",
    "fwd urg flags": "fwd_urg_flags",
    "bwd urg flags": "bwd_urg_flags",
    "fwd header length": "fwd_header_length",
    "bwd header length": "bwd_header_length",
    "fwd packets/s": "fwd_packets_per_sec",
    "bwd packets/s": "bwd_packets_per_sec",
    "min packet length": "packet_length_min",
    "max packet length": "packet_length_max",
    "packet length mean": "packet_length_mean",
    "packet length std": "packet_length_std",
    "packet length variance": "packet_length_variance",
    "fin flag count": "fin_flag_count",
    "syn flag count": "syn_flag_count",
    "rst flag count": "rst_flag_count",
    "psh flag count": "psh_flag_count",
    "ack flag count": "ack_flag_count",
    "urg flag count": "urg_flag_count",
    "cwe flag count": "cwe_flag_count",
    "ece flag count": "ece_flag_count",
    "down/up ratio": "down_up_ratio",
    "average packet size": "avg_packet_size",
    "avg fwd segment size": "avg_fwd_segment_size",
    "avg bwd segment size": "avg_bwd_segment_size",
}

LABEL_NORMALIZATION = {
    "BENIGN": "Benign",
    "DDoS": "DoS/DDoS",
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "Heartbleed": "DoS/DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}


def load_and_audit():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR.resolve()}."
        )

    schema_features = json.loads(
        SCHEMA_PATH.read_text()
    )["features"]

    frames = []

    for path in csv_files:
        print(f"Reading {path.name} ...")

        df = pd.read_csv(
            path,
            encoding="latin1",
            low_memory=False,
        )

        # Preserve the original CSV line number.
        # CSV line 1 is the header; first data row is line 2.
        df["source_row"] = df.index + 2

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        if "label" not in df.columns:
            print(f"Skipping {path.name}: no label column")
            continue

        rename_dict = {
            source: target
            for source, target in CIC_TO_V1_MAP.items()
            if source in df.columns
        }

        df = df.rename(columns=rename_dict)

        missing = set(schema_features) - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing schema features: "
                f"{sorted(missing)}"
            )

        sub = df[
            schema_features + ["label", "source_row"]
        ].copy()

        sub["source_file"] = path.name
        frames.append(sub)

    if not frames:
        raise ValueError("No usable CICIDS CSV files were found.")

    full_df = pd.concat(frames, ignore_index=True)
    initial_rows = len(full_df)

    # Normalize labels without changing the original label text.
    full_df["raw_label"] = (
        full_df["label"].astype("string").str.strip()
    )

    # Convert all model features to numeric values.
    numeric = full_df[schema_features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Track invalidity without silently replacing bad values with zero.
    reasons = pd.Series(
        "",
        index=full_df.index,
        dtype="object",
    )

    for feature in schema_features:
        values = numeric[feature]

        missing_or_non_numeric = values.isna()
        infinite = np.isinf(values)
        negative = values < 0

        reasons.loc[missing_or_non_numeric] += (
            f"missing_or_non_numeric:{feature};"
        )

        reasons.loc[infinite] += (
            f"infinite_value:{feature};"
        )

        reasons.loc[negative] += (
            f"negative_value:{feature};"
        )

    missing_label = (
        full_df["raw_label"].isna()
        | full_df["raw_label"].eq("")
    )

    reasons.loc[missing_label] += "missing_label;"

    # Use numeric values for all subsequent processing.
    full_df[schema_features] = numeric

    invalid_mask = reasons.ne("")
    invalid_rows = full_df.loc[invalid_mask].copy()

    invalid_rows["rejection_reason"] = (
        reasons.loc[invalid_mask]
    )

    # Save every rejected row with its original source location.
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    invalid_path = INTERIM_DIR / "cicids2017_invalid_rows.csv"

    invalid_rows.to_csv(invalid_path, index=False)

    valid_df = full_df.loc[~invalid_mask].copy()

    # Identify duplicates before removing them.
    # Keep the first occurrence, as in the previous pipeline.
    duplicate_mask = valid_df.duplicated(
        subset=schema_features + ["label"],
        keep="first",
    )

    duplicate_rows = valid_df.loc[duplicate_mask].copy()

    duplicate_path = (
        INTERIM_DIR / "cicids2017_duplicate_rows.csv"
    )

    duplicate_rows.to_csv(duplicate_path, index=False)

    clean_df = valid_df.loc[~duplicate_mask].copy()

    clean_df["attack_family"] = clean_df["raw_label"].map(
        lambda label: LABEL_NORMALIZATION.get(
            label,
            "Web Attack" if "Web Attack" in label else label,
        )
    )

    clean_df["is_malicious"] = (
        clean_df["attack_family"] != "Benign"
    ).astype(int)

    # Write the cleaned dataset.
    out_path = INTERIM_DIR / "cicids2017_clean.parquet"
    clean_df.to_parquet(out_path, index=False)

    # Summarize rejection reasons.
    reason_counts = {}

    for reason in reasons.loc[invalid_mask]:
        for item in reason.split(";"):
            if item:
                category = item.split(":", 1)[0]
                reason_counts[category] = (
                    reason_counts.get(category, 0) + 1
                )

    attack_counts = (
        clean_df["attack_family"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "input_rows": int(initial_rows),
        "invalid_rows": int(len(invalid_rows)),
        "duplicate_rows": int(len(duplicate_rows)),
        "retained_rows": int(len(clean_df)),
        "rejection_reason_counts": reason_counts,
        "attack_family_counts": {
            str(key): int(value)
            for key, value in attack_counts.items()
        },
        "source_files": [
            str(path.name) for path in csv_files
        ],
        "features": len(schema_features),
        "outputs": {
            "clean_dataset": str(out_path),
            "invalid_rows": str(invalid_path),
            "duplicate_rows": str(duplicate_path),
        },
    }

    summary_path = INTERIM_DIR / "cicids2017_audit_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("\n========== PREPROCESSING AUDIT ==========")
    print(f"Input rows:       {initial_rows:,}")
    print(f"Invalid rows:     {len(invalid_rows):,}")
    print(f"Duplicate rows:   {len(duplicate_rows):,}")
    print(f"Retained rows:    {len(clean_df):,}")
    print(f"Features:         {len(schema_features)}")

    print("\nRejection reasons:")
    print(
        pd.Series(reason_counts)
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nAttack-family distribution:")
    counts = clean_df["attack_family"].value_counts()
    percentages = (
        counts / len(clean_df) * 100
    ).round(3)

    print(
        pd.DataFrame({
            "count": counts,
            "percent": percentages,
        }).to_string()
    )

    print("\nGenerated files:")
    print(f"Clean dataset:  {out_path}")
    print(f"Invalid audit:  {invalid_path}")
    print(f"Duplicate audit:{duplicate_path}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    load_and_audit()
