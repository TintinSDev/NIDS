import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.models.calibration import PlattCalibrator


SCHEMA_PATH = Path("src/features/features_v1.json")
DATA_DIR = Path("data/processed/v3")
ART_DIR = Path("artifacts/models/v3")

RANDOM_STATE = 42
TARGET_MAX_FPR = 0.01


def load_dataset(name, features):
    path = DATA_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing v3 dataset: {path}")

    df = pd.read_parquet(path)

    required = features + [
        "is_malicious",
        "attack_family",
        "source_file",
        "source_row",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError(f"{name} contains missing required values")

    if df[features].isna().any().any():
        raise ValueError(f"{name} contains missing feature values")

    values = df[features].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains infinite feature values")

    labels = set(df["is_malicious"].astype(int).unique())

    if labels != {0, 1}:
        raise ValueError(
            f"{name} must contain both binary classes; found {labels}"
        )

    return df


def binary_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "fpr": fpr,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def select_threshold(y_true, probabilities, max_fpr=TARGET_MAX_FPR):
    """
    Maximize validation recall while keeping validation FPR <= max_fpr.
    Tied probabilities are evaluated as one group.
    """
    y = np.asarray(y_true, dtype=np.int8)
    probs = np.asarray(probabilities, dtype=np.float64)

    if len(y) != len(probs) or len(y) == 0:
        raise ValueError("Threshold selection received invalid arrays")

    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    sorted_y = y[order]

    cumulative_tp = np.cumsum(sorted_y == 1)
    cumulative_fp = np.cumsum(sorted_y == 0)

    group_ends = np.r_[
        np.flatnonzero(sorted_probs[:-1] != sorted_probs[1:]),
        len(sorted_probs) - 1,
    ]

    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    fpr = fp / negatives
    recall = tp / positives

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0,
    )

    eligible = np.flatnonzero(fpr <= max_fpr)

    # Include a no-alert threshold as a valid fallback.
    if not len(eligible):
        return float(np.nextafter(np.max(probs), np.inf))

    best = max(
        eligible,
        key=lambda i: (
            recall[i],
            precision[i],
            sorted_probs[group_ends[i]],
        ),
    )

    return float(sorted_probs[group_ends[best]])


def family_metrics(df, probabilities, threshold):
    results = {}

    families = df["attack_family"].astype(str).to_numpy()
    labels = df["is_malicious"].to_numpy(dtype=int)
    predictions = (probabilities >= threshold).astype(int)

    for family in sorted(np.unique(families)):
        mask = families == family
        y_true = labels[mask]
        family_predictions = predictions[mask]

        results[family] = {
            "rows": int(mask.sum()),
            "malicious_rows": int(y_true.sum()),
            "detected_malicious": int(
                np.sum((y_true == 1) & (family_predictions == 1))
            ),
            "missed_malicious": int(
                np.sum((y_true == 1) & (family_predictions == 0))
            ),
            "false_positives": int(
                np.sum((y_true == 0) & (family_predictions == 1))
            ),
            "malicious_recall": (
                float(
                    recall_score(
                        y_true,
                        family_predictions,
                        zero_division=0,
                    )
                )
                if np.sum(y_true) > 0
                else None
            ),
        }

    return results


def multiclass_metrics(df, features, model, encoder):
    y_true_names = df["attack_family"].astype(str)

    known_mask = y_true_names.isin(encoder.classes_)
    known_df = df.loc[known_mask]

    if known_df.empty:
        return {
            "evaluated_rows": 0,
            "note": "No known training classes in this dataset.",
        }

    y_true = encoder.transform(
        known_df["attack_family"].astype(str)
    )

    predictions = model.predict(known_df[features]).astype(int)

    return {
        "evaluated_rows": int(len(known_df)),
        "excluded_unseen_family_rows": int(len(df) - len(known_df)),
        "classes": encoder.classes_.tolist(),
        "report": classification_report(
            y_true,
            predictions,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=list(range(len(encoder.classes_))),
        ).tolist(),
    }


def class_distribution(df):
    return {
        str(key): int(value)
        for key, value in (
            df["attack_family"].astype(str).value_counts().sort_index().items()
        )
    }


def source_distribution(df):
    return {
        str(key): int(value)
        for key, value in (
            df["source_file"].astype(str).value_counts().sort_index().items()
        )
    }


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)

    schema = json.loads(SCHEMA_PATH.read_text())
    features = schema["features"]

    print("=" * 60)
    print("NIDS V3 MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Artifact directory: {ART_DIR}")
    print(f"Features: {len(features)}")
    print(f"Maximum validation FPR: {TARGET_MAX_FPR:.2%}")

    print("\nLoading v3 partitions...")

    train_df = load_dataset("train", features)
    calibration_df = load_dataset("calibration", features)
    val_df = load_dataset("val", features)
    test_df = load_dataset("test", features)

    partitions = {
        "train": train_df,
        "calibration": calibration_df,
        "validation": val_df,
        "test": test_df,
    }

    # Verify no source rows overlap before training.
    identity_sets = {}

    for name, df in partitions.items():
        identities = set(
            zip(
                df["source_file"].astype(str),
                df["source_row"].astype(str),
            )
        )

        if len(identities) != len(df):
            raise ValueError(f"{name} contains duplicate source row identities")

        identity_sets[name] = identities

    names = list(identity_sets)

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = identity_sets[left] & identity_sets[right]

            if overlap:
                raise ValueError(
                    f"Data leakage: {left} and {right} share "
                    f"{len(overlap)} source rows"
                )

    print("PASS: no source-row overlap across v3 partitions.")

    for name, df in partitions.items():
        print(
            f"{name:12s}: {len(df):>9,} rows | "
            f"families={class_distribution(df)}"
        )

    X_train = train_df[features]
    y_train = train_df["is_malicious"].astype(int)

    X_cal = calibration_df[features]
    y_cal = calibration_df["is_malicious"].astype(int)

    X_val = val_df[features]
    y_val = val_df["is_malicious"].astype(int)

    X_test = test_df[features]
    y_test = test_df["is_malicious"].astype(int)

    # ---------------------------------------------------------
    # 1. Train binary Random Forest.
    # ---------------------------------------------------------
    print("\n[1/7] Training Random Forest...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    rf_base = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    rf_base.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 2. Train binary XGBoost.
    # ---------------------------------------------------------
    print("\n[2/7] Training binary XGBoost...")

    xgb_base = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgb_base.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 3. Calibrate using calibration partition only.
    # ---------------------------------------------------------
    print("\n[3/7] Calibrating binary models...")

    rf_cal = PlattCalibrator(rf_base).fit(X_cal, y_cal)
    xgb_cal = PlattCalibrator(xgb_base).fit(X_cal, y_cal)

    # ---------------------------------------------------------
    # 4. Train Isolation Forest on benign training traffic.
    # ---------------------------------------------------------
    print("\n[4/7] Training Isolation Forest...")

    benign_train = X_train.loc[y_train == 0]

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    iso.fit(benign_train)

    # ---------------------------------------------------------
    # 5. Train multiclass XGBoost on all training families.
    # ---------------------------------------------------------
    print("\n[5/7] Training multiclass XGBoost...")

    encoder = LabelEncoder()

    y_train_mc = encoder.fit_transform(
        train_df["attack_family"].astype(str)
    )

    xgb_mc = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgb_mc.fit(X_train, y_train_mc)

    print("Multiclass classes:", encoder.classes_.tolist())

    expected_families = {"Benign", "DoS/DDoS", "Brute Force", "PortScan"}

    if set(encoder.classes_) != expected_families:
        raise ValueError(
            "Unexpected multiclass training classes. "
            f"Expected {sorted(expected_families)}, "
            f"found {encoder.classes_.tolist()}"
        )

    # ---------------------------------------------------------
    # 6. Select separate thresholds using validation only.
    # ---------------------------------------------------------
    print("\n[6/7] Selecting thresholds on validation data...")

    xgb_val_probs = xgb_cal.predict_proba(X_val)[:, 1]
    rf_val_probs = rf_cal.predict_proba(X_val)[:, 1]

    xgb_threshold = select_threshold(y_val, xgb_val_probs)
    rf_threshold = select_threshold(y_val, rf_val_probs)

    xgb_val_metrics = binary_metrics(
        y_val,
        xgb_val_probs,
        xgb_threshold,
    )

    rf_val_metrics = binary_metrics(
        y_val,
        rf_val_probs,
        rf_threshold,
    )

    iso_val_scores = -iso.decision_function(X_val)

    print(f"XGBoost validation threshold: {xgb_threshold:.10f}")
    print(f"Random Forest validation threshold: {rf_threshold:.10f}")
    print(f"XGBoost validation FPR: {xgb_val_metrics['fpr']:.4%}")
    print(f"Random Forest validation FPR: {rf_val_metrics['fpr']:.4%}")

    # ---------------------------------------------------------
    # 7. Evaluate test once with frozen thresholds.
    # ---------------------------------------------------------
    print("\n[7/7] Evaluating held-out test partition...")

    xgb_test_probs = xgb_cal.predict_proba(X_test)[:, 1]
    rf_test_probs = rf_cal.predict_proba(X_test)[:, 1]

    iso_test_scores = -iso.decision_function(X_test)

    xgb_test_metrics = binary_metrics(
        y_test,
        xgb_test_probs,
        xgb_threshold,
    )

    rf_test_metrics = binary_metrics(
        y_test,
        rf_test_probs,
        rf_threshold,
    )

    xgb_family_results = family_metrics(
        test_df,
        xgb_test_probs,
        xgb_threshold,
    )

    rf_family_results = family_metrics(
        test_df,
        rf_test_probs,
        rf_threshold,
    )

    xgb_val_multiclass = multiclass_metrics(
        val_df,
        features,
        xgb_mc,
        encoder,
    )

    xgb_test_multiclass = multiclass_metrics(
        test_df,
        features,
        xgb_mc,
        encoder,
    )

    # ---------------------------------------------------------
    # Save v3 artifacts only.
    # ---------------------------------------------------------
    print("\nSaving v3 artifacts...")

    joblib.dump(rf_cal, ART_DIR / "rf_calibrated.joblib")
    joblib.dump(xgb_cal, ART_DIR / "xgb_binary_calibrated.joblib")
    joblib.dump(iso, ART_DIR / "isolation_forest.joblib")
    joblib.dump(xgb_mc, ART_DIR / "xgb_multiclass.joblib")
    joblib.dump(encoder, ART_DIR / "label_encoder.joblib")

    calibration_metadata = {
        "artifact_version": "v3",
        "random_state": RANDOM_STATE,
        "feature_schema": str(SCHEMA_PATH),
        "feature_count": len(features),
        "threshold_selection": {
            "dataset": "v3/val.parquet",
            "target_max_fpr": TARGET_MAX_FPR,
            "objective": "maximum recall under validation FPR constraint",
            "xgb_binary": {
                "threshold": xgb_threshold,
                "validation_fpr": xgb_val_metrics["fpr"],
            },
            "rf_binary": {
                "threshold": rf_threshold,
                "validation_fpr": rf_val_metrics["fpr"],
            },
        },
        "calibration_dataset": "v3/calibration.parquet",
        "training_dataset": "v3/train.parquet",
        "test_dataset": "v3/test.parquet",
        "test_used_for_threshold_selection": False,
    }

    (ART_DIR / "calibration_v3.json").write_text(
        json.dumps(calibration_metadata, indent=2)
    )

    metrics = {
        "artifact_version": "v3",
        "random_state": RANDOM_STATE,
        "feature_count": len(features),
        "split": {
            name: {
                "rows": int(len(df)),
                "attack_families": class_distribution(df),
                "source_captures": source_distribution(df),
            }
            for name, df in partitions.items()
        },
        "training": {
            "rows": int(len(train_df)),
            "binary_class_counts": {
                str(k): int(v)
                for k, v in y_train.value_counts().to_dict().items()
            },
            "multiclass_classes": encoder.classes_.tolist(),
            "benign_rows_used_for_isolation_forest": int(len(benign_train)),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "calibration": {
            "rows": int(len(calibration_df)),
            "method": "Platt scaling",
        },
        "validation": {
            "rows": int(len(val_df)),
            "xgb_binary": xgb_val_metrics,
            "rf_binary": rf_val_metrics,
            "isolation_forest_pr_auc": float(
                average_precision_score(y_val, iso_val_scores)
            ),
            "multiclass": xgb_val_multiclass,
        },
        "test": {
            "rows": int(len(test_df)),
            "xgb_binary": xgb_test_metrics,
            "rf_binary": rf_test_metrics,
            "isolation_forest_pr_auc": float(
                average_precision_score(y_test, iso_test_scores)
            ),
            "xgb_per_attack_family": xgb_family_results,
            "rf_per_attack_family": rf_family_results,
            "multiclass": xgb_test_multiclass,
        },
        "integrity": {
            "source_row_overlap": False,
            "all_four_attack_families_in_training": True,
            "test_used_for_threshold_selection": False,
        },
        "limitations": [
            "This is a row-level stratified split, not a session-level split.",
            "All three source captures contribute rows to every partition.",
            "Source IPs, source ports and timestamps are unavailable for session-level isolation.",
            "The test partition is held out by row, but is not an independent capture.",
            "Validation FPR is constrained to 1%; test FPR is measured and may differ.",
            "Isolation Forest scores are anomaly scores, not calibrated malicious probabilities.",
        ],
    }

    (ART_DIR / "metrics_v3.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\n" + "=" * 60)
    print("V3 TRAINING COMPLETE")
    print("=" * 60)

    print(json.dumps({
        "artifacts_directory": str(ART_DIR),
        "training_rows": len(train_df),
        "calibration_rows": len(calibration_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "multiclass_classes": encoder.classes_.tolist(),
        "xgb_threshold": xgb_threshold,
        "rf_threshold": rf_threshold,
        "xgb_test": xgb_test_metrics,
        "rf_test": rf_test_metrics,
        "xgb_per_attack_family": xgb_family_results,
    }, indent=2))


if __name__ == "__main__":
    main()
