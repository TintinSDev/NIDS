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
PROC_DIR = Path("data/processed")
ART_DIR = Path("artifacts/models/v2")

RANDOM_STATE = 42
TARGET_MAX_FPR = 0.01


def load_dataset(name, features):
    path = PROC_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_parquet(path)

    required = features + ["is_malicious", "attack_family"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if df[features].isna().any().any():
        raise ValueError(f"{name} contains missing feature values")

    if not np.isfinite(df[features].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} contains infinite feature values")

    return df


def binary_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
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
        "pr_auc": (
            float(average_precision_score(y_true, probabilities))
            if np.sum(y_true) > 0
            else None
        ),
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
    Select the threshold with maximum validation recall under the FPR limit.
    Uses a sorted cumulative pass instead of repeatedly computing confusion
    matrices for every unique probability.
    """
    y = np.asarray(y_true, dtype=np.int8)
    probs = np.asarray(probabilities, dtype=np.float64)

    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    sorted_y = y[order]

    cumulative_tp = np.cumsum(sorted_y == 1)
    cumulative_fp = np.cumsum(sorted_y == 0)

    # Evaluate only at the final row of each tied probability group.
    group_ends = np.r_[
        np.flatnonzero(sorted_probs[:-1] != sorted_probs[1:]),
        len(sorted_probs) - 1,
    ]

    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    fpr = fp / negatives if negatives else np.zeros_like(fp, dtype=float)
    recall = tp / positives if positives else np.zeros_like(tp, dtype=float)
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0,
    )

    eligible = np.flatnonzero(fpr <= max_fpr)

    if len(eligible) == 0:
        raise RuntimeError(
            f"No threshold satisfies maximum FPR of {max_fpr:.2%}"
        )

    # Highest recall, then highest precision, then highest threshold.
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

    for family in sorted(df["attack_family"].astype(str).unique()):
        mask = df["attack_family"].astype(str).to_numpy() == family

        y_true = df["is_malicious"].to_numpy(dtype=int)[mask]
        family_probs = probabilities[mask]
        predictions = (family_probs >= threshold).astype(int)

        results[family] = {
            "rows": int(mask.sum()),
            "malicious_rows": int(y_true.sum()),
            "detected_malicious": int(
                np.sum((y_true == 1) & (predictions == 1))
            ),
            "missed_malicious": int(
                np.sum((y_true == 1) & (predictions == 0))
            ),
            "false_positives": int(
                np.sum((y_true == 0) & (predictions == 1))
            ),
            "malicious_recall": (
                float(recall_score(y_true, predictions, zero_division=0))
                if np.sum(y_true) > 0
                else None
            ),
        }

    return results


def multiclass_metrics(df, features, model, encoder):
    known_mask = df["attack_family"].isin(encoder.classes_)
    known_df = df.loc[known_mask]

    if known_df.empty:
        return {
            "evaluated_rows": 0,
            "note": "No known training classes in this dataset.",
        }

    X = known_df[features]
    y_true = encoder.transform(known_df["attack_family"])
    predictions = model.predict(X).astype(int)

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
    }


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)

    schema = json.loads(SCHEMA_PATH.read_text())
    features = schema["features"]

    print("Loading v2 datasets...")
    train_df = load_dataset("train", features)
    calibration_df = load_dataset("calibration", features)
    val_df = load_dataset("val", features)
    test_df = load_dataset("test", features)

    X_train = train_df[features]
    y_train = train_df["is_malicious"].astype(int)

    X_cal = calibration_df[features]
    y_cal = calibration_df["is_malicious"].astype(int)

    X_val = val_df[features]
    y_val = val_df["is_malicious"].astype(int)

    X_test = test_df[features]
    y_test = test_df["is_malicious"].astype(int)

    for name, labels in [
        ("training", y_train),
        ("calibration", y_cal),
        ("validation", y_val),
        ("test", y_test),
    ]:
        if set(labels.unique()) != {0, 1}:
            raise ValueError(
                f"{name} must contain both binary classes; "
                f"found {sorted(labels.unique())}"
            )

    print(
        f"Rows: train={len(train_df):,}, "
        f"calibration={len(calibration_df):,}, "
        f"validation={len(val_df):,}, test={len(test_df):,}"
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    # 1. Train base binary models on training data only.
    print("\nTraining Random Forest...")
    rf_base = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_base.fit(X_train, y_train)

    print("Training XGBoost binary classifier...")
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

    # 2. Calibrate using the dedicated calibration partition only.
    print("Calibrating binary models...")
    rf_cal = PlattCalibrator(rf_base).fit(X_cal, y_cal)
    xgb_cal = PlattCalibrator(xgb_base).fit(X_cal, y_cal)

    # 3. Train anomaly detector on benign training traffic only.
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    iso.fit(X_train.loc[y_train == 0])

    # 4. Train multiclass model on the attack families present in train.
    print("Training multiclass classifier...")
    encoder = LabelEncoder()
    y_train_mc = encoder.fit_transform(train_df["attack_family"].astype(str))

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

    # 5. Choose threshold using validation data only.
    print("Selecting threshold on validation data...")
    xgb_val_probs = xgb_cal.predict_proba(X_val)[:, 1]
    selected_threshold = select_threshold(y_val, xgb_val_probs)

    print(f"Selected threshold: {selected_threshold:.8f}")

    validation_metrics = binary_metrics(
        y_val, xgb_val_probs, selected_threshold
    )

    rf_val_probs = rf_cal.predict_proba(X_val)[:, 1]
    iso_val_scores = -iso.decision_function(X_val)

    # 6. Evaluate the held-out test set once after threshold selection.
    print("Evaluating held-out test set...")
    xgb_test_probs = xgb_cal.predict_proba(X_test)[:, 1]
    rf_test_probs = rf_cal.predict_proba(X_test)[:, 1]
    iso_test_scores = -iso.decision_function(X_test)

    test_metrics = binary_metrics(
        y_test, xgb_test_probs, selected_threshold
    )

    # 7. Save v2 artifacts. Nothing in v1 is modified.
    print("Saving v2 artifacts...")
    joblib.dump(rf_cal, ART_DIR / "rf_calibrated.joblib")
    joblib.dump(xgb_cal, ART_DIR / "xgb_binary_calibrated.joblib")
    joblib.dump(iso, ART_DIR / "isolation_forest.joblib")
    joblib.dump(xgb_mc, ART_DIR / "xgb_multiclass.joblib")
    joblib.dump(encoder, ART_DIR / "label_encoder.joblib")

    calibration_metadata = {
        "artifact_version": "v2",
        "selected_high_conf_threshold": selected_threshold,
        "threshold_selection": {
            "dataset": "val",
            "target_max_fpr": TARGET_MAX_FPR,
            "actual_validation_fpr": validation_metrics["fpr"],
            "selection_objective": "maximum recall under FPR constraint",
        },
        "calibration_dataset": "calibration",
        "training_dataset": "train",
        "test_dataset": "test",
    }

    (ART_DIR / "calibration_v2.json").write_text(
        json.dumps(calibration_metadata, indent=2)
    )

    metrics = {
        "artifact_version": "v2",
        "feature_count": len(features),
        "training": {
            "rows": int(len(train_df)),
            "class_counts": {
                str(k): int(v)
                for k, v in y_train.value_counts().to_dict().items()
            },
            "multiclass_classes": encoder.classes_.tolist(),
        },
        "calibration": {
            "rows": int(len(calibration_df)),
        },
        "validation": {
            "rows": int(len(val_df)),
            "xgb_binary": validation_metrics,
            "rf_pr_auc": float(
                average_precision_score(y_val, rf_val_probs)
            ),
            "isolation_forest_pr_auc": float(
                average_precision_score(y_val, iso_val_scores)
            ),
            "multiclass": multiclass_metrics(
                val_df, features, xgb_mc, encoder
            ),
        },
        "test": {
            "rows": int(len(test_df)),
            "xgb_binary": test_metrics,
            "rf_binary": binary_metrics(
                y_test, rf_test_probs, selected_threshold
            ),
            "isolation_forest_pr_auc": float(
                average_precision_score(y_test, iso_test_scores)
            ),
            "per_attack_family": family_metrics(
                test_df, xgb_test_probs, selected_threshold
            ),
            "multiclass": multiclass_metrics(
                test_df, features, xgb_mc, encoder
            ),
        },
        "limitations": [
            "Training contains only the attack families present in the DDoS capture.",
            "PortScan and Brute Force are unseen attack families for the multiclass model.",
            "The final test set is from a separate capture, but source IPs, ports and timestamps are unavailable for session-level split verification.",
            "Threshold selection is based on the PortScan validation capture and may not generalize to other traffic distributions.",
        ],
    }

    (ART_DIR / "metrics_v2.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\nV2 training complete.")
    print(json.dumps({
        "artifacts_directory": str(ART_DIR),
        "selected_threshold": selected_threshold,
        "validation_xgb": validation_metrics,
        "test_xgb": test_metrics,
        "test_per_attack_family": metrics["test"]["per_attack_family"],
    }, indent=2))


if __name__ == "__main__":
    main()
