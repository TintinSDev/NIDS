import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("data/interim/cicids2017_clean.parquet")
SCHEMA_PATH = Path("src/features/features_v1.json")
OUTPUT_DIR = Path("artifacts/evaluations/independent_capture_v1")

RANDOM_STATE = 42
MAX_VALIDATION_FPR = 0.01

CAPTURES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def split_training_pool(pool):
    """Create train/calibration/validation splits from non-held-out captures."""
    train, temporary = train_test_split(
        pool,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=pool["is_malicious"],
    )

    calibration, validation = train_test_split(
        temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary["is_malicious"],
    )

    return (
        train.reset_index(drop=True),
        calibration.reset_index(drop=True),
        validation.reset_index(drop=True),
    )


def make_models(scale_pos_weight):
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {"random_forest": rf, "xgboost": xgb}


def logit_transform(probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-7,
        1.0 - 1e-7,
    )
    return np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)


class PlattProbabilityCalibrator:
    """Platt scaling fitted only on the calibration partition."""

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    def fit(self, raw_probabilities, y):
        self.model.fit(logit_transform(raw_probabilities), y)
        return self

    def predict(self, raw_probabilities):
        return self.model.predict_proba(
            logit_transform(raw_probabilities)
        )[:, 1]


def choose_threshold(y_true, probabilities):
    """Maximize validation recall while keeping validation FPR <= 1%."""
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    eligible = np.flatnonzero(fpr <= MAX_VALIDATION_FPR)

    if len(eligible) == 0:
        raise RuntimeError("No threshold satisfies the validation FPR limit.")

    # Highest recall first, then lowest FPR, then higher threshold.
    best = max(
        eligible,
        key=lambda i: (tpr[i], -fpr[i], thresholds[i]),
    )

    return float(thresholds[best]), float(fpr[best]), float(tpr[best])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "pr_auc": float(average_precision_score(
            y_true, probabilities
        )),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def family_metrics(test, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {}

    for family, group in test.groupby("attack_family"):
        positions = group.index.to_numpy()
        family_predictions = predictions[positions]
        family_labels = group["is_malicious"].to_numpy()

        malicious_count = int(family_labels.sum())
        detected = int(((family_predictions == 1) & (family_labels == 1)).sum())
        false_positives = int(
            ((family_predictions == 1) & (family_labels == 0)).sum()
        )

        result[str(family)] = {
            "rows": int(len(group)),
            "malicious_rows": malicious_count,
            "detected_malicious": detected,
            "missed_malicious": malicious_count - detected,
            "false_positives": false_positives,
            "malicious_recall": (
                float(detected / malicious_count)
                if malicious_count else None
            ),
        }

    return result


def identity_set(df):
    return set(zip(df["source_file"], df["source_row"]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SCHEMA_PATH.open() as f:
        schema = json.load(f)

    features = schema["features"]
    df = pd.read_parquet(DATA_PATH)

    required = (
        features
        + ["is_malicious", "attack_family", "source_file", "source_row"]
    )
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Null values found in required columns.")

    if not set(df["is_malicious"].unique()).issubset({0, 1}):
        raise ValueError("is_malicious must contain only 0 and 1.")

    if not np.isfinite(df[features].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite feature values found.")

    available = set(df["source_file"].unique())
    if available != set(CAPTURES):
        raise ValueError(
            f"Capture mismatch. Expected {CAPTURES}; found {sorted(available)}"
        )

    print("=" * 68)
    print("SENTINEL — INDEPENDENT CAPTURE EVALUATION")
    print("=" * 68)
    print(f"Dataset rows: {len(df):,}")
    print(f"Features: {len(features)}")
    print(f"Validation FPR target: {MAX_VALIDATION_FPR:.2%}")
    print("Evaluation: leave-one-capture-out")
    print("Models: Random Forest and XGBoost")
    print()

    all_results = {
        "evaluation": "leave_one_capture_out",
        "dataset": str(DATA_PATH),
        "feature_count": len(features),
        "random_state": RANDOM_STATE,
        "max_validation_fpr": MAX_VALIDATION_FPR,
        "capture_folds": {},
    }

    X_all = df[features]

    for fold_number, held_out_capture in enumerate(CAPTURES, start=1):
        print("\n" + "=" * 68)
        print(f"FOLD {fold_number}/3 — HELD OUT:")
        print(held_out_capture)
        print("=" * 68)

        pool = df[df["source_file"] != held_out_capture].copy()
        test = df[df["source_file"] == held_out_capture].copy()

        train, calibration, validation = split_training_pool(pool)

        train_ids = identity_set(train)
        calibration_ids = identity_set(calibration)
        validation_ids = identity_set(validation)
        test_ids = identity_set(test)

        assert not (train_ids & calibration_ids)
        assert not (train_ids & validation_ids)
        assert not (calibration_ids & validation_ids)
        assert not (train_ids & test_ids)
        assert not (calibration_ids & test_ids)
        assert not (validation_ids & test_ids)

        print(
            f"Train: {len(train):,} | "
            f"Calibration: {len(calibration):,} | "
            f"Validation: {len(validation):,} | "
            f"Test: {len(test):,}"
        )
        print("Training attack families:",
              sorted(train["attack_family"].unique()))
        print("Held-out test families:",
              test["attack_family"].value_counts().to_dict())

        y_train = train["is_malicious"].astype(int).to_numpy()
        y_calibration = calibration["is_malicious"].astype(int).to_numpy()
        y_validation = validation["is_malicious"].astype(int).to_numpy()
        y_test = test["is_malicious"].astype(int).to_numpy()

        scale_pos_weight = (
            (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        )

        models = make_models(scale_pos_weight)
        fold_results = {
            "held_out_capture": held_out_capture,
            "rows": {
                "train": len(train),
                "calibration": len(calibration),
                "validation": len(validation),
                "test": len(test),
            },
            "training_attack_families": sorted(
                train["attack_family"].unique().tolist()
            ),
            "test_attack_families": test["attack_family"].value_counts().to_dict(),
            "integrity": {
                "no_train_test_overlap": not bool(train_ids & test_ids),
                "no_calibration_test_overlap": not bool(calibration_ids & test_ids),
                "no_validation_test_overlap": not bool(validation_ids & test_ids),
            },
            "models": {},
        }

        X_train = train[features]
        X_calibration = calibration[features]
        X_validation = validation[features]
        X_test = test[features]

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)

            raw_calibration = model.predict_proba(X_calibration)[:, 1]
            raw_validation = model.predict_proba(X_validation)[:, 1]
            raw_test = model.predict_proba(X_test)[:, 1]

            calibrator = PlattProbabilityCalibrator()
            calibrator.fit(raw_calibration, y_calibration)

            validation_probabilities = calibrator.predict(raw_validation)
            test_probabilities = calibrator.predict(raw_test)

            threshold, validation_fpr, validation_recall = choose_threshold(
                y_validation, validation_probabilities
            )

            validation_metrics = calculate_metrics(
                y_validation, validation_probabilities, threshold
            )
            test_metrics = calculate_metrics(
                y_test, test_probabilities, threshold
            )

            test_family_results = family_metrics(
                test.reset_index(drop=True),
                test_probabilities,
                threshold,
            )

            fold_results["models"][model_name] = {
                "threshold": threshold,
                "validation": validation_metrics,
                "test": test_metrics,
                "test_per_attack_family": test_family_results,
                "calibration_method": "Platt scaling (logistic regression)",
                "validation_threshold_selection": {
                    "target_max_fpr": MAX_VALIDATION_FPR,
                    "achieved_fpr": validation_fpr,
                    "achieved_recall": validation_recall,
                },
            }

            print(
                f"{model_name}: threshold={threshold:.8f} | "
                f"validation FPR={validation_fpr:.3%}"
            )
            print(
                f"  TEST precision={test_metrics['precision']:.4f} | "
                f"recall={test_metrics['recall']:.4f} | "
                f"F1={test_metrics['f1']:.4f} | "
                f"FPR={test_metrics['fpr']:.4%}"
            )
            print(
                f"  TEST confusion matrix: "
                f"{test_metrics['confusion_matrix']}"
            )

        all_results["capture_folds"][f"fold_{fold_number}"] = fold_results

        # Save after every fold so completed results survive an interruption.
        output_path = OUTPUT_DIR / "independent_capture_metrics.json"
        with output_path.open("w") as f:
            json.dump(json_safe(all_results), f, indent=2)

        print(f"\nFold {fold_number} saved.")

    summary = {}
    for model_name in ("random_forest", "xgboost"):
        summary[model_name] = {}
        for metric in ("precision", "recall", "f1", "pr_auc", "fpr"):
            values = [
                fold["models"][model_name]["test"][metric]
                for fold in all_results["capture_folds"].values()
            ]
            summary[model_name][metric] = {
                "per_fold": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    all_results["aggregate_test_metrics"] = summary

    output_path = OUTPUT_DIR / "independent_capture_metrics.json"
    with output_path.open("w") as f:
        json.dump(json_safe(all_results), f, indent=2)

    print("\n" + "=" * 68)
    print("INDEPENDENT CAPTURE EVALUATION COMPLETE")
    print("=" * 68)

    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()} — mean across three capture folds")
        for metric in ("precision", "recall", "f1", "pr_auc", "fpr"):
            item = metrics[metric]
            print(
                f"  {metric:10s}: {item['mean']:.5f} "
                f"+/- {item['std']:.5f}"
            )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
