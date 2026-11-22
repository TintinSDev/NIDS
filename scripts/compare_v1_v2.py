
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.decision.engine import NIDSDetectionEngine
from src.features.flow_builder import vectorize_dataframe


ROOT = Path(".")
V1_DIR = ROOT / "artifacts/models/v1"
V2_DIR = ROOT / "artifacts/models/v2"

VAL_FILE = ROOT / "data/processed/val.parquet"
TEST_FILE = ROOT / "data/processed/test.parquet"

OUTPUT_DIR = ROOT / "artifacts/evaluation"
OUTPUT_FILE = OUTPUT_DIR / "v1_v2_comparison.json"

CHUNK_SIZE = 5000
MAX_FPR = 0.01

MODEL_FILES = {
    "xgb": "xgb_binary_calibrated.joblib",
    "rf": "rf_calibrated.joblib",
}


def select_threshold(y_true, probabilities):
    """Maximize validation recall while keeping FPR <= MAX_FPR."""

    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    eligible = np.flatnonzero(
        (fpr <= MAX_FPR) & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        raise RuntimeError("No threshold satisfies the FPR constraint.")

    best_recall = np.max(tpr[eligible])
    best_candidates = eligible[tpr[eligible] == best_recall]

    # If recall ties, choose the more conservative threshold.
    best_index = best_candidates[
        np.argmax(thresholds[best_candidates])
    ]

    return {
        "threshold": float(thresholds[best_index]),
        "validation_fpr": float(fpr[best_index]),
        "validation_recall": float(tpr[best_index]),
    }


def load_models():
    """
    Initialize v2 engine first so its PlattCalibrator compatibility
    registration is available when loading the saved joblib artifacts.
    """

    v2_engine = NIDSDetectionEngine(artifacts_dir=V2_DIR)

    models = {
        "v1": {},
        "v2": {
            "xgb": v2_engine.xgb_bin,
            "rf": v2_engine.rf_bin,
        },
    }

    for model_name, filename in MODEL_FILES.items():
        models["v1"][model_name] = joblib.load(V1_DIR / filename)

    return models, v2_engine.features


def get_probabilities(model, dataframe, features):
    """Predict in chunks to keep memory usage manageable."""

    output = []

    for start in range(0, len(dataframe), CHUNK_SIZE):
        chunk = dataframe.iloc[start:start + CHUNK_SIZE]
        X = vectorize_dataframe(chunk, features)

        probabilities = model.predict_proba(X)[:, 1]
        output.append(probabilities)

    return np.concatenate(output)


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def family_breakdown(dataframe, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    result = {}

    for family, group in dataframe.groupby("attack_family", sort=True):
        positions = dataframe.index.get_indexer(group.index)

        actual = group["is_malicious"].astype(int).to_numpy()
        predicted = predictions[positions]

        attacks = int(actual.sum())
        detected = int(((actual == 1) & (predicted == 1)).sum())

        result[str(family)] = {
            "rows": int(len(group)),
            "actual_attacks": attacks,
            "detected_attacks": detected,
            "missed_attacks": int(
                ((actual == 1) & (predicted == 0)).sum()
            ),
            "false_positives": int(
                ((actual == 0) & (predicted == 1)).sum()
            ),
            "attack_recall": (
                float(detected / attacks) if attacks else None
            ),
        }

    return result


def main():
    print("Loading validation and test datasets...")

    val = pd.read_parquet(VAL_FILE)
    test = pd.read_parquet(TEST_FILE)

    required = {"is_malicious", "attack_family"}

    for name, dataframe in [("validation", val), ("test", test)]:
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    y_val = val["is_malicious"].astype(int).to_numpy()
    y_test = test["is_malicious"].astype(int).to_numpy()

    print(f"Validation rows: {len(val):,}")
    print(f"Test rows:       {len(test):,}")
    print(f"Validation classes: {dict(pd.Series(y_val).value_counts())}")
    print(f"Test classes:       {dict(pd.Series(y_test).value_counts())}")

    print("\nLoading calibrated model artifacts...")
    models, features = load_models()

    print(f"Features: {len(features)}")

    report = {
        "comparison": "v1_vs_v2",
        "threshold_selection": (
            "Maximum validation recall subject to FPR <= 1%"
        ),
        "max_validation_fpr": MAX_FPR,
        "validation_rows": len(val),
        "test_rows": len(test),
        "models": {},
    }

    for version in ("v1", "v2"):
        report["models"][version] = {}

        for model_name in ("xgb", "rf"):
            print(f"\n========== {version.upper()} {model_name.upper()} ==========")

            model = models[version][model_name]

            print("Scoring validation data...")
            val_probs = get_probabilities(model, val, features)

            print("Selecting threshold on validation data...")
            selection = select_threshold(y_val, val_probs)
            threshold = selection["threshold"]

            val_metrics = evaluate(y_val, val_probs, threshold)

            print("Scoring held-out test data...")
            test_probs = get_probabilities(model, test, features)

            test_metrics = evaluate(y_test, test_probs, threshold)

            test_metrics["per_attack_family"] = family_breakdown(
                test,
                test_probs,
                threshold,
            )

            report["models"][version][model_name] = {
                "threshold_selection": selection,
                "validation": val_metrics,
                "test": test_metrics,
            }

            print(f"Selected threshold: {threshold:.10f}")
            print(
                "Validation: "
                f"Precision={val_metrics['precision']:.4f}, "
                f"Recall={val_metrics['recall']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, "
                f"FPR={val_metrics['fpr']:.4%}"
            )
            print(
                "Test:       "
                f"Precision={test_metrics['precision']:.4f}, "
                f"Recall={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"FPR={test_metrics['fpr']:.4%}"
            )
            print("Test confusion matrix:")
            print(test_metrics["confusion_matrix"])

            for family, stats in test_metrics["per_attack_family"].items():
                print(
                    f"  {family}: "
                    f"detected={stats['detected_attacks']:,}, "
                    f"missed={stats['missed_attacks']:,}, "
                    f"false_positives={stats['false_positives']:,}, "
                    f"recall={stats['attack_recall']}"
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w") as file:
        json.dump(report, file, indent=2)

    print("\n========== COMPARISON COMPLETE ==========")
    print(f"Report saved to: {OUTPUT_FILE}")
    print("No model artifacts, datasets, or database records were modified.")


if __name__ == "__main__":
    main()
