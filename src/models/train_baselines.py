import json
from pathlib import Path
import numpy as np
import pandas as pd
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
import xgboost as xgb

SCHEMA_PATH = Path("src/features/features_v1.json")
PROC_DIR = Path("data/processed")
ART_DIR = Path("artifacts/models/v3")


def compute_fpr(y_true_bin, y_pred_bin):
    tn, fp, _, _ = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def main():
    features = json.loads(SCHEMA_PATH.read_text())["features"]
    train_df = pd.read_parquet(PROC_DIR / "train.parquet")
    val_df = pd.read_parquet(PROC_DIR / "val.parquet")

    X_train, y_train_bin = train_df[features], train_df["is_malicious"].astype(int)
    X_val, y_val_bin = val_df[features], val_df["is_malicious"].astype(int)

    # Class imbalance ratio for binary XGBoost
    neg, pos = (y_train_bin == 0).sum(), (y_train_bin == 1).sum()
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    # 1. Baseline Random Forest (Binary)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train_bin)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    # 2. XGBoost Binary Classifier
    xgb_bin = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    xgb_bin.fit(X_train, y_train_bin)
    xgb_probs = xgb_bin.predict_proba(X_val)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    # 3. Isolation Forest trained strictly on Benign training flows
    X_train_benign = X_train[y_train_bin == 0]
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        n_jobs=-1,
        random_state=42,
    )
    iso.fit(X_train_benign)
    # Higher score = more anomalous
    iso_anom_score = -iso.decision_function(X_val)
    iso_preds = (iso.predict(X_val) == -1).astype(int)

    # 4. Multi-class XGBoost on attack_family (fit on classes present in train)
    le = LabelEncoder()
    y_train_mc = le.fit_transform(train_df["attack_family"])
    known_mask = val_df["attack_family"].isin(le.classes_)
    val_mc_df = val_df[known_mask]
    X_val_mc = val_mc_df[features]
    y_val_mc = le.transform(val_mc_df["attack_family"])

    xgb_mc = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb_mc.fit(X_train, y_train_mc)
    mc_preds = xgb_mc.predict(X_val_mc)

    metrics = {
        "random_forest_binary": {
            "precision": float(precision_score(y_val_bin, rf_preds, zero_division=0)),
            "recall": float(recall_score(y_val_bin, rf_preds, zero_division=0)),
            "f1": float(f1_score(y_val_bin, rf_preds, zero_division=0)),
            "pr_auc": float(average_precision_score(y_val_bin, rf_probs)) if y_val_bin.sum() > 0 else None,
            "fpr": compute_fpr(y_val_bin, rf_preds),
        },
        "xgboost_binary": {
            "precision": float(precision_score(y_val_bin, xgb_preds, zero_division=0)),
            "recall": float(recall_score(y_val_bin, xgb_preds, zero_division=0)),
            "f1": float(f1_score(y_val_bin, xgb_preds, zero_division=0)),
            "pr_auc": float(average_precision_score(y_val_bin, xgb_probs)) if y_val_bin.sum() > 0 else None,
            "fpr": compute_fpr(y_val_bin, xgb_preds),
        },
        "isolation_forest_benign_only": {
            "precision": float(precision_score(y_val_bin, iso_preds, zero_division=0)),
            "recall": float(recall_score(y_val_bin, iso_preds, zero_division=0)),
            "f1": float(f1_score(y_val_bin, iso_preds, zero_division=0)),
            "pr_auc": float(average_precision_score(y_val_bin, iso_anom_score)) if y_val_bin.sum() > 0 else None,
            "fpr": compute_fpr(y_val_bin, iso_preds),
        },
        "xgboost_multiclass_val_known_classes": {
            "classes": le.classes_.tolist(),
            "report": classification_report(
                y_val_mc,
                mc_preds,
                labels=list(range(len(le.classes_))),
                target_names=le.classes_.tolist(),
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "metrics_v1.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
