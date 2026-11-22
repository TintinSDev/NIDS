import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from src.features.flow_builder import load_schema_features, vectorize_dataframe

ART_DIR = Path("artifacts/models/v3")


def _ensure_platt_calibrator_registered() -> None:
    """Allow unpickling PlattCalibrator whether saved under __main__ or calibrate_and_save."""
    try:
        from src.models.calibrate_and_save import PlattCalibrator  # type: ignore
    except Exception:
        from sklearn.linear_model import LogisticRegression

        class PlattCalibrator:  # type: ignore
            def __init__(self, estimator, pos_idx: int = 1):
                self.estimator = estimator
                self.pos_idx = pos_idx
                self.calibrator = LogisticRegression()

            def fit(self, X, y):
                raw_probs = np.clip(self.estimator.predict_proba(X)[:, self.pos_idx], 1e-6, 1 - 1e-6)
                logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
                self.calibrator.fit(logits, y)
                return self

            def predict_proba(self, X):
                raw_probs = np.clip(self.estimator.predict_proba(X)[:, self.pos_idx], 1e-6, 1 - 1e-6)
                logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
                cal_pos = self.calibrator.predict_proba(logits)[:, 1]
                return np.column_stack([1.0 - cal_pos, cal_pos])

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "PlattCalibrator"):
        setattr(main_mod, "PlattCalibrator", PlattCalibrator)

    cal_mod_name = "src.models.calibrate_and_save"
    cal_mod = sys.modules.get(cal_mod_name)
    if cal_mod is None:
        cal_mod = types.ModuleType(cal_mod_name)
        sys.modules[cal_mod_name] = cal_mod
    if not hasattr(cal_mod, "PlattCalibrator"):
        setattr(cal_mod, "PlattCalibrator", PlattCalibrator)


class NIDSDetectionEngine:
    """Combines calibrated binary classifiers, anomaly scoring, and multi-class family prediction."""

    def __init__(self, artifacts_dir: Path = ART_DIR):
        _ensure_platt_calibrator_registered()
        self.artifacts_dir = Path(artifacts_dir)
        self.model_version = "v3.0.0"
        self.features: List[str] = load_schema_features()

        self.xgb_bin = joblib.load(self.artifacts_dir / "xgb_binary_calibrated.joblib")
        self.rf_bin = joblib.load(self.artifacts_dir / "rf_calibrated.joblib")
        self.iso = joblib.load(self.artifacts_dir / "isolation_forest.joblib")
        self.xgb_mc = joblib.load(self.artifacts_dir / "xgb_multiclass.joblib")
        self.le = joblib.load(self.artifacts_dir / "label_encoder.joblib")

        cal_path = self.artifacts_dir / "calibration_v3.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            thresholds = cal.get("threshold_selection", {})
            xgb_thresh = float(
                thresholds.get("xgb_binary", {}).get(
                    "threshold",
                    cal.get("selected_high_conf_threshold", 0.5),
                )
            )
            rf_thresh = float(
                thresholds.get("rf_binary", {}).get("threshold", 0.5)
            )
        else:
            xgb_thresh = 0.5
            rf_thresh = 0.5

        self.malicious_threshold = float(
            os.getenv("NIDS_MALICIOUS_THRESHOLD", str(xgb_thresh))
        )
        self.rf_malicious_threshold = float(
            os.getenv("NIDS_RF_MALICIOUS_THRESHOLD", str(rf_thresh))
        )
        self.suspicious_threshold = float(os.getenv("NIDS_SUSPICIOUS_THRESHOLD", "0.55"))
        self.anomaly_threshold = float(os.getenv("NIDS_ANOMALY_THRESHOLD", "0.08"))

    def _top_feature_contributions(self, X_df: pd.DataFrame, row_idx: int, top_k: int = 5) -> List[Dict[str, Any]]:
        base_est = getattr(self.xgb_bin, "estimator", self.xgb_bin)
        booster_imp = getattr(base_est, "feature_importances_", None)
        if booster_imp is None or len(booster_imp) != len(self.features):
            booster_imp = np.ones(len(self.features), dtype=np.float64)

        row_vals = X_df.iloc[row_idx].to_numpy(dtype=np.float64)
        weights = booster_imp * np.log1p(np.abs(row_vals))
        top_indices = np.argsort(weights)[::-1][:top_k]

        evidence = []
        for idx in top_indices:
            fname = self.features[int(idx)]
            evidence.append(
                {
                    "feature": fname,
                    "value": float(row_vals[int(idx)]),
                    "importance_score": float(weights[int(idx)]),
                }
            )
        return evidence

    def predict_dataframe(self, df: pd.DataFrame, model_version: str = "v3.0.0") -> List[Dict[str, Any]]:
        X_df = vectorize_dataframe(df, self.features)

        xgb_probs = self.xgb_bin.predict_proba(X_df)[:, 1]
        rf_probs = self.rf_bin.predict_proba(X_df)[:, 1]
        anomaly_scores = (-self.iso.decision_function(X_df)).astype(float)
        mc_probs = self.xgb_mc.predict_proba(X_df)
        mc_pred_idx = np.argmax(mc_probs, axis=1)
        mc_labels = self.le.inverse_transform(mc_pred_idx)
        mc_confidences = mc_probs[np.arange(len(X_df)), mc_pred_idx]

        results: List[Dict[str, Any]] = []
        for i in range(len(X_df)):
            p_xgb = float(xgb_probs[i])
            p_rf = float(rf_probs[i])
            anom = float(anomaly_scores[i])
            ensemble_conf = max(p_xgb, p_rf)
            attack_family = str(mc_labels[i])
            mc_conf = float(mc_confidences[i])

            models_agree_malicious = (
                p_xgb >= self.malicious_threshold
                and p_rf >= self.rf_malicious_threshold
            )
            anomaly_fired = anom >= self.anomaly_threshold

            calibrated_xgb_fired = p_xgb >= self.malicious_threshold

            mid_range_conf = (
                p_xgb >= self.suspicious_threshold
                or p_rf >= self.suspicious_threshold
            )

            if models_agree_malicious and attack_family != "Benign":
                level = "confirmed_malicious"
                confidence = round(float((p_xgb + p_rf) / 2.0), 4)
                final_class = attack_family
            elif (
                calibrated_xgb_fired
                or anomaly_fired
                or mid_range_conf
                or (attack_family != "Benign" and mc_conf >= 0.6)
            ):
                level = "suspicious"
                confidence = round(float(max(ensemble_conf, mc_conf if attack_family != "Benign" else 0.0)), 4)
                final_class = attack_family if attack_family != "Benign" else "Anomaly/Suspicious"
            else:
                level = "benign"
                confidence = round(float(1.0 - ensemble_conf), 4)
                final_class = "Benign"

            results.append(
                {
                    "level": level,
                    "attack_class": final_class,
                    "confidence": confidence,
                    "anomaly_score": round(anom, 6),
                    "model_version": model_version,
                    "detector_scores": {
                        "xgb_binary_prob": round(p_xgb, 6),
                        "rf_binary_prob": round(p_rf, 6),
                        "isolation_forest_anomaly_score": round(anom, 6),
                        "multiclass_family": attack_family,
                        "multiclass_prob": round(mc_conf, 6),
                    },
                    "shap_values": self._top_feature_contributions(X_df, i, top_k=5),
                }
            )

        return results


def score_sample_from_val(n: int = 5) -> List[Dict[str, Any]]:
    val_df = pd.read_parquet("data/processed/val.parquet").head(n)
    engine = NIDSDetectionEngine()
    return engine.predict_dataframe(val_df)


if __name__ == "__main__":
    sample_out = score_sample_from_val(5)
    print(json.dumps(sample_out, indent=2))
