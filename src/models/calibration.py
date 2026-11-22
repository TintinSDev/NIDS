import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Calibrate an estimator's positive-class probabilities using Platt scaling."""

    def __init__(self, estimator, pos_idx: int = 1):
        self.estimator = estimator
        self.pos_idx = pos_idx
        self.calibrator = LogisticRegression()

    def fit(self, X, y):
        raw_probs = np.clip(
            self.estimator.predict_proba(X)[:, self.pos_idx],
            1e-6,
            1 - 1e-6,
        )
        logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
        self.calibrator.fit(logits, np.asarray(y).astype(int))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        raw_probs = np.clip(
            self.estimator.predict_proba(X)[:, self.pos_idx],
            1e-6,
            1 - 1e-6,
        )
        logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logits)[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
