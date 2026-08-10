"""
Confidence calibration module for hybrid lightweight scorer.

Provides isotonic regression and Platt scaling calibration.
Computes Expected Calibration Error (ECE) and reliability diagrams.
"""

import numpy as np
from typing import Optional, Tuple


class ConfidenceCalibrator:
    """
    Calibrate classifier probabilities using isotonic regression or Platt scaling.

    Wraps sklearn's CalibratedClassifierCV with additional diagnostics.
    """

    def __init__(self, base_classifier=None):
        self.base = base_classifier
        self.calibrated = None
        self.method = None

    def fit(self, X, y, method: str = "isotonic", cv: int = 5):
        from sklearn.calibration import CalibratedClassifierCV

        self.method = method
        self.calibrated = CalibratedClassifierCV(
            self.base, method=method, cv=cv
        )
        self.calibrated.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X)
        if self.base is not None:
            return self.base.predict_proba(X)
        raise RuntimeError("No model fitted")

    def predict_proba_calibrated(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
        return float(ece)

    @staticmethod
    def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_ce = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                max_ce = max(max_ce, abs(bin_acc - bin_conf))
        return float(max_ce)

    @staticmethod
    def compute_reliability_data(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> dict:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_centers.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2))
                bin_accuracies.append(float(y_true[mask].mean()))
                bin_counts.append(int(mask.sum()))

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "ece": ConfidenceCalibrator.compute_ece(y_true, y_prob, n_bins),
        }

    @staticmethod
    def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, dict]:
        from sklearn.metrics import f1_score, precision_score, recall_score

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        results = {}

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced_accuracy":
                from sklearn.metrics import balanced_accuracy_score
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == "mcc":
                from sklearn.metrics import matthews_corrcoef
                score = matthews_corrcoef(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = t
                results = {
                    "threshold": float(t),
                    "score": float(score),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "fpr": float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).sum() > 0 else 0.0,
                    "fnr": float((y_pred[y_true == 1] == 0).mean()) if (y_true == 1).sum() > 0 else 0.0,
                }

        return best_threshold, results

    def save(self, path: str):
        import joblib
        joblib.dump({
            "calibrated": self.calibrated,
            "method": self.method,
        }, path)

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.calibrated = data["calibrated"]
        self.method = data["method"]
