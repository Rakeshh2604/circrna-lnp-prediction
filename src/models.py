"""Cross-validation harness for the LNPDB transfection model.

`cv_evaluate` runs k-fold CV, returns per-fold metrics, and produces out-of-fold
predictions for every row — so downstream code can compute aggregate metrics,
plot pred-vs-actual, and inspect residuals without re-fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


@dataclass
class CVResult:
    """Per-fold and aggregate CV metrics plus the full out-of-fold prediction vector."""
    model_name: str
    n_splits: int
    rmse_per_fold: list[float] = field(default_factory=list)
    r2_per_fold: list[float] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None  # length == n_samples

    @property
    def rmse_mean(self) -> float:
        return float(np.mean(self.rmse_per_fold))

    @property
    def rmse_std(self) -> float:
        return float(np.std(self.rmse_per_fold))

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.r2_per_fold))

    @property
    def r2_std(self) -> float:
        return float(np.std(self.r2_per_fold))

    def summary(self) -> str:
        return (
            f"{self.model_name:24s}  "
            f"RMSE {self.rmse_mean:.4f} ± {self.rmse_std:.4f}  "
            f"R²   {self.r2_mean:+.4f} ± {self.r2_std:.4f}"
        )

    def to_row(self) -> dict:
        return {
            "model": self.model_name,
            "n_splits": self.n_splits,
            "rmse_mean": self.rmse_mean,
            "rmse_std": self.rmse_std,
            "r2_mean": self.r2_mean,
            "r2_std": self.r2_std,
        }


def cv_evaluate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVResult:
    """K-fold CV.  Clones `model` per fold so the input is left untouched."""
    name = name or type(model).__name__
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(X), np.nan, dtype="float64")
    result = CVResult(model_name=name, n_splits=n_splits)

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    for train_idx, val_idx in kf.split(X_arr):
        fitted = clone(model)
        fitted.fit(X_arr[train_idx], y_arr[train_idx])
        pred = fitted.predict(X_arr[val_idx])
        oof[val_idx] = pred
        result.rmse_per_fold.append(float(np.sqrt(mean_squared_error(y_arr[val_idx], pred))))
        result.r2_per_fold.append(float(r2_score(y_arr[val_idx], pred)))

    result.oof_predictions = oof
    return result
