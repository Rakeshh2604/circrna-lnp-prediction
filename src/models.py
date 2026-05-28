"""Model training utilities."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def cv_evaluate(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                random_state: int = 42) -> dict:
    """K-fold CV. Returns mean and std of RMSE and R^2."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses, r2s = [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, pred)))
        r2s.append(r2_score(y_val, pred))
    return {
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
    }
