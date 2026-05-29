"""04 — XGBoost with light hyperparameter tuning.

Tunes XGBoost over a small grid using 3-fold CV inside a RandomizedSearchCV, then
re-evaluates the best estimator with the project's standard 5-fold CV (same seed
as the baselines) so the result is directly comparable to ridge/RF.

Updates:
  - outputs/metrics_baseline.csv      append the xgboost row
  - outputs/oof_predictions.parquet   add an oof_xgboost column
  - outputs/figures/modeling/01_model_comparison.png   include xgboost
  - outputs/figures/modeling/04_xgboost_pred_vs_actual.png
  - outputs/figures/modeling/05_xgboost_residuals.png
  - outputs/best_xgboost_params.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features import load_features  # noqa: E402
from src.models import cv_evaluate  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "modeling"
METRICS_PATH = REPO_ROOT / "outputs" / "metrics_baseline.csv"
OOF_PATH = REPO_ROOT / "outputs" / "oof_predictions.parquet"
BEST_PARAMS_PATH = REPO_ROOT / "outputs" / "best_xgboost_params.json"

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_iter: int = 30, seed: int = 42) -> dict:
    """RandomizedSearch over a small space; return best params dict."""
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=600,
        random_state=seed,
        n_jobs=-1,
    )
    space = {
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),       # 0.6 .. 1.0
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda": loguniform(0.1, 10.0),
        "reg_alpha": loguniform(0.001, 1.0),
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=seed),
        n_jobs=-1,
        random_state=seed,
        verbose=0,
        refit=False,
    )
    search.fit(X.to_numpy(), y.to_numpy())
    return dict(search.best_params_)


def fig_model_comparison(metrics: pd.DataFrame) -> str:
    """Re-draw model_comparison.png now that xgboost is included."""
    color_map = {
        "mean_baseline": "#999999",
        "ridge_l2": "#3b6cb7",
        "random_forest": "#2a9d4b",
        "xgboost": "#d4612a",
    }
    metrics = metrics.set_index("model")
    order = [m for m in color_map if m in metrics.index]
    metrics = metrics.loc[order]
    colors = [color_map[m] for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(metrics.index, metrics["rmse_mean"], yerr=metrics["rmse_std"],
                color=colors, capsize=4)
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_title("5-fold CV RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(metrics.index, metrics["r2_mean"], yerr=metrics["r2_std"],
                color=colors, capsize=4)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].set_title("5-fold CV R²")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return str(_save(fig, "01_model_comparison"))


def fig_pred_vs_actual(y: pd.Series, oof: np.ndarray, name: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y, oof, s=4, alpha=0.25, color="#d4612a", edgecolor="none")
    lo, hi = float(min(y.min(), oof.min())), float(max(y.max(), oof.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Actual experiment_value (z-scored)")
    ax.set_ylabel(f"OOF prediction ({name})")
    rho = float(np.corrcoef(y, oof)[0, 1])
    ax.set_title(f"{name} — predicted vs actual (Pearson r = {rho:.3f})")
    ax.legend(loc="upper left")
    return str(_save(fig, "04_xgboost_pred_vs_actual"))


def fig_residuals(y: pd.Series, oof: np.ndarray) -> str:
    resid = y.to_numpy() - oof
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(resid, bins=60, ax=axes[0], color="#d4612a")
    axes[0].set_xlabel("Residual (actual − predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"XGBoost residuals  (mean {resid.mean():+.3f}, std {resid.std():.3f})")
    axes[0].axvline(0, color="black", linewidth=0.6)

    axes[1].scatter(oof, resid, s=4, alpha=0.25, color="#d4612a", edgecolor="none")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_xlabel("OOF prediction")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")
    fig.tight_layout()
    return str(_save(fig, "05_xgboost_residuals"))


def main() -> None:
    X, y, _ = load_features()
    print(f"Loaded features: X {X.shape}, y {len(y)}")

    print("\nTuning XGBoost (RandomizedSearch, n_iter=30 over 3-fold inner CV)...")
    best = tune_xgboost(X, y, n_iter=30, seed=42)
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2, default=float))
    print(f"  Best params: {json.dumps(best, default=float)}")

    final = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        **best,
    )
    print("\nFinal 5-fold CV with tuned XGBoost (same seed as baselines)...")
    res = cv_evaluate(final, X, y, name="xgboost", n_splits=5, random_state=42)
    print(f"  {res.summary()}")

    # Update metrics file and OOF table.
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        metrics = metrics[metrics["model"] != "xgboost"]
    else:
        metrics = pd.DataFrame()
    metrics = pd.concat([metrics, pd.DataFrame([res.to_row()])], ignore_index=True)
    metrics.to_csv(METRICS_PATH, index=False)

    oof = pd.read_parquet(OOF_PATH)
    oof["oof_xgboost"] = res.oof_predictions
    oof.to_parquet(OOF_PATH, index=False)

    # Figures.
    p = fig_model_comparison(metrics)
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")
    p = fig_pred_vs_actual(y, res.oof_predictions, "XGBoost")
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")
    p = fig_residuals(y, res.oof_predictions)
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")

    print(f"\nFinal model comparison:\n{metrics.to_string(index=False)}")


if __name__ == "__main__":
    main()
