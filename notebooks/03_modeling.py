"""03 — Baseline modeling: mean baseline, Ridge, Random Forest.

Runs 5-fold CV on the primary feature matrix and reports RMSE / R² per model.
Saves:
  - outputs/metrics_baseline.csv         per-model aggregate metrics
  - outputs/oof_predictions.parquet      out-of-fold preds for every model
  - outputs/figures/modeling/01_model_comparison.png
  - outputs/figures/modeling/02_pred_vs_actual_rf.png
  - outputs/figures/modeling/03_residuals_rf.png

Run from the repo root:
    python notebooks/03_modeling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features import load_features  # noqa: E402
from src.models import cv_evaluate  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "modeling"
METRICS_PATH = REPO_ROOT / "outputs" / "metrics_baseline.csv"
OOF_PATH = REPO_ROOT / "outputs" / "oof_predictions.parquet"

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_models() -> dict:
    """Return {name: estimator} for the baseline + two real models."""
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge_l2": RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
    }


def fig_model_comparison(rows: list[dict]) -> str:
    """RMSE and R² bar charts side-by-side."""
    df = pd.DataFrame(rows).set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#999999", "#3b6cb7", "#2a9d4b"]

    axes[0].bar(df.index, df["rmse_mean"], yerr=df["rmse_std"],
                color=colors, capsize=4)
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_title("5-fold CV RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(df.index, df["r2_mean"], yerr=df["r2_std"],
                color=colors, capsize=4)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("R² (higher is better; 0 = mean baseline)")
    axes[1].set_title("5-fold CV R²")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return str(_save(fig, "01_model_comparison"))


def fig_pred_vs_actual(y: pd.Series, oof_rf: np.ndarray) -> str:
    """Predicted vs actual for the random forest, with y=x reference."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y, oof_rf, s=4, alpha=0.25, color="#2a9d4b", edgecolor="none")
    lo, hi = float(min(y.min(), oof_rf.min())), float(max(y.max(), oof_rf.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Actual experiment_value (z-scored)")
    ax.set_ylabel("OOF prediction (Random Forest)")
    rho = float(np.corrcoef(y, oof_rf)[0, 1])
    ax.set_title(f"Random Forest — predicted vs actual (Pearson r = {rho:.3f})")
    ax.legend(loc="upper left")
    return str(_save(fig, "02_pred_vs_actual_rf"))


def fig_residuals(y: pd.Series, oof_rf: np.ndarray) -> str:
    """Residual distribution for the random forest."""
    resid = y.to_numpy() - oof_rf
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(resid, bins=60, ax=axes[0], color="#2a9d4b")
    axes[0].set_xlabel("Residual (actual − predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Residual distribution  (mean {resid.mean():+.3f}, std {resid.std():.3f})")
    axes[0].axvline(0, color="black", linewidth=0.6)

    axes[1].scatter(oof_rf, resid, s=4, alpha=0.25, color="#2a9d4b", edgecolor="none")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_xlabel("OOF prediction")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs predicted")
    fig.tight_layout()
    return str(_save(fig, "03_residuals_rf"))


def main() -> None:
    X, y, meta = load_features()
    print(f"Loaded features: X {X.shape}, y {len(y)}, feature_meta groups: {meta['group'].nunique()}")

    models = make_models()
    print(f"\nRunning 5-fold CV on {len(models)} models...")
    results = {}
    oof_table = pd.DataFrame({"y_true": y.to_numpy()})
    for name, est in models.items():
        print(f"  fitting {name}...", flush=True)
        res = cv_evaluate(est, X, y, name=name, n_splits=5, random_state=42)
        results[name] = res
        oof_table[f"oof_{name}"] = res.oof_predictions
        print(f"    {res.summary()}")

    # Save metrics + OOF predictions.
    metrics_rows = [res.to_row() for res in results.values()]
    pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)
    oof_table.to_parquet(OOF_PATH, index=False)
    print(f"\nSaved metrics -> {METRICS_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved OOF preds -> {OOF_PATH.relative_to(REPO_ROOT)}")

    # Figures.
    print("\nGenerating figures...")
    p = fig_model_comparison(metrics_rows)
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")
    rf_oof = results["random_forest"].oof_predictions
    p = fig_pred_vs_actual(y, rf_oof)
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")
    p = fig_residuals(y, rf_oof)
    print(f"  -> {Path(p).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
