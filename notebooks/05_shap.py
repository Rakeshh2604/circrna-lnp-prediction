"""05 — SHAP interpretation of the tuned XGBoost model.

Trains the final XGBoost on the full primary subset (using the best params from
Day 8), computes SHAP values on the full dataset, and produces the standard
interpretive figures:

  - 01_shap_summary_dot.png        beeswarm of top 20 features
  - 02_shap_bar_global.png         mean |SHAP| bar chart (global importance)
  - 03_shap_bar_by_group.png       importance aggregated by feature group
  - 04_shap_dependence_*.png       partial-dependence-style scatter for top 5
                                   numeric features, colored by an interaction
                                   feature SHAP auto-picks

Also persists outputs/shap_values.parquet (so downstream code — e.g. the
circRNA-cut analysis in Day 11 — can reuse the SHAP values without recomputing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features import load_features  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "shap"
BEST_PARAMS_PATH = REPO_ROOT / "outputs" / "best_xgboost_params.json"
SHAP_OUT_PATH = REPO_ROOT / "outputs" / "shap_values.parquet"

sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)


def _save_current(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def fig_shap_summary_beeswarm(shap_values, X) -> str:
    shap.summary_plot(shap_values, X, max_display=20, show=False, plot_size=(8, 9))
    return str(_save_current("01_shap_summary_dot"))


def fig_shap_bar_global(shap_values, X) -> str:
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20,
                      show=False, plot_size=(8, 9))
    return str(_save_current("02_shap_bar_global"))


def fig_shap_bar_by_group(shap_values, X, feature_meta) -> str:
    abs_shap = pd.DataFrame(np.abs(shap_values), columns=X.columns)
    by_feature = abs_shap.mean(axis=0)
    grouped = by_feature.groupby(feature_meta.set_index("feature").loc[by_feature.index, "group"]).sum()
    grouped = grouped.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(grouped.index, grouped.values, color="#d4612a")
    ax.set_xlabel("Sum of mean |SHAP| across features in the group")
    ax.set_title("Feature-group contribution to XGBoost predictions")
    fig.tight_layout()
    path = FIG_DIR / "03_shap_bar_by_group.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_shap_dependence_top5(shap_values, X) -> list[str]:
    """Dependence plots for the top 5 features by mean |SHAP|."""
    abs_mean = np.abs(shap_values).mean(axis=0)
    top = pd.Series(abs_mean, index=X.columns).sort_values(ascending=False).head(5)
    out = []
    for i, feat in enumerate(top.index, start=1):
        shap.dependence_plot(feat, shap_values, X, show=False, alpha=0.4)
        out.append(str(_save_current(f"04_dependence_{i:02d}_{feat}")))
    return out


def main() -> None:
    X, y, meta = load_features()
    best = json.loads(BEST_PARAMS_PATH.read_text())
    print(f"Loaded features: X {X.shape}")
    print(f"Using best params: {best}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        **best,
    )
    print("Fitting XGBoost on full primary subset...")
    model.fit(X.to_numpy(), y.to_numpy())

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"  shap_values shape: {shap_values.shape}")

    # Persist SHAP for downstream (circRNA cut).
    pd.DataFrame(shap_values, columns=X.columns).to_parquet(SHAP_OUT_PATH, index=False)
    print(f"  saved -> {SHAP_OUT_PATH.relative_to(REPO_ROOT)}")

    print("\nGenerating figures...")
    print(f"  -> {Path(fig_shap_summary_beeswarm(shap_values, X)).relative_to(REPO_ROOT)}")
    print(f"  -> {Path(fig_shap_bar_global(shap_values, X)).relative_to(REPO_ROOT)}")
    print(f"  -> {Path(fig_shap_bar_by_group(shap_values, X, meta)).relative_to(REPO_ROOT)}")
    for p in fig_shap_dependence_top5(shap_values, X):
        print(f"  -> {Path(p).relative_to(REPO_ROOT)}")

    abs_mean = np.abs(shap_values).mean(axis=0)
    top10 = pd.Series(abs_mean, index=X.columns).sort_values(ascending=False).head(10)
    print("\nTop 10 features by mean |SHAP|:")
    print(top10.to_string())


if __name__ == "__main__":
    main()
