"""06 — Leave-One-Publication-Out (LOPO) robustness test.

For each publication with at least 30 LNPs in the primary subset, hold out all
rows from that publication, train XGBoost (with the Day 8 best params) on
everything else, and score on the held-out publication. Compare per-publication
RMSE/R² to the random 5-fold CV result.

This is the honest "can the model predict LNPs from a study it has never seen"
test. We expect mean LOPO R² to be lower than the random-CV R²; the size of
that gap is the project's most important number on robustness.

Outputs:
  - outputs/lopo_per_publication.csv
  - outputs/figures/lopo/01_lopo_vs_random.png
  - outputs/figures/lopo/02_r2_distribution.png
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
from sklearn.metrics import mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_processed  # noqa: E402
from src.features import load_features, PRIMARY_METHOD  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "lopo"
RESULTS_PATH = REPO_ROOT / "outputs" / "lopo_per_publication.csv"
BEST_PARAMS_PATH = REPO_ROOT / "outputs" / "best_xgboost_params.json"
RANDOM_CV_METRICS_PATH = REPO_ROOT / "outputs" / "metrics_baseline.csv"

MIN_HELD_OUT = 30  # publications smaller than this give noisy R²; skip for scoring

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def get_publication_groups() -> pd.Series:
    """Re-derive publication_pmid for each row of X.parquet from the cleaned data."""
    df = load_processed()
    sub = df[df["experiment_method"] == PRIMARY_METHOD].reset_index(drop=True)
    return sub["publication_pmid"]


def run_lopo(X: pd.DataFrame, y: pd.Series, groups: pd.Series, best_params: dict) -> pd.DataFrame:
    """LOPO over publications with >= MIN_HELD_OUT rows."""
    pub_counts = groups.value_counts()
    eligible = pub_counts[pub_counts >= MIN_HELD_OUT].index.tolist()
    print(f"  {len(eligible)} eligible publications (>= {MIN_HELD_OUT} LNPs each); "
          f"{len(pub_counts) - len(eligible)} smaller publications are still used as "
          f"training data when others are held out.")

    rows = []
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    groups_arr = groups.to_numpy()

    for i, pub in enumerate(eligible, start=1):
        val_mask = groups_arr == pub
        train_mask = ~val_mask
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_arr[train_mask], y_arr[train_mask])
        pred = model.predict(X_arr[val_mask])
        rmse = float(np.sqrt(mean_squared_error(y_arr[val_mask], pred)))
        r2 = float(r2_score(y_arr[val_mask], pred))
        n_held = int(val_mask.sum())
        n_train = int(train_mask.sum())
        rows.append({
            "publication_pmid": int(pub) if not pd.isna(pub) else None,
            "n_held_out": n_held,
            "n_train": n_train,
            "rmse": rmse,
            "r2": r2,
        })
        print(f"  [{i:2d}/{len(eligible)}] PMID {int(pub):>8d}  "
              f"n_held={n_held:>4d}  RMSE={rmse:.3f}  R²={r2:+.3f}")
    return pd.DataFrame(rows)


def fig_lopo_vs_random(results: pd.DataFrame, random_r2: float) -> str:
    """Per-publication R² ordered, with the random-CV R² as reference."""
    df = results.sort_values("r2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c44e52" if r < 0 else "#3b6cb7" for r in df["r2"]]
    ax.bar(range(len(df)), df["r2"], color=colors)
    ax.axhline(random_r2, color="#2a9d4b", linestyle="--", linewidth=1.5,
               label=f"Random 5-fold CV R² = {random_r2:+.3f}")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["publication_pmid"].astype(int).astype(str), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("R² on held-out publication")
    ax.set_title(f"LOPO per-publication R² — {len(df)} publications, mean = {df['r2'].mean():+.3f}")
    ax.legend()
    fig.tight_layout()
    return str(_save(fig, "01_lopo_vs_random"))


def fig_r2_distribution(results: pd.DataFrame, random_r2: float) -> str:
    """Distribution of LOPO R² values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(results["r2"], bins=20, ax=ax, color="#3b6cb7", edgecolor="white")
    ax.axvline(results["r2"].mean(), color="#3b6cb7", linewidth=1.5,
               label=f"LOPO mean = {results['r2'].mean():+.3f}")
    ax.axvline(random_r2, color="#2a9d4b", linestyle="--", linewidth=1.5,
               label=f"Random CV = {random_r2:+.3f}")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Held-out publication R²")
    ax.set_ylabel("Count")
    ax.set_title("LOPO R² distribution")
    ax.legend()
    fig.tight_layout()
    return str(_save(fig, "02_r2_distribution"))


def main() -> None:
    X, y, _ = load_features()
    groups = get_publication_groups()
    best_params = json.loads(BEST_PARAMS_PATH.read_text())
    print(f"Loaded features: X {X.shape}, publications: {groups.nunique()}")
    print(f"Best XGBoost params from Day 8: {json.dumps(best_params)}")

    print("\nRunning LOPO...")
    results = run_lopo(X, y, groups, best_params)
    results = results.sort_values("r2").reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    random_r2 = pd.read_csv(RANDOM_CV_METRICS_PATH).query("model == 'xgboost'")["r2_mean"].iloc[0]
    print(f"\nLOPO summary:")
    print(f"  mean R²:   {results['r2'].mean():+.3f}")
    print(f"  median R²: {results['r2'].median():+.3f}")
    print(f"  min R²:    {results['r2'].min():+.3f}")
    print(f"  max R²:    {results['r2'].max():+.3f}")
    print(f"  random-CV R² (for comparison): {random_r2:+.3f}")
    print(f"  publications with R² > 0:     {(results['r2'] > 0).sum()} / {len(results)}")
    print(f"  publications with R² > 0.1:   {(results['r2'] > 0.1).sum()} / {len(results)}")

    print("\nGenerating figures...")
    print(f"  -> {Path(fig_lopo_vs_random(results, random_r2)).relative_to(REPO_ROOT)}")
    print(f"  -> {Path(fig_r2_distribution(results, random_r2)).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
