"""07 — circRNA-adjacent interpretation.

Honest framing: LNPDB contains no circRNA cargo. We use the **mRNA subset** as
the closest physicochemical analog (long, single-stranded, structured) and
within that, focus on the **upper tertile of IL:nucleic-acid mass ratio** as
a proxy for the demanding-delivery regime that a long, structured cargo would
impose. The question is: when we condition on this "circRNA-like" envelope,
do the same chemistry features dominate the model's predictions, or do
different features rise?

Outputs:
  - outputs/figures/circrna/01_group_shap_by_subset.png
  - outputs/figures/circrna/02_top_features_by_subset.png
  - outputs/figures/circrna/03_predicted_efficiency_envelope.png
  - outputs/circrna_cut_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_processed  # noqa: E402
from src.features import load_features, PRIMARY_METHOD  # noqa: E402

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "circrna"
SUMMARY_PATH = REPO_ROOT / "outputs" / "circrna_cut_summary.md"
SHAP_PATH = REPO_ROOT / "outputs" / "shap_values.parquet"
OOF_PATH = REPO_ROOT / "outputs" / "oof_predictions.parquet"

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def get_subset_masks(X: pd.DataFrame, raw_sub: pd.DataFrame) -> dict[str, np.ndarray]:
    """Three nested subsets: all, mRNA, mRNA + IL:NA mass ratio strictly > 10.

    The threshold of 10 is the standard IL:NA mass ratio in this dataset (70% of
    mRNA rows are at exactly 10). Cutting at > 10 picks out the ~18% of mRNA
    formulations where the curator chose to load *more* ionizable lipid than the
    standard — a proxy for cargoes that need extra delivery support, which is
    the regime a long, structured cargo like circRNA would impose.
    """
    is_mrna = (raw_sub["cargo"] == "mRNA").to_numpy()
    mr = raw_sub["il_to_nucleicacid_massratio"].to_numpy()
    threshold = 10.0
    circ_mask = is_mrna & (mr > threshold)
    print(f"  All:                                                       n = {len(X):,}")
    print(f"  mRNA only:                                                 n = {int(is_mrna.sum()):,}")
    print(f"  mRNA + IL:NA mass ratio > {threshold}  (circRNA-like regime):  n = {int(circ_mask.sum()):,}")
    return {
        "All LNPs": np.ones(len(X), dtype=bool),
        "mRNA": is_mrna,
        "circRNA-like\n(mRNA, IL:NA mass ratio > 10)": circ_mask,
    }


def fig_group_shap_by_subset(shap_df: pd.DataFrame, meta: pd.DataFrame,
                              masks: dict[str, np.ndarray]) -> str:
    """Grouped mean |SHAP|, normalized within each subset so the bars compare shape."""
    group_map = meta.set_index("feature")["group"]
    by_subset = {}
    for name, mask in masks.items():
        abs_shap = shap_df.loc[mask].abs().mean(axis=0)
        grouped = abs_shap.groupby(group_map).sum()
        by_subset[name] = grouped / grouped.sum()  # normalize -> share of importance

    df = pd.DataFrame(by_subset)
    order = df.mean(axis=1).sort_values(ascending=True).index.tolist()
    df = df.loc[order]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    width = 0.27
    xs = np.arange(len(order))
    colors = ["#999999", "#3b6cb7", "#d4612a"]
    for i, (name, vals) in enumerate(df.items()):
        ax.barh(xs + (i - 1) * width, vals.values, height=width, label=name, color=colors[i])
    ax.set_yticks(xs)
    ax.set_yticklabels(order)
    ax.set_xlabel("Share of total |SHAP| (within subset)")
    ax.set_title("Feature-group importance — does the circRNA-like envelope shift the lever?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return str(_save(fig, "01_group_shap_by_subset"))


def fig_top_features_by_subset(shap_df: pd.DataFrame, masks: dict[str, np.ndarray]) -> str:
    """Top 10 features by mean |SHAP|, side-by-side across subsets."""
    rows = []
    for name, mask in masks.items():
        s = shap_df.loc[mask].abs().mean(axis=0).sort_values(ascending=False).head(10)
        for feat, val in s.items():
            rows.append({"subset": name, "feature": feat, "mean_abs_shap": val})
    df = pd.DataFrame(rows)
    feat_order = (df.groupby("feature")["mean_abs_shap"].max().sort_values(ascending=True).index.tolist())
    df["feature"] = pd.Categorical(df["feature"], categories=feat_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    subset_names = list(masks.keys()) if "masks" in locals() else df["subset"].unique().tolist()
    base_colors = ["#999999", "#3b6cb7", "#d4612a"]
    palette = dict(zip(subset_names, base_colors))
    sns.barplot(data=df, y="feature", x="mean_abs_shap", hue="subset",
                palette=palette, ax=ax)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    ax.set_title("Top-10 feature importance, by subset")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return str(_save(fig, "02_top_features_by_subset"))


def fig_predicted_efficiency_envelope(oof_xgb: np.ndarray, masks: dict[str, np.ndarray]) -> str:
    """OOF prediction distribution by subset — is the model bullish or bearish on the envelope?"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    palette = ["#999999", "#3b6cb7", "#d4612a"]
    for color, (name, mask) in zip(palette, masks.items()):
        sns.kdeplot(oof_xgb[mask], ax=ax, label=f"{name} (n={mask.sum():,})",
                    color=color, fill=True, alpha=0.15, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("OOF predicted experiment_value (z-scored, XGBoost)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted-efficiency distribution by subset")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return str(_save(fig, "03_predicted_efficiency_envelope"))


def main() -> None:
    X, y, meta = load_features()
    df = load_processed()
    raw_sub = df[df["experiment_method"] == PRIMARY_METHOD].reset_index(drop=True)
    shap_df = pd.read_parquet(SHAP_PATH)
    oof = pd.read_parquet(OOF_PATH)["oof_xgboost"].to_numpy()
    print(f"Loaded: X {X.shape}, SHAP {shap_df.shape}, raw subset {raw_sub.shape}")

    masks = get_subset_masks(X, raw_sub)
    print("\nGenerating figures...")
    p1 = fig_group_shap_by_subset(shap_df, meta, masks)
    p2 = fig_top_features_by_subset(shap_df, masks)
    p3 = fig_predicted_efficiency_envelope(oof, masks)
    for p in (p1, p2, p3):
        print(f"  -> {Path(p).relative_to(REPO_ROOT)}")

    # Summary stats.
    rows = []
    for name, mask in masks.items():
        rows.append({
            "subset": name.replace("\n(", " ("),
            "n": int(mask.sum()),
            "median_predicted_efficiency": float(np.median(oof[mask])),
            "p90_predicted_efficiency": float(np.quantile(oof[mask], 0.9)),
        })

    # Top 5 features in each subset by mean |SHAP|.
    summary_lines = ["# circRNA-cut summary\n"]
    summary_lines.append("## Subsets and prediction envelopes\n")
    summary_lines.append("| Subset | n | median predicted | 90th percentile predicted |")
    summary_lines.append("|---|---:|---:|---:|")
    for r in rows:
        summary_lines.append(
            f"| {r['subset']} | {r['n']:,} | {r['median_predicted_efficiency']:+.3f} | "
            f"{r['p90_predicted_efficiency']:+.3f} |"
        )

    summary_lines.append("\n## Top 5 features by mean |SHAP|, per subset\n")
    for name, mask in masks.items():
        clean_name = name.replace("\n(", " (")
        s = shap_df.loc[mask].abs().mean(axis=0).sort_values(ascending=False).head(5)
        summary_lines.append(f"\n**{clean_name} (n={int(mask.sum()):,})**\n")
        summary_lines.append("| Feature | mean \\|SHAP\\| |")
        summary_lines.append("|---|---:|")
        for feat, val in s.items():
            summary_lines.append(f"| `{feat}` | {val:.4f} |")

    summary_lines.append("\n## Reading these numbers\n")
    summary_lines.append(
        "If the top features in the circRNA-like subset look very similar to the overall "
        "top features, the model is saying the same chemistry levers matter even in the "
        "demanding-delivery regime — and those features become defensible hypotheses for "
        "circRNA delivery experiments (subject to the LOPO caveat that cross-study transfer "
        "is the unsolved problem)."
    )
    SUMMARY_PATH.write_text("\n".join(summary_lines))
    print(f"\nSummary -> {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
