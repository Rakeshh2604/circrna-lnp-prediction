"""01 — Exploratory Data Analysis of cleaned LNPDB.

Produces ~12 figures to outputs/figures/eda/ and a structured summary in
outputs/eda_summary.md. Designed to be run end-to-end:

    python -m notebooks.01_eda

(or just `python notebooks/01_eda.py` from the repo root.)
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

FIG_DIR = REPO_ROOT / "outputs" / "figures" / "eda"
SUMMARY_PATH = REPO_ROOT / "outputs" / "eda_summary.md"
PRIMARY_METHOD = "luminescence_normalized"

sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def fig_target_distribution_overall(df: pd.DataFrame) -> str:
    """Histogram of the target on the primary modeling subset."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(sub["experiment_value"], bins=60, ax=ax, color="#3b6cb7")
    ax.set_xlabel(f"Experiment_value ({PRIMARY_METHOD}, z-scored)")
    ax.set_ylabel("Count")
    ax.set_title(f"Target distribution — {PRIMARY_METHOD} (n={len(sub):,})")
    return str(_save(fig, "01_target_distribution"))


def fig_target_by_method(df: pd.DataFrame) -> str:
    """Violin of target across the largest measurement methods."""
    top_methods = (
        df["experiment_method"].value_counts().head(6).index.tolist()
    )
    sub = df[df["experiment_method"].isin(top_methods)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.violinplot(
        data=sub, x="experiment_method", y="experiment_value",
        order=top_methods, inner="quartile", ax=ax, cut=0,
    )
    ax.set_xlabel("Experiment_method")
    ax.set_ylabel("Experiment_value")
    ax.set_title("Target distribution by measurement method (top 6)")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    return str(_save(fig, "02_target_by_method"))


def fig_target_by_cargo(df: pd.DataFrame) -> str:
    """Target distribution by cargo (mRNA / siRNA / pDNA) on the primary subset."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    order = ["mRNA", "siRNA", "pDNA"]
    n_per = sub.groupby("cargo").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=sub, x="cargo", y="experiment_value", order=order, ax=ax)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}\n(n={n_per[c]:,})" for c in order])
    ax.set_xlabel("")
    ax.set_ylabel(f"Experiment_value ({PRIMARY_METHOD})")
    ax.set_title("Transfection efficiency by cargo type")
    return str(_save(fig, "03_target_by_cargo"))


def fig_target_by_cell_line(df: pd.DataFrame) -> str:
    """Target by cell line (top 10 by sample size) on the primary subset."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    top = sub["model_type"].value_counts().head(10).index.tolist()
    sub = sub[sub["model_type"].isin(top)]
    medians = sub.groupby("model_type")["experiment_value"].median().sort_values()
    order = medians.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=sub, x="model_type", y="experiment_value", order=order, ax=ax)
    ax.set_xlabel("Cell line / model")
    ax.set_ylabel(f"Experiment_value ({PRIMARY_METHOD})")
    ax.set_title("Transfection efficiency by cell line (top 10 by sample size)")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    return str(_save(fig, "04_target_by_cell_line"))


def fig_mol_percent_distributions(df: pd.DataFrame) -> str:
    """4-panel: mol% distributions for IL, HL, CHL, PEG on the full clean df."""
    components = [
        ("il_molratio", "Ionizable lipid mol%"),
        ("hl_molratio", "Helper lipid mol%"),
        ("chl_molratio", "Cholesterol mol%"),
        ("peg_molratio", "PEG-lipid mol%"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (col, title) in zip(axes.flat, components):
        sns.histplot(df[col].dropna(), bins=40, ax=ax, color="#3b6cb7")
        ax.set_title(title)
        ax.set_xlabel("mol %")
    fig.suptitle("Component mol% distributions (cleaned dataset, n={:,})".format(len(df)))
    fig.tight_layout()
    return str(_save(fig, "05_mol_pct_distributions"))


def fig_ionizable_lipid_frequency(df: pd.DataFrame) -> str:
    """How often does each ionizable lipid appear? Mostly singletons."""
    counts = df["il_name"].value_counts()
    bins_def = [(1, 1), (2, 4), (5, 9), (10, 49), (50, counts.max())]
    labels, vals = [], []
    for lo, hi in bins_def:
        n = int(((counts >= lo) & (counts <= hi)).sum())
        labels.append(f"{lo}" if lo == hi else f"{lo}–{hi}")
        vals.append(n)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color="#3b6cb7")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of LNPs using a given ionizable lipid")
    ax.set_ylabel("Count of unique ionizable lipids")
    ax.set_title(
        f"Ionizable-lipid frequency — {df['il_name'].nunique():,} unique ILs across {len(df):,} LNPs"
    )
    return str(_save(fig, "06_il_frequency"))


def fig_helper_pegs(df: pd.DataFrame) -> str:
    """Helper lipid and PEG-lipid composition counts (side-by-side)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    hl_counts = df["hl_name"].fillna("(none)").value_counts()
    sns.barplot(x=hl_counts.values, y=hl_counts.index, ax=axes[0], color="#3b6cb7")
    axes[0].set_title(f"Helper lipid ({df['hl_name'].nunique()} unique + (none))")
    axes[0].set_xlabel("LNP count")

    peg_counts = df["peg_name"].fillna("(none)").value_counts()
    sns.barplot(x=peg_counts.values, y=peg_counts.index, ax=axes[1], color="#3b6cb7")
    axes[1].set_title(f"PEG-lipid ({df['peg_name'].nunique()} unique + (none))")
    axes[1].set_xlabel("LNP count")

    fig.tight_layout()
    return str(_save(fig, "07_helper_peg_composition"))


def _descriptor_columns() -> list[str]:
    return [
        "heavy_atoms", "rings", "aromatic_rings", "rotatable_bonds",
        "van_der_waals_molecular_volume", "topological_polar_surface_area",
        "hydrogen_bond_donors", "hydrogen_bond_acceptors", "logp",
        "molar_refractivity", "fraction_sp3_carbons", "sp3_carbons",
        "nitrogen_count", "molecular_weight",
    ]


def fig_descriptor_correlation_heatmap(df: pd.DataFrame) -> str:
    """Pairwise Spearman correlations among RDKit descriptors."""
    cols = _descriptor_columns()
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Pairwise Spearman correlations — RDKit descriptors")
    fig.tight_layout()
    return str(_save(fig, "08_descriptor_corr_heatmap"))


def fig_descriptor_vs_target(df: pd.DataFrame) -> str:
    """Spearman correlation of each RDKit descriptor with the target."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    cols = _descriptor_columns()
    rho = sub[cols + ["experiment_value"]].corr(method="spearman")["experiment_value"].drop("experiment_value")
    rho = rho.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c44e52" if r < 0 else "#3b6cb7" for r in rho.values]
    ax.barh(rho.index, rho.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"Spearman ρ with target ({PRIMARY_METHOD}, n={len(sub):,})")
    ax.set_title("Descriptor–target rank correlation")
    fig.tight_layout()
    return str(_save(fig, "09_descriptor_vs_target")), rho


def fig_publication_sample_size(df: pd.DataFrame) -> str:
    """Sample size per publication (sorted descending)."""
    counts = df["publication_pmid"].fillna("(no PMID)").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(counts)), counts.values, color="#3b6cb7")
    ax.set_xlabel("Publication (sorted by sample size, anonymized)")
    ax.set_ylabel("LNPs contributed")
    ax.set_title(
        f"LNPs per publication — {len(counts)} publications, "
        f"median {int(np.median(counts))}, max {int(counts.max()):,}"
    )
    return str(_save(fig, "10_publication_sample_size"))


def fig_publication_target_variation(df: pd.DataFrame) -> str:
    """Per-publication boxplot of target on the primary subset (top 15 by n)."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    top = sub["publication_pmid"].value_counts().head(15).index.tolist()
    sub = sub[sub["publication_pmid"].isin(top)]
    medians = sub.groupby("publication_pmid")["experiment_value"].median().sort_values()
    order = medians.index.astype(int).astype(str).tolist()
    sub = sub.copy()
    sub["publication_pmid"] = sub["publication_pmid"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=sub, x="publication_pmid", y="experiment_value", order=order, ax=ax)
    ax.set_xlabel("Publication PMID (top 15 by sample size, sorted by median target)")
    ax.set_ylabel(f"Experiment_value ({PRIMARY_METHOD})")
    ax.set_title("Per-publication target distribution — well-centered after within-method z-scoring")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return str(_save(fig, "11_publication_target_variation"))


def fig_cell_x_cargo_heatmap(df: pd.DataFrame) -> str:
    """Sample-size heatmap: cell line × cargo on the primary subset."""
    sub = df[df["experiment_method"] == PRIMARY_METHOD]
    top_cells = sub["model_type"].value_counts().head(10).index.tolist()
    sub = sub[sub["model_type"].isin(top_cells)]
    xt = pd.crosstab(sub["model_type"], sub["cargo"]).loc[top_cells]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(xt, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Sample size: cell line × cargo (primary subset)")
    ax.set_xlabel("Cargo")
    ax.set_ylabel("Cell line / model")
    fig.tight_layout()
    return str(_save(fig, "12_cell_x_cargo_heatmap"))


def main() -> None:
    df = load_processed()
    _print_section("Loaded clean dataset")
    print(f"  shape: {df.shape}")

    primary = df[df["experiment_method"] == PRIMARY_METHOD]
    _print_section(f"Primary modeling subset: {PRIMARY_METHOD}")
    print(f"  n = {len(primary):,}")
    print(f"  unique ionizable lipids: {primary['il_name'].nunique():,}")
    print(f"  unique publications: {primary['publication_pmid'].nunique()}")

    _print_section("Generating figures")
    paths = []
    paths.append(fig_target_distribution_overall(df))
    paths.append(fig_target_by_method(df))
    paths.append(fig_target_by_cargo(df))
    paths.append(fig_target_by_cell_line(df))
    paths.append(fig_mol_percent_distributions(df))
    paths.append(fig_ionizable_lipid_frequency(df))
    paths.append(fig_helper_pegs(df))
    paths.append(fig_descriptor_correlation_heatmap(df))
    p, rho = fig_descriptor_vs_target(df)
    paths.append(p)
    paths.append(fig_publication_sample_size(df))
    paths.append(fig_publication_target_variation(df))
    paths.append(fig_cell_x_cargo_heatmap(df))

    for p in paths:
        print(f"  -> {Path(p).relative_to(REPO_ROOT)}")

    _print_section("Top RDKit descriptors by |Spearman ρ| with target")
    print(rho.reindex(rho.abs().sort_values(ascending=False).index).to_string())

    # Write structured summary.
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# EDA summary — LNPDB cleaned dataset\n")
    lines.append(f"- Cleaned dataset: **{len(df):,} LNPs × {df.shape[1]} columns**")
    lines.append(f"- Primary modeling subset (`{PRIMARY_METHOD}`): **n = {len(primary):,}**")
    lines.append(f"  - Unique ionizable lipids: {primary['il_name'].nunique():,}")
    lines.append(f"  - Unique publications: {primary['publication_pmid'].nunique()}")
    lines.append(f"  - Cargo split: " + ", ".join(
        f"{k} {v:,}" for k, v in primary["cargo"].value_counts().items()
    ))
    lines.append("")
    lines.append("## Descriptor–target rank correlations (Spearman)\n")
    lines.append("| Descriptor | ρ |")
    lines.append("|---|---:|")
    for name, val in rho.reindex(rho.abs().sort_values(ascending=False).index).items():
        lines.append(f"| `{name}` | {val:+.3f} |")
    lines.append("")
    lines.append("## Figures\n")
    for p in paths:
        rel = Path(p).relative_to(REPO_ROOT)
        lines.append(f"- ![{Path(p).stem}]({rel})")
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"\nWrote summary -> {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
