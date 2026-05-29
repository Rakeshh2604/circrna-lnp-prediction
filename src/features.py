"""Feature engineering for the LNPDB transfection-prediction model.

Design rationale lives in `writeup/report.md` Section 2; the short version:

- Ionizable lipid identity is unusable as a categorical (84% are singletons) — we
  represent IL via its 17 pre-computed RDKit descriptors instead.
- Helper, cholesterol, and PEG-lipid identities are low-cardinality and one-hottable,
  but cholesterol is 99.8% standard cholesterol so we drop it. PEG-lipid has a long
  tail of rare lipids that we pool into `(other)`.
- We exclude `dose_ug_nucleicacid`, `publication_pmid`, and `cargo_type` because they
  are study-identifying proxies that would defeat the leave-one-publication-out test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
X_PATH = PROCESSED_DIR / "X.parquet"
Y_PATH = PROCESSED_DIR / "y.parquet"
META_PATH = PROCESSED_DIR / "feature_meta.parquet"

PRIMARY_METHOD = "luminescence_normalized"

# Numeric: composition mol%, IL chemistry descriptors, formulation parameters.
NUMERIC_MOL_PCT = ["il_molratio", "hl_molratio", "chl_molratio", "peg_molratio"]
NUMERIC_FORMULATION = ["il_to_nucleicacid_massratio"]
NUMERIC_IL_CHEMISTRY = [
    "heavy_atoms", "rings", "aromatic_rings", "rotatable_bonds",
    "van_der_waals_molecular_volume", "topological_polar_surface_area",
    "hydrogen_bond_donors", "hydrogen_bond_acceptors", "logp",
    "molar_refractivity", "fraction_sp3_carbons", "sp3_carbons",
    "nitrogen_count", "molecular_weight",
]
BINARY_IL_CHEMISTRY = ["has_ester", "has_carbonate", "has_disulfide"]

CATEGORICAL_DIRECT = ["hl_name", "cargo", "model", "model_type", "mixing_method", "aqueous_buffer"]
# peg_name is one-hot but with a long tail pooled into (other); see _collapse_peg.
KEEP_PEGS = {"DMG-PEG2000", "DMPE-PEG2000"}


@dataclass
class FeatureSpec:
    """Provenance: which engineered columns come from which raw groups.

    Useful at SHAP time to aggregate importance by group, and at LOPO time to
    confirm no publication-leaking columns slipped in.
    """
    target_method: str
    n_rows: int
    n_features: int
    groups: dict[str, list[str]] = field(default_factory=dict)
    dropped_for_leakage: list[str] = field(default_factory=list)


def _collapse_peg(s: pd.Series) -> pd.Series:
    out = s.fillna("(none)").astype(str).copy()
    mask = ~out.isin(KEEP_PEGS) & (out != "(none)")
    out.loc[mask] = "(other)"
    return out


def build_features(
    df: pd.DataFrame,
    target_method: str = PRIMARY_METHOD,
) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """Build (X, y, spec) for the primary modeling subset.

    Parameters
    ----------
    df : DataFrame
        Cleaned LNPDB frame (as produced by `src.data_loader.clean_lnpdb`).
    target_method : str
        Filter applied to `experiment_method`. Defaults to `luminescence_normalized`.
    """
    sub = df[df["experiment_method"] == target_method].copy()
    y = sub["experiment_value"].astype("float64").reset_index(drop=True)
    y.name = "experiment_value"

    parts: list[pd.DataFrame] = []
    groups: dict[str, list[str]] = {}

    # 1. Composition (mol%)
    comp = sub[NUMERIC_MOL_PCT].astype("float64").reset_index(drop=True)
    parts.append(comp)
    groups["composition"] = NUMERIC_MOL_PCT.copy()

    # 2. Formulation numerics
    form = sub[NUMERIC_FORMULATION].astype("float64").reset_index(drop=True)
    parts.append(form)
    groups["formulation_numeric"] = NUMERIC_FORMULATION.copy()

    # 3. IL chemistry: numeric + binary
    chem_num = sub[NUMERIC_IL_CHEMISTRY].astype("float64").reset_index(drop=True)
    parts.append(chem_num)
    groups["il_chemistry_numeric"] = NUMERIC_IL_CHEMISTRY.copy()

    chem_bin = sub[BINARY_IL_CHEMISTRY].astype("int8").reset_index(drop=True)
    parts.append(chem_bin)
    groups["il_chemistry_binary"] = BINARY_IL_CHEMISTRY.copy()

    # 4. Direct one-hot categoricals
    for col in CATEGORICAL_DIRECT:
        s = sub[col].fillna("(none)").astype(str).reset_index(drop=True)
        oh = pd.get_dummies(s, prefix=col, dtype="int8")
        parts.append(oh)
        groups[col] = oh.columns.tolist()

    # 5. PEG (long tail collapsed)
    peg = _collapse_peg(sub["peg_name"]).reset_index(drop=True)
    peg_oh = pd.get_dummies(peg, prefix="peg_name", dtype="int8")
    parts.append(peg_oh)
    groups["peg_name"] = peg_oh.columns.tolist()

    X = pd.concat(parts, axis=1)

    # Sanity: drop fully-constant columns (zero variance) — these can appear after
    # one-hot if the cleaned frame happens to have only one value in a group.
    constants = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constants:
        X = X.drop(columns=constants)
        for grp, cols in groups.items():
            groups[grp] = [c for c in cols if c not in constants]

    spec = FeatureSpec(
        target_method=target_method,
        n_rows=len(X),
        n_features=X.shape[1],
        groups=groups,
        dropped_for_leakage=[
            "dose_ug_nucleicacid",
            "publication_pmid",
            "publication_link",
            "cargo_type",
            "chl_name",
            "experiment_id",
            "formulation_id",
            "lnp_id",
        ],
    )
    return X, y, spec


def save_features(X: pd.DataFrame, y: pd.Series, spec: FeatureSpec) -> tuple[Path, Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(X_PATH, index=False)
    y.to_frame().to_parquet(Y_PATH, index=False)
    meta_rows = []
    for grp, cols in spec.groups.items():
        for c in cols:
            meta_rows.append({"feature": c, "group": grp})
    pd.DataFrame(meta_rows).to_parquet(META_PATH, index=False)
    return X_PATH, Y_PATH, META_PATH


def load_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_parquet(X_PATH)
    y = pd.read_parquet(Y_PATH).iloc[:, 0]
    meta = pd.read_parquet(META_PATH)
    return X, y, meta


if __name__ == "__main__":
    from src.data_loader import load_processed

    df = load_processed()
    X, y, spec = build_features(df)
    save_features(X, y, spec)
    print(f"target_method: {spec.target_method}")
    print(f"n_rows: {spec.n_rows:,}")
    print(f"n_features: {spec.n_features}")
    print("\nFeature groups:")
    for grp, cols in spec.groups.items():
        print(f"  {grp:25s}  {len(cols):3d} cols")
    print("\nNA check on X:", X.isna().sum().sum())
    print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"\nSaved:\n  {X_PATH.relative_to(REPO_ROOT)}\n  {Y_PATH.relative_to(REPO_ROOT)}\n  {META_PATH.relative_to(REPO_ROOT)}")
