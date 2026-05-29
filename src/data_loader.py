"""Data loader and cleaner for LNPDB.

Pipeline:
    load_lnpdb_raw() -> clean_lnpdb() -> save_processed() -> load_processed()

Cleaning policy is documented in `clean_lnpdb` and surfaced via a `CleaningReport`
so callers can audit exactly what was dropped and why.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_PARQUET = PROCESSED_DIR / "lnpdb_clean.parquet"


MOL_RATIO_COLS = [
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
    "fifthcomponent_molratio",
]


@dataclass
class CleaningReport:
    """Audit trail for clean_lnpdb. Counts are cumulative drops from the raw frame."""
    n_raw: int
    n_after_drop_na_experiment_block: int
    n_after_drop_null_target: int
    n_after_drop_mol_sum_outliers: int
    n_final: int
    dropped_na_experiment_block: int
    dropped_null_target: int
    dropped_mol_sum_outliers: int

    def summary(self) -> str:
        return (
            f"raw {self.n_raw} -> "
            f"-{self.dropped_na_experiment_block} NA-experiment rows -> "
            f"-{self.dropped_null_target} null target -> "
            f"-{self.dropped_mol_sum_outliers} mol%-sum outliers -> "
            f"final {self.n_final}"
        )


def load_lnpdb_raw(filename: str = "LNPDB.csv") -> pd.DataFrame:
    """Load the raw LNPDB CSV. NA / None / empty strings become NaN."""
    path = RAW_DIR / filename
    return pd.read_csv(path, na_values=["NA", "None", ""], low_memory=False)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case + replace dots so columns are valid Python identifiers."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]
    return df


def clean_lnpdb(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the project's cleaning policy.

    Drops (in order, cumulative on the result):
      1. Rows that are NaN across the experiment block — physical-characterization
         experiments (size, zeta, hemolysis) without a transfection measurement.
         Detected as: Cargo + Cargo_type + Model both NaN.
      2. Rows with null Experiment_value (no target to train on).
      3. Rows whose mol-ratio components don't sum to ~100 (data-entry artifacts).

    Returns the cleaned frame and a CleaningReport describing what was removed.
    """
    n_raw = len(df)

    # 1. Drop rows that are NaN across the entire experiment block.
    experiment_block_na = df["Cargo"].isna() & df["Cargo_type"].isna() & df["Model"].isna()
    df = df.loc[~experiment_block_na].copy()
    n_after_drop_na_experiment_block = len(df)

    # 2. Drop rows with no target.
    df = df.loc[df["Experiment_value"].notna()].copy()
    n_after_drop_null_target = len(df)

    # 3. Drop rows whose mol-ratio components don't sum to ~100.
    mol_sum = df[MOL_RATIO_COLS].sum(axis=1, skipna=True)
    df = df.loc[np.isclose(mol_sum, 100.0)].copy()
    n_after_drop_mol_sum_outliers = len(df)

    # Standardize column naming once the row set is final.
    df = _standardize_columns(df)
    df = df.reset_index(drop=True)

    report = CleaningReport(
        n_raw=n_raw,
        n_after_drop_na_experiment_block=n_after_drop_na_experiment_block,
        n_after_drop_null_target=n_after_drop_null_target,
        n_after_drop_mol_sum_outliers=n_after_drop_mol_sum_outliers,
        n_final=len(df),
        dropped_na_experiment_block=n_raw - n_after_drop_na_experiment_block,
        dropped_null_target=n_after_drop_na_experiment_block - n_after_drop_null_target,
        dropped_mol_sum_outliers=n_after_drop_null_target - n_after_drop_mol_sum_outliers,
    )
    return df, report


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> Path:
    """Write the cleaned frame to parquet. Creates the parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_processed(path: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    """Load the cleaned dataset (produced by save_processed)."""
    return pd.read_parquet(path)


if __name__ == "__main__":
    raw = load_lnpdb_raw()
    clean, report = clean_lnpdb(raw)
    out_path = save_processed(clean)
    print(report.summary())
    print(f"saved -> {out_path}")
