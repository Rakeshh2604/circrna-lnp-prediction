"""Data loader for LNPDB dataset."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_lnpdb_raw(filename: str = "lnpdb_full.csv") -> pd.DataFrame:
    """Load the raw LNPDB CSV. Replace filename when the actual download is confirmed."""
    return pd.read_csv(RAW_DIR / filename)


def clean_lnpdb(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, types, and drop duplicates."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    return df


def load_processed() -> pd.DataFrame:
    """Load the cleaned dataset (available after Day 3)."""
    return pd.read_parquet(PROCESSED_DIR / "lnpdb_clean.parquet")
