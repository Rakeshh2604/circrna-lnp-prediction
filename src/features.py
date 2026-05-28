"""Feature engineering for LNPDB.

Fleshed out on Day 6 once the cleaned schema is known.
"""
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, keep numeric mole fractions, optionally add RDKit descriptors."""
    raise NotImplementedError("Populate after Day 3 cleaning + Day 6 feature design.")
