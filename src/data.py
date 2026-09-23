"""Loading and cleaning helpers for the freight rate prediction dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the 48,000-row labeled training set (2025-01-01 to 2025-10-31)."""
    df = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    return df


def load_validation(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the 12,000-row unlabeled out-of-time validation set (2025-11-01 to 2025-12-31)."""
    df = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    return df


def load_december(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the 31-row fixed-lane December chart input set (Lexington -> Fort Wayne).

    This file intentionally has only 7 columns (no lat/lon, market_index, or
    quote_signal) and is not run through clean_loads — it is a fixed scenario
    used only to render the candidate_december.png chart via score.py.
    """
    df = pd.read_csv(data_dir / "december_chart_inputs.csv", parse_dates=["date"])
    return df


def clean_loads(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a train_test/validation-shaped loads dataframe.

    Decisions and justification:
    - weight sign-flip: ~292 negative values in train / ~145 in validation have
      abs(value) distributions matching the positive-weight population (same
      5000-47500 range, similar mean/std). This points to a sign-flip artifact
      (e.g. an upstream subtraction or unit-conversion bug) rather than a
      distinct erroneous population, so we take .abs() instead of dropping
      these rows and losing ~0.6-1.2% of records.
    - missing weight / market_index (~0.6% and ~0.8-2% respectively): imputed
      with the column median (robust to outliers, avoids row loss) after the
      sign-flip fix. A boolean flag column is added for each so any
      downstream model can still learn from "was this value imputed" if that
      carries signal, rather than silently hiding the missingness.
    - Everything else (pickup/delivery, lat/lon, distance, equipment, date,
      quote_signal, posted_rate if present) is left untouched: no known data
      quality issues were confirmed for these columns during EDA.
    """
    out = df.copy()

    out["weight_was_missing"] = out["weight"].isna()
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out["weight"].median())

    out["market_index_was_missing"] = out["market_index"].isna()
    out["market_index"] = out["market_index"].fillna(out["market_index"].median())

    return out
