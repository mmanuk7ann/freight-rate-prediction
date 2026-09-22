"""Leakage-safe, time-based cross-validation for the freight rate dataset.

Random/shuffled CV would leak future rate levels (e.g. the market_index
seasonal arc, or trend drift) into training folds and overstate performance.
Instead we use forward-chaining/expanding-window splits: each fold trains on
everything before a cutoff date and validates on a held-out window right
after it, mirroring the real train_test -> validation time gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_based_folds(df: pd.DataFrame, n_folds: int = 3, val_window_weeks: int = 4) -> list[dict]:
    """Build `n_folds` expanding-window (forward-chaining) train/val splits.

    Fold i trains on all rows with date < val_start_i and validates on rows
    with val_start_i <= date <= val_end_i, where the val windows are
    `val_window_weeks` long and tile backward from the dataset's max date
    (no gap, no overlap). With the default n_folds=3, val_window_weeks=4 on
    train_test.csv (Jan-Oct 2025), cutoffs land at roughly early Aug / early
    Sep / early Oct, each holding out a 4-week window - directly mimicking
    the real ~1-month gap to validation.csv's Nov-Dec period.

    No shuffling anywhere: dates are used as-is, so training data for fold i
    is always strictly earlier than its validation window.

    Returns a list of dicts, one per fold, in chronological order:
    {"fold": int, "val_start": Timestamp, "val_end": Timestamp,
     "train_idx": np.ndarray, "val_idx": np.ndarray}
    """
    dates = pd.to_datetime(df["date"])
    max_date = dates.max()
    val_window = pd.Timedelta(weeks=val_window_weeks)

    # cutoffs[i] = start of fold i's validation window; cutoffs[n_folds] is a
    # sentinel one day past max_date so the final fold's window includes it.
    cutoffs = [max_date - (n_folds - i) * val_window for i in range(n_folds)]
    cutoffs.append(max_date + pd.Timedelta(days=1))

    folds = []
    for i in range(n_folds):
        val_start, val_end = cutoffs[i], cutoffs[i + 1]
        train_mask = dates < val_start
        val_mask = (dates >= val_start) & (dates < val_end)
        folds.append(
            {
                "fold": i,
                "val_start": val_start,
                "val_end": val_end - pd.Timedelta(days=1),  # inclusive, human-readable
                "train_idx": df.index[train_mask].to_numpy(),
                "val_idx": df.index[val_mask].to_numpy(),
            }
        )
    return folds


def assert_no_leakage(train_idx: np.ndarray, val_idx: np.ndarray, dates: pd.Series) -> None:
    """Raise AssertionError if the split shows any sign of temporal leakage.

    Checks: no index shared between train/val, no calendar date shared
    between train/val, and every train date strictly precedes every val date.
    Fails loudly (rather than warning) because a silent leak here would
    invalidate any CV score computed from these folds.
    """
    assert not (set(train_idx) & set(val_idx)), "Leakage: train_idx and val_idx share row indices"

    train_dates = pd.to_datetime(dates.loc[train_idx])
    val_dates = pd.to_datetime(dates.loc[val_idx])

    overlap = set(train_dates.unique()) & set(val_dates.unique())
    assert not overlap, f"Leakage: dates present in both train and val: {sorted(overlap)}"

    assert train_dates.max() < val_dates.min(), (
        f"Leakage: latest train date {train_dates.max().date()} is not before "
        f"earliest val date {val_dates.min().date()}"
    )
