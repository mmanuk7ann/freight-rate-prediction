"""Feature engineering for the freight rate prediction dataset.

Call build_features() on an already-cleaned dataframe (i.e. the output of
src.data.clean_loads). Nothing here imputes or fixes data quality issues —
that stays in clean_loads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8

# Equipment is intentionally left untouched by build_features. It has 3
# categories (Dry Van, Flatbed, Reefer) with a confirmed $/mile ordering
# (Dry Van < Flatbed < Reefer), so downstream it should be either one-hot
# encoded or passed as a native categorical dtype (e.g. LightGBM/CatBoost);
# an ordinal encoding following the $/mile ordering is also defensible given
# the confirmed monotonic relationship, but that's a modeling-stage decision.


def _haversine_miles(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in miles, independent of any road-network routing
    baked into the provided `distance` column (useful as a cross-check /
    circuity signal since it correlates with but isn't identical to `distance`).
    """
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Initial compass bearing (0-360 deg) of the route; cheap to compute from
    the same lat/lon inputs and may capture directional rate effects (e.g.
    backhaul-heavy corridors) that haversine distance alone can't.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def _us_thanksgiving(year: int) -> pd.Timestamp:
    """4th Thursday of November."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    return first_thursday + pd.Timedelta(weeks=3)


def _us_christmas(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=12, day=25)


def _near_holiday(dates: pd.Series, window_days: int = 3) -> pd.Series:
    """True if a date falls within `window_days` of US Thanksgiving or Christmas.

    Relevant because validation.csv covers Nov-Dec, when holiday-adjacent
    shipping patterns (pre-holiday rushes, post-holiday lulls) can shift
    rates independent of distance/equipment.
    """
    years = dates.dt.year.unique()
    holidays: list[pd.Timestamp] = []
    for year in years:
        holidays.append(_us_thanksgiving(int(year)))
        holidays.append(_us_christmas(int(year)))
    holidays = pd.to_datetime(holidays)

    # vectorized min-abs-distance to nearest holiday, in days
    diffs = np.abs(dates.values[:, None] - holidays.values[None, :]) / np.timedelta64(1, "D")
    return pd.Series(diffs.min(axis=1) <= window_days, index=dates.index)


# Fixed distance-bucket cutoffs (not re-derived per call, so they apply
# identically and without leakage to any future data). Chosen as the
# train_test.csv distance terciles (~688mi / ~1324mi), rounded, so the three
# buckets are roughly balanced on this dataset rather than using generic
# industry short/medium/long-haul cutoffs (e.g. 250mi/750mi), which would
# dump most of this particular dataset (median distance ~953mi) into a
# single "long_haul" bucket and not actually split anything.
DISTANCE_BUCKET_EDGES = [0, 700, 1300, np.inf]
DISTANCE_BUCKET_LABELS = ["short_haul", "medium_haul", "long_haul"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready features to a cleaned loads dataframe. Does not mutate input.

    Adds:
    - haversine_distance_mi: great-circle pickup->delivery distance (see rationale above).
      Deliberately does NOT use pickup/delivery city names as features: 8 cities in
      validation.csv never appear in train_test.csv, so any city-identity encoding
      (one-hot, target encoding, etc.) would have no learned signal for them at
      inference time. lat/lon-derived features generalize to unseen cities instead.
    - bearing_deg: initial compass bearing of the route (see rationale above).
    - day_of_week: Monday=0..Sunday=6; captures weekday/weekend posting patterns.
    - month: calendar month (1-12); coarse seasonality signal, redundant with but
      cheaper than the sin/cos pair below for tree-based models that split on it directly.
    - day_of_year_sin / day_of_year_cos: cyclical encoding of day-of-year so Dec 31
      and Jan 1 are treated as adjacent rather than maximally far apart.
    - near_holiday: see _near_holiday rationale above.
    - weight_per_mile: weight / distance -- a density proxy (heavy-for-its-distance
      vs light-for-its-distance) distinct from either raw feature alone.
    - distance_bucket: short/medium/long haul categorical (see DISTANCE_BUCKET_EDGES).
      Gives a tree an explicit, cheap split point for haul-length x equipment or
      haul-length x season interactions that it could in principle rediscover from
      continuous `distance` alone, but only by spending extra splits/depth to do so.

    EXPERIMENTAL, TESTED, NOT ADOPTED: weight_per_mile and distance_bucket were
    added to the GBM's feature set in src/models.py, tuned, and compared against
    the production configuration across all 3 CV folds -- the resulting model
    improved mean MAE but wasn't better on every fold, so it wasn't promoted
    (see src/models.py's NEW_GBM_NUMERIC_FEATURES comment and
    reports/model_comparison.md). Still computed here since they're harmless,
    generically useful columns and other code (src/models.py's NEW_GBM_* path)
    still exercises them for that comparison's reproducibility.

    equipment is left as-is; see the module-level comment for how to encode it.
    """
    out = df.copy()

    out["haversine_distance_mi"] = _haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["bearing_deg"] = _bearing_deg(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )

    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    day_of_year_frac = out["date"].dt.dayofyear / 365.25
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year_frac)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year_frac)

    out["near_holiday"] = _near_holiday(out["date"])

    out["weight_per_mile"] = out["weight"] / out["distance"]
    out["distance_bucket"] = pd.cut(
        out["distance"], bins=DISTANCE_BUCKET_EDGES, labels=DISTANCE_BUCKET_LABELS
    ).astype(str)

    return out
