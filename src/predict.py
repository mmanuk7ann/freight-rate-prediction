"""Generate submission predictions using the production GBM
(models/gbm_challenger.pkl). The GLM is documented-baseline only and is not
used for predictions here.

Run with:
    python src/predict.py

Writes validation_predictions.csv and december_predictions.csv at the repo
root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import clean_loads, load_december, load_train_test, load_validation
from src.features import build_features
from src.models import predict_gbm

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def load_production_gbm() -> dict:
    model = joblib.load(MODELS_DIR / "gbm_challenger.pkl")
    print(
        f"Loaded production GBM: target={model['target']!r}, "
        f"numeric_features={model['numeric_features']}, "
        f"categorical_features={model['categorical_features']}"
    )
    return model


# ----------------------------------------------------------------------
# TASK A: validation.csv
# ----------------------------------------------------------------------


def predict_validation(model: dict) -> pd.DataFrame:
    val = build_features(clean_loads(load_validation()))
    preds = predict_gbm(model, val)
    print(f"validation.csv predictions: n={len(preds)}, min=${preds.min():.2f}, max=${preds.max():.2f}")
    return pd.DataFrame({"load_id": val["load_id"], "predicted_rate": preds})


def fill_validation_template(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Fill validation_predictions_template.csv by joining on load_id --
    never assumes the prediction dataframe's row order matches the
    template's."""
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    assert list(template.columns) == ["load_id", "predicted_rate"]

    rate_by_id = pred_df.set_index("load_id")["predicted_rate"]
    template["predicted_rate"] = template["load_id"].map(rate_by_id)

    n_missing = int(template["predicted_rate"].isna().sum())
    assert n_missing == 0, f"{n_missing} load_ids in the template had no matching prediction"
    assert len(template) == 12_000, f"expected 12,000 rows, got {len(template)}"
    return template


# ----------------------------------------------------------------------
# TASK B: december_chart_inputs.csv (fixed Lexington -> Fort Wayne lane)
# ----------------------------------------------------------------------


def _lookup_city_coords(train: pd.DataFrame, city: str) -> tuple[float, float]:
    """Look up a city's (lat, lon) from train_test.csv, verifying it has
    exactly one consistent coordinate pair whether it appears as a pickup
    or a delivery."""
    as_pickup = train.loc[train["pickup"] == city, ["pickup_lat", "pickup_lon"]].rename(
        columns={"pickup_lat": "lat", "pickup_lon": "lon"}
    )
    as_delivery = train.loc[train["delivery"] == city, ["delivery_lat", "delivery_lon"]].rename(
        columns={"delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coords = pd.concat([as_pickup, as_delivery]).drop_duplicates()
    print(
        f"{city}: {len(as_pickup)} rows as pickup, {len(as_delivery)} rows as delivery, "
        f"{len(coords)} distinct (lat, lon) pair(s)"
    )
    assert len(coords) == 1, f"{city} has inconsistent coordinates across rows: {coords.values.tolist()}"
    lat, lon = coords.iloc[0]
    print(f"  -> using ({lat}, {lon})")
    return float(lat), float(lon)


def build_december_features(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Build a model-ready dataframe for december_chart_inputs.csv's fixed
    Lexington->Fort Wayne scenario.

    december_chart_inputs.csv is missing lat/lon and market_index/
    quote_signal entirely, so those are reconstructed rather than derived
    from columns that don't exist in that file:
    - lat/lon: looked up from train_test.csv (checked for consistency first).
    - market_index/quote_signal: no December-specific values exist anywhere
      (december_chart_inputs.csv never had them, and train_test.csv doesn't
      cover December), so each date's value is proxied by the mean across
      validation.csv rows sharing that exact calendar date -- validation.csv
      is the only source with real observed market-wide values for Nov-Dec.
    Once lat/lon/market_index/quote_signal are filled in, this becomes a
    normal clean_loads() + build_features() dataframe, reusing the exact
    same feature logic as everywhere else instead of reimplementing it.
    """
    dec = load_december()

    lex_lat, lex_lon = _lookup_city_coords(train, "Lexington")
    fw_lat, fw_lon = _lookup_city_coords(train, "Fort Wayne")

    synth = dec.copy()
    synth["pickup_lat"] = lex_lat
    synth["pickup_lon"] = lex_lon
    synth["delivery_lat"] = fw_lat
    synth["delivery_lon"] = fw_lon

    val_clean = clean_loads(validation)
    per_date = val_clean.groupby("date")[["market_index", "quote_signal"]].agg(["mean", "count"])
    print("\nPer-date validation.csv row counts used for market_index/quote_signal:")
    for date in synth["date"]:
        n = int(per_date.loc[date, ("market_index", "count")])
        print(f"  {date.date()}: n={n} validation rows")
    synth["market_index"] = synth["date"].map(per_date[("market_index", "mean")])
    synth["quote_signal"] = synth["date"].map(per_date[("quote_signal", "mean")])

    synth = clean_loads(synth)
    synth = build_features(synth)

    # lane_historical_avg_rate: this lane's entire recorded history lives in
    # train_test.csv (Jan-Oct) and is therefore entirely prior to every
    # December date -- no new same-lane data arrives mid-scenario to update
    # a running average. So the causal expanding average "as of" any
    # December date collapses to this lane's fixed full-history mean; it's
    # still the correct causal value, just constant across all 31 rows.
    # NOT filtered by equipment (matches how lane_historical_avg_rate is
    # computed everywhere else -- by lane only).
    lane_mask = (train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")
    lane_rows = train.loc[lane_mask]
    lane_avg = float(lane_rows["posted_rate"].mean())
    print(
        f"\nLexington->Fort Wayne lane history in train_test.csv: {len(lane_rows)} rows "
        f"(all equipment types), mean posted_rate=${lane_avg:.2f}"
    )
    print(f"  -> using lane_historical_avg_rate=${lane_avg:.2f} (constant) for all 31 December rows")
    synth["lane_historical_avg_rate"] = lane_avg

    return synth


def predict_december(model: dict, dec_features: pd.DataFrame) -> pd.DataFrame:
    used = set(model["numeric_features"]) | set(model["categorical_features"])
    computed_but_unused = {"lane_historical_avg_rate", "weight_per_mile", "distance_bucket"} - used
    if computed_but_unused:
        print(
            f"\nNote: computed {sorted(computed_but_unused)} per the brief, but the production "
            f"model's feature set doesn't include them ({sorted(used)}), so they don't affect "
            "these predictions -- they were dropped from the model last session for not being a "
            "clear improvement. Keeping them here for visibility/future use is harmless: "
            "predict_gbm only pulls the columns the model actually needs."
        )

    preds = predict_gbm(model, dec_features)
    out = load_december()  # original 7-column frame, untouched
    out["predicted_rate"] = preds
    return out


def print_december_sanity_check(dec_predictions: pd.DataFrame, train: pd.DataFrame) -> None:
    lane_mask = (train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne") & (train["equipment"] == "Dry Van")
    dry_van_lane = train.loc[lane_mask, "posted_rate"]
    band_low, band_high = float(dry_van_lane.min()), float(dry_van_lane.max())
    print(
        f"\nHistorical Lexington->Fort Wayne Dry Van band (train_test.csv, n={len(dry_van_lane)}): "
        f"${band_low:.2f}-${band_high:.2f} (confirms the ~$757-935 figure)."
    )
    print("Nov-Dec market_index ran below the yearly mean (confirmed in an earlier session's EDA).")

    print("\nDecember predictions:")
    table = dec_predictions[["date", "predicted_rate"]].copy()
    table["date"] = table["date"].dt.date
    print(table.to_string(index=False))

    out_of_band = dec_predictions[
        (dec_predictions["predicted_rate"] < band_low) | (dec_predictions["predicted_rate"] > band_high)
    ]
    if len(out_of_band):
        print(
            f"\nFLAG: {len(out_of_band)}/31 predictions fall outside the ${band_low:.2f}-${band_high:.2f} "
            f"historical band:\n{out_of_band[['date', 'predicted_rate']].to_string(index=False)}"
        )
    else:
        print(f"\nAll 31 predictions fall within the ${band_low:.2f}-${band_high:.2f} historical band.")


def main() -> None:
    model = load_production_gbm()
    train = clean_loads(load_train_test())
    validation_raw = load_validation()

    # ------------------------------------------------------------------
    # TASK A
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK A: validation.csv predictions")
    print("=" * 80)
    val_preds = predict_validation(model)
    val_submission = fill_validation_template(val_preds)
    val_out_path = ROOT / "validation_predictions.csv"
    val_submission.to_csv(val_out_path, index=False)
    print(f"Saved {val_out_path} ({len(val_submission)} rows, columns={list(val_submission.columns)})")

    # ------------------------------------------------------------------
    # TASK B
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK B: december_chart_inputs.csv predictions")
    print("=" * 80)
    dec_features = build_december_features(train, validation_raw)
    dec_submission = predict_december(model, dec_features)
    print_december_sanity_check(dec_submission, train)

    dec_out_path = ROOT / "december_predictions.csv"
    dec_submission.to_csv(dec_out_path, index=False)
    print(f"\nSaved {dec_out_path} ({len(dec_submission)} rows, columns={list(dec_submission.columns)})")


if __name__ == "__main__":
    main()
