"""Baseline interpretable Gamma GLM (log link) for posted_rate.

Gamma/log is used instead of OLS because posted_rate is strictly positive
and right-skewed: the log link guarantees positive predictions and models
multiplicative (percentage) effects, and the Gamma variance function
(variance proportional to mean^2) fits a target whose spread grows with its
level, unlike OLS's constant-variance assumption.

Run directly to fit/evaluate the baseline model end-to-end:
    python src/models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import clean_loads, load_train_test
from src.features import build_features
from src.validation import assert_no_leakage, time_based_folds

MODELS_DIR = ROOT / "models"

# market_index/quote_signal are included even though EDA found weak (<0.1)
# correlation with the distance-detrended residual -- that's an expected,
# reportable finding, not a reason to drop them from the baseline.
#
# near_holiday is deliberately excluded here: train_test.csv only spans
# Jan-Oct, so the feature is constant (always False) in every training
# slice we fit on. A constant column is perfectly collinear with the
# intercept, which makes the GLM design matrix singular and its coefficient
# unidentifiable (statsmodels warns and returns a numerically meaningless
# near-zero estimate). It contributes zero information in-sample either
# way, so dropping it here avoids the warning/garbage coefficient without
# changing predictions. Revisit for the GBM challenger, which can hold a
# constant feature without erroring (it just never splits on it).
_BASE_TERMS = "market_index + quote_signal + day_of_year_sin + day_of_year_cos"
FORMULA_NO_INTERACTION = f"posted_rate ~ distance + C(equipment) + {_BASE_TERMS}"
FORMULA_INTERACTION = f"posted_rate ~ distance * C(equipment) + {_BASE_TERMS}"


def _fit_gamma_glm(formula: str, df: pd.DataFrame):
    return smf.glm(formula, data=df, family=sm.families.Gamma(link=sm.families.links.Log())).fit()


def fit_baseline_glm(train_df: pd.DataFrame, verbose: bool = False):
    """Fit the baseline Gamma GLM (log link) on train_df.

    Fits both `distance + equipment` (equipment only shifts the intercept,
    i.e. parallel $/mile lines) and `distance * equipment` (equipment also
    scales the $/mile slope), compares them with a likelihood-ratio test,
    and returns whichever the data supports: the interaction model if it's
    a significant improvement (p < 0.05), otherwise the simpler additive
    model, since a non-significant interaction just adds noise to the
    interpretable coefficients we're reporting.
    """
    reduced = _fit_gamma_glm(FORMULA_NO_INTERACTION, train_df)
    full = _fit_gamma_glm(FORMULA_INTERACTION, train_df)

    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    keep_interaction = p_value < 0.05

    if verbose:
        verdict = "keeping distance:equipment interaction" if keep_interaction else "dropping interaction (not significant)"
        print(f"  LR test distance:equipment -> stat={lr_stat:.2f}, df={df_diff:.0f}, p={p_value:.4g} -> {verdict}")

    return full if keep_interaction else reduced


def predict_glm(model, df: pd.DataFrame) -> np.ndarray:
    """Predict posted_rate (response scale, not link scale) for new rows."""
    return np.asarray(model.predict(df))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
    }


def main() -> None:
    train = build_features(clean_loads(load_train_test()))
    dates = train["date"]

    folds = time_based_folds(train, n_folds=3)

    print("Per-fold evaluation (expanding-window, forward-chaining, no shuffling):\n")
    fold_metrics = []
    fold_models = {}
    for fold in folds:
        assert_no_leakage(fold["train_idx"], fold["val_idx"], dates)
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]

        print(f"Fold {fold['fold']} (train n={len(train_slice):,}, val n={len(val_slice):,}):")
        model = fit_baseline_glm(train_slice, verbose=True)
        fold_models[fold["fold"]] = model

        preds = predict_glm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        fold_metrics.append(metrics)
        print(f"  MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%\n")

    metrics_df = pd.DataFrame(fold_metrics, index=[f"fold_{f['fold']}" for f in folds])
    print("=" * 80)
    print("Per-fold metrics:")
    print("=" * 80)
    print(metrics_df.round(3).to_string())
    print("\nMean ± std across folds:")
    summary = pd.DataFrame({"mean": metrics_df.mean(), "std": metrics_df.std()}).round(3)
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("Fold 2 coefficient summary (train slice closest to the real train->validation gap):")
    print("=" * 80)
    print(fold_models[2].summary())

    print("\n" + "=" * 80)
    print("Fitting final production model on the full train_test.csv (48,000 rows)")
    print("=" * 80)
    final_model = fit_baseline_glm(train, verbose=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "baseline_glm.pkl"
    joblib.dump(final_model, out_path)
    print(f"Saved final model to {out_path}")


if __name__ == "__main__":
    main()
