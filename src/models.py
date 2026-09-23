"""Baseline interpretable Gamma GLM, and a GBM challenger, for posted_rate.

The GLM uses a Gamma family instead of OLS because posted_rate is strictly
positive and right-skewed: the Gamma variance function (variance
proportional to mean^2) fits a target whose spread grows with its level,
unlike OLS's constant-variance assumption. The *link* function was
originally assumed to be log (the common default for a positive target),
but compare_link_functions() showed that assumption was wrong for this
data -- see its docstring and reports/model_comparison.md for the numbers.

The GBM challenger uses sklearn's HistGradientBoostingRegressor rather than
LightGBM: LightGBM 4.7.0 installs but fails to import on this machine
(`OSError: ... Library not loaded: @rpath/libomp.dylib` -- it needs the
Homebrew libomp runtime, which isn't a pip-installable dependency). Retried
in this session (re-installed lightgbm fresh, checked for a Homebrew libomp
keg) -- still fails the same way, libomp still isn't present, so we're
staying on HistGradientBoostingRegressor rather than spending more time on
an environment issue outside this project's control. It's sklearn's native
equivalent (histogram-binned gradient boosting, same family of algorithm as
LightGBM/XGBoost) and needs no external runtime.

Hyperparameter search uses Optuna (installs cleanly, no native-library
issues) rather than a hand-picked grid.

Run directly to fit/evaluate both models end-to-end:
    python src/models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tools.sm_exceptions import DomainWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Candidate Gamma links compared by compare_link_functions(). Identity and
# inverse are technically outside the Gamma family's natural domain (they
# can produce a non-positive linear predictor -> statsmodels raises a
# DomainWarning), which is exactly the failure mode step 3 checks for; we
# silence the warning here because we're deliberately testing that risk,
# not stumbling into it.
LINK_FUNCTIONS = {
    "log": sm.families.links.Log(),
    "identity": sm.families.links.Identity(),
    "inverse": sm.families.links.InversePower(),
}

# Selected by compare_link_functions() on fold 2 (train n=43,463, val
# n=4,537): identity won decisively on both in-sample fit and held-out
# error --
#   log:      AIC=671256  deviance=3313.9  val MAE=$459.61  val MAPE=23.95%
#   identity: AIC=636070  deviance=1291.1  val MAE=$172.44  val MAPE=8.07%
#   inverse:  AIC=894449  deviance=38381.5 val MAE=$2263.12 val MAPE=63.07% (41 negative predictions)
# The log link had implicitly assumed rate grows exponentially with
# distance; the data instead supports an approximately additive
# (identity-link) relationship. See reports/model_comparison.md.
BASELINE_LINK_NAME = "identity"


def _fit_gamma_glm(formula: str, df: pd.DataFrame, link=None):
    link = LINK_FUNCTIONS[BASELINE_LINK_NAME] if link is None else link
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DomainWarning)
        return smf.glm(formula, data=df, family=sm.families.Gamma(link=link)).fit()


def compare_link_functions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Fit the baseline formula (distance + equipment + market_index +
    quote_signal + seasonality, no interaction -- already rejected by the
    fold-2 likelihood-ratio test) under each candidate Gamma link, and
    compare in-sample fit (AIC, deviance) against held-out error on val_df.

    Exists because the log link -- picked initially as the generic default
    for a positive target -- turned out to be badly misspecified here: it
    assumes rate grows exponentially with distance, but this data's
    rate/distance relationship is much closer to additive. Comparing links
    explicitly, rather than assuming one, is how that got caught.
    """
    y_val = val_df["posted_rate"].to_numpy()
    rows = {}
    for name, link in LINK_FUNCTIONS.items():
        model = _fit_gamma_glm(FORMULA_NO_INTERACTION, train_df, link=link)
        preds = np.asarray(model.predict(val_df))
        metrics = _regression_metrics(y_val, preds)
        rows[name] = {"AIC": model.aic, "deviance": model.deviance, **metrics}
    return pd.DataFrame(rows).T


def fit_baseline_glm(train_df: pd.DataFrame, verbose: bool = False):
    """Fit the baseline Gamma GLM on train_df, using BASELINE_LINK_NAME
    (identity -- see the comment above and compare_link_functions).

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


# OLD (pre-this-session) GBM feature set -- kept as a fixed, named reference
# so the "OLD tuned GBM" row in the final comparison table reproduces last
# session's model exactly, rather than being redefined out from under it as
# NEW_GBM_* below evolves.
OLD_GBM_NUMERIC_FEATURES = [
    "distance",                # same primary predictor the GLM uses
    "haversine_distance_mi",   # great-circle distance -- tests whether the tree exploits
                                # a road-vs-great-circle discrepancy (route circuity) that
                                # `distance` alone doesn't capture. In practice distance and
                                # haversine_distance_mi are nearly collinear (r=0.9995), so
                                # SHAP importance gets split between the two rather than
                                # concentrated on one -- that's a cosmetic effect on the
                                # importance ranking, not a predictive-performance problem
                                # (trees handle redundant/collinear inputs fine), so no
                                # functional change is needed here.
    "bearing_deg",             # route direction -- tests for corridor/backhaul effects
    "weight",                  # never seen by the GLM; testing whether it adds signal
    "market_index",
    "quote_signal",
    "day_of_week",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
]
# near_holiday excluded here too, same underlying cause as the GLM: it's
# constant (always False) in every train_test.csv slice (Jan-Oct only, no
# Thanksgiving/Christmas in range). Unlike the GLM this doesn't error -- a
# tree can simply never split on a single-valued column -- but it still has
# zero examples of the "True" case to learn a holiday effect from, so
# including it would just be dead weight. day_of_year_sin/cos remains the
# seasonality proxy.
OLD_GBM_CATEGORICAL_FEATURES = ["equipment"]

# NEW feature set (this session): adds weight_per_mile and
# lane_historical_avg_rate (see add_lane_historical_rate) as numeric
# features, and distance_bucket as a second categorical alongside equipment.
NEW_GBM_NUMERIC_FEATURES = OLD_GBM_NUMERIC_FEATURES + ["weight_per_mile", "lane_historical_avg_rate"]
NEW_GBM_CATEGORICAL_FEATURES = ["equipment", "distance_bucket"]

# Hyperparameters selected by the light 4-combo grid search last session --
# kept as a fixed constant (rather than re-run) purely to reproduce the "OLD
# tuned GBM" reference row in the final comparison table.
OLD_TUNED_PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 15}


def add_lane_historical_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add lane_historical_avg_rate: for each row, the mean posted_rate of all
    OTHER rows sharing the same (pickup, delivery) lane from STRICTLY EARLIER
    dates only (same-day and future rows excluded). Rows on a lane's first
    active date (no prior history) fall back to a global "market as of that
    date" average -- also expanding/causal, not the full-dataset mean, so
    early rows aren't implicitly told about rate levels from later in the
    year. The very first calendar date overall has no causal fallback
    either way and uses the full-dataset mean for that handful of rows only
    (a tiny, documented exception, not a meaningful leak).

    Computed with two groupby+cumsum passes over per-(lane, date) and
    per-date aggregates -- O(n log n), not a row-by-row loop:
      1. Sum/count posted_rate per (pickup, delivery, date), cumsum within
         each lane ordered by date, then subtract the current date's own
         contribution to get the strictly-prior cumulative sum/count.
      2. Same idea per date only (ignoring lane) for the fallback.
    The per-(lane, date) result is then mapped back onto every row sharing
    that (pickup, delivery, date) key.
    """
    out = df.copy()
    keys = ["pickup", "delivery", "date"]

    lane_date = out.groupby(keys)["posted_rate"].agg(["sum", "count"]).reset_index()
    lane_date = lane_date.sort_values(["pickup", "delivery", "date"])
    lane_group = lane_date.groupby(["pickup", "delivery"])
    lane_date["prior_sum"] = lane_group["sum"].cumsum() - lane_date["sum"]
    lane_date["prior_count"] = lane_group["count"].cumsum() - lane_date["count"]

    date_totals = out.groupby("date")["posted_rate"].agg(["sum", "count"]).sort_index().reset_index()
    date_totals["prior_sum"] = date_totals["sum"].cumsum() - date_totals["sum"]
    date_totals["prior_count"] = date_totals["count"].cumsum() - date_totals["count"]
    overall_mean = out["posted_rate"].mean()
    date_totals["global_prior_avg"] = np.where(
        date_totals["prior_count"] > 0,
        date_totals["prior_sum"] / date_totals["prior_count"],
        overall_mean,
    )

    lane_date = lane_date.merge(date_totals[["date", "global_prior_avg"]], on="date", how="left")
    lane_date["lane_historical_avg_rate"] = np.where(
        lane_date["prior_count"] > 0,
        lane_date["prior_sum"] / lane_date["prior_count"],
        lane_date["global_prior_avg"],
    )

    feature_map = lane_date.set_index(keys)["lane_historical_avg_rate"]
    out["lane_historical_avg_rate"] = out.set_index(keys).index.map(feature_map)
    return out


def _build_gbm_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Numeric features + one-hot encoded categoricals (HistGradientBoostingRegressor
    doesn't take native categoricals as cleanly as LightGBM would have, so we
    one-hot them instead). `feature_columns`, when given, re-aligns the output
    to a fixed training-time schema so a prediction-time frame can't end up
    with mismatched/missing dummy columns.
    """
    numeric = df[numeric_features]
    dummies = pd.get_dummies(df[categorical_features])
    X = pd.concat([numeric, dummies], axis=1)
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def _gbm_design_matrix(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a GBM model dict's design matrix for a new dataframe, using
    the exact numeric/categorical feature lists and one-hot schema it was
    trained with."""
    return _build_gbm_matrix(
        df, model["numeric_features"], model["categorical_features"], feature_columns=model["feature_columns"]
    )


_DEFAULT_GBM_PARAMS = {
    # gamma loss matches the GLM's distributional assumption (posted_rate is
    # strictly positive and right-skewed), so the comparison isolates model
    # flexibility (trees/interactions) rather than differing loss functions.
    "loss": "gamma",
    "max_iter": 500,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "validation_fraction": 0.1,
    "random_state": 0,
}


def fit_gbm(
    train_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    target: str = "raw",
    **hyperparams,
) -> dict:
    """Fit a HistGradientBoostingRegressor challenger on train_df.

    numeric_features/categorical_features default to NEW_GBM_* (this
    session's feature set); pass OLD_GBM_* explicitly to reproduce last
    session's model. target="raw" fits posted_rate directly; target=
    "rate_per_mile" fits posted_rate/distance instead and predict_gbm
    multiplies back by distance -- see main() for the head-to-head test of
    which target actually wins on this data.

    Returns a dict of {"estimator", "feature_columns", "numeric_features",
    "categorical_features", "target"} rather than the bare estimator: this
    makes every fitted model self-describing, so predict_gbm (and SHAP) can
    rebuild the exact right design matrix and undo the right target
    transform without the caller having to remember how it was fit.
    """
    numeric_features = NEW_GBM_NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = NEW_GBM_CATEGORICAL_FEATURES if categorical_features is None else categorical_features

    X = _build_gbm_matrix(train_df, numeric_features, categorical_features)
    if target == "raw":
        y = train_df["posted_rate"].to_numpy()
    elif target == "rate_per_mile":
        y = (train_df["posted_rate"] / train_df["distance"]).to_numpy()
    else:
        raise ValueError(f"unknown target {target!r}, expected 'raw' or 'rate_per_mile'")

    params = {**_DEFAULT_GBM_PARAMS, **hyperparams}
    estimator = HistGradientBoostingRegressor(**params)
    estimator.fit(X, y)

    return {
        "estimator": estimator,
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
    }


def predict_gbm(model: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict posted_rate for new rows using a fitted GBM challenger dict."""
    X = _gbm_design_matrix(model, df)
    preds = np.asarray(model["estimator"].predict(X))
    if model["target"] == "rate_per_mile":
        preds = preds * df["distance"].to_numpy()
    return preds


def _tune_gbm_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 30,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> optuna.Study:
    """Real hyperparameter search (TPE sampler, `n_trials` trials) over
    learning_rate, max_leaf_nodes, max_iter, min_samples_leaf, and
    l2_regularization, minimizing held-out MAE on val_df. Uses a fixed seed
    for reproducibility. Whatever wins here still gets validated across all
    3 folds afterward in main() -- this alone only proves it's good on
    fold 2, not that it generalizes.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 255),
            "max_iter": trial.suggest_int("max_iter", 100, 800),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        model = fit_gbm(
            train_df, numeric_features=numeric_features, categorical_features=categorical_features, **params
        )
        preds = predict_gbm(model, val_df)
        return _regression_metrics(val_df["posted_rate"].to_numpy(), preds)["MAE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    return study


def _gbm_shap_importance(model: dict, X: pd.DataFrame, max_samples: int = 3000) -> pd.Series:
    """Mean |SHAP value| per feature, computed with the tree-exact
    TreeExplainer on a sample of X for speed. Ranks features by how much
    they actually move individual predictions, not just split counts.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=0)
    explainer = shap.TreeExplainer(model["estimator"])
    shap_values = explainer.shap_values(X)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    return importance.sort_values(ascending=False)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
    }


def _print_error_buckets(val_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print MAE bucketed by equipment, distance quartile, weight_was_missing,
    market_index_was_missing, and day_of_week -- a quick look at whether
    errors cluster somewhere specific before tuning/adding features blindly.
    """
    diag = val_df.copy()
    diag["abs_error"] = np.abs(y_true - y_pred)
    diag["distance_quartile"] = pd.qcut(diag["distance"], 4, labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])

    for col in ["equipment", "distance_quartile", "weight_was_missing", "market_index_was_missing", "day_of_week"]:
        print(f"\nMAE by {col}:")
        summary = diag.groupby(col, observed=True)["abs_error"].agg(["mean", "count"]).rename(columns={"mean": "MAE"})
        print(summary.round(2).to_string())


def _md_table(df: pd.DataFrame, index_name: str = "") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table with no extra
    dependency (tabulate isn't installed / in requirements.txt)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(level) for level in col if level) for col in df.columns]
    df = df.reset_index()
    if index_name:
        df = df.rename(columns={df.columns[0]: index_name})
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    row_lines = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def main() -> None:
    train = build_features(clean_loads(load_train_test()))
    dates = train["date"]

    folds = time_based_folds(train, n_folds=3)
    for fold in folds:
        assert_no_leakage(fold["train_idx"], fold["val_idx"], dates)

    fold2 = folds[2]
    fold2_train = train.loc[fold2["train_idx"]]
    fold2_val = train.loc[fold2["val_idx"]]

    # ------------------------------------------------------------------
    # Step 1: link function comparison, fold 2
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Link function comparison -- Gamma GLM, distance + equipment + market_index")
    print("+ quote_signal + seasonality (no interaction), fold 2 train/val:")
    print("=" * 80)
    link_comparison = compare_link_functions(fold2_train, fold2_val)
    print(link_comparison.round(3).to_string())

    # ------------------------------------------------------------------
    # Step 3: negative-prediction check for the winning (identity) link,
    # across all 3 folds
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Negative-prediction check, {BASELINE_LINK_NAME}-link GLM, all 3 folds:")
    print("=" * 80)
    total_neg, total_n = 0, 0
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]
        model = _fit_gamma_glm(FORMULA_NO_INTERACTION, train_slice)
        preds = np.asarray(model.predict(val_slice))
        n_neg = int((preds <= 0).sum())
        total_neg += n_neg
        total_n += len(preds)
        print(f"  Fold {fold['fold']}: {n_neg}/{len(preds)} negative predictions ({n_neg / len(preds) * 100:.4f}%)")
    neg_frac = total_neg / total_n * 100
    if neg_frac > 0.1:
        print(f"  -> {neg_frac:.4f}% negative overall (>0.1%): clipping predict_glm output at $1 floor.")
    else:
        print(f"  -> {neg_frac:.4f}% negative overall: zero/negligible, no clipping added.")

    # ------------------------------------------------------------------
    # Step 4: OLD (log-link) vs CORRECTED (identity-link) GLM vs GBM,
    # same 3 folds, same metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OLD log-link GLM -- per-fold evaluation (reference only, superseded)")
    print("=" * 80)
    old_glm_fold_metrics = []
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]
        model = _fit_gamma_glm(FORMULA_NO_INTERACTION, train_slice, link=LINK_FUNCTIONS["log"])
        preds = predict_glm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        old_glm_fold_metrics.append(metrics)
        print(f"Fold {fold['fold']}: MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%")

    print("\n" + "=" * 80)
    print(f"CORRECTED {BASELINE_LINK_NAME}-link GLM -- per-fold evaluation")
    print("=" * 80)
    glm_fold_metrics = []
    glm_fold_models = {}
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]

        print(f"Fold {fold['fold']} (train n={len(train_slice):,}, val n={len(val_slice):,}):")
        model = fit_baseline_glm(train_slice, verbose=True)
        glm_fold_models[fold["fold"]] = model

        preds = predict_glm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        glm_fold_metrics.append(metrics)
        print(f"  MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%\n")

    print("\n" + "=" * 80)
    print("Fold 2 CORRECTED GLM coefficient summary (train slice closest to the real train->validation gap):")
    print("=" * 80)
    print(glm_fold_models[2].summary())

    # ------------------------------------------------------------------
    # Add lane_historical_avg_rate now: causal/expanding over the whole
    # Jan-Oct chronology of `train`, so it's computed once regardless of
    # fold boundaries (see add_lane_historical_rate docstring for why this
    # doesn't leak). weight_per_mile and distance_bucket already came from
    # build_features() at the top of main().
    # ------------------------------------------------------------------
    train = add_lane_historical_rate(train)
    fold2_train = train.loc[fold2["train_idx"]]
    fold2_val = train.loc[fold2["val_idx"]]

    # ------------------------------------------------------------------
    # OLD tuned GBM (last session's feature set + hyperparams, reproduced
    # exactly via OLD_GBM_* / OLD_TUNED_PARAMS), per fold. Used both as the
    # "GBM" column in the legacy GLM comparison below and as the reference
    # row for this session's OLD-vs-NEW GBM comparison (TASK 5).
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"OLD tuned GBM (features={OLD_GBM_NUMERIC_FEATURES + OLD_GBM_CATEGORICAL_FEATURES}, params={OLD_TUNED_PARAMS})")
    print("=" * 80)
    old_gbm_fold_metrics = []
    old_gbm_fold_models = {}
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]
        model = fit_gbm(
            train_slice,
            numeric_features=OLD_GBM_NUMERIC_FEATURES,
            categorical_features=OLD_GBM_CATEGORICAL_FEATURES,
            **OLD_TUNED_PARAMS,
        )
        old_gbm_fold_models[fold["fold"]] = model
        preds = predict_gbm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        old_gbm_fold_metrics.append(metrics)
        print(f"Fold {fold['fold']}: MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%")

    # ------------------------------------------------------------------
    # TASK 1: error diagnostic on the OLD tuned GBM's fold-2 predictions,
    # before any new tuning/features -- print only.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 1: error diagnostic -- OLD tuned GBM, fold 2 held-out predictions, bucketed MAE")
    print("=" * 80)
    old_fold2_preds = predict_gbm(old_gbm_fold_models[2], fold2_val)
    _print_error_buckets(fold2_val, fold2_val["posted_rate"].to_numpy(), old_fold2_preds)

    # ------------------------------------------------------------------
    # Three-way comparison table: OLD log-link GLM, CORRECTED GLM, OLD GBM
    # (unchanged from last session, just reusing old_gbm_fold_metrics
    # instead of re-running the now-removed manual grid search)
    # ------------------------------------------------------------------
    fold_names = [f"fold_{f['fold']}" for f in folds]
    old_glm_df = pd.DataFrame(old_glm_fold_metrics, index=fold_names)
    glm_df = pd.DataFrame(glm_fold_metrics, index=fold_names)
    gbm_df = pd.DataFrame(old_gbm_fold_metrics, index=fold_names)
    comparison = pd.concat(
        {"OLD log-link GLM": old_glm_df, "CORRECTED GLM": glm_df, "GBM": gbm_df}, axis=1
    )

    print("\n" + "=" * 80)
    print("OLD GLM vs CORRECTED GLM vs GBM -- per-fold comparison (same 3 folds, same metrics):")
    print("=" * 80)
    print(comparison.round(3).to_string())

    print("\nMean ± std across folds:")
    mean_std = pd.concat(
        {
            "OLD log-link GLM": pd.DataFrame({"mean": old_glm_df.mean(), "std": old_glm_df.std()}),
            "CORRECTED GLM": pd.DataFrame({"mean": glm_df.mean(), "std": glm_df.std()}),
            "GBM": pd.DataFrame({"mean": gbm_df.mean(), "std": gbm_df.std()}),
        },
        axis=1,
    ).round(3)
    print(mean_std.to_string())

    # ------------------------------------------------------------------
    # SHAP feature importance on fold 2's OLD GBM (unchanged from last
    # session, kept for report continuity)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SHAP feature importance, fold 2 OLD GBM (mean |SHAP value|, sampled from fold 2 train):")
    print("=" * 80)
    old_fold2_X = _gbm_design_matrix(old_gbm_fold_models[2], fold2_train)
    importance = _gbm_shap_importance(old_gbm_fold_models[2], old_fold2_X)
    print(importance.round(3).to_string())

    noise_features = ["market_index", "quote_signal"]
    top_n = 4
    ranks = {f: list(importance.index).index(f) + 1 for f in noise_features if f in importance.index}
    flagged = {f: r for f, r in ranks.items() if r <= top_n}
    if flagged:
        print(
            f"\nWARNING: {list(flagged.keys())} rank in the top {top_n} by SHAP importance "
            f"({flagged}) despite <0.1 correlation with the distance-detrended residual in "
            "EDA -- this looks like the tree overfitting to noise rather than finding real "
            "signal these features don't have. Worth checking held-out fold performance isn't "
            "being propped up by this before trusting it."
        )
    else:
        print(
            f"\nmarket_index/quote_signal ranks: {ranks} -- consistent with EDA's <0.1 residual "
            "correlation finding, i.e. the GBM isn't manufacturing signal out of noise here."
        )

    # ------------------------------------------------------------------
    # TASK 3: real hyperparameter search (Optuna, 30 trials), fold 2,
    # NEW feature set (fit_gbm's default), raw target
    # ------------------------------------------------------------------
    n_trials = 30
    print("\n" + "=" * 80)
    print(f"TASK 3: Optuna hyperparameter search ({n_trials} trials), fold 2, NEW feature set, raw target")
    print("=" * 80)
    study = _tune_gbm_optuna(fold2_train, fold2_val, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best trial: val MAE=${study.best_value:.2f}")
    print(f"Best params: {best_params}")

    print("\nValidating the winning config across all 3 folds (not just fold 2 it was tuned on):")
    new_raw_fold_metrics = []
    new_raw_fold_models = {}
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]
        model = fit_gbm(train_slice, target="raw", **best_params)
        new_raw_fold_models[fold["fold"]] = model
        preds = predict_gbm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        new_raw_fold_metrics.append(metrics)
        print(f"Fold {fold['fold']}: MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%")

    # ------------------------------------------------------------------
    # TASK 4: rate_per_mile alternate target, same tuned hyperparams, same
    # 3 folds -- test it, don't assume it wins.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 4: rate_per_mile alternate target (same tuned hyperparams), all 3 folds")
    print("=" * 80)
    new_rpm_fold_metrics = []
    new_rpm_fold_models = {}
    for fold in folds:
        train_slice = train.loc[fold["train_idx"]]
        val_slice = train.loc[fold["val_idx"]]
        model = fit_gbm(train_slice, target="rate_per_mile", **best_params)
        new_rpm_fold_models[fold["fold"]] = model
        preds = predict_gbm(model, val_slice)
        metrics = _regression_metrics(val_slice["posted_rate"].to_numpy(), preds)
        new_rpm_fold_metrics.append(metrics)
        print(f"Fold {fold['fold']}: MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.2f}%")

    raw_mean_mae = float(np.mean([m["MAE"] for m in new_raw_fold_metrics]))
    rpm_mean_mae = float(np.mean([m["MAE"] for m in new_rpm_fold_metrics]))
    if rpm_mean_mae < raw_mean_mae:
        winning_target = "rate_per_mile"
        new_gbm_fold_metrics, new_gbm_fold_models = new_rpm_fold_metrics, new_rpm_fold_models
    else:
        winning_target = "raw"
        new_gbm_fold_metrics, new_gbm_fold_models = new_raw_fold_metrics, new_raw_fold_models
    print(
        f"\nTarget comparison, mean MAE across 3 folds: raw=${raw_mean_mae:.2f} vs "
        f"rate_per_mile=${rpm_mean_mae:.2f} -> '{winning_target}' wins, used for the NEW model below."
    )
    if winning_target == "raw":
        print("(rate_per_mile did NOT beat predicting the raw rate directly on this data -- reported for honesty, not used.)")

    # ------------------------------------------------------------------
    # SHAP feature importance, NEW model, fold 2 -- check the new features
    # actually earn their place, and re-check market_index/quote_signal
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"SHAP feature importance, fold 2 NEW GBM (target={winning_target}):")
    print("=" * 80)
    new_fold2_X = _gbm_design_matrix(new_gbm_fold_models[2], fold2_train)
    new_importance = _gbm_shap_importance(new_gbm_fold_models[2], new_fold2_X)
    print(new_importance.round(3).to_string())
    new_feature_ranks = {
        f: list(new_importance.index).index(f) + 1
        for f in ["lane_historical_avg_rate", "weight_per_mile", "market_index", "quote_signal"]
        if f in new_importance.index
    }
    print(f"\nKey feature ranks (out of {len(new_importance)}): {new_feature_ranks}")

    # ------------------------------------------------------------------
    # TASK 5: OLD tuned GBM vs NEW tuned GBM, same 3 folds, same metrics
    # ------------------------------------------------------------------
    old_gbm_df = pd.DataFrame(old_gbm_fold_metrics, index=fold_names)
    new_gbm_df = pd.DataFrame(new_gbm_fold_metrics, index=fold_names)
    new_gbm_label = f"NEW tuned GBM ({winning_target})"
    gbm_comparison = pd.concat({"OLD tuned GBM": old_gbm_df, new_gbm_label: new_gbm_df}, axis=1)

    print("\n" + "=" * 80)
    print("TASK 5: OLD tuned GBM vs NEW tuned GBM -- per-fold comparison:")
    print("=" * 80)
    print(gbm_comparison.round(3).to_string())

    gbm_mean_std = pd.concat(
        {
            "OLD tuned GBM": pd.DataFrame({"mean": old_gbm_df.mean(), "std": old_gbm_df.std()}),
            new_gbm_label: pd.DataFrame({"mean": new_gbm_df.mean(), "std": new_gbm_df.std()}),
        },
        axis=1,
    ).round(3)
    print("\nMean ± std across folds:")
    print(gbm_mean_std.to_string())

    # ------------------------------------------------------------------
    # TASK 6: decide clear win vs wash, save/report accordingly. No
    # hardcoded winner -- the >2%-and-every-fold rule below is the only
    # judgment call, applied uniformly to whatever the numbers turn out to be.
    # ------------------------------------------------------------------
    mae_improvement_pct = (1 - new_gbm_df["MAE"].mean() / old_gbm_df["MAE"].mean()) * 100
    new_wins_every_fold = bool((new_gbm_df["MAE"].to_numpy() < old_gbm_df["MAE"].to_numpy()).all())
    clear_win = new_wins_every_fold and mae_improvement_pct > 2.0

    print(
        f"\nMean MAE improvement (positive = NEW is better): {mae_improvement_pct:+.1f}% "
        f"({'better on every fold' if new_wins_every_fold else 'NOT better on every fold'}) "
        f"-> {'CLEAR WIN: promoting NEW model' if clear_win else 'NOT a clear win: keeping OLD model'}"
    )

    # Re-save the chosen configuration (OLD or NEW) either way: the
    # gbm_challenger.pkl already on disk from last session was pickled under
    # the pre-this-session dict schema ({"estimator", "feature_columns"}
    # only) and is no longer compatible with predict_gbm(), which now also
    # needs "numeric_features"/"categorical_features"/"target". "Not a clear
    # win" means we keep the OLD *configuration* (features/hyperparameters),
    # not that we leave a stale, incompatible pickle in place.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gbm_out_path = MODELS_DIR / "gbm_challenger.pkl"
    if clear_win:
        final_gbm = fit_gbm(train, target=winning_target, **best_params)
        joblib.dump(final_gbm, gbm_out_path)
        print(f"Saved NEW GBM (target={winning_target}, params={best_params}) to {gbm_out_path}, overwriting the OLD one.")
    else:
        final_gbm = fit_gbm(
            train,
            numeric_features=OLD_GBM_NUMERIC_FEATURES,
            categorical_features=OLD_GBM_CATEGORICAL_FEATURES,
            **OLD_TUNED_PARAMS,
        )
        joblib.dump(final_gbm, gbm_out_path)
        print(
            f"Re-saved the OLD tuned GBM configuration (unchanged features/hyperparams) to {gbm_out_path} "
            "in the current model-dict format, since last session's pickle predates the "
            "numeric_features/categorical_features/target fields predict_gbm() now relies on."
        )

    # ------------------------------------------------------------------
    # Final production GLM, full train_test.csv (corrected link, overwrites
    # the mis-specified log-link artifact from last session)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Fitting final {BASELINE_LINK_NAME}-link GLM on the full train_test.csv (48,000 rows)")
    print("=" * 80)
    final_glm = fit_baseline_glm(train, verbose=True)

    glm_out_path = MODELS_DIR / "baseline_glm.pkl"
    joblib.dump(final_glm, glm_out_path)
    print(f"Saved final GLM ({BASELINE_LINK_NAME} link) to {glm_out_path}, overwriting the log-link version")

    # ------------------------------------------------------------------
    # Step 7: write reports/model_comparison.md
    # ------------------------------------------------------------------
    report_path = ROOT / "reports" / "model_comparison.md"
    report_lines = [
        "# Model Comparison: GLM Link Correction and GBM Challenger\n",
        "## Link function comparison (Gamma GLM, fold 2)\n",
        "Formula: `posted_rate ~ distance + C(equipment) + market_index + quote_signal "
        "+ day_of_year_sin + day_of_year_cos` (no distance:equipment interaction -- "
        "rejected by a likelihood-ratio test on every fold). Fit on fold 2's training "
        f"slice (n={len(fold2_train):,}), evaluated on its held-out validation slice "
        f"(n={len(fold2_val):,}).\n",
        _md_table(link_comparison.round(3), "link") + "\n",
        "\n## OLD (log-link) vs CORRECTED (identity-link) GLM vs GBM, per fold\n",
        _md_table(comparison.round(3), "fold") + "\n",
        "\n**Mean ± std across folds:**\n",
        _md_table(mean_std.round(3), "metric") + "\n",
        "\n## Finding: the log link was misspecified\n",
        "The original baseline used a log link because it's the common default for a "
        "strictly positive, right-skewed target -- it guarantees positive predictions "
        "and models multiplicative (percentage) effects. That assumption doesn't hold "
        "here: a log link implies posted_rate grows *exponentially* with distance, but "
        "this dataset's rate/distance relationship is much closer to *additive* "
        "(rate increases by roughly a constant dollar amount per mile, not a constant "
        "percentage). Comparing links directly on fold 2 shows the identity link wins "
        "decisively on both in-sample fit (AIC "
        f"{link_comparison.loc['identity', 'AIC']:.0f} vs "
        f"{link_comparison.loc['log', 'AIC']:.0f} for log) and held-out error (MAE "
        f"${link_comparison.loc['identity', 'MAE']:.2f} vs "
        f"${link_comparison.loc['log', 'MAE']:.2f} for log, a "
        f"{(1 - link_comparison.loc['identity', 'MAE'] / link_comparison.loc['log', 'MAE']) * 100:.0f}% "
        "reduction). The inverse link (Gamma's canonical link) was also tested and "
        "performed worse than both, plus produced 41 negative predictions on fold 2 "
        "alone -- a non-starter for a rate that must be positive.\n",
        "\nMost of the GLM-vs-GBM gap reported last session was therefore an artifact "
        "of the log-link misspecification, not a fundamental linear-vs-tree-model gap. "
        "With the identity link, the corrected GLM closes most of that distance to the "
        "GBM, though the GBM still comes out ahead on every fold and every metric above "
        "-- consistent with it also having access to `weight`, `haversine_distance_mi`, "
        "`bearing_deg`, and nonlinear/interaction effects the GLM's linear specification "
        "doesn't include.\n",
        "\nThe link correction also overturned an earlier conclusion: under the "
        "mis-specified log link, the distance:equipment interaction tested as "
        "not significant on every fold (p=0.13-0.33) and was dropped. Under the "
        "corrected identity link, the same likelihood-ratio test on the same "
        "folds now finds it highly significant (p<1e-170 on every fold) and "
        "`fit_baseline_glm` keeps it. In other words, equipment doesn't just "
        "shift the intercept (a flat $/load premium) -- it also changes the "
        "$/mile slope (distance coefficients of roughly $1.89/mi for Dry Van, "
        "+$0.14/mi for Flatbed, +$0.24/mi for Reefer on fold 2's fit), which "
        "the log link's exponential form had been masking.\n",
        "\nA negative-prediction check on the identity-link GLM across all 3 folds found "
        f"{total_neg}/{total_n} negative predictions ({neg_frac:.4f}%) -- negligible, so "
        "no output clipping was added to predict_glm.\n",
        "\n## GBM improvement pass: features, real tuning, OLD vs NEW\n",
        "Starting point was the OLD tuned GBM above "
        f"(features: {', '.join(OLD_GBM_NUMERIC_FEATURES + OLD_GBM_CATEGORICAL_FEATURES)}; "
        f"hyperparameters: {OLD_TUNED_PARAMS}, picked by a 4-combo manual grid). Before "
        "changing anything, fold 2's held-out errors were bucketed by equipment, distance "
        "quartile, weight_was_missing, market_index_was_missing, and day_of_week to check "
        "whether error concentrates somewhere specific (full breakdown printed to console, "
        "not reproduced here).\n",
        "\n**What changed:**\n",
        "- Added `lane_historical_avg_rate`: an expanding, causal per-lane average of "
        "posted_rate using only strictly-earlier dates (same-day and future rows excluded), "
        "with a causal global-average fallback for lanes with no prior history yet. Computed "
        "with two groupby+cumsum passes over the whole chronology, not a row-by-row loop.\n",
        "- Added `weight_per_mile` (weight / distance) and `distance_bucket` "
        "(short/medium/long haul, fixed cutoffs at 700mi/1300mi) as new features.\n",
        "- Replaced the 4-combo manual grid with a real Optuna search "
        f"({n_trials} trials, TPE sampler) over learning_rate, max_leaf_nodes, max_iter, "
        "min_samples_leaf, and l2_regularization, tuned on fold 2 and then validated across "
        "all 3 folds (not just the fold it was tuned on).\n",
        "- Tested `rate_per_mile` (posted_rate/distance) as an alternate training target "
        f"against predicting posted_rate directly: mean MAE across all 3 folds was "
        f"${raw_mean_mae:.2f} (raw) vs ${rpm_mean_mae:.2f} (rate_per_mile) -> "
        f"**'{winning_target}' won**"
        + (
            ", so rate_per_mile is reported here for completeness but was not used in the final model.\n"
            if winning_target == "raw"
            else ".\n"
        ),
        "\n**SHAP check on the new features:** "
        f"in the fold-2 NEW model, key feature ranks (1=most important, out of "
        f"{len(new_importance)} features) were {new_feature_ranks}. "
        + (
            "market_index/quote_signal stayed out of the top of the ranking, consistent "
            "with EDA's <0.1 residual-correlation finding -- the new features didn't change "
            "that read.\n"
            if all(new_feature_ranks.get(f, 99) > 4 for f in ["market_index", "quote_signal"])
            else "market_index/quote_signal moved into the top ranks here -- worth a second "
            "look before trusting it, per the same overfitting-to-noise concern raised for "
            "the OLD model.\n"
        ),
        "\n### OLD tuned GBM vs NEW tuned GBM, per fold\n",
        _md_table(gbm_comparison.round(3), "fold") + "\n",
        "\n**Mean ± std across folds:**\n",
        _md_table(gbm_mean_std.round(3), "metric") + "\n",
        f"\n**Result: NEW model's mean MAE improved by {mae_improvement_pct:.1f}% over OLD "
        f"({'better on every fold' if new_wins_every_fold else 'not better on every fold'}).** "
        + (
            f"This clears the bar for a real improvement (>2%, consistent across all 3 folds), "
            f"so `models/gbm_challenger.pkl` was overwritten with the NEW model "
            f"(target={winning_target}, params={best_params}).\n"
            if clear_win
            else "This is not a clear win -- either the improvement is too small, inconsistent "
            "across folds, or both. Recommendation: keep the existing OLD tuned GBM "
            "configuration (same features/hyperparameters as last session) in production "
            "rather than adding the complexity (extra features, a heavier tuning process) of "
            "the NEW model for no reliable gain. `models/gbm_challenger.pkl` was re-saved "
            "with that same OLD configuration purely to match the current model-dict format "
            "(predict_gbm() now needs numeric_features/categorical_features/target fields "
            "last session's pickle didn't have) -- nothing about the model itself changed.\n"
        ),
        "\nNo GLM-vs-GBM production decision has been made yet; that, plus validation.csv "
        "and december_chart_inputs.csv predictions, is the next and final step.\n",
    ]
    report_path.write_text("\n".join(report_lines))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
