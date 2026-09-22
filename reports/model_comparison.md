# Model Comparison: GLM Link Correction and GBM Challenger

## Link function comparison (Gamma GLM, fold 2)

Formula: `posted_rate ~ distance + C(equipment) + market_index + quote_signal + day_of_year_sin + day_of_year_cos` (no distance:equipment interaction -- rejected by a likelihood-ratio test on every fold). Fit on fold 2's training slice (n=43,463), evaluated on its held-out validation slice (n=4,537).

| link     | AIC        | deviance  | MAE      | RMSE      | MAPE   |
| -------- | ---------- | --------- | -------- | --------- | ------ |
| log      | 671256.126 | 3313.861  | 459.612  | 944.076   | 23.952 |
| identity | 636069.462 | 1291.137  | 172.445  | 650.093   | 8.07   |
| inverse  | 894448.595 | 38381.522 | 2263.123 | 26690.266 | 63.069 |


## OLD (log-link) vs CORRECTED (identity-link) GLM vs GBM, per fold

| fold   | OLD log-link GLM MAE | OLD log-link GLM RMSE | OLD log-link GLM MAPE | CORRECTED GLM MAE | CORRECTED GLM RMSE | CORRECTED GLM MAPE | GBM MAE | GBM RMSE | GBM MAPE |
| ------ | -------------------- | --------------------- | --------------------- | ----------------- | ------------------ | ------------------ | ------- | -------- | -------- |
| fold_0 | 437.488              | 927.638               | 22.206                | 155.086           | 611.328            | 6.665              | 122.28  | 601.98   | 5.813    |
| fold_1 | 484.571              | 1009.175              | 21.524                | 157.849           | 669.585            | 6.454              | 135.644 | 667.696  | 5.27     |
| fold_2 | 459.612              | 944.076               | 23.952                | 159.109           | 644.962            | 7.555              | 126.285 | 639.697  | 6.103    |


**Mean ± std across folds:**

| metric | OLD log-link GLM mean | OLD log-link GLM std | CORRECTED GLM mean | CORRECTED GLM std | GBM mean | GBM std |
| ------ | --------------------- | -------------------- | ------------------ | ----------------- | -------- | ------- |
| MAE    | 460.557               | 23.556               | 157.348            | 2.058             | 128.07   | 6.858   |
| RMSE   | 960.296               | 43.121               | 641.958            | 29.245            | 636.458  | 32.977  |
| MAPE   | 22.561                | 1.253                | 6.891              | 0.584             | 5.729    | 0.423   |


## Finding: the log link was misspecified

The original baseline used a log link because it's the common default for a strictly positive, right-skewed target -- it guarantees positive predictions and models multiplicative (percentage) effects. That assumption doesn't hold here: a log link implies posted_rate grows *exponentially* with distance, but this dataset's rate/distance relationship is much closer to *additive* (rate increases by roughly a constant dollar amount per mile, not a constant percentage). Comparing links directly on fold 2 shows the identity link wins decisively on both in-sample fit (AIC 636069 vs 671256 for log) and held-out error (MAE $172.44 vs $459.61 for log, a 62% reduction). The inverse link (Gamma's canonical link) was also tested and performed worse than both, plus produced 41 negative predictions on fold 2 alone -- a non-starter for a rate that must be positive.


Most of the GLM-vs-GBM gap reported last session was therefore an artifact of the log-link misspecification, not a fundamental linear-vs-tree-model gap. With the identity link, the corrected GLM closes most of that distance to the GBM, though the GBM still comes out ahead on every fold and every metric above -- consistent with it also having access to `weight`, `haversine_distance_mi`, `bearing_deg`, and nonlinear/interaction effects the GLM's linear specification doesn't include.


The link correction also overturned an earlier conclusion: under the mis-specified log link, the distance:equipment interaction tested as not significant on every fold (p=0.13-0.33) and was dropped. Under the corrected identity link, the same likelihood-ratio test on the same folds now finds it highly significant (p<1e-170 on every fold) and `fit_baseline_glm` keeps it. In other words, equipment doesn't just shift the intercept (a flat $/load premium) -- it also changes the $/mile slope (distance coefficients of roughly $1.89/mi for Dry Van, +$0.14/mi for Flatbed, +$0.24/mi for Reefer on fold 2's fit), which the log link's exponential form had been masking.


A negative-prediction check on the identity-link GLM across all 3 folds found 0/13156 negative predictions (0.0000%) -- negligible, so no output clipping was added to predict_glm.


No model has been selected for production yet; that decision is pending further review of both models' behavior on validation.csv.
