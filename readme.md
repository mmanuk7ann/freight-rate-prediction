# Freight Rate Prediction

This project predicts freight shipment prices from route, equipment, weight, and market data, built for Spotter AI's Machine Learning Engineer assessment.

## Overview

I trained the model on a year of historical freight loads and evaluated it on two months it never saw during training. Since that evaluation window comes entirely after the training period, I validated the model using time series cross-validation instead of a random split, so the accuracy numbers below reflect genuine forward-looking performance, not just a good fit on familiar data.

## What I found

Distance turned out to be the biggest factor in price by far, with a 0.91 correlation to rate. Equipment type mattered too: Dry Van loads were cheapest, then Flatbed, then Reefer. Prices also followed a seasonal pattern across the year, sitting lower heading into November and December, which happens to be the exact window I had to predict.

I also caught a data quality issue early on: about 0.6% of the weight values were negative, which isn't physically possible. Checking the numbers showed it was a simple sign error, not bad data, so I corrected it rather than throwing those rows away. Separately, 8 cities appeared only in the evaluation data and never in training, which meant I couldn't rely on city names as a feature. I used coordinates and distance instead, so the model handles a city it's never seen just as well as one it has.

## Results

| Model | MAE | MAPE |
|---|---|---|
| Gamma GLM (baseline) | $157.3 | 6.89% |
| Gradient-boosted trees (production) | $128.1 | 5.73% |

Both numbers are averaged across three time-based validation folds, each trained on an expanding window of the past and tested on the month right after it, mirroring the real gap between training and evaluation.

I also used the production model to forecast prices for a fixed lane across every day of December 2025. The forecast shows a real weekly pattern, prices run a bit higher mid-week and lower on weekends, which I checked against the actual training data before trusting it. It held up.

## Repository structure

src/
  data.py         cleaning and imputation
  features.py     feature engineering
  validation.py   time series cross-validation
  models.py       the GLM and GBM models
  predict.py      generates the final submission files
reports/
  eda_findings.md       full exploratory analysis
  model_comparison.md   full model comparison

## How to run

python -m pip install -r requirements.txt
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv

## Full report and walkthrough

The full write-up with methodology and validation details is in `reports/`, along with the accompanying PDF/DOCX report.


---

## Original assessment brief (Spotter AI)

*Everything below this line is the original brief provided with the assessment, unchanged.*
