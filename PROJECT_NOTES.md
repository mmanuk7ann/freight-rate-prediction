# Freight Rate Prediction — Project Notes

Working notes and code for the freight rate prediction take-home.

**Note on filenames:** this project's filesystem is case-insensitive, so
`README.md` and `readme.md` are the same file here — this document is kept
as `PROJECT_NOTES.md` instead so it doesn't collide with the assessment's
own [`readme.md`](readme.md) (see also `Freight_Rate_ML_Assessment.pdf`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
data/                          raw CSVs (train_test, validation, december_chart_inputs,
                                validation_predictions_template)
src/
  __init__.py
  data.py                      load_train_test(), load_validation(), load_december(),
                                clean_loads(df)
notebooks_or_scripts/
  eda.py                       runnable EDA script (see below)
reports/
  eda_findings.md              written EDA summary
  figures/                     saved PNG plots
score.py, readme.md            assessment-provided scorer and instructions (unchanged)
```

## Running the EDA

```bash
source .venv/bin/activate
python notebooks_or_scripts/eda.py
```

This loads the raw data, programmatically confirms the known data quality
facts (row counts, date ranges, the weight sign-flip artifact, missingness,
correlations, equipment $/mile ordering, market_index seasonality, and
validation-only cities), and writes:

- `reports/figures/*.png`
- `reports/eda_findings.md`

## Data cleaning

`src/data.py::clean_loads(df)` fixes the weight sign-flip (`.abs()`) and
median-imputes missing `weight` / `market_index`, adding
`weight_was_missing` / `market_index_was_missing` flag columns. See the
function's docstring for the reasoning behind each step.

No modeling yet — this stage is data setup, cleaning, and EDA only.
