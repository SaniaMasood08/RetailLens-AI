# RetailLens-AI

RetailLens-AI is an end-to-end retail sales decision-intelligence project built with the UCI Online Retail transaction history. It cleans historical transaction records, analyzes net recorded revenue, creates leakage-aware next-calendar-day forecasting features, evaluates candidate regressors with chronological partitions, and serves the already-packaged Random Forest forecast in a Streamlit dashboard. The dashboard combines the historical forecast with descriptive product, customer-country, trend, cancellation, and anomaly summaries to support review and planning; it does not estimate profit or guarantee future demand.

## Problem statement

Retail teams need a clear view of historical revenue patterns and a consistent short-horizon forecast to inform routine sales and inventory reviews. Transaction-level records are not directly suitable for that view: they need cleaning, aggregation, time-aware features, and evaluation that respects the order in which observations occurred.

## Project objective

Build a reproducible, interpretable demonstration that turns historical retail transactions into daily net recorded revenue, evaluates next-calendar-day forecasts without random shuffling, packages the selected estimator for inference, and presents its output alongside useful historical summaries.

## Key capabilities

- Transaction preprocessing with exact-duplicate removal and explicit handling of cancellations and negative unit prices.
- Daily net recorded revenue aggregation and calendar-day feature construction.
- Lag, rolling-average, and known-in-advance calendar predictors for a one-step-ahead target.
- Chronological training, validation, and held-out test evaluation with MAE, RMSE, and R².
- A packaged, constrained `RandomForestRegressor` and a prediction helper that validates feature names and values.
- Streamlit views for revenue trends, the historical forecast, seller decision support, product and country summaries, rule-based alerts, model metrics, and feature importance.

## System workflow and architecture

```text
Raw historical transactions
  → preprocessing
  → feature engineering
  → chronological train / validation / test split
  → model comparison
  → selected model
  → packaged predictor
  → Streamlit decision-intelligence dashboard
```

The packaged predictor is loaded from `models/`; the dashboard does not fit or retrain it when run.

## Dataset and revenue definition

The project is based on the UCI Online Retail dataset. The raw workbook and source ZIP are local input files under `data/raw/`, which is intentionally Git-ignored; the raw dataset is not tracked in this repository. Anyone cloning the project must obtain and supply the raw dataset separately to rerun preprocessing or model training. The processed transaction data spans 2010-12-01 through 2011-12-09. The source workbook contains invoice, product, quantity, unit price, customer, country, and invoice-date fields. The packaged model, processed features, model results, and dashboard are available from the repository without the raw files.

The target is **next-calendar-day net recorded revenue**, calculated for each transaction as `Quantity × UnitPrice` and summed by date. Negative quantities and cancellation invoice records are retained and contribute to the net total. Exact duplicate source rows are removed. The preprocessing code excludes negative `UnitPrice` adjustment rows while retaining zero prices. Revenue is **not profit**: the source has no costs or margins from which to calculate profit.

Within the observed date range, dates without transaction rows are filled with zero recorded revenue when creating the daily calendar series. This is the project's stated assumption, not evidence that no business activity occurred outside the records.

## Preprocessing and feature engineering

`src/preprocessing.py` reads the workbook, removes exact duplicate rows, parses invoice dates, marks cancellation invoices, removes negative unit-price adjustments, retains negative quantities, calculates line revenue, and writes `cleaned_transactions.csv` and `daily_revenue.csv` under `data/processed/`.

`src/feature_engineering.py` creates one row per forecast date `Date` and a target date one calendar day later (`target_date = Date + 1 day`). Its revenue predictors are:

- `lag_1`: revenue on `Date - 1 day`.
- `lag_7`: revenue on `Date - 7 days`.
- `lag_14`: revenue on `Date - 14 days`.
- `rolling_mean_7`: mean revenue over `Date - 7` through `Date - 1`.
- `rolling_mean_28`: mean revenue over `Date - 28` through `Date - 1`.
- `day_of_week` and `month`: calendar values for `target_date`, which are known in advance.

The target `target_DailyRevenue` is revenue on `target_date`. Thus, for a row whose `Date` is *t*, all revenue predictors stop at *t - 1 day*. For the latest saved feature row, `Date` is 2011-12-08, `target_date` is 2011-12-09, and the latest revenue date available to the model is 2011-12-07. The dashboard uses that same 2011-12-07 information cutoff for its forecast baseline and Seller Decision Center summaries.

## Forecast and evaluation methodology

The target is next-calendar-day net recorded revenue. The 345 generated rows are kept in chronological order and split contiguously into 70% training, 15% validation, and 15% test partitions:

| Partition | Rows | Forecast-date range |
|---|---:|---|
| Training | 241 | 2010-12-29 to 2011-08-26 |
| Validation | 52 | 2011-08-27 to 2011-10-17 |
| Test | 52 | 2011-10-18 to 2011-12-08 |

Candidate models are compared using validation MAE. The test partition is held out from model selection and hyperparameter tuning. After selection, the final estimator is fit on the combined 241 training and 52 validation rows (293 total) and evaluated on the 52 test rows. The stored test metrics are evaluation evidence for this single chronological period; they are not a guarantee of future performance.

### Leakage prevention

- Revenue lags and rolling means use only dates before each feature row's `Date`.
- Calendar predictors describe the target date and are known before that date.
- `Date`, `target_date`, and the target revenue column are not model predictors.
- Rows remain in time order; the train, validation, and test partitions do not overlap.
- Validation MAE selects the candidate. Test observations are excluded from fitting and selection, then scored once.
- At inference, `src/predict.py` requires exactly one row with the saved predictor fields and rejects missing or non-finite values.

## Model comparison and selected model

The selected model is the existing **constrained Random Forest** (`RandomForestRegressor`) with:

```text
n_estimators=300
max_depth=8
min_samples_leaf=3
random_state=42
n_jobs=1
```

Validation results in `data/processed/model_improvement_results.csv`:

| Candidate | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest_constrained (selected) | 13,621.16 | 19,925.18 | 0.143 |
| RandomForest_baseline | 13,990.69 | 20,247.49 | 0.115 |
| ExtraTrees | 14,085.12 | 20,155.68 | 0.123 |
| GradientBoosting | 14,764.37 | 20,654.55 | 0.079 |
| HistGradientBoosting | 15,178.53 | 21,215.73 | 0.028 |

Final selected-model metrics on the held-out test partition:

| Metric | Test result |
|---|---:|
| MAE | 12,282.34 |
| RMSE | 17,453.85 |
| R² | 0.482 |

For context, the original Random Forest baseline's recorded test MAE was 13,323.01 (RMSE 18,697.97; R² 0.406). The selected model's recorded test score is lower on this test period. This comparison is descriptive and does not establish performance on a different time period or retailer. Metrics are in the dataset's recorded monetary units; no currency is inferred by the dashboard.

## Streamlit application

The application is organized into six navigable pages: **Overview**, **Sales Explorer**, **Product & Customer Insights**, **Revenue Forecast**, **Decision Center**, and **Methodology**. Historical exploration pages share sidebar date, country, and product filters; the selected transaction rows consistently drive their displayed totals, charts, and tables. The forecast remains the packaged historical estimate and is not changed by exploration filters.

The Decision Center preserves the project's transparent ±10% forecast-to-baseline rule and historical activity summaries. These are review cues, not optimal stock calculations, causal findings, or promises of demand. The interface does not retrain the model.

## How the system can support business decisions

The dashboard can help a reviewer monitor recorded revenue patterns, compare the packaged forecast with the recent history available at its cutoff, identify products or countries for further investigation, and review cancellation activity. These outputs can inform a human review of inventory or operations. They do not replace inventory systems, cost and margin analysis, domain knowledge, or a current demand forecast.

## Limitations

- The model is trained and evaluated on a historical 2010–2011 dataset.
- The displayed forecast uses the latest historical feature row and is a demonstration, not live/current 2026 data.
- Missing calendar dates within the observed period are treated as zero recorded revenue under the project's assumption.
- Revenue is not profit; costs, margins, and stock-on-hand are not available.
- Negative quantities and cancellation records are retained in net revenue.
- Anomaly signals use a rule-based historical IQR threshold and do not explain causes.
- Product and country insights are descriptive associations, not causal findings.
- Test evidence covers one 52-row chronological held-out period only.
- The source data may not represent another retailer, market, or present-day conditions.

## Project structure

```text
RetailLens-AI/
├── app.py                              # Streamlit Overview page
├── pages/                              # Streamlit multipage application
├── requirements.txt                    # Runtime and notebook dependencies
├── README.md                           # Project documentation
├── data/
│   ├── raw/                             # Source workbook and ZIP (ignored by Git)
│   └── processed/                       # Cleaned data, features, and result CSVs
├── models/
│   ├── model_metadata.json              # Predictor, fit, split, and metric metadata
│   └── retail_revenue_model.joblib      # Packaged Random Forest estimator
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── src/
    ├── dashboard.py                     # Shared Streamlit styling, loading, and filters
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── train_models.py
    ├── improve_models.py
    ├── package_model.py
    └── predict.py
```

## Installation and setup

Use Python 3.10 or later and create an isolated environment from the project root:

```bash
python -m venv .venv
source .venv/bin/activate             # macOS / Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The preprocessing step expects the source workbook at `data/raw/Online Retail.xlsx`. The workbook and raw data directory are ignored by Git; obtain the UCI Online Retail dataset separately if they are not present in your checkout. The processed CSVs and packaged model are currently included in this project.

## Run the dashboard

From the project root, with the environment activated:

```bash
streamlit run app.py
```

The app reads the processed CSVs and packaged model from the repository. It does not retrain the model.

## Run the preprocessing and model pipeline

The scripts can reproduce the current workflow when the source workbook is available:

```bash
python src/preprocessing.py
python src/feature_engineering.py
python src/train_models.py
python src/improve_models.py
python src/package_model.py
```

These commands rewrite processed datasets/results; `package_model.py` fits the selected configuration on training plus validation data and overwrites the packaged model and metadata. Do not run the packaging step when the goal is only to use or inspect the existing packaged model. Running `app.py` does not invoke these scripts.

## Validation commands

The repository does not currently include a standalone automated test suite. The following checks compile the Python files and exercise the Streamlit app using its testing interface:

```bash
./.venv/bin/python -m py_compile app.py src/*.py pages/*.py
./.venv/bin/python - <<'PY'
from streamlit.testing.v1 import AppTest

from pathlib import Path

for path in [Path("app.py"), *sorted(Path("pages").glob("*.py"))]:
    at = AppTest.from_file(str(path))
    at.run(timeout=30)
    print(path, "exceptions:", len(at.exception))
    if at.exception:
        for error in at.exception:
            print(error)
PY
git diff --check
```

## Internship and portfolio positioning

RetailLens-AI demonstrates a complete applied data-science workflow: business framing, transaction preprocessing, time-aware feature construction, chronological evaluation, model packaging, prediction validation, and an interactive dashboard with stated assumptions and limitations. It is suitable as an internship or portfolio case study in retail analytics and forecasting. Reported metrics are specific to this dataset and held-out historical period; they should not be presented as evidence of production readiness or guaranteed business impact.
