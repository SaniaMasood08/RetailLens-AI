"""Dataset, forecast, and interpretation documentation."""

import pandas as pd
import streamlit as st

from src.dashboard import get_forecast, load_data, page_shell

page_shell("Methodology", "How the project measures revenue, constructs the forecast, and frames its limitations.")
transactions, daily, features, results = load_data()
_, metadata, latest = get_forecast()
method = metadata["training_validation_test_methodology"]

st.subheader("Dataset and measure")
st.markdown("This project uses the UCI Online Retail transaction history, with source fields for invoice, product, quantity, unit price, customer, country, and invoice date. The tracked processed data covers **2010-12-01 through 2011-12-09**.")
st.markdown("**Recorded revenue = Quantity × UnitPrice**, summed across retained rows by calendar date. Negative quantities and cancellation records are retained. Exact duplicates are removed and negative unit-price adjustment rows are excluded. Revenue is not profit: costs and margins are not present.")
st.subheader("Forecast target and features")
st.markdown("The target is next-calendar-day net recorded revenue. For a row dated *t*, revenue lags and rolling averages use information strictly before *t*. Day of week and month describe the target date and are known in advance. The model excludes dates, target fields, and future revenue from its predictors.")
st.code("Revenue predictors: lag_1, lag_7, lag_14, rolling_mean_7, rolling_mean_28\nKnown calendar predictors: day_of_week, month\nTarget: revenue on the next calendar day", language="text")
st.subheader("Evaluation design")
st.markdown(f"Rows remain chronological in a contiguous **70% / 15% / 15%** training, validation, and test split. The saved design contains {method['row_counts']['training']} training, {method['row_counts']['validation']} validation, and {method['row_counts']['test']} test rows. Candidate models were selected using validation MAE. The final packaged estimator was fitted on training plus validation rows; test rows were held out for final evaluation.")
st.subheader("Packaged estimator")
cutoff = pd.Timestamp(latest["Date"]) - pd.Timedelta(days=1)
st.markdown(f"The application loads the existing **{metadata['model_name']}** from `models/retail_revenue_model.joblib`. It does not train or change the model. The displayed forecast targets **{latest['target_date']:%Y-%m-%d}** from the latest packaged feature row; its revenue information cutoff is **{cutoff:%Y-%m-%d}**.")
st.subheader("Assumptions and limitations")
st.markdown("""
- Dates without transaction rows inside the observed date range are filled with zero recorded revenue for daily modeling; this is a data convention, not proof of no business activity.
- The prediction is a historical demonstration based on 2010–2011 data, not a current forecast.
- Test metrics cover one chronological 52-row period and do not guarantee performance on another period or retailer.
- Product and country summaries are descriptive; they do not establish causal drivers, customer demographics, or growth potential.
- Cancellation counts and IQR alerts are rule-based review cues. They do not explain causes or calculate optimal inventory.
- The data contains no cost, margin, stock-on-hand, or current market information.
""")
st.subheader("Available source files")
st.write(f"Processed transaction rows: {len(transactions):,} · daily revenue rows: {len(daily):,} · forecast feature rows: {len(features):,}")
st.caption("See README.md for the reproducible preprocessing and model workflow. Running the dashboard only loads committed data and artifacts.")
