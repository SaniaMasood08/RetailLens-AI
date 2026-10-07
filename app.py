"""Streamlit demonstration dashboard for the packaged RetailLens-AI model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import load_model_bundle, predict_next_day


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_PATH = PROCESSED_DIR / "daily_revenue.csv"
FEATURES_PATH = PROCESSED_DIR / "ml_features.csv"
IMPROVEMENT_RESULTS_PATH = PROCESSED_DIR / "model_improvement_results.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "retail_revenue_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="RetailLens-AI | Sales Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["Date"]).sort_values("Date")
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date", "target_date"])
    results = pd.read_csv(IMPROVEMENT_RESULTS_PATH)
    return daily, features, results


@st.cache_resource
def load_packaged_model():
    return load_model_bundle(MODEL_PATH, METADATA_PATH)


def format_revenue(value: float) -> str:
    """Format revenue without implying a currency absent from the labels."""
    return f"{value:,.2f}"


try:
    daily, features, improvement_results = load_dashboard_data()
    model, metadata = load_packaged_model()
except Exception as error:
    st.error(f"The dashboard could not load a required project artifact: {error}")
    st.stop()

if daily.empty or features.empty:
    st.error("The processed revenue or feature dataset is empty.")
    st.stop()

predictor_columns = list(metadata["predictor_columns"])
features = features.sort_values("target_date")
feature_row = features.tail(1).loc[:, predictor_columns]
latest_feature = features.iloc[-1]
try:
    forecast = predict_next_day(feature_row, model=model, metadata=metadata)
except Exception as error:
    st.error(f"Could not generate the historical demonstration prediction: {error}")
    st.stop()

# Reconstruct the agreed calendar-day target convention: missing transaction
# dates represent zero recorded revenue within the observed data range.
daily = daily.set_index("Date")
calendar_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="Date")
calendar_revenue = daily["DailyRevenue"].reindex(calendar_index).fillna(0.0)
monthly_revenue = calendar_revenue.resample("MS").sum()
strongest_month_date = monthly_revenue.idxmax()
highest_day = calendar_revenue.idxmax()
highest_day_revenue = float(calendar_revenue.loc[highest_day])
average_calendar_day = float(calendar_revenue.mean())
total_revenue = float(calendar_revenue.sum())
daily_std = float(calendar_revenue.std())
coefficient_of_variation = daily_std / average_calendar_day if average_calendar_day else float("nan")

test_result = improvement_results.loc[
    (improvement_results["model"] == metadata["selection_label"])
    & (improvement_results["split"] == "test_selected")
]
if len(test_result) != 1:
    st.error("Could not find the selected model's held-out test metrics.")
    st.stop()
test_metrics = test_result.iloc[0]

st.title("RetailLens-AI")
st.subheader("Industry-Oriented Sales Forecasting")
st.write(
    "This project demonstrates next-day **net recorded revenue** forecasting "
    "from historical daily sales patterns."
)
st.caption(
    f"Packaged model: {metadata['model_name']} · "
    "forecast horizon: next calendar day"
)

st.markdown("### Historical performance at a glance")
kpi_columns = st.columns(4)
kpi_columns[0].metric("Total recorded revenue", format_revenue(total_revenue))
kpi_columns[1].metric("Recorded transaction days", f"{len(daily):,}")
kpi_columns[2].metric("Average per calendar day", format_revenue(average_calendar_day))
kpi_columns[3].metric("Highest daily revenue", format_revenue(highest_day_revenue))
st.caption(
    "Average and total use every calendar date in the observed range; dates "
    "without processed transactions count as zero recorded revenue."
)

st.markdown("### Historical daily recorded revenue")
figure, axis = plt.subplots(figsize=(13, 4.3))
axis.plot(calendar_revenue.index, calendar_revenue.values, color="#2563eb", linewidth=1.35)
axis.fill_between(calendar_revenue.index, calendar_revenue.values, 0, color="#2563eb", alpha=0.08)
axis.set_title("Daily Net Recorded Revenue Over Time", loc="left", fontsize=13, weight="bold")
axis.set_xlabel("Date")
axis.set_ylabel("Daily recorded revenue (dataset units)")
axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
axis.spines[["top", "right"]].set_visible(False)
figure.autofmt_xdate()
figure.tight_layout()
st.pyplot(figure, width="stretch")
plt.close(figure)

forecast_column, metrics_column = st.columns([1.2, 1])
with forecast_column:
    st.markdown("### Next-day forecast demonstration")
    st.info(
        "Historical demonstration using the latest feature row in the project. "
        "This is not a current 2026 forecast."
    )
    forecast_kpis = st.columns(3)
    forecast_kpis[0].metric("Forecast origin", latest_feature["Date"].strftime("%Y-%m-%d"))
    forecast_kpis[1].metric("Forecast target", latest_feature["target_date"].strftime("%Y-%m-%d"))
    forecast_kpis[2].metric("Predicted recorded revenue", format_revenue(forecast))

    st.markdown("**Predictor values used**")
    feature_labels = {
        "lag_1": "Previous day revenue",
        "lag_7": "Revenue 7 days earlier",
        "lag_14": "Revenue 14 days earlier",
        "rolling_mean_7": "Mean revenue over prior 7 days",
        "rolling_mean_28": "Mean revenue over prior 28 days",
        "day_of_week": "Target day of week (Monday = 0)",
        "month": "Target month",
    }
    feature_display = pd.DataFrame(
        {
            "Feature": [feature_labels[name] for name in predictor_columns],
            "Value": [latest_feature[name] for name in predictor_columns],
        }
    )
    st.dataframe(feature_display, hide_index=True, width="stretch")

with metrics_column:
    st.markdown("### Model performance")
    st.caption("Held-out test performance · chronological final test period")
    score_columns = st.columns(3)
    score_columns[0].metric("MAE", format_revenue(float(test_metrics["MAE"])))
    score_columns[1].metric("RMSE", format_revenue(float(test_metrics["RMSE"])))
    score_columns[2].metric("R²", f"{float(test_metrics['R2']):.3f}")
    methodology = metadata["training_validation_test_methodology"]
    st.caption(
        f"Selected by {methodology['selection_metric']}; final fit used "
        f"{methodology['row_counts']['final_model_fit_training_plus_validation']} "
        "training + validation rows. Test rows were excluded from fitting."
    )

st.markdown("### Business insights from the historical series")
insight_columns = st.columns(3)
with insight_columns[0]:
    st.markdown("**Strongest recorded month**")
    st.write(f"{strongest_month_date.strftime('%B %Y')}: {format_revenue(float(monthly_revenue.max()))}")
    st.caption("Monthly totals aggregate the recorded calendar-day series.")
with insight_columns[1]:
    st.markdown("**Highest-revenue day**")
    st.write(f"{highest_day.strftime('%Y-%m-%d')}: {format_revenue(highest_day_revenue)}")
    st.caption("This is the maximum daily total in the processed historical period.")
with insight_columns[2]:
    st.markdown("**Daily revenue variability**")
    st.write(f"Standard deviation: {format_revenue(daily_std)}")
    st.caption(f"Coefficient of variation: {coefficient_of_variation:.2f} across calendar days.")

st.write(
    "**Planning implication:** Recorded revenue varies across days and includes "
    "large peaks. Use the time series to support revenue monitoring and review "
    "planning assumptions; revenue alone does not show item-level demand or "
    "inventory requirements."
)

with st.expander("Methodology and interpretation"):
    st.markdown(
        "- The target is **next-calendar-day net recorded revenue**.\n"
        "- Negative quantities and cancellation records were retained in net revenue.\n"
        "- Calendar dates with no processed transaction rows were represented as zero recorded revenue.\n"
        "- The model was trained and evaluated only on the historical 2010–2011 dataset using chronological splits.\n"
        "- The displayed forecast uses the latest historical feature row available in this project; it is a historical demonstration, not a live or current forecast."
    )
