"""RetailLens-AI decision intelligence built around the packaged forecast."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import load_model_bundle, predict_next_day


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSACTIONS_PATH = PROCESSED_DIR / "cleaned_transactions.csv"
DAILY_PATH = PROCESSED_DIR / "daily_revenue.csv"
FEATURES_PATH = PROCESSED_DIR / "ml_features.csv"
MODEL_RESULTS_PATH = PROCESSED_DIR / "model_results.csv"
IMPROVEMENT_RESULTS_PATH = PROCESSED_DIR / "model_improvement_results.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "retail_revenue_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="RetailLens-AI | Retail Decision Intelligence",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, ...]:
    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        parse_dates=["InvoiceDate", "Date"],
    )
    daily = pd.read_csv(DAILY_PATH, parse_dates=["Date"]).sort_values("Date")
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date", "target_date"])
    model_results = pd.read_csv(MODEL_RESULTS_PATH)
    improvement_results = pd.read_csv(IMPROVEMENT_RESULTS_PATH)
    return transactions, daily, features, model_results, improvement_results


@st.cache_resource
def load_packaged_model():
    return load_model_bundle(MODEL_PATH, METADATA_PATH)


def format_revenue(value: float) -> str:
    """Format recorded revenue without implying a currency absent from labels."""
    return f"{value:,.2f}"


def format_change(current: float, previous: float) -> str:
    if previous == 0:
        return "not defined (previous average is zero)"
    return f"{(current - previous) / abs(previous) * 100:+.1f}%"


try:
    transactions, daily, features, model_results, improvement_results = load_dashboard_data()
    model, metadata = load_packaged_model()
except Exception as error:
    st.error(f"The dashboard could not load a required project artifact: {error}")
    st.stop()

if daily.empty or features.empty or transactions.empty:
    st.error("A required processed dataset is empty.")
    st.stop()

predictor_columns = list(metadata["predictor_columns"])
features = features.sort_values("target_date").reset_index(drop=True)
latest_feature = features.iloc[-1]
feature_row = features.tail(1).loc[:, predictor_columns]
try:
    forecast = predict_next_day(feature_row, model=model, metadata=metadata)
except Exception as error:
    st.error(f"Could not generate the historical demonstration prediction: {error}")
    st.stop()

# Missing dates are zero recorded revenue under the existing calendar-day convention.
daily = daily.set_index("Date").sort_index()
calendar_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="Date")
calendar_revenue = daily["DailyRevenue"].reindex(calendar_index).fillna(0.0).astype(float)
rolling_7 = calendar_revenue.rolling(7, min_periods=7).mean()
monthly_revenue = calendar_revenue.resample("MS").sum()
total_revenue = float(calendar_revenue.sum())
average_calendar_day = float(calendar_revenue.mean())
highest_day = calendar_revenue.idxmax()
lowest_day = calendar_revenue.idxmin()
recent_7 = calendar_revenue.tail(7)
previous_7 = calendar_revenue.iloc[-14:-7]
recent_7_average = float(recent_7.mean())
previous_7_average = float(previous_7.mean()) if len(previous_7) == 7 else float("nan")
strongest_month = monthly_revenue.idxmax()
weakest_month = monthly_revenue.idxmin()

# Use only values available by the prediction origin for forecast interpretation.
forecast_origin = pd.Timestamp(latest_feature["Date"])
# Feature engineering uses revenue strictly before Date, so this is the latest
# day of revenue available to both the model and its decision-center comparison.
forecast_information_cutoff = forecast_origin - pd.Timedelta(days=1)
origin_history = calendar_revenue.loc[:forecast_information_cutoff]
forecast_recent_7 = origin_history.tail(7)
forecast_baseline = float(forecast_recent_7.mean())
origin_previous_7 = origin_history.iloc[-14:-7]
origin_recent_change_pct = (
    (float(forecast_recent_7.mean()) - float(origin_previous_7.mean())) / abs(float(origin_previous_7.mean())) * 100
    if len(origin_previous_7) == 7 and float(origin_previous_7.mean()) != 0
    else float("nan")
)
forecast_difference_pct = (
    (forecast - forecast_baseline) / abs(forecast_baseline) * 100
    if forecast_baseline != 0
    else float("nan")
)

test_result = improvement_results.loc[
    (improvement_results["model"] == metadata["selection_label"])
    & (improvement_results["split"] == "test_selected")
]
validation_results = improvement_results.loc[
    improvement_results["split"] == "validation"
].sort_values("MAE")
selected_validation = validation_results.loc[
    validation_results["model"] == metadata["selection_label"]
]
if len(test_result) != 1:
    st.error("Could not find the selected model's held-out test metrics.")
    st.stop()
test_metrics = test_result.iloc[0]

# Reuse the transaction rows for descriptive product and customer-country summaries.
transactions["Product"] = transactions["Description"].fillna("").astype(str).str.strip()
missing_product = transactions["Product"] == ""
transactions.loc[missing_product, "Product"] = (
    "StockCode " + transactions.loc[missing_product, "StockCode"].astype(str)
)
transactions["Country"] = transactions["Country"].fillna("Unknown")
product_summary = transactions.groupby(["StockCode", "Product"], dropna=False).agg(
    recorded_revenue=("Revenue", "sum"),
    quantity=("Quantity", "sum"),
    transaction_frequency=("InvoiceNo", "nunique"),
).reset_index()
transactions["Date"] = pd.to_datetime(transactions["Date"])
origin_transactions = transactions.loc[
    transactions["Date"] <= forecast_information_cutoff
].copy()
recent_start = forecast_information_cutoff - pd.Timedelta(days=6)
previous_start = forecast_information_cutoff - pd.Timedelta(days=13)
recent_transactions = origin_transactions.loc[origin_transactions["Date"] >= recent_start]
previous_transactions = origin_transactions.loc[
    origin_transactions["Date"].between(previous_start, recent_start - pd.Timedelta(days=1))
]
origin_product_summary = origin_transactions.groupby(["StockCode", "Product"], dropna=False).agg(
    recorded_revenue=("Revenue", "sum"), quantity=("Quantity", "sum"),
    transaction_frequency=("InvoiceNo", "nunique"),
).reset_index()
recent_product_summary = recent_transactions.groupby(["StockCode", "Product"], dropna=False).agg(
    recent_revenue=("Revenue", "sum"), recent_quantity=("Quantity", "sum"),
    recent_transaction_frequency=("InvoiceNo", "nunique"),
).reset_index()
previous_product_summary = previous_transactions.groupby(["StockCode", "Product"], dropna=False).agg(
    previous_revenue=("Revenue", "sum"), previous_quantity=("Quantity", "sum"),
).reset_index()
product_watch = origin_product_summary.merge(recent_product_summary, on=["StockCode", "Product"], how="left").fillna(
    {"recent_revenue": 0.0, "recent_quantity": 0.0, "recent_transaction_frequency": 0}
)
product_watch = product_watch.merge(previous_product_summary, on=["StockCode", "Product"], how="left").fillna(
    {"previous_revenue": 0.0, "previous_quantity": 0.0}
)
product_watch["recent_change_pct"] = (
    (product_watch["recent_revenue"] - product_watch["previous_revenue"])
    / product_watch["previous_revenue"].abs().replace(0, float("nan")) * 100
)
country_summary = transactions.groupby("Country").agg(
    recorded_revenue=("Revenue", "sum"),
    quantity=("Quantity", "sum"),
    transaction_activity=("InvoiceNo", "nunique"),
).reset_index()
origin_country_summary = origin_transactions.groupby("Country").agg(
    recorded_revenue=("Revenue", "sum"), quantity=("Quantity", "sum"),
    transaction_activity=("InvoiceNo", "nunique"),
).reset_index()
cancellation_mask = transactions["IsCancellation"].astype(str).str.lower().isin(["true", "1"])
cancellation_count = int(cancellation_mask.sum())
cancellation_rate = cancellation_count / len(transactions) * 100
weekday_average = calendar_revenue.groupby(calendar_revenue.index.day_name()).mean()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_average = weekday_average.reindex(weekday_order).dropna()
strongest_weekday = weekday_average.idxmax()
highest_country = country_summary.loc[country_summary["recorded_revenue"].idxmax()]
highest_product = product_summary.loc[product_summary["recorded_revenue"].idxmax()]
top_five_product_revenue = product_summary.nlargest(5, "recorded_revenue")["recorded_revenue"].sum()
top_five_product_share = top_five_product_revenue / total_revenue * 100 if total_revenue else float("nan")

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    if len(importance_values) != len(predictor_columns):
        st.error("Packaged model feature importances do not match the saved predictor metadata.")
        st.stop()
    feature_importance = pd.DataFrame(
        {"Feature": predictor_columns, "Importance": importance_values}
    ).sort_values("Importance", ascending=False)
else:
    feature_importance = pd.DataFrame(columns=["Feature", "Importance"])

st.title("RetailLens-AI")
st.subheader("Retail sales decision intelligence")
st.write(
    "Explore recorded revenue trends, product and customer-country summaries, "
    "and the existing next-day revenue model. Revenue is not profit."
)
st.caption(
    f"Packaged model: {metadata['model_name']} · "
    "forecast horizon: next calendar day · historical data period only"
)

st.markdown("### 1. Executive sales overview")
kpi_columns = st.columns(4)
kpi_columns[0].metric("Total recorded revenue", format_revenue(total_revenue))
kpi_columns[1].metric("Recorded transaction days", f"{len(daily):,}")
kpi_columns[2].metric("Average per calendar day", format_revenue(average_calendar_day))
kpi_columns[3].metric("Recent 7-day average", format_revenue(recent_7_average))
more_kpis = st.columns(2)
more_kpis[0].metric(
    "Highest recorded daily revenue",
    format_revenue(float(calendar_revenue.loc[highest_day])),
    help=highest_day.strftime("%Y-%m-%d"),
)
more_kpis[1].metric(
    "Lowest recorded daily revenue",
    format_revenue(float(calendar_revenue.loc[lowest_day])),
    help=lowest_day.strftime("%Y-%m-%d"),
)
st.caption(
    "Calendar dates with no processed transactions are represented as zero "
    "recorded revenue under the project's existing assumption. Revenue does "
    "not represent profit."
)

st.markdown("### 2. Sales trend intelligence")
trend_tab, monthly_tab = st.tabs(["Daily and rolling trend", "Monthly revenue"])
with trend_tab:
    figure, axis = plt.subplots(figsize=(13, 4.4))
    axis.plot(calendar_revenue.index, calendar_revenue.values, color="#94a3b8", linewidth=0.9, label="Daily recorded revenue")
    axis.plot(rolling_7.index, rolling_7.values, color="#2563eb", linewidth=2, label="Rolling 7-day average")
    axis.set_title("Daily Recorded Revenue and Rolling 7-Day Average", loc="left", fontsize=13, weight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Recorded revenue (dataset units)")
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.autofmt_xdate()
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    if len(previous_7) == 7:
        st.write(
            f"Latest 7-day average: **{format_revenue(recent_7_average)}** · "
            f"previous 7-day average: **{format_revenue(previous_7_average)}** · "
            f"change: **{format_change(recent_7_average, previous_7_average)}**"
        )
    st.caption("The comparison uses the latest seven and preceding seven calendar days in the observed series.")

with monthly_tab:
    figure, axis = plt.subplots(figsize=(13, 4.0))
    axis.bar(monthly_revenue.index, monthly_revenue.values, width=22, color="#2563eb")
    axis.set_title("Monthly Recorded Revenue", loc="left", fontsize=13, weight="bold")
    axis.set_xlabel("Month")
    axis.set_ylabel("Recorded revenue (dataset units)")
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    figure.autofmt_xdate()
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    period_columns = st.columns(2)
    period_columns[0].metric(
        "Strongest recorded month",
        strongest_month.strftime("%B %Y"),
        format_revenue(float(monthly_revenue.loc[strongest_month])),
    )
    period_columns[1].metric(
        "Weakest recorded month",
        weakest_month.strftime("%B %Y"),
        format_revenue(float(monthly_revenue.loc[weakest_month])),
    )

st.markdown("### 3. Forecast & decision center")
st.info(
    "Historical next-day forecast demonstration using the latest available "
    "feature row. This is not a live 2026 forecast. The recent baseline ends at "
    f"the latest revenue date available to the model "
    f"({forecast_information_cutoff:%Y-%m-%d}) and does not include "
    "forecast-origin or target-date revenue."
)
forecast_columns = st.columns(3)
forecast_columns[0].metric("Forecast origin", forecast_origin.strftime("%Y-%m-%d"))
forecast_columns[1].metric("Target date", pd.Timestamp(latest_feature["target_date"]).strftime("%Y-%m-%d"))
forecast_columns[2].metric("Predicted recorded revenue", format_revenue(forecast))
if not pd.isna(forecast_difference_pct):
    forecast_difference = forecast - forecast_baseline
    relative_difference = abs(forecast_difference_pct)
    if relative_difference <= 10:
        forecast_interpretation = "CLOSE TO BASELINE"
    elif forecast_difference > 0:
        forecast_interpretation = "ABOVE RECENT BASELINE"
    else:
        forecast_interpretation = "BELOW RECENT BASELINE"
    compare_columns = st.columns(4)
    compare_columns[0].metric("Recent 7-day average baseline", format_revenue(forecast_baseline))
    compare_columns[1].metric("Absolute difference", format_revenue(abs(forecast_difference)))
    compare_columns[2].metric("Percentage difference", f"{forecast_difference_pct:+.1f}%")
    compare_columns[3].metric("Interpretation", forecast_interpretation)
else:
    st.write("The forecast comparison is undefined because the recent 7-day average is zero.")
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

with st.expander("Forecast scenario estimates"):
    st.caption(
        "Sensitivity analysis around the model forecast. Conservative and "
        "optimistic values are scenario estimates, not additional model predictions."
    )
    scenario_adjustment = st.slider(
        "Scenario adjustment around the model forecast (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Applies a symmetric percentage change to the packaged model forecast.",
    )
    adjustment = scenario_adjustment / 100
    scenario_columns = st.columns(3)
    scenario_columns[0].metric("Conservative scenario estimate", format_revenue(forecast * (1 - adjustment)))
    scenario_columns[1].metric("Expected / model forecast", format_revenue(forecast))
    scenario_columns[2].metric("Optimistic scenario estimate", format_revenue(forecast * (1 + adjustment)))
    st.caption(f"Scenario assumption: ±{scenario_adjustment}% of the historical model forecast.")

# Historical IQR thresholds for the latest-origin status panel exclude the target date.
q1 = float(origin_history.quantile(0.25))
q3 = float(origin_history.quantile(0.75))
iqr = q3 - q1
high_threshold = q3 + 1.5 * iqr
low_threshold = max(0.0, q1 - 1.5 * iqr)

st.markdown("### 4. Seller Decision Center")
st.caption("Rule-based decision support combines the existing model forecast with historical transaction summaries. It does not guarantee demand, identify causes, or calculate an optimal stock level.")
if forecast_difference_pct > 10:
    forecast_signal, forecast_direction = "Prepare for a stronger sales day", "Above recent baseline"
elif forecast_difference_pct < -10:
    forecast_signal, forecast_direction = "Consider a conservative inventory position", "Below recent baseline"
elif pd.notna(forecast_difference_pct):
    forecast_signal, forecast_direction = "Maintain normal inventory and operations", "Near recent baseline"
else:
    forecast_signal, forecast_direction = "Forecast comparison unavailable", "Unavailable"
st.info(f"**Model-based forecast signal:** {forecast_signal}. {forecast_direction} uses the transparent ±10% threshold. This is a decision-support signal based on the model forecast and recent recorded revenue, not a guaranteed demand prediction.")

st.markdown("**Current Decision Snapshot**")
if len(origin_history) >= 14:
    snapshot_recent, snapshot_previous = float(origin_history.tail(7).mean()), float(origin_history.iloc[-14:-7].mean())
    trend_status = "Improving" if snapshot_recent > snapshot_previous else "Declining" if snapshot_recent < snapshot_previous else "Stable"
else:
    trend_status = "Insufficient history"
latest_origin_revenue = float(origin_history.iloc[-1])
anomaly_status = "Unusually high" if latest_origin_revenue > high_threshold else "Unusually low" if latest_origin_revenue < low_threshold else "Normal"
snapshot = st.columns(3)
snapshot[0].metric("Historical revenue trend", trend_status)
snapshot[1].metric("Historical anomaly status", anomaly_status)
snapshot[2].metric("Model forecast vs baseline", forecast_direction)
st.write(f"**Seller signal:** {forecast_signal}. Review product and cancellation activity before the next business day.")

st.markdown("**Inventory Planning Signal**")
if forecast_difference_pct > 10:
    inventory_watch = product_watch.nlargest(5, "recorded_revenue")
    st.write("Historically high-revenue products may deserve closer inventory attention. Recent transaction activity is historical context, not expected demand.")
    st.dataframe(inventory_watch[["Product", "recent_revenue", "recent_quantity", "recorded_revenue"]].rename(columns={"recent_revenue": "Recent 7-day recorded revenue", "recent_quantity": "Recent 7-day quantity", "recorded_revenue": "Historical recorded revenue"}), hide_index=True, width="stretch")
elif forecast_difference_pct < -10:
    conservative_watch = product_watch.loc[(product_watch["quantity"] > 0) & (product_watch["recent_quantity"] < product_watch["quantity"] / max(1, len(origin_history)) * 7)].nlargest(5, "quantity")
    if conservative_watch.empty:
        st.write("No products met the high historical volume and lower recent activity rule for a stock review.")
    else:
        st.write("Review existing stock for historically high-volume products with lower recent activity before replenishment. This is a historical review, not a demand forecast.")
        st.dataframe(conservative_watch[["Product", "recent_quantity", "quantity", "recent_revenue"]].rename(columns={"recent_quantity": "Recent 7-day quantity", "quantity": "Historical quantity", "recent_revenue": "Recent 7-day recorded revenue"}), hide_index=True, width="stretch")
else:
    st.write("No directional inventory rule was triggered; consult the historical product summaries below.")

st.markdown("**Products to Watch — historical activity**")
watch_display = product_watch.nlargest(10, "recent_revenue")
st.dataframe(watch_display[["Product", "recent_revenue", "recent_quantity", "recorded_revenue", "transaction_frequency"]].rename(columns={"recent_revenue": "Recent revenue", "recent_quantity": "Recent quantity", "recorded_revenue": "Historical revenue", "transaction_frequency": "Transaction frequency (distinct invoices)"}), hide_index=True, width="stretch")
st.caption("Historical recorded metrics include negative quantities and cancellation lines under the project definition.")

st.markdown("**Products Showing Reduced Recent Activity**")
reduced_products = product_watch.loc[(product_watch["previous_revenue"] > 0) & (product_watch["recent_revenue"] < product_watch["previous_revenue"] * 0.75)].sort_values("recent_change_pct").head(10)
if reduced_products.empty:
    st.write("No products met the rule: recent 7-day recorded revenue < 75% of preceding 7-day revenue.")
else:
    st.dataframe(reduced_products[["Product", "recent_revenue", "previous_revenue", "recent_change_pct"]].rename(columns={"recent_revenue": "Recent 7-day revenue", "previous_revenue": "Previous 7-day revenue", "recent_change_pct": "Change (%)"}), hide_index=True, width="stretch")
    st.caption("Recorded activity has decreased in the latest period and may warrant review; this does not predict product failure.")

st.markdown("**Customer Market Overview — historical analysis**")
origin_country_summary["revenue_contribution_pct"] = origin_country_summary["recorded_revenue"] / origin_country_summary["recorded_revenue"].sum() * 100 if origin_country_summary["recorded_revenue"].sum() else 0.0
country_watch = origin_country_summary.nlargest(10, "recorded_revenue")
st.dataframe(country_watch[["Country", "recorded_revenue", "revenue_contribution_pct", "quantity", "transaction_activity"]].rename(columns={"recorded_revenue": "Historical revenue", "revenue_contribution_pct": "Revenue contribution (%)", "quantity": "Quantity", "transaction_activity": "Transaction activity (distinct invoices)"}), hide_index=True, width="stretch")
if not country_watch.empty:
    lead_market = country_watch.iloc[0]
    st.write(f"{lead_market['Country']} accounts for {lead_market['revenue_contribution_pct']:.1f}% of recorded revenue in this dataset.")
st.caption("Country totals describe historical activity and do not establish future growth potential.")

st.markdown("**Cancellation Watch — operational review**")
cancel_detail = origin_transactions.loc[origin_transactions.index.isin(transactions.index[cancellation_mask])]
if cancel_detail.empty:
    st.write("No cancellation records are available through the forecast origin.")
else:
    cancel_by_product = cancel_detail.groupby(["StockCode", "Product"], dropna=False).agg(cancellation_count=("InvoiceNo", "size")).reset_index()
    all_by_product = origin_transactions.groupby(["StockCode", "Product"], dropna=False).agg(total_transaction_count=("InvoiceNo", "size")).reset_index()
    cancel_by_product = cancel_by_product.merge(all_by_product, on=["StockCode", "Product"], how="left")
    cancel_by_product["cancellation_pct"] = cancel_by_product["cancellation_count"] / cancel_by_product["total_transaction_count"] * 100
    cancel_by_product = cancel_by_product.nlargest(10, "cancellation_count")
    st.dataframe(cancel_by_product[["Product", "cancellation_count", "total_transaction_count", "cancellation_pct"]].rename(columns={"cancellation_count": "Cancellation transaction count", "total_transaction_count": "Total transaction count", "cancellation_pct": "Cancellation share (%)"}), hide_index=True, width="stretch")
    st.caption("These products have a relatively high number of cancellation records and may warrant operational review. Counts do not explain why cancellations happened.")

st.markdown("**What should I review today?**")
recommendations = []
if pd.notna(origin_recent_change_pct) and origin_recent_change_pct > 0:
    recommendations.append(f"Recorded revenue through the forecast origin is above the preceding 7-day average ({origin_recent_change_pct:+.1f}%).")
if forecast_difference_pct > 10:
    recommendations.append("Review inventory for the highest historical recorded-revenue products before the next business day.")
elif forecast_difference_pct < -10:
    recommendations.append("The next-day forecast is below the recent baseline; consider a conservative replenishment approach.")
if not reduced_products.empty:
    recommendations.append(f"{reduced_products.iloc[0]['Product']} has a notable decline in recent recorded activity.")
if not cancel_detail.empty and not cancel_by_product.empty:
    recommendations.append(f"{cancel_by_product.iloc[0]['Product']} has relatively high cancellation activity and may warrant operational review.")
if recommendations:
    for recommendation in recommendations[:5]:
        st.markdown(f"- {recommendation}")
else:
    st.write("No review recommendation rules were triggered by the available data.")
st.caption("Recommendations are rule-based decision support, not guarantees of future demand or explanations of causes.")

st.markdown("### 5. Product intelligence")
st.caption(
    "Product summaries use cleaned transaction lines. Recorded revenue and "
    "quantity are aggregated as present in the data, including cancellation lines. "
    "Rankings describe one metric and do not establish that an item is objectively best."
)
transactions["Product"] = transactions["Description"].fillna("").astype(str).str.strip()
transactions.loc[transactions["Product"] == "", "Product"] = (
    "StockCode " + transactions.loc[transactions["Product"] == "", "StockCode"].astype(str)
)
product_tabs = st.tabs(["Highest recorded revenue", "Highest recorded quantity", "Most frequently transacted"])
for tab, column, label in zip(
    product_tabs,
    ["recorded_revenue", "quantity", "transaction_frequency"],
    ["Recorded revenue", "Recorded quantity", "Distinct invoice count"],
):
    with tab:
        top_products = product_summary.nlargest(10, column).copy()
        top_products[label] = top_products[column].map(
            format_revenue if column == "recorded_revenue" else lambda value: f"{value:,.0f}"
        )
        st.dataframe(top_products[["Product", label]], hide_index=True, width="stretch")
cancellation_products = transactions.loc[transactions["IsCancellation"].astype(str).str.lower().isin(["true", "1"])]
if cancellation_products.empty:
    st.write("No cancellation records are marked in the cleaned transaction data.")
else:
    cancellation_summary = cancellation_products.groupby(["StockCode", "Product"], dropna=False).agg(
        cancellation_lines=("InvoiceNo", "size"),
        cancellation_invoices=("InvoiceNo", "nunique"),
    ).reset_index().sort_values("cancellation_lines", ascending=False).head(10)
    st.markdown("**Products associated with cancellation records**")
    st.caption("These counts identify products on rows flagged as cancellations; they do not establish a reason for cancellation.")
    st.dataframe(
        cancellation_summary[["Product", "cancellation_lines", "cancellation_invoices"]],
        hide_index=True,
        width="stretch",
    )

st.markdown("### 6. Customer-country intelligence")
st.caption("This is an aggregation by the customer-country field recorded in the dataset; no demographic attributes are inferred.")
transactions["Country"] = transactions["Country"].fillna("Unknown")
country_summary = transactions.groupby("Country").agg(
    recorded_revenue=("Revenue", "sum"),
    quantity=("Quantity", "sum"),
).reset_index()
country_summary["revenue_contribution_pct"] = (
    country_summary["recorded_revenue"] / country_summary["recorded_revenue"].sum() * 100
    if country_summary["recorded_revenue"].sum() != 0
    else 0.0
)
country_tabs = st.tabs(["Recorded revenue", "Recorded quantity"])
for tab, column, label in zip(
    country_tabs,
    ["recorded_revenue", "quantity"],
    ["Recorded revenue", "Recorded quantity"],
):
    with tab:
        top_countries = country_summary.nlargest(10, column).copy()
        top_countries[label] = top_countries[column].map(
            format_revenue if column == "recorded_revenue" else lambda value: f"{value:,.0f}"
        )
        if column == "recorded_revenue":
            top_countries["Share of total recorded revenue"] = top_countries["revenue_contribution_pct"].map(lambda value: f"{value:.1f}%")
            st.dataframe(top_countries[["Country", label, "Share of total recorded revenue"]], hide_index=True, width="stretch")
        else:
            st.dataframe(top_countries[["Country", label]], hide_index=True, width="stretch")

st.markdown("### 7. Business alerts")
st.caption(
    "Rule-based historical flags describe unusual values in this series; they "
    "do not explain why a change occurred."
)
q1 = float(calendar_revenue.quantile(0.25))
q3 = float(calendar_revenue.quantile(0.75))
iqr = q3 - q1
high_threshold = q3 + 1.5 * iqr
low_threshold = max(0.0, q1 - 1.5 * iqr)
high_days = calendar_revenue[calendar_revenue > high_threshold]
low_days = calendar_revenue[calendar_revenue < low_threshold]
alert_columns = st.columns(2)
with alert_columns[0]:
    st.markdown("**Unusually high recorded-revenue days**")
    st.write(f"Rule: daily revenue > Q3 + 1.5 × IQR = {format_revenue(high_threshold)}")
    st.write(f"Flagged days: {len(high_days)}")
    if not high_days.empty:
        st.dataframe(
            pd.DataFrame({"Date": high_days.index.strftime("%Y-%m-%d"), "Recorded revenue": high_days.values})
            .sort_values("Recorded revenue", ascending=False).head(5),
            hide_index=True,
            width="stretch",
        )
with alert_columns[1]:
    st.markdown("**Unusually low recorded-revenue days**")
    st.write(f"Rule: daily revenue < max(0, Q1 − 1.5 × IQR) = {format_revenue(low_threshold)}")
    st.write(f"Flagged days: {len(low_days)}")
    if not low_days.empty:
        st.dataframe(
            pd.DataFrame({"Date": low_days.index.strftime("%Y-%m-%d"), "Recorded revenue": low_days.values})
            .sort_values("Recorded revenue").head(5),
            hide_index=True,
            width="stretch",
        )
if len(previous_7) == 7:
    trend_direction = "increased" if recent_7_average > previous_7_average else "decreased" if recent_7_average < previous_7_average else "was unchanged"
    st.info(
        f"Recent recorded revenue {trend_direction}: the 7-day average moved from "
        f"{format_revenue(previous_7_average)} to {format_revenue(recent_7_average)} "
        f"({format_change(recent_7_average, previous_7_average)}). This comparison "
        "is descriptive and does not identify a cause."
    )

anomaly_days = pd.concat([high_days, low_days]).sort_index()
st.markdown("**Anomaly explorer**")
if anomaly_days.empty:
    st.write("No dates meet the displayed IQR thresholds in the observed series.")
else:
    selected_anomaly = st.selectbox(
        "Select a high or low revenue anomaly date",
        options=list(anomaly_days.index),
        format_func=lambda date: f"{date:%Y-%m-%d} — {format_revenue(float(anomaly_days.loc[date]))}",
    )
    anomaly_revenue = float(anomaly_days.loc[selected_anomaly])
    if anomaly_revenue > high_threshold:
        anomaly_type = "high"
        anomaly_deviation = anomaly_revenue - high_threshold
        normal_range = f"upper IQR threshold {format_revenue(high_threshold)}"
    else:
        anomaly_type = "low"
        anomaly_deviation = low_threshold - anomaly_revenue
        normal_range = f"lower IQR threshold {format_revenue(low_threshold)}"
    st.write(
        f"Selected {anomaly_type} anomaly: **{format_revenue(anomaly_revenue)}**; "
        f"deviation beyond the {normal_range}: **{format_revenue(anomaly_deviation)}**."
    )
    anomaly_transactions = transactions.loc[transactions["Date"] == selected_anomaly]
    if anomaly_transactions.empty:
        st.write("No matching transaction detail is available for this date.")
    else:
        anomaly_columns = st.columns(2)
        anomaly_products = anomaly_transactions.groupby("Product")["Revenue"].sum().nlargest(5)
        anomaly_country = anomaly_transactions.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
        with anomaly_columns[0]:
            st.markdown("**Top products by net recorded revenue that date**")
            st.dataframe(
                anomaly_products.rename("Net recorded revenue").reset_index(),
                hide_index=True,
                width="stretch",
            )
        with anomaly_columns[1]:
            st.markdown("**Customer-country contribution that date**")
            country_detail = anomaly_country.rename("Net recorded revenue").reset_index()
            country_detail["Share of date total"] = (
                country_detail["Net recorded revenue"] / anomaly_revenue * 100
                if anomaly_revenue != 0
                else float("nan")
            )
            st.dataframe(country_detail.head(5), hide_index=True, width="stretch")
    st.caption("These are date-level associations in the processed transactions; they do not explain the anomaly's cause.")

st.markdown("### 8. Descriptive business insights")
st.caption("Automatically generated from the processed historical data. These summaries describe this dataset and do not establish real-world causes or generalize to other retailers.")
insight_columns = st.columns(3)
insight_columns[0].metric(
    "Strongest recorded month",
    strongest_month.strftime("%B %Y"),
    format_revenue(float(monthly_revenue.loc[strongest_month])),
)
insight_columns[1].metric(
    "Weakest observed month",
    weakest_month.strftime("%B %Y"),
    format_revenue(float(monthly_revenue.loc[weakest_month])),
)
insight_columns[2].metric(
    "Strongest weekday by average",
    strongest_weekday,
    format_revenue(float(weekday_average.loc[strongest_weekday])),
)
insight_columns = st.columns(3)
insight_columns[0].metric(
    "Highest-revenue customer country",
    highest_country["Country"],
    format_revenue(float(highest_country["recorded_revenue"])),
)
insight_columns[1].metric(
    "Highest-revenue product",
    str(highest_product["Product"])[:36],
    format_revenue(float(highest_product["recorded_revenue"])),
    help=str(highest_product["Product"]),
)
insight_columns[2].metric(
    "Cancellation transaction lines",
    f"{cancellation_count:,} ({cancellation_rate:.1f}%)",
    f"{transactions.loc[cancellation_mask, 'InvoiceNo'].nunique():,} distinct invoices",
)
if len(previous_7) == 7:
    trend_word = "up" if recent_7_average > previous_7_average else "down" if recent_7_average < previous_7_average else "unchanged"
    st.write(
        f"**Recent trend:** the latest 7-day average is {trend_word} versus the prior "
        f"7 days ({format_change(recent_7_average, previous_7_average)}). "
        f"**Concentration:** the five highest-revenue products account for "
        f"{top_five_product_share:.1f}% of total net recorded revenue in the processed data."
    )

st.markdown("### 9. Model performance")
st.write(f"**Selected model:** {metadata['model_name']} · selection label: `{metadata['selection_label']}`")
st.caption("Validation results were used for model comparison and selection. The chronological final test set was not used for model selection or fitting.")
st.markdown("**Validation comparison (sorted by MAE)**")
st.dataframe(validation_results[["model", "MAE", "RMSE", "R2"]].reset_index(drop=True), hide_index=True, width="stretch")
if len(selected_validation) == 1:
    st.markdown("**Selected model validation metrics**")
    validation_columns = st.columns(3)
    validation_row = selected_validation.iloc[0]
    validation_columns[0].metric("Validation MAE", format_revenue(float(validation_row["MAE"])))
    validation_columns[1].metric("Validation RMSE", format_revenue(float(validation_row["RMSE"])))
    validation_columns[2].metric("Validation R²", f"{float(validation_row['R2']):.3f}")
st.markdown("**Final held-out test metrics**")
test_columns = st.columns(3)
test_columns[0].metric("MAE", format_revenue(float(test_metrics["MAE"])))
test_columns[1].metric("RMSE", format_revenue(float(test_metrics["RMSE"])))
test_columns[2].metric("R²", f"{float(test_metrics['R2']):.3f}")
methodology = metadata["training_validation_test_methodology"]
st.caption(
    f"Selection metric: {methodology['selection_metric']} · final packaged model fit used "
    f"{methodology['row_counts']['final_model_fit_training_plus_validation']} training + "
    "validation rows; test rows were excluded."
)

st.markdown("### 10. Model explainability")
if feature_importance.empty:
    st.info("Feature importance is not available for the loaded packaged model.")
else:
    importance_figure, importance_axis = plt.subplots(figsize=(9, 3.8))
    importance_plot = feature_importance.sort_values("Importance")
    importance_axis.barh(importance_plot["Feature"], importance_plot["Importance"], color="#2563eb")
    importance_axis.set_title("Packaged Random Forest Feature Importance", loc="left", fontsize=13, weight="bold")
    importance_axis.set_xlabel("Importance")
    importance_axis.set_ylabel("Feature")
    importance_axis.grid(axis="x", color="#d1d5db", linewidth=0.7, alpha=0.75)
    importance_axis.spines[["top", "right"]].set_visible(False)
    importance_figure.tight_layout()
    st.pyplot(importance_figure, width="stretch")
    plt.close(importance_figure)
    st.dataframe(feature_importance.reset_index(drop=True), hide_index=True, width="stretch")
    st.caption("Feature importance measures model reliance within this fitted model; it does not show causation or business impact.")

st.markdown("### 11. Methodology and limitations")
st.markdown(
    "Raw transaction data → preprocessing → daily recorded revenue → "
    "leakage-safe lag and rolling features → chronological train/validation/test "
    "split → validation model comparison → constrained Random Forest → "
    "packaged prediction → dashboard business summaries."
)
with st.expander("Interpretation and limitations"):
    st.markdown(
        "- The target is next-calendar-day net recorded revenue; revenue is not profit.\n"
        "- Negative quantities and cancellation records remain part of net recorded revenue.\n"
        "- Missing calendar dates within the observed range count as zero recorded revenue.\n"
        "- Product and customer-country summaries are descriptive aggregations of the provided transaction fields.\n"
        "- Historical flags use the explicitly shown IQR rule and do not imply a cause.\n"
        "- The packaged forecast uses the latest historical feature row available in this project; it is not a live or current 2026 forecast.\n"
        "- The selected model was fitted on training + validation rows. The final chronological test period was reserved for evaluation."
    )
