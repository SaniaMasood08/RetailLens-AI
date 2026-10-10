"""Rule-based, evidence-linked review cues."""

import pandas as pd
import streamlit as st

from src.dashboard import get_forecast, load_data, page_shell, revenue

page_shell("Decision Center", "Transparent review cues drawn from historical activity and the packaged forecast.")
transactions, daily, _, _ = load_data()
forecast, metadata, latest = get_forecast()
cutoff = pd.Timestamp(latest["Date"]) - pd.Timedelta(days=1)
history = daily.loc[daily["Date"] <= cutoff].set_index("Date")["DailyRevenue"]
cal = pd.date_range(history.index.min(), history.index.max(), freq="D")
history = history.reindex(cal, fill_value=0.0)
recent, previous = history.tail(7), history.iloc[-14:-7]
baseline = float(recent.mean())
change = (forecast-baseline) / abs(baseline) * 100 if baseline else float("nan")

st.warning(f"All cues are historical. Forecast target: {pd.Timestamp(latest['target_date']):%Y-%m-%d}; information cutoff: {cutoff:%Y-%m-%d}.")
if pd.isna(change):
    st.info("Forecast comparison is unavailable because the recent baseline is zero.")
elif change > 10:
    st.success(f"Forecast is {change:+.1f}% above its recent baseline. Review the highest historical activity items before making an inventory decision.")
elif change < -10:
    st.warning(f"Forecast is {change:+.1f}% below its recent baseline. Review replenishment plans alongside current stock information.")
else:
    st.info(f"Forecast is within the rule-based ±10% band around the recent baseline ({change:+.1f}%).")
st.caption("The ±10% rule is a project threshold, not a statistical confidence interval or guaranteed demand signal.")

trend = "up" if len(previous) == 7 and recent.mean() > previous.mean() else "down" if len(previous) == 7 and recent.mean() < previous.mean() else "flat / insufficient history"
q1, q3 = history.quantile([.25, .75]); iqr = q3-q1
upper, lower = q3 + 1.5*iqr, max(0.0, q1 - 1.5*iqr)
last_value = float(history.iloc[-1])
anomaly = "High" if last_value > upper else "Low" if last_value < lower else "Within IQR thresholds"
cols = st.columns(3)
cols[0].metric("Recent 7-day direction", trend)
cols[1].metric("Latest cutoff-date anomaly", anomaly, help="Historical IQR rule applied through the forecast information cutoff.")
cols[2].metric("Latest cutoff-date revenue", revenue(last_value), help=cutoff.strftime("%Y-%m-%d"))

origin_transactions = transactions.loc[transactions["Date"] <= cutoff]
recent_transactions = origin_transactions.loc[origin_transactions["Date"] >= cutoff-pd.Timedelta(days=6)]
prior_transactions = origin_transactions.loc[origin_transactions["Date"].between(cutoff-pd.Timedelta(days=13), cutoff-pd.Timedelta(days=7))]
current = recent_transactions.groupby(["StockCode", "Product"], dropna=False)["Revenue"].sum().rename("Recent 7-day revenue")
prior = prior_transactions.groupby(["StockCode", "Product"], dropna=False)["Revenue"].sum().rename("Previous 7-day revenue")
products = pd.concat([current, prior], axis=1).fillna(0)
products["Change (%)"] = (products["Recent 7-day revenue"]-products["Previous 7-day revenue"])/products["Previous 7-day revenue"].abs().replace(0, float("nan"))*100
products = products.reset_index()

left, right = st.columns(2)
with left:
    st.subheader("Products with highest recent revenue")
    st.caption("Recent 7-calendar-day activity through the forecast cutoff.")
    st.dataframe(products.nlargest(10, "Recent 7-day revenue")[["Product", "Recent 7-day revenue", "Previous 7-day revenue"]], width="stretch", hide_index=True)
with right:
    st.subheader("Largest observed declines")
    declines = products.loc[products["Previous 7-day revenue"] > 0].nsmallest(10, "Change (%)")
    if declines.empty:
        st.write("No comparable products have positive revenue in the preceding window.")
    else:
        st.dataframe(declines[["Product", "Recent 7-day revenue", "Previous 7-day revenue", "Change (%)"]], width="stretch", hide_index=True)

cancellations = origin_transactions.loc[origin_transactions["IsCancellation"]]
st.subheader("Cancellation activity")
if cancellations.empty:
    st.write("No cancellation rows are recorded through the forecast cutoff.")
else:
    cancel_products = cancellations.groupby("Product").agg(Cancellation_lines=("InvoiceNo", "size"), Distinct_invoices=("InvoiceNo", "nunique")).sort_values("Cancellation_lines", ascending=False).head(10)
    st.dataframe(cancel_products, width="stretch")
st.caption("These cues describe observed transactions. They do not explain causes, estimate optimal stock, or replace current inventory and operational information.")
