"""Historical inference view for the existing packaged model."""

import pandas as pd
import streamlit as st

from src.dashboard import get_forecast, load_data, page_shell, revenue

page_shell("Revenue Forecast", "Review the packaged one-day-ahead model output and its historical reference points.")
try:
    forecast, metadata, latest = get_forecast()
except Exception as error:
    st.error(f"The packaged historical forecast could not be loaded: {error}")
    st.stop()
transactions, daily, features, comparisons = load_data()
origin = pd.Timestamp(latest["Date"])
target_date = pd.Timestamp(latest["target_date"])
cutoff = origin - pd.Timedelta(days=1)
history = daily.loc[daily["Date"] <= cutoff].set_index("Date")["DailyRevenue"]
calendar = pd.date_range(history.index.min(), history.index.max(), freq="D")
history = history.reindex(calendar, fill_value=0.0)
baseline = float(history.tail(7).mean())
selected = metadata["selection_label"]
validation = comparisons.loc[(comparisons["model"] == selected) & (comparisons["split"] == "validation")]
test = comparisons.loc[(comparisons["model"] == selected) & (comparisons["split"] == "test_selected")]

st.warning(f"Historical demonstration only. The packaged feature row forecasts **{target_date:%d %b %Y}** using information available through **{cutoff:%d %b %Y}**. This is not a current forecast.")
k = st.columns(4)
k[0].metric("Forecast date", target_date.strftime("%d %b %Y"))
k[1].metric("Model estimate", revenue(forecast), help="Output of the saved RandomForestRegressor and existing prediction interface.")
k[2].metric("Recent 7-calendar-day baseline", revenue(baseline), help="Mean revenue through the model's historical information cutoff.")
k[3].metric("Difference vs baseline", f"{(forecast-baseline)/abs(baseline)*100:+.1f}%" if baseline else "Undefined")
st.subheader("Historical context")
context = pd.DataFrame({"Recent daily revenue": history.tail(28)})
st.line_chart(context, color="#9aa8bc", height=280)
st.caption(f"Forecast origin: {origin:%Y-%m-%d}. Revenue inputs stop before the origin, consistent with the stored feature timing. The baseline is a simple comparison, not a second model.")
st.subheader("Evaluation evidence")
if len(validation) == 1 and len(test) == 1:
    val = validation.iloc[0]
    metric_cols = st.columns(3)
    for col, name in zip(metric_cols, ["MAE", "RMSE", "R2"]):
        col.metric(f"Validation {name}", f"{float(val[name]):.3f}" if name == "R2" else revenue(val[name]))
    test_cols = st.columns(3)
    row = test.iloc[0]
    for col, name in zip(test_cols, ["MAE", "RMSE", "R2"]):
        col.metric(f"Held-out test {name}", f"{float(row[name]):.3f}" if name == "R2" else revenue(row[name]))
    st.caption("Validation MAE was used for model selection. The final chronological test segment was reserved for evaluation and is not a guarantee of future performance.")
else:
    st.info("Evaluation rows for the packaged model are unavailable in the results file.")
with st.expander("Model inputs"):
    labels = {"lag_1":"Previous day revenue", "lag_7":"Revenue 7 days earlier", "lag_14":"Revenue 14 days earlier", "rolling_mean_7":"Mean over prior 7 days", "rolling_mean_28":"Mean over prior 28 days", "day_of_week":"Target weekday (Monday = 0)", "month":"Target month"}
    st.dataframe(pd.DataFrame({"Feature": [labels[x] for x in metadata["predictor_columns"]], "Value": [latest[x] for x in metadata["predictor_columns"]]}), hide_index=True, width="stretch")
st.info("The model was fitted and evaluated on 2010–2011 data. Revenue is not profit, and this interface does not retrain or alter the packaged model.")
