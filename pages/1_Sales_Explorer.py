"""Filtered transaction exploration."""

import streamlit as st

from src.dashboard import daily_from_transactions, empty_state, filters, load_data, metric_strip, page_shell

page_shell("Sales Explorer", "Investigate transaction history by date, customer country, and product.")
transactions = load_data()[0]
rows = filters(transactions, "explorer")
if rows.empty:
    empty_state()
    st.stop()
metric_strip(rows)
tab_trend, tab_products, tab_rows = st.tabs(["Revenue over time", "Category comparison", "Transactions"])
with tab_trend:
    series = daily_from_transactions(rows)
    st.caption("Daily totals from the selected transaction rows; unlike the model series, no missing dates are filled with zero.")
    st.line_chart(series.rename("Recorded revenue"), color="#148a7c", height=350)
    monthly = rows.assign(Month=rows["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month")["Revenue"].sum()
    st.bar_chart(monthly.rename("Recorded revenue"), color="#4678c8", height=250)
with tab_products:
    dimension = st.radio("Compare by", ["Product", "Country"], horizontal=True, key="explorer_dimension")
    column = "Product" if dimension == "Product" else "Country"
    summary = rows.groupby(column).agg(Recorded_revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"), Invoices=("InvoiceNo", "nunique")).sort_values("Recorded_revenue", ascending=False)
    chart_data = summary["Recorded_revenue"].head(15).reset_index()
    st.bar_chart(chart_data, x=column, y="Recorded_revenue", color="#4678c8", height=350)
    st.dataframe(summary, width="stretch", height=380)
with tab_rows:
    st.caption("Sortable transaction details. Revenue follows the project’s net recorded revenue definition.")
    columns = ["Date", "InvoiceNo", "StockCode", "Product", "Country", "Quantity", "UnitPrice", "Revenue", "IsCancellation"]
    st.dataframe(rows[columns].sort_values("Date", ascending=False), width="stretch", height=520, hide_index=True)
