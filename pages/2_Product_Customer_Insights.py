"""Descriptive product, country, and cancellation analysis."""

import streamlit as st

from src.dashboard import empty_state, filters, load_data, page_shell

page_shell("Product & Customer Insights", "Compare recorded activity across products and customer-country fields.")
transactions = load_data()[0]
rows = filters(transactions, "insights")
if rows.empty:
    empty_state()
    st.stop()
product, country, cancellation = st.tabs(["Products", "Countries", "Cancellations"])
with product:
    product_summary = rows.groupby(["StockCode", "Product"], dropna=False).agg(
        Recorded_revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"),
        Distinct_invoices=("InvoiceNo", "nunique"), Transaction_lines=("InvoiceNo", "size"),
    ).sort_values("Recorded_revenue", ascending=False)
    st.caption("Product totals include retained negative quantities and cancellation lines.")
    product_chart = product_summary["Recorded_revenue"].head(12).reset_index()
    st.bar_chart(product_chart, x="Product", y="Recorded_revenue", color="#148a7c", height=320)
    st.dataframe(product_summary, width="stretch", height=440)
with country:
    country_summary = rows.groupby("Country").agg(
        Recorded_revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"),
        Distinct_invoices=("InvoiceNo", "nunique"), Transaction_lines=("InvoiceNo", "size"),
    ).sort_values("Recorded_revenue", ascending=False)
    st.caption("Country is the value recorded in the source transaction; it does not imply demographics or market potential.")
    st.bar_chart(country_summary["Recorded_revenue"].head(15), color="#4678c8", height=320)
    st.dataframe(country_summary, width="stretch", height=440)
with cancellation:
    cancelled = rows.loc[rows["IsCancellation"]]
    c = st.columns(3)
    c[0].metric("Cancellation lines", f"{len(cancelled):,}")
    c[1].metric("Cancellation invoices", f"{cancelled['InvoiceNo'].nunique():,}")
    c[2].metric("Net recorded revenue on these rows", f"{cancelled['Revenue'].sum():,.2f}")
    if cancelled.empty:
        st.info("No cancellation rows match the current filters.")
    else:
        detail = cancelled.groupby(["StockCode", "Product"], dropna=False).agg(
            Cancellation_lines=("InvoiceNo", "size"), Distinct_invoices=("InvoiceNo", "nunique"),
            Recorded_revenue=("Revenue", "sum"),
        ).sort_values("Cancellation_lines", ascending=False)
        st.dataframe(detail, width="stretch", height=450)
        st.caption("Cancellation counts identify flagged rows only; they do not establish why a cancellation occurred.")
