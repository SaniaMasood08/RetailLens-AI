"""Shared data access and presentation helpers for the Streamlit pages."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model_bundle, predict_next_day


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL = ROOT / "models" / "retail_revenue_model.joblib"
METADATA = ROOT / "models" / "model_metadata.json"


@st.cache_data(show_spinner="Loading the processed retail history…")
def load_data():
    """Load the committed historical datasets once per Streamlit session."""
    try:
        transactions = pd.read_csv(DATA / "cleaned_transactions.csv", parse_dates=["InvoiceDate", "Date"])
        daily = pd.read_csv(DATA / "daily_revenue.csv", parse_dates=["Date"]).sort_values("Date")
        features = pd.read_csv(DATA / "ml_features.csv", parse_dates=["Date", "target_date"])
        comparisons = pd.read_csv(DATA / "model_improvement_results.csv")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"RetailLens could not load a required processed data file. Check data/processed and try again. Details: {error}")
        st.stop()
    transactions["Product"] = transactions["Description"].fillna("").astype(str).str.strip()
    blank = transactions["Product"].eq("")
    transactions.loc[blank, "Product"] = "StockCode " + transactions.loc[blank, "StockCode"].astype(str)
    transactions["Country"] = transactions["Country"].fillna("Unknown")
    transactions["IsCancellation"] = transactions["IsCancellation"].astype(str).str.lower().isin(["true", "1"])
    return transactions, daily, features, comparisons


@st.cache_resource
def load_bundle():
    return load_model_bundle(MODEL, METADATA)


def get_forecast():
    try:
        model, metadata = load_bundle()
        features = load_data()[2].sort_values("target_date")
        if features.empty:
            raise ValueError("the saved feature dataset is empty")
        latest = features.iloc[-1]
        forecast = predict_next_day(features.tail(1).loc[:, metadata["predictor_columns"]], model, metadata)
    except Exception as error:
        st.error(f"The packaged historical forecast is unavailable. Check the model artifacts and feature data. Details: {error}")
        st.stop()
    return float(forecast), metadata, latest


def page_shell(title: str, subtitle: str):
    st.set_page_config(page_title=f"{title} · RetailLens-AI", page_icon="◈", layout="wide")
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
      [data-testid="stSidebar"] { background: #101b2d; }
      [data-testid="stSidebar"] * { color: #e5edf7; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebdd0; }
      h1,h2,h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; color:#16243a; }
      h1 { font-weight: 800; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e4eaf2; padding:16px 18px; border-radius:14px; box-shadow:0 3px 12px #172b4d08; }
      [data-testid="stMetricLabel"] { color:#68778c; }
      [data-testid="stMetricValue"] { color:#16243a; font-weight:700; }
      .rl-brand { font:800 19px Manrope,sans-serif; color:white; padding:5px 0 20px; }
      .rl-brand span { color:#57d3be; }
      .rl-eyebrow { color:#168879; text-transform:uppercase; letter-spacing:.12em; font-size:11px; font-weight:700; }
      .rl-subtitle { color:#718096; margin-top:-10px; }
      @media(max-width:700px) { .block-container{padding:1rem;} }
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="rl-brand">◈ Retail<span>Lens</span>-AI</div>', unsafe_allow_html=True)
        st.caption("SALES DECISION INTELLIGENCE")
        st.divider()
        st.markdown("Built on historical UCI Online Retail transactions.")
        st.caption("Revenue is recorded revenue, not profit.")
    st.markdown(f'<div class="rl-eyebrow">RetailLens-AI / Decision intelligence</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="rl-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def filters(transactions: pd.DataFrame, key: str):
    """Render common historical filters and return scoped transaction rows."""
    with st.sidebar:
        st.markdown("#### Explore history")
        minimum, maximum = transactions["Date"].min().date(), transactions["Date"].max().date()
        period = st.date_input("Date range", value=(minimum, maximum), min_value=minimum, max_value=maximum, key=f"{key}_date")
        countries = sorted(transactions["Country"].dropna().unique())
        chosen_countries = st.multiselect("Country", countries, key=f"{key}_country", placeholder="All countries")
        chosen_products = st.multiselect("Product", sorted(transactions["Product"].dropna().unique()), key=f"{key}_product", placeholder="All products")
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        return transactions.iloc[0:0].copy()
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    result = transactions.loc[transactions["Date"].between(start, end)]
    if chosen_countries:
        result = result.loc[result["Country"].isin(chosen_countries)]
    if chosen_products:
        result = result.loc[result["Product"].isin(chosen_products)]
    return result.copy()


def revenue(value) -> str:
    return f"{float(value):,.2f}"


def daily_from_transactions(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float, name="Revenue")
    return rows.groupby("Date")["Revenue"].sum().sort_index()


def metric_strip(rows: pd.DataFrame, label="Filtered"):
    days = rows["Date"].nunique() if not rows.empty else 0
    invoices = rows["InvoiceNo"].nunique() if not rows.empty else 0
    cols = st.columns(4)
    cols[0].metric(f"{label} recorded revenue", revenue(rows["Revenue"].sum()) if not rows.empty else "0.00")
    cols[1].metric("Transaction lines", f"{len(rows):,}")
    cols[2].metric("Distinct invoices", f"{invoices:,}")
    cols[3].metric("Active dates", f"{days:,}")


def empty_state(message="No transactions match these filters."):
    st.info(message)
