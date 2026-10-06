"""Clean the Online Retail workbook and build a daily net-revenue series."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WORKBOOK = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def preprocess() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return cleaned transactions, daily revenue, and data-derived summary."""
    # Read the source only; all outputs are written to data/processed so the
    # original workbook remains untouched.
    transactions = pd.read_excel(RAW_WORKBOOK)
    original_row_count = len(transactions)

    # Deduplicate on every source column before adding any derived columns.
    exact_duplicate_count = int(transactions.duplicated(keep="first").sum())
    transactions = transactions.drop_duplicates(keep="first").copy()

    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])

    # Cancellation invoices are retained and marked; their negative quantities
    # contribute to net recorded revenue just like other transaction rows.
    transactions["IsCancellation"] = (
        transactions["InvoiceNo"].astype("string").str.startswith("C", na=False)
    )

    # Negative-quantity rows are intentionally retained. Negative unit prices
    # are excluded because these two source rows are bad-debt adjustments,
    # rather than normal product sales. Zero unit prices remain valid records.
    negative_unit_price_mask = transactions["UnitPrice"] < 0
    negative_unit_price_count = int(negative_unit_price_mask.sum())
    transactions = transactions.loc[~negative_unit_price_mask].copy()

    transactions["Revenue"] = transactions["Quantity"] * transactions["UnitPrice"]
    transactions["Date"] = transactions["InvoiceDate"].dt.normalize()

    # Sum all retained records by calendar date to measure net recorded revenue.
    daily_revenue = (
        transactions.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .rename(columns={"Revenue": "DailyRevenue"})
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(PROCESSED_DIR / "cleaned_transactions.csv", index=False)
    daily_revenue.to_csv(PROCESSED_DIR / "daily_revenue.csv", index=False)

    summary: dict[str, object] = {
        "original row count": original_row_count,
        "exact duplicate rows removed": exact_duplicate_count,
        "negative UnitPrice rows removed": negative_unit_price_count,
        "final transaction row count": len(transactions),
        "minimum date": daily_revenue["Date"].min().date(),
        "maximum date": daily_revenue["Date"].max().date(),
        "number of daily records": len(daily_revenue),
        "minimum DailyRevenue": daily_revenue["DailyRevenue"].min(),
        "maximum DailyRevenue": daily_revenue["DailyRevenue"].max(),
        "total DailyRevenue": daily_revenue["DailyRevenue"].sum(),
    }
    return transactions, daily_revenue, summary


if __name__ == "__main__":
    _, _, preprocessing_summary = preprocess()
    print("Preprocessing summary:")
    for label, value in preprocessing_summary.items():
        print(f"{label}: {value}")
