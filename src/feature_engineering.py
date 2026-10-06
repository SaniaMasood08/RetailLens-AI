"""Build leakage-safe next-calendar-day revenue forecasting features."""

from pathlib import Path
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_PATH = PROCESSED_DIR / "daily_revenue.csv"
TRANSACTIONS_PATH = PROCESSED_DIR / "cleaned_transactions.csv"
OUTPUT_PATH = PROCESSED_DIR / "ml_features.csv"


def create_features() -> tuple[pd.DataFrame, int, int]:
    """Create one-step-ahead features, zero-filling dates without transactions.

    Each row's ``Date`` is the forecast origin t. All revenue features use
    dates before t. ``target_DailyRevenue`` is revenue on t + 1 calendar day.
    Calendar fields describe the target day, which is known before forecasting.
    """
    daily = pd.read_csv(DAILY_PATH, parse_dates=["Date"])
    required_columns = {"Date", "DailyRevenue"}
    if not required_columns.issubset(daily.columns):
        raise ValueError(f"daily_revenue.csv must contain {sorted(required_columns)}")
    if daily["Date"].isna().any() or daily["DailyRevenue"].isna().any():
        raise ValueError("Daily data contains missing Date or DailyRevenue values")
    if daily["Date"].duplicated().any():
        raise ValueError("daily_revenue.csv must contain no duplicate dates")

    daily = daily.sort_values("Date").copy()
    daily["Date"] = daily["Date"].dt.normalize()

    # Check against retained transaction dates before filling gaps. This makes
    # zero mean exactly no recorded transaction rows for that calendar date.
    transaction_dates = pd.read_csv(
        TRANSACTIONS_PATH, usecols=["Date"], parse_dates=["Date"]
    )["Date"].dropna().dt.normalize().drop_duplicates()
    daily_dates = pd.DatetimeIndex(daily["Date"])
    transaction_date_index = pd.DatetimeIndex(transaction_dates)
    if len(daily_dates.difference(transaction_date_index)):
        raise ValueError("Daily revenue contains dates absent from transaction data")
    if len(transaction_date_index.difference(daily_dates)):
        raise ValueError("Transaction data contains dates absent from daily revenue")

    start_date = daily["Date"].min()
    end_date = daily["Date"].max()
    calendar = pd.date_range(start_date, end_date, freq="D", name="Date")
    original_daily_dates = pd.DatetimeIndex(daily["Date"])
    missing_date_count = len(calendar.difference(original_daily_dates))

    # Per the agreed target definition, no transaction rows on a date means
    # zero *recorded* revenue. Do not extend beyond the observed date range.
    revenue = daily.set_index("Date")["DailyRevenue"].reindex(calendar).fillna(0.0)
    frame = pd.DataFrame(index=calendar)
    frame.index.name = "Date"

    # For a forecast made on t, lag_1 is t-1; all rolling windows end at t-1.
    prior_revenue = revenue.shift(1)
    frame["lag_1"] = revenue.shift(1)
    frame["lag_7"] = revenue.shift(7)
    frame["lag_14"] = revenue.shift(14)
    frame["rolling_mean_7"] = prior_revenue.rolling(window=7, min_periods=7).mean()
    frame["rolling_mean_28"] = prior_revenue.rolling(window=28, min_periods=28).mean()

    # Calendar values for t+1 are known in advance and do not use revenue.
    frame["target_date"] = frame.index + pd.Timedelta(days=1)
    frame["day_of_week"] = frame["target_date"].dt.dayofweek
    frame["month"] = frame["target_date"].dt.month

    # The last date has no following-day target in the provided data; first
    # dates lack a full 28-day history. Drop only rows with unavailable values.
    frame["target_DailyRevenue"] = revenue.shift(-1)
    candidate_row_count = len(frame)
    result = frame.dropna().reset_index()
    removed_row_count = candidate_row_count - len(result)

    # Assert the feature timing on representative rows: every revenue feature
    # equals values before its forecast date, and target is the following date.
    for row_index in sorted({0, len(result) // 2, len(result) - 1}):
        row = result.iloc[row_index]
        forecast_date = row["Date"]
        assert row["target_date"] == forecast_date + pd.Timedelta(days=1)
        assert row["lag_1"] == revenue.loc[forecast_date - pd.Timedelta(days=1)]
        assert row["lag_7"] == revenue.loc[forecast_date - pd.Timedelta(days=7)]
        assert row["lag_14"] == revenue.loc[forecast_date - pd.Timedelta(days=14)]
        expected_7 = revenue.loc[
            forecast_date - pd.Timedelta(days=7):forecast_date - pd.Timedelta(days=1)
        ].mean()
        expected_28 = revenue.loc[
            forecast_date - pd.Timedelta(days=28):forecast_date - pd.Timedelta(days=1)
        ].mean()
        assert math.isclose(row["rolling_mean_7"], expected_7, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(row["rolling_mean_28"], expected_28, rel_tol=1e-12, abs_tol=1e-12)
        assert row["target_DailyRevenue"] == revenue.loc[row["target_date"]]

    expected_columns = [
        "Date", "target_date", "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_28", "day_of_week", "month",
        "target_DailyRevenue",
    ]
    result = result[expected_columns]
    if result.isna().any().any():
        raise AssertionError("Generated feature table unexpectedly contains NaN values")
    if not result["Date"].is_monotonic_increasing:
        raise AssertionError("Forecast dates are not chronologically ordered")
    if not result["target_date"].is_monotonic_increasing:
        raise AssertionError("Target dates are not chronologically ordered")

    result.to_csv(OUTPUT_PATH, index=False)
    return result, missing_date_count, removed_row_count


if __name__ == "__main__":
    features, gap_count, removed_count = create_features()
    print(f"Calendar dates zero-filled after transaction-date verification: {gap_count}")
    print(f"Rows removed for unavailable lag, rolling, or target values: {removed_count}")
    print(f"Feature rows written: {len(features)}")
    print(f"Output: {OUTPUT_PATH}")
