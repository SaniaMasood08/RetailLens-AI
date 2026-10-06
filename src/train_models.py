"""Train baseline revenue forecasters with chronological validation and test."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_results.csv"

PREDICTORS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
]
TARGET = "target_DailyRevenue"
FORBIDDEN_PREDICTORS = {"Date", "target_date", TARGET}


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for one chronological evaluation set."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def run_training() -> pd.DataFrame:
    """Compare baselines on validation, then test the selected model once."""
    data = pd.read_csv(FEATURES_PATH, parse_dates=["Date", "target_date"])
    if data.empty:
        raise ValueError("The feature table is empty")

    missing_columns = set(PREDICTORS + [TARGET, "Date", "target_date"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")
    if set(PREDICTORS) & FORBIDDEN_PREDICTORS:
        raise AssertionError("A forbidden date or target column is configured as a predictor")
    if data[PREDICTORS + [TARGET]].isna().any().any():
        raise ValueError("Predictors or target contain NaN values")
    if not np.isfinite(data[PREDICTORS + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Predictors or target contain infinite values")

    # Rows must remain in time order. Date is the forecast origin and
    # target_date is the following day; neither date column enters model X.
    if not data["Date"].is_monotonic_increasing or data["Date"].duplicated().any():
        raise AssertionError("Forecast-origin dates are not strictly chronological")
    if not data["target_date"].is_monotonic_increasing:
        raise AssertionError("Target dates are not chronologically ordered")
    if not (data["target_date"] == data["Date"] + pd.Timedelta(days=1)).all():
        raise AssertionError("Each target must be the next calendar day")
    if not (data["Date"] < data["target_date"]).all():
        raise AssertionError("A predictor row must precede its target date")

    # Deterministic contiguous split: 70% train, 15% validation, 15% test.
    # Integer boundaries preserve all 345 rows: 241 / 52 / 52.
    row_count = len(data)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")

    train_dates = set(train["Date"])
    validation_dates = set(validation["Date"])
    test_dates = set(test["Date"])
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()
    assert train["target_date"].max() < validation["target_date"].min()
    assert validation["target_date"].max() < test["target_date"].min()

    print(f"Dataset shape: {data.shape}")
    print(f"Predictor features: {PREDICTORS}")
    print(f"Target: {TARGET}")
    print(f"Train: {len(train)} rows; forecast dates {train['Date'].min().date()} to {train['Date'].max().date()}; target dates {train['target_date'].min().date()} to {train['target_date'].max().date()}")
    print(f"Validation: {len(validation)} rows; forecast dates {validation['Date'].min().date()} to {validation['Date'].max().date()}; target dates {validation['target_date'].min().date()} to {validation['target_date'].max().date()}")
    print(f"Test: {len(test)} rows; forecast dates {test['Date'].min().date()} to {test['Date'].max().date()}; target dates {test['target_date'].min().date()} to {test['target_date'].max().date()}")
    print("Leakage assertions passed: chronological, disjoint splits; Date, target_date, and target excluded from predictors; target follows forecast date.")

    # Fit each candidate using training rows only; validation MAE chooses the
    # model. The untouched test partition is not evaluated during selection.
    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
    X_train = train[PREDICTORS]
    y_train = train[TARGET]
    X_validation = validation[PREDICTORS]
    y_validation = validation[TARGET]
    X_test = test[PREDICTORS]
    y_test = test[TARGET]
    validation_results: list[dict[str, object]] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = calculate_metrics(y_validation, model.predict(X_validation))
        validation_results.append({"model": model_name, "split": "validation", **metrics})

    # This selection set is validation-only by construction and assertion.
    assert validation_results
    assert all(result["split"] == "validation" for result in validation_results)
    selected_result = min(validation_results, key=lambda result: result["MAE"])
    selected_model_name = str(selected_result["model"])
    print("\nValidation model comparison (selection based on lowest validation MAE):")
    for result in validation_results:
        print(
            f"{result['model']}: MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, R2={result['R2']:.4f}"
        )
    print(f"Selected model: {selected_model_name} (lowest validation MAE={selected_result['MAE']:.4f})")
    print("Test partition was not used for model selection: True")

    # Refit only the selected model on train + validation, then evaluate once
    # on test. Do not inspect test metrics until after selection is complete.
    combined_train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = models[selected_model_name]
    final_model.fit(combined_train_validation[PREDICTORS], combined_train_validation[TARGET])
    test_metrics = calculate_metrics(y_test, final_model.predict(X_test))
    results = pd.DataFrame(
        validation_results + [
            {"model": selected_model_name, "split": "test", **test_metrics}
        ],
        columns=["model", "split", "MAE", "RMSE", "R2"],
    )
    if not np.isfinite(results[["MAE", "RMSE", "R2"]].to_numpy()).all():
        raise AssertionError("Evaluation results contain NaN or infinite metrics")
    results.to_csv(RESULTS_PATH, index=False)

    print("\nFinal test evaluation (selected model retrained on train + validation; evaluated once):")
    print(f"Final test MAE: {test_metrics['MAE']:.4f}")
    print(f"Final test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Final test R2: {test_metrics['R2']:.4f}")
    print(f"Saved output path: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_training()
