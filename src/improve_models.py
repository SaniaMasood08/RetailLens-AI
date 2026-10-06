"""Compare a small set of tree-based regressors on chronological validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_improvement_results.csv"

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


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def chronological_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the existing 70/15/15 contiguous split: 241 / 52 / 52 rows."""
    if not data["Date"].is_monotonic_increasing or data["Date"].duplicated().any():
        raise AssertionError("Feature rows must be sorted by unique forecast date")
    if not data["target_date"].is_monotonic_increasing:
        raise AssertionError("Target dates must be chronologically ordered")
    if not (data["target_date"] == data["Date"] + pd.Timedelta(days=1)).all():
        raise AssertionError("Target date must be the next calendar day")

    n_rows = len(data)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    date_sets = [set(part["Date"]) for part in (train, validation, test)]
    assert date_sets[0].isdisjoint(date_sets[1])
    assert date_sets[0].isdisjoint(date_sets[2])
    assert date_sets[1].isdisjoint(date_sets[2])
    assert train["Date"].max() < validation["Date"].min() < test["Date"].min()
    assert train["target_date"].max() < validation["target_date"].min()
    assert validation["target_date"].max() < test["target_date"].min()
    return train, validation, test


def run_comparison() -> pd.DataFrame:
    data = pd.read_csv(FEATURES_PATH, parse_dates=["Date", "target_date"])
    required = set(PREDICTORS + [TARGET, "Date", "target_date"])
    if not required.issubset(data.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(required - set(data.columns))}")
    if set(PREDICTORS) & FORBIDDEN_PREDICTORS:
        raise AssertionError("Date or target was included among predictors")
    if data[PREDICTORS + [TARGET]].isna().any().any():
        raise ValueError("Feature table contains missing predictor or target values")
    if not np.isfinite(data[PREDICTORS + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains infinite predictor or target values")

    train, validation, test = chronological_splits(data)
    if (len(train), len(validation), len(test)) != (241, 52, 52):
        raise AssertionError("Expected the established chronological 241/52/52 split")
    assert not (FORBIDDEN_PREDICTORS & set(PREDICTORS))
    # Date/target_date are metadata only; calendar predictors (day_of_week,
    # month) are known in advance and no future revenue is provided as input.
    assert "Date" not in PREDICTORS and "target_date" not in PREDICTORS

    print(f"Dataset shape: {data.shape}")
    print(f"Predictor features: {PREDICTORS}")
    print(f"Train: {len(train)} rows, {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Validation: {len(validation)} rows, {validation['Date'].min().date()} to {validation['Date'].max().date()}")
    print(f"Test: {len(test)} rows, {test['Date'].min().date()} to {test['Date'].max().date()}")
    print("Leakage checks passed: chronological non-overlapping splits; no date/target columns in predictors; target is next day.")

    candidates = {
        "RandomForest_baseline": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=1
        ),
        "RandomForest_constrained": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            min_samples_leaf=5, random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=1.0,
            random_state=42, n_jobs=1,
        ),
    }
    X_train, y_train = train[PREDICTORS], train[TARGET]
    X_validation, y_validation = validation[PREDICTORS], validation[TARGET]
    validation_rows: list[dict[str, object]] = []
    fitted_candidates = {}

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        score = metrics(y_validation, estimator.predict(X_validation))
        validation_rows.append({"model": name, "split": "validation", **score})
        fitted_candidates[name] = estimator

    # Selection uses only validation MAE; test partition has not been read by
    # a model, scored, or consulted during this decision.
    assert all(row["split"] == "validation" for row in validation_rows)
    selected_row = min(validation_rows, key=lambda row: row["MAE"])
    selected_name = str(selected_row["model"])
    print("\nValidation metrics (selected solely by lowest validation MAE):")
    for row in validation_rows:
        print(f"{row['model']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R2={row['R2']:.4f}")
    print(f"Selected model: {selected_name}; validation MAE={selected_row['MAE']:.4f}")
    print("Test data used for selection/tuning: False")

    # Refit the validation winner on train + validation and evaluate it once.
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = candidates[selected_name]
    final_model.fit(train_validation[PREDICTORS], train_validation[TARGET])
    test_score = metrics(test[TARGET], final_model.predict(test[PREDICTORS]))
    result_rows = validation_rows + [
        {"model": selected_name, "split": "test_selected", **test_score}
    ]

    # Load the already-recorded baseline test score only after selection and
    # final evaluation, for a reporting comparison (never for model choice).
    prior_results = pd.read_csv(BASELINE_RESULTS_PATH)
    baseline_test = prior_results.loc[
        (prior_results["model"] == "RandomForestRegressor")
        & (prior_results["split"] == "test")
    ]
    if len(baseline_test) != 1:
        raise ValueError("Expected one existing Random Forest baseline test result")
    baseline_reference = baseline_test.iloc[0]
    result_rows.append({
        "model": "RandomForest_baseline_reference",
        "split": "test_baseline_reference",
        "MAE": float(baseline_reference["MAE"]),
        "RMSE": float(baseline_reference["RMSE"]),
        "R2": float(baseline_reference["R2"]),
    })

    results = pd.DataFrame(
        result_rows,
        columns=["model", "split", "MAE", "RMSE", "R2"],
    )
    if not np.isfinite(results[["MAE", "RMSE", "R2"]].to_numpy()).all():
        raise AssertionError("Comparison output contains NaN or infinite metrics")
    results.to_csv(OUTPUT_PATH, index=False)

    baseline_validation = next(row for row in validation_rows if row["model"] == "RandomForest_baseline")
    validation_gain = baseline_validation["MAE"] - selected_row["MAE"]
    validation_gain_percent = validation_gain / baseline_validation["MAE"] * 100
    test_gain = float(baseline_reference["MAE"]) - test_score["MAE"]
    test_gain_percent = test_gain / float(baseline_reference["MAE"]) * 100
    print("\nFinal selected-model test metrics (retrained on train + validation; evaluated once):")
    print(f"MAE={test_score['MAE']:.4f}, RMSE={test_score['RMSE']:.4f}, R2={test_score['R2']:.4f}")
    print("\nComparison with current Random Forest baseline:")
    print(f"Validation MAE: baseline={baseline_validation['MAE']:.4f}, selected={selected_row['MAE']:.4f}, reduction={validation_gain:.4f} ({validation_gain_percent:.2f}%)")
    print(f"Test MAE: baseline reference={float(baseline_reference['MAE']):.4f}, selected={test_score['MAE']:.4f}, reduction={test_gain:.4f} ({test_gain_percent:.2f}%)")
    print(f"Test RMSE: baseline reference={float(baseline_reference['RMSE']):.4f}, selected={test_score['RMSE']:.4f}")
    print(f"Test R2: baseline reference={float(baseline_reference['R2']):.4f}, selected={test_score['R2']:.4f}")
    print(f"Saved results: {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_comparison()
