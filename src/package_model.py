"""Fit and package the selected final model without using test rows to fit."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from predict import predict_next_day


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
IMPROVEMENT_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_improvement_results.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "retail_revenue_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

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
SELECTED_MODEL_NAME = "RandomForest_constrained"
HYPERPARAMETERS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": 1,
}


def package_final_model() -> tuple[Path, Path, float]:
    """Fit train+validation, save artifacts, and smoke-test one test row."""
    data = pd.read_csv(FEATURES_PATH, parse_dates=["Date", "target_date"])
    expected_columns = set(PREDICTORS + [TARGET, "Date", "target_date"])
    if not expected_columns.issubset(data.columns):
        raise ValueError(f"Feature table lacks columns: {sorted(expected_columns - set(data.columns))}")
    if not data["Date"].is_monotonic_increasing or data["Date"].duplicated().any():
        raise AssertionError("Feature rows must be chronologically ordered by forecast date")
    if not (data["target_date"] == data["Date"] + pd.Timedelta(days=1)).all():
        raise AssertionError("Target dates must follow forecast dates by one calendar day")
    if data[PREDICTORS + [TARGET]].isna().any().any():
        raise ValueError("Feature table contains missing predictor or target values")
    if not np.isfinite(data[PREDICTORS + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains infinite predictor or target values")

    # Keep the exact contiguous split used during model selection: 241 train,
    # 52 validation, and 52 test rows. Only the first 293 rows are fitted.
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]
    train_validation = pd.concat([train, validation], ignore_index=True)
    fit_dates = set(train_validation["Date"])
    test_dates = set(test["Date"])
    assert len(train) == 241 and len(validation) == 52 and len(test) == 52
    assert len(train_validation) == 293
    assert fit_dates.isdisjoint(test_dates)
    assert train_validation["Date"].max() < test["Date"].min()
    assert train_validation["target_date"].max() < test["target_date"].min()

    # Confirm the selected configuration and recorded test metrics from the
    # model-improvement output. Test scores are metadata only and do not affect
    # fitting, model choice, or hyperparameters here.
    comparison = pd.read_csv(IMPROVEMENT_RESULTS_PATH)
    selected_test = comparison.loc[
        (comparison["model"] == SELECTED_MODEL_NAME)
        & (comparison["split"] == "test_selected")
    ]
    if len(selected_test) != 1:
        raise ValueError("Expected one recorded test result for the selected model")
    test_metrics = {
        metric: float(selected_test.iloc[0][metric])
        for metric in ("MAE", "RMSE", "R2")
    }
    if not np.isfinite(list(test_metrics.values())).all():
        raise ValueError("Recorded test metrics are not finite")

    estimator = RandomForestRegressor(**HYPERPARAMETERS)
    # Fitting uses exactly train + validation; the test partition is excluded.
    estimator.fit(train_validation[PREDICTORS], train_validation[TARGET])
    if list(estimator.feature_names_in_) != PREDICTORS:
        raise AssertionError("Fitted predictor column order differs from the expected order")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODEL_PATH, compress=3)
    metadata = {
        "model_name": "RandomForestRegressor (Constrained Random Forest)",
        "selection_label": SELECTED_MODEL_NAME,
        "hyperparameters": HYPERPARAMETERS,
        "predictor_columns": PREDICTORS,
        "target_column": TARGET,
        "training_validation_test_methodology": {
            "split": "strict chronological contiguous 70% / 15% / 15% split",
            "row_counts": {
                "training": len(train),
                "validation": len(validation),
                "test": len(test),
                "final_model_fit_training_plus_validation": len(train_validation),
            },
            "forecast_date_ranges": {
                "training": [train["Date"].min().date().isoformat(), train["Date"].max().date().isoformat()],
                "validation": [validation["Date"].min().date().isoformat(), validation["Date"].max().date().isoformat()],
                "test": [test["Date"].min().date().isoformat(), test["Date"].max().date().isoformat()],
                "final_fit_training_plus_validation": [train_validation["Date"].min().date().isoformat(), train_validation["Date"].max().date().isoformat()],
            },
            "selection_metric": "validation MAE",
            "test_used_for_fitting": False,
            "test_evaluation": "recorded once after model selection; not used for fitting or tuning",
        },
        "test_metrics": test_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Smoke-test prediction using only predictor columns from one test row.
    # Its target and dates are not passed to the prediction function.
    smoke_features = test.loc[:, PREDICTORS].iloc[[0]].copy()
    assert list(smoke_features.columns) == PREDICTORS
    smoke_prediction = predict_next_day(smoke_features)

    print(f"Selected model: {metadata['model_name']}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    print(f"Final fit rows: {len(train_validation)} (training {len(train)} + validation {len(validation)})")
    print(f"Final fit forecast-date range: {train_validation['Date'].min().date()} to {train_validation['Date'].max().date()}")
    print(f"Test rows excluded from fitting: {len(test)}; overlap with fit rows: {len(fit_dates & test_dates)}")
    print(f"Predictor order: {PREDICTORS}")
    print(f"Recorded final test metrics: {test_metrics}")
    print(f"Smoke prediction from one test feature row: {smoke_prediction:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    return MODEL_PATH, METADATA_PATH, smoke_prediction


if __name__ == "__main__":
    package_final_model()
