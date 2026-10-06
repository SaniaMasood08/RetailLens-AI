"""Load the packaged revenue model and predict next-day recorded revenue."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "retail_revenue_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_model_bundle(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Load the fitted estimator and its feature-order metadata."""
    with Path(metadata_path).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    model = joblib.load(model_path)
    return model, metadata


def predict_next_day(
    feature_row: pd.DataFrame | pd.Series | Mapping[str, float],
    model: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> float:
    """Predict next-day revenue from exactly one row of named predictors.

    The input must contain exactly the predictor fields recorded in metadata.
    It may be a one-row DataFrame, Series, or mapping. Values are reordered to
    the saved model's expected column order before prediction.
    """
    if model is None or metadata is None:
        loaded_model, loaded_metadata = load_model_bundle(model_path, metadata_path)
        model = loaded_model if model is None else model
        metadata = loaded_metadata if metadata is None else metadata

    expected_columns = list(metadata["predictor_columns"])
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError("feature_row must contain exactly one row")
        row = feature_row.copy()
    elif isinstance(feature_row, pd.Series):
        row = feature_row.to_frame().T
    elif isinstance(feature_row, Mapping):
        row = pd.DataFrame([dict(feature_row)])
    else:
        raise TypeError("feature_row must be a one-row DataFrame, Series, or mapping")

    if len(set(row.columns)) != len(row.columns):
        raise ValueError("feature_row contains duplicate column names")
    if set(row.columns) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(row.columns))
        unexpected = sorted(set(row.columns) - set(expected_columns))
        raise ValueError(f"Predictor columns mismatch; missing={missing}, unexpected={unexpected}")
    ordered_row = row.loc[:, expected_columns]
    if ordered_row.isna().any().any():
        raise ValueError("feature_row contains missing predictor values")
    values = ordered_row.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("feature_row contains infinite predictor values")

    prediction = model.predict(ordered_row)[0]
    return float(prediction)
