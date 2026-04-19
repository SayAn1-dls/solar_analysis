"""Forecast helpers that reuse the Milestone 1 model without editing it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.load_data import load_data
from src.preprocessing.preprocessing import FEATURES, encode_features

DEFAULT_HYPERPARAMETERS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}


def _ensure_numeric_source_key(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SOURCE_KEY to deterministic integer ids without altering Milestone 1 code."""
    if pd.api.types.is_numeric_dtype(df["SOURCE_KEY"]):
        return df

    prepared = df.copy()
    prepared["SOURCE_KEY"] = pd.factorize(prepared["SOURCE_KEY"])[0]
    return prepared


def load_or_train_forecaster() -> RandomForestRegressor:
    """Load the trained model if present, otherwise train a compatible model in memory."""
    model_path = Path("models/solar_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)

    df = _ensure_numeric_source_key(encode_features(load_data()))
    df = df.sort_values("DATE_TIME")
    X = df[FEATURES]
    y = df["DC_POWER"]

    model = RandomForestRegressor(**DEFAULT_HYPERPARAMETERS)
    model.fit(X, y)
    return model


def load_reference_dataset() -> pd.DataFrame:
    """Load the historical dataset used for variability baselines."""
    return _ensure_numeric_source_key(encode_features(load_data())).sort_values("DATE_TIME").reset_index(drop=True)


def _season_factor(month: int) -> float:
    return {
        1: 0.60,
        2: 0.70,
        3: 0.90,
        4: 1.00,
        5: 1.10,
        6: 1.15,
        7: 1.10,
        8: 1.00,
        9: 0.90,
        10: 0.80,
        11: 0.70,
        12: 0.60,
    }.get(month, 1.0)


def _solar_factor(hour: int, month: int) -> float:
    if 6 <= hour <= 18:
        factor = float(np.sin(np.pi * (hour - 6) / 12))
    else:
        factor = 0.0
    return factor * _season_factor(month)


def _build_demand_profile(hours: List[int], base_demand_kw: float, critical_load_share: float) -> List[Dict[str, float]]:
    base_w = base_demand_kw * 1000
    profile: List[Dict[str, float]] = []

    for hour in hours:
        evening_peak = 1.16 if 18 <= hour <= 22 else 1.0
        midday_relief = 0.92 if 11 <= hour <= 15 else 1.0
        early_morning_ramp = 1.07 if 6 <= hour <= 8 else 1.0
        demand_w = base_w * evening_peak * midday_relief * early_morning_ramp
        profile.append(
            {
                "demand_w": demand_w,
                "critical_demand_w": demand_w * critical_load_share,
            }
        )

    return profile


def _historical_hourly_profile(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    profile: Dict[int, Dict[str, float]] = {}
    grouped = df.groupby("hour")["DC_POWER"]
    for hour, series in grouped:
        mean_value = float(series.mean())
        std_value = float(series.std()) if len(series) > 1 else 0.0
        p90 = float(series.quantile(0.90))
        profile[int(hour)] = {
            "mean_w": mean_value,
            "std_w": std_value,
            "p90_w": p90,
        }
    return profile


def build_forecast_context(scenario: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create an hourly forecast curve plus a structured summary for the agent."""
    model = load_or_train_forecaster()
    historical_df = load_reference_dataset()
    hourly_profile = _historical_hourly_profile(historical_df)

    start_hour = int(scenario["start_hour"])
    horizon = int(scenario["forecast_horizon_hours"])
    month = int(scenario["month"])
    base_irradiation = float(scenario["base_irradiation"])
    source_key = int(scenario["source_key"])
    ambient_temp = float(scenario["ambient_temperature"])
    module_temp = float(scenario["module_temperature"])
    critical_load_share = float(scenario["critical_load_share"])

    forecast_hours = [(start_hour + offset) % 24 for offset in range(horizon)]
    demand_profile = _build_demand_profile(
        hours=forecast_hours,
        base_demand_kw=float(scenario["grid_demand_kw"]),
        critical_load_share=critical_load_share,
    )

    rows: List[Dict[str, Any]] = []
    previous_prediction = None

    for index, hour in enumerate(forecast_hours):
        solar_factor = _solar_factor(hour, month)
        future_irradiation = max(base_irradiation * solar_factor, 0.0)
        future_module_temp = module_temp * (0.90 + 0.25 * solar_factor)
        future_ambient_temp = ambient_temp * (0.95 + 0.10 * solar_factor)

        features = pd.DataFrame(
            [
                [
                    source_key,
                    future_ambient_temp,
                    future_module_temp,
                    future_irradiation,
                    hour,
                    month,
                ]
            ],
            columns=FEATURES,
        )

        predicted_w = max(0.0, float(model.predict(features)[0]))
        if hour < 6 or hour > 18:
            predicted_w = 0.0

        demand = demand_profile[index]
        historical = hourly_profile.get(hour, {"mean_w": 0.0, "std_w": 0.0, "p90_w": 0.0})
        ramp_delta_w = 0.0 if previous_prediction is None else predicted_w - previous_prediction
        variability_ratio = (
            historical["std_w"] / historical["mean_w"]
            if historical["mean_w"] > 0
            else 0.0
        )

        rows.append(
            {
                "hour": hour,
                "predicted_w": predicted_w,
                "predicted_kw": predicted_w / 1000,
                "demand_w": demand["demand_w"],
                "demand_kw": demand["demand_w"] / 1000,
                "critical_demand_w": demand["critical_demand_w"],
                "net_margin_w": predicted_w - demand["demand_w"],
                "net_margin_kw": (predicted_w - demand["demand_w"]) / 1000,
                "critical_margin_w": predicted_w - demand["critical_demand_w"],
                "ramp_delta_w": ramp_delta_w,
                "ramp_delta_kw": ramp_delta_w / 1000,
                "solar_factor": solar_factor,
                "historical_mean_w": historical["mean_w"],
                "historical_std_w": historical["std_w"],
                "historical_p90_w": historical["p90_w"],
                "historical_variability_ratio": variability_ratio,
            }
        )
        previous_prediction = predicted_w

    predicted_values = [row["predicted_w"] for row in rows]
    net_margins = [row["net_margin_w"] for row in rows]
    peak_row = max(rows, key=lambda item: item["predicted_w"])
    steepest_drop = min(rows[1:], key=lambda item: item["ramp_delta_w"], default=rows[0])
    surplus_hours = [row["hour"] for row in rows if row["net_margin_w"] > 0]
    deficit_hours = [row["hour"] for row in rows if row["net_margin_w"] < 0]
    demand_covered_hours = sum(1 for row in rows if row["critical_margin_w"] >= 0)

    summary = {
        "start_hour": start_hour,
        "forecast_horizon_hours": horizon,
        "month": month,
        "estimated_energy_kwh": round(sum(predicted_values) / 1000, 2),
        "average_generation_kw": round(float(np.mean(predicted_values)) / 1000, 2),
        "peak_generation_kw": round(peak_row["predicted_w"] / 1000, 2),
        "peak_hour": int(peak_row["hour"]),
        "minimum_generation_kw": round(float(np.min(predicted_values)) / 1000, 2),
        "coverage_ratio": round(demand_covered_hours / len(rows), 3),
        "surplus_hours": surplus_hours,
        "deficit_hours": deficit_hours,
        "largest_net_gap_kw": round(min(net_margins) / 1000, 2),
        "largest_surplus_kw": round(max(net_margins) / 1000, 2),
        "steepest_drop_kw": round(steepest_drop["ramp_delta_w"] / 1000, 2),
        "steepest_drop_hour": int(steepest_drop["hour"]),
        "model_source": "models/solar_model.pkl" if Path("models/solar_model.pkl").exists() else "in-memory fallback",
    }

    return rows, summary
