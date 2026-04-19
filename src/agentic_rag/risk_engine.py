"""Risk analysis for solar forecast volatility and grid operations."""

from __future__ import annotations

from typing import Any, Dict, List


def _severity_label(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def _merge_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not windows:
        return []

    merged: List[Dict[str, Any]] = [windows[0].copy()]
    for window in windows[1:]:
        current = merged[-1]
        if (
            current["type"] == window["type"]
            and window["start_hour"] <= current["end_hour"] + 1
        ):
            current["end_hour"] = window["end_hour"]
            current["hours"].extend(window["hours"])
            current["max_score"] = max(current["max_score"], window["max_score"])
            current["severity"] = _severity_label(current["max_score"])
            current["details"] = window["details"]
        else:
            merged.append(window.copy())
    return merged


def assess_variability_and_risk(forecast_rows: List[Dict[str, Any]], scenario: Dict[str, Any]) -> Dict[str, Any]:
    reserve_margin_pct = float(scenario["reserve_margin_pct"])
    reserve_factor = reserve_margin_pct / 100

    windows: List[Dict[str, Any]] = []
    deficit_hours = 0
    surplus_hours = 0
    ramp_down_events = 0
    volatility_hours = 0

    peak_w = max((row["predicted_w"] for row in forecast_rows), default=0.0)
    ramp_threshold_w = max(1800.0, peak_w * 0.24)

    for row in forecast_rows:
        demand_threshold_w = row["demand_w"] * (1 + reserve_factor)
        net_shortfall_w = demand_threshold_w - row["predicted_w"]

        if net_shortfall_w > 0:
            severity_score = min(1.0, net_shortfall_w / max(demand_threshold_w, 1.0))
            windows.append(
                {
                    "type": "deficit",
                    "severity": _severity_label(severity_score),
                    "max_score": severity_score,
                    "start_hour": row["hour"],
                    "end_hour": row["hour"],
                    "hours": [row["hour"]],
                    "details": (
                        f"Predicted generation is {net_shortfall_w / 1000:.2f} kW below "
                        f"demand plus reserve margin."
                    ),
                }
            )
            deficit_hours += 1

        if row["net_margin_w"] > max(1200.0, row["demand_w"] * 0.10):
            severity_score = min(1.0, row["net_margin_w"] / max(row["demand_w"], 1.0))
            windows.append(
                {
                    "type": "surplus",
                    "severity": _severity_label(severity_score),
                    "max_score": severity_score,
                    "start_hour": row["hour"],
                    "end_hour": row["hour"],
                    "hours": [row["hour"]],
                    "details": (
                        f"Expected solar surplus of {row['net_margin_w'] / 1000:.2f} kW is available "
                        "for storage charging or flexible loads."
                    ),
                }
            )
            surplus_hours += 1

        if row["ramp_delta_w"] <= -ramp_threshold_w:
            severity_score = min(1.0, abs(row["ramp_delta_w"]) / max(peak_w, 1.0))
            windows.append(
                {
                    "type": "ramp_down",
                    "severity": _severity_label(severity_score),
                    "max_score": severity_score,
                    "start_hour": row["hour"],
                    "end_hour": row["hour"],
                    "hours": [row["hour"]],
                    "details": (
                        f"Rapid drop of {abs(row['ramp_delta_w']) / 1000:.2f} kW indicates a ramp-down risk."
                    ),
                }
            )
            ramp_down_events += 1

        if row["historical_variability_ratio"] >= 0.45 and row["predicted_w"] > 0:
            severity_score = min(1.0, row["historical_variability_ratio"])
            windows.append(
                {
                    "type": "variability",
                    "severity": _severity_label(severity_score),
                    "max_score": severity_score,
                    "start_hour": row["hour"],
                    "end_hour": row["hour"],
                    "hours": [row["hour"]],
                    "details": (
                        f"Historical variability for this hour is high "
                        f"(std/mean={row['historical_variability_ratio']:.2f})."
                    ),
                }
            )
            volatility_hours += 1

    merged_windows = _merge_windows(sorted(windows, key=lambda item: (item["type"], item["start_hour"])))

    raw_risk_score = (
        deficit_hours * 14
        + surplus_hours * 4
        + ramp_down_events * 12
        + volatility_hours * 8
    )
    risk_score = min(100, int(raw_risk_score))

    if risk_score >= 75:
        risk_level = "Critical"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reserve_margin_pct": reserve_margin_pct,
        "deficit_hours": deficit_hours,
        "surplus_hours": surplus_hours,
        "ramp_down_events": ramp_down_events,
        "volatility_hours": volatility_hours,
        "windows": merged_windows,
    }
