from src.agentic_rag.risk_engine import assess_variability_and_risk


def test_assess_variability_and_risk_detects_deficit_and_ramp():
    scenario = {"reserve_margin_pct": 15}
    rows = [
        {
            "hour": 10,
            "predicted_w": 18000.0,
            "demand_w": 16000.0,
            "net_margin_w": 2000.0,
            "ramp_delta_w": 0.0,
            "historical_variability_ratio": 0.20,
        },
        {
            "hour": 11,
            "predicted_w": 8000.0,
            "demand_w": 15000.0,
            "net_margin_w": -7000.0,
            "ramp_delta_w": -10000.0,
            "historical_variability_ratio": 0.52,
        },
    ]

    risk = assess_variability_and_risk(rows, scenario)

    assert risk["risk_score"] > 0
    assert risk["risk_level"] in {"Moderate", "High", "Critical"}
    assert any(window["type"] == "deficit" for window in risk["windows"])
    assert any(window["type"] == "ramp_down" for window in risk["windows"])
