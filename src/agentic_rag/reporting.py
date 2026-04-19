"""Structured report generation for Milestone 2."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List

from src.agentic_rag.prompting import SYSTEM_PROMPT, build_synthesis_prompt


def build_structured_report(state: Dict[str, Any]) -> Dict[str, Any]:
    forecast_summary = state["forecast_summary"]
    risk_analysis = state["risk_analysis"]
    windows = risk_analysis.get("windows", [])
    action_plan = state.get("action_plan", [])

    executive_summary = (
        f"The agent forecasts {forecast_summary['estimated_energy_kwh']:.2f} kWh over the next "
        f"{forecast_summary['forecast_horizon_hours']} hours, with peak solar output of "
        f"{forecast_summary['peak_generation_kw']:.2f} kW at {forecast_summary['peak_hour']}:00. "
        f"Overall grid risk is rated {risk_analysis['risk_level']} "
        f"(score {risk_analysis['risk_score']}/100)."
    )

    risk_periods = [
        {
            "type": window["type"],
            "hours": f"{window['start_hour']}:00-{window['end_hour']}:59",
            "severity": window["severity"],
            "details": window["details"],
        }
        for window in windows
    ]

    supporting_references = [
        {
            "id": item["id"],
            "title": item["title"],
            "source": item["source"],
            "score": item["score"],
        }
        for item in state.get("retrieved_guidelines", [])
    ]

    responsible_ai_notes = [
        "Recommendations are guidance for operator review, not autonomous dispatch commands.",
        "Forecast quality depends on the Milestone 1 model and the selected scenario assumptions.",
        "Reserve actions should be validated against local grid codes, battery constraints, and live telemetry.",
    ]

    return {
        "question": state["question"],
        "scenario": state["scenario"],
        "executive_summary": executive_summary,
        "forecast_summary": forecast_summary,
        "risk_periods": risk_periods,
        "recommended_actions": action_plan,
        "supporting_references": supporting_references,
        "responsible_ai_notes": responsible_ai_notes,
    }


def _maybe_generate_with_ollama(prompt: str) -> str | None:
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        return None

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def render_report_markdown(state: Dict[str, Any], structured_report: Dict[str, Any]) -> str:
    llm_prompt = build_synthesis_prompt(
        question=state["question"],
        scenario=state["scenario"],
        forecast_summary=state["forecast_summary"],
        risk_analysis=state["risk_analysis"],
        action_plan=state.get("action_plan", []),
        retrieved_guidelines=state.get("retrieved_guidelines", []),
    )
    llm_response = _maybe_generate_with_ollama(llm_prompt)
    if llm_response:
        return llm_response

    lines: List[str] = [
        "# Agentic Grid Optimization Report",
        "",
        "## Executive Summary",
        structured_report["executive_summary"],
        "",
        "## Forecast Summary",
        (
            f"- Estimated energy: {structured_report['forecast_summary']['estimated_energy_kwh']:.2f} kWh\n"
            f"- Peak generation: {structured_report['forecast_summary']['peak_generation_kw']:.2f} kW at "
            f"{structured_report['forecast_summary']['peak_hour']}:00\n"
            f"- Demand coverage ratio: {structured_report['forecast_summary']['coverage_ratio']:.0%}\n"
            f"- Largest deficit: {structured_report['forecast_summary']['largest_net_gap_kw']:.2f} kW\n"
            f"- Largest surplus: {structured_report['forecast_summary']['largest_surplus_kw']:.2f} kW"
        ),
        "",
        "## Risk Periods",
    ]

    if structured_report["risk_periods"]:
        for item in structured_report["risk_periods"]:
            lines.append(
                f"- {item['type']} | {item['hours']} | {item['severity']}: {item['details']}"
            )
    else:
        lines.append("- No high-risk windows detected for the selected horizon.")

    lines.extend(["", "## Recommended Actions"])
    for action in structured_report["recommended_actions"]:
        lines.append(
            f"- {action['category']}: {action['recommendation']} "
            f"(Priority: {action['priority']}, basis: {action['basis']})"
        )

    lines.extend(["", "## Supporting References"])
    for item in structured_report["supporting_references"]:
        lines.append(
            f"- {item['id']} | {item['title']} | score={item['score']} | {item['source']}"
        )

    lines.extend(["", "## Responsible AI Notes"])
    for item in structured_report["responsible_ai_notes"]:
        lines.append(f"- {item}")

    return "\n".join(lines)
