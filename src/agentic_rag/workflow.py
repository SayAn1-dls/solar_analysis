"""LangGraph-style workflow for the solar Agentic RAG assistant."""

from __future__ import annotations

from typing import Any, Dict, List

from src.agentic_rag.forecast_bridge import build_forecast_context
from src.agentic_rag.reporting import build_structured_report, render_report_markdown
from src.agentic_rag.retrieval import GuidelineRetriever, build_retrieval_query
from src.agentic_rag.risk_engine import assess_variability_and_risk
from src.agentic_rag.state import GridOptimizationState


def _trace(state: Dict[str, Any], step: str, detail: str) -> List[Dict[str, str]]:
    trace = list(state.get("trace", []))
    trace.append({"step": step, "detail": detail})
    return trace


def _prepare_context(state: GridOptimizationState) -> GridOptimizationState:
    forecast_rows, forecast_summary = build_forecast_context(state["scenario"])
    return {
        "forecast_rows": forecast_rows,
        "forecast_summary": forecast_summary,
        "trace": _trace(
            state,
            "prepare_context",
            f"Built {len(forecast_rows)} hourly forecast points using {forecast_summary['model_source']}.",
        ),
    }


def _analyze_risk(state: GridOptimizationState) -> GridOptimizationState:
    risk_analysis = assess_variability_and_risk(state["forecast_rows"], state["scenario"])
    retrieval_query = build_retrieval_query(
        question=state["question"],
        forecast_summary=state["forecast_summary"],
        risk_analysis=risk_analysis,
    )
    return {
        "risk_analysis": risk_analysis,
        "retrieval_query": retrieval_query,
        "trace": _trace(
            state,
            "analyze_risk",
            f"Assigned {risk_analysis['risk_level']} risk with score {risk_analysis['risk_score']}/100.",
        ),
    }


def _retrieve_guidelines(state: GridOptimizationState) -> GridOptimizationState:
    retriever = GuidelineRetriever()
    documents = retriever.search(state["retrieval_query"], top_k=5)
    references = [
        {
            "id": item["id"],
            "title": item["title"],
            "source": item["source"],
            "score": item["score"],
        }
        for item in documents
    ]
    return {
        "retrieved_guidelines": documents,
        "supporting_references": references,
        "retriever_backend": retriever.backend,
        "trace": _trace(
            state,
            "retrieve_guidelines",
            f"Retrieved {len(documents)} knowledge chunks using the {retriever.backend} backend.",
        ),
    }


def _plan_actions(state: GridOptimizationState) -> GridOptimizationState:
    scenario = state["scenario"]
    forecast_summary = state["forecast_summary"]
    risk_analysis = state["risk_analysis"]
    actions: List[Dict[str, Any]] = []

    deficit_windows = [item for item in risk_analysis["windows"] if item["type"] == "deficit"]
    if deficit_windows:
        first_window = deficit_windows[0]
        actions.append(
            {
                "category": "Grid balancing",
                "priority": "High",
                "basis": "Forecast deficit window",
                "recommendation": (
                    f"Reserve battery discharge or grid import for {first_window['start_hour']}:00-"
                    f"{first_window['end_hour']}:59 to protect the {scenario['reserve_margin_pct']}% reserve margin. "
                    f"Available storage envelope: {scenario['battery_capacity_kwh']} kWh at "
                    f"{scenario['battery_soc_pct']}% state of charge with {scenario['battery_power_kw']} kW power limit."
                ),
            }
        )
        actions.append(
            {
                "category": "Flexible load control",
                "priority": "High",
                "basis": "Low coverage period",
                "recommendation": (
                    "Delay non-critical pumping, cooling, or EV charging away from deficit windows and keep "
                    "critical load support prioritized."
                ),
            }
        )

    surplus_windows = [item for item in risk_analysis["windows"] if item["type"] == "surplus"]
    if surplus_windows:
        first_window = surplus_windows[0]
        actions.append(
            {
                "category": "Storage charging",
                "priority": "Medium",
                "basis": "Forecast surplus window",
                "recommendation": (
                    f"Charge storage and schedule flexible daytime loads during {first_window['start_hour']}:00-"
                    f"{first_window['end_hour']}:59 to absorb expected solar surplus, staying within the "
                    f"{scenario['battery_power_kw']} kW battery power ceiling."
                ),
            }
        )

    if risk_analysis["ramp_down_events"] > 0:
        actions.append(
            {
                "category": "Ramp management",
                "priority": "High",
                "basis": "Rapid solar drop",
                "recommendation": (
                    "Stage backup supply before the steepest drop hour and avoid committing new discretionary loads "
                    "during rapid output decline."
                ),
            }
        )

    actions.append(
        {
            "category": "Utilization strategy",
            "priority": "Medium",
            "basis": "Forecast peak and surplus pattern",
            "recommendation": (
                f"Anchor energy-intensive processes near the predicted peak at {forecast_summary['peak_hour']}:00 "
                "to maximize direct solar consumption."
            ),
        }
    )
    actions.append(
        {
            "category": "Operator assurance",
            "priority": "Low",
            "basis": "Responsible AI guardrail",
            "recommendation": (
                "Treat the report as decision support, validate against live telemetry, and keep manual approval for "
                "dispatch or curtailment actions."
            ),
        }
    )

    return {
        "action_plan": actions,
        "trace": _trace(
            state,
            "plan_actions",
            f"Built {len(actions)} structured recommendations for balancing, storage, and utilization.",
        ),
    }


def _llm_reason(state: GridOptimizationState) -> GridOptimizationState:
    from groq import Groq
    import os

    try:
        
        import streamlit as st
        groq_key = os.getenv("GROQ_API_KEY")
        try:
            if not groq_key and hasattr(st, "secrets"):
                groq_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        client = Groq(api_key=groq_key)


        summary = state["forecast_summary"]
        risk = state["risk_analysis"]
        actions = state["action_plan"]

        prompt = f"""
You are a solar grid optimization expert.

Forecast Summary:
{summary}

Risk Analysis:
{risk}

Planned Actions:
{actions}

Explain:
1. Why this risk level exists
2. Why these actions are recommended
3. What could go wrong if ignored

Be practical, not theoretical.
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        llm_reasoning = response.choices[0].message.content
    except Exception as e:
        llm_reasoning = f"LLM reasoning failed or GROQ_API_KEY is missing. Error: {str(e)}"

    return {
        "llm_reasoning": llm_reasoning,
        "trace": _trace(
            state,
            "llm_reason",
            "Generated AI reasoning behind actions.",
        ),
    }



def _generate_report(state: GridOptimizationState) -> GridOptimizationState:
    structured_report = build_structured_report(state)
    report_markdown = render_report_markdown(state, structured_report)
    return {
        "optimization_report": structured_report,
        "report_markdown": report_markdown,
        "trace": _trace(
            state,
            "generate_report",
            "Rendered the final optimization report with supporting references.",
        ),
    }


def _run_fallback(initial_state: GridOptimizationState) -> GridOptimizationState:
    state: GridOptimizationState = dict(initial_state)
    for node in (_prepare_context, _analyze_risk, _retrieve_guidelines, _plan_actions, _llm_reason, _generate_report):
        state.update(node(state))
    return state


def run_agentic_grid_optimizer(question: str, scenario: Dict[str, Any]) -> GridOptimizationState:
    """Run the Milestone 2 workflow, using LangGraph when available."""
    initial_state: GridOptimizationState = {
        "question": question,
        "scenario": scenario,
        "trace": [],
    }

    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GridOptimizationState)
        graph.add_node("prepare_context", _prepare_context)
        graph.add_node("analyze_risk", _analyze_risk)
        graph.add_node("retrieve_guidelines", _retrieve_guidelines)
        graph.add_node("plan_actions", _plan_actions)
        graph.add_node("llm_reason", _llm_reason)
        graph.add_node("generate_report", _generate_report)
        graph.add_edge(START, "prepare_context")
        graph.add_edge("prepare_context", "analyze_risk")
        graph.add_edge("analyze_risk", "retrieve_guidelines")
        graph.add_edge("retrieve_guidelines", "plan_actions")
        graph.add_edge("plan_actions", "llm_reason")
        graph.add_edge("llm_reason", "generate_report")
        graph.add_edge("generate_report", END)
        compiled = graph.compile()
        return compiled.invoke(initial_state)
    except Exception:
        return _run_fallback(initial_state)
