"""Standalone Streamlit app for Milestone 2 Agentic RAG."""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.agentic_rag.workflow import run_agentic_grid_optimizer

st.set_page_config(
    page_title="Solar Grid Optimization Assistant",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .hero {
        background: linear-gradient(135deg, rgba(11,61,145,0.15), rgba(244,162,97,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        margin: 0;
        color: #F8FAFC;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: #CBD5E1;
        font-size: 1rem;
    }
    .small-note {
        color: #94A3B8;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1>Agentic AI Grid Optimization Assistant</h1>
    <p>Milestone 2 runs as a separate layer on top of your completed ML forecaster. It builds a solar forecast, analyzes variability risk, retrieves grid guidelines, and produces a structured optimization report.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scenario Inputs")
    source_key = st.number_input("Inverter ID (encoded)", min_value=0, value=0)
    ambient_temperature = st.number_input("Ambient Temperature (°C)", value=28.0)
    module_temperature = st.number_input("Module Temperature (°C)", value=40.0)
    base_irradiation = st.number_input("Base Irradiation", min_value=0.0, max_value=1.5, value=0.75)
    start_hour = st.slider("Starting Hour", 0, 23, 8)
    month = st.slider("Month", 1, 12, 6)
    forecast_horizon_hours = st.slider("Forecast Horizon", 6, 24, 12)
    grid_demand_kw = st.number_input("Average Grid Demand (kW)", min_value=1.0, value=22.0)
    reserve_margin_pct = st.slider("Reserve Margin (%)", 5, 40, 15)
    critical_load_share = st.slider("Critical Load Share", 0.2, 1.0, 0.6, 0.05)
    battery_capacity_kwh = st.number_input("Battery Capacity (kWh)", min_value=0.0, value=50.0)
    battery_soc_pct = st.slider("Battery State of Charge (%)", 0, 100, 55)
    battery_power_kw = st.number_input("Battery Power Limit (kW)", min_value=0.0, value=15.0)

question = st.text_area(
    "Operator Goal",
    value=(
        "Generate a grid optimization plan that minimizes deficit risk, uses storage well, "
        "and shifts flexible loads into strong solar periods."
    ),
    height=110,
)

st.caption(
    "Optional: set `OLLAMA_MODEL` to let an open-source local model rewrite the final report. "
    "Without it, the app uses a deterministic agentic fallback."
)

scenario = {
    "source_key": source_key,
    "ambient_temperature": ambient_temperature,
    "module_temperature": module_temperature,
    "base_irradiation": base_irradiation,
    "start_hour": start_hour,
    "month": month,
    "forecast_horizon_hours": forecast_horizon_hours,
    "grid_demand_kw": grid_demand_kw,
    "reserve_margin_pct": reserve_margin_pct,
    "critical_load_share": critical_load_share,
    "battery_capacity_kwh": battery_capacity_kwh,
    "battery_soc_pct": battery_soc_pct,
    "battery_power_kw": battery_power_kw,
}

if st.button("Run Agentic RAG Optimization", type="primary"):
    with st.spinner("Analyzing forecast, risks, and retrieved grid guidance..."):
        state = run_agentic_grid_optimizer(question=question, scenario=scenario)

    report = state["optimization_report"]
    forecast_df = pd.DataFrame(state["forecast_rows"])
    references_df = pd.DataFrame(state["supporting_references"])

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Risk Level", state["risk_analysis"]["risk_level"])
    top2.metric("Risk Score", f"{state['risk_analysis']['risk_score']}/100")
    top3.metric("Peak Solar", f"{report['forecast_summary']['peak_generation_kw']:.2f} kW")
    top4.metric("Coverage Ratio", f"{report['forecast_summary']['coverage_ratio']:.0%}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Optimization Report", "Forecast & Risk", "Retrieved Guidelines", "Agent Trace"]
    )

    with tab1:
        st.markdown(state["report_markdown"])
        st.download_button(
            "Download Report (Markdown)",
            data=state["report_markdown"].encode("utf-8"),
            file_name="agentic_grid_optimization_report.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download Report (JSON)",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="agentic_grid_optimization_report.json",
            mime="application/json",
        )

    with tab2:
        st.subheader("Hourly Forecast Table")
        st.dataframe(
            forecast_df[
                [
                    "hour",
                    "predicted_kw",
                    "demand_kw",
                    "net_margin_kw",
                    "ramp_delta_kw",
                    "historical_mean_w",
                    "historical_variability_ratio",
                ]
            ],
            width="stretch",
        )
        st.line_chart(
            forecast_df.set_index("hour")[["predicted_kw", "demand_kw"]],
            color=["#10B981", "#F97316"],
        )

        st.subheader("Risk Windows")
        if report["risk_periods"]:
            st.json(report["risk_periods"])
        else:
            st.success("No significant risk windows detected in this scenario.")

    with tab3:
        st.caption(f"Retriever backend: {state.get('retriever_backend', 'tfidf-cosine')}")
        st.dataframe(references_df, width="stretch")
        for item in state["retrieved_guidelines"]:
            with st.expander(f"{item['title']} ({item['id']})"):
                st.write(item["content"])
                st.caption(item["source"])

    with tab4:
        st.dataframe(pd.DataFrame(state["trace"]), width="stretch")
        st.json(
            {
                "scenario": state["scenario"],
                "forecast_summary": state["forecast_summary"],
                "risk_analysis": state["risk_analysis"],
            }
        )
else:
    st.info("Set a scenario and run the workflow to generate the Milestone 2 optimization report.")
