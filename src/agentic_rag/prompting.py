"""Prompt templates for optional LLM-backed report synthesis."""

from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are a solar-grid optimization assistant.
Use only the provided forecast summary, risk analysis, and retrieved guideline snippets.
Be conservative, practical, and explicit about uncertainty.
Always produce:
1. A short executive summary.
2. Risk periods with hour ranges.
3. Battery/grid/load-shifting recommendations.
4. Supporting references tagged by source id.
5. A responsible-AI note that avoids overclaiming.
Do not invent data or cite sources that were not retrieved.
""".strip()


def build_synthesis_prompt(
    question: str,
    scenario: Dict[str, Any],
    forecast_summary: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    action_plan: List[Dict[str, Any]],
    retrieved_guidelines: List[Dict[str, Any]],
) -> str:
    references = "\n".join(
        f"- {item['id']} | {item['title']} | {item['content']}"
        for item in retrieved_guidelines
    )
    actions = "\n".join(
        f"- {item['category']}: {item['recommendation']}"
        for item in action_plan
    )
    return f"""
User goal:
{question}

Scenario:
{scenario}

Forecast summary:
{forecast_summary}

Risk analysis:
{risk_analysis}

Draft actions:
{actions}

Retrieved references:
{references}

Write a concise markdown optimization report with the required sections.
""".strip()
