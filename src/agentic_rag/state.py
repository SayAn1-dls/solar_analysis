"""Typed state definitions for the Agentic RAG workflow."""

from typing import Any, Dict, List, TypedDict


class GridOptimizationState(TypedDict, total=False):
    question: str
    scenario: Dict[str, Any]
    forecast_rows: List[Dict[str, Any]]
    forecast_summary: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    retrieval_query: str
    retrieved_guidelines: List[Dict[str, Any]]
    retriever_backend: str
    action_plan: List[Dict[str, Any]]
    llm_reasoning: str
    optimization_report: Dict[str, Any]
    report_markdown: str
    supporting_references: List[Dict[str, Any]]
    trace: List[Dict[str, str]]
