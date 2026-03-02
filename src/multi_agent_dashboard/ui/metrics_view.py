# ui/metrics_view.py
"""
Cost and latency visualization module for the Multi-Agent Dashboard.

This module provides visualization components for displaying agent execution
metrics including token usage, latency, and cost breakdowns. It renders both
summary metrics and detailed per-agent breakdowns in a consistent format
across current and historical runs.

Key responsibilities:
- Render cost and latency summary metrics with formatted display
- Display per-agent breakdown tables with token counts and costs
- Format numeric values for human-readable display
- Provide reusable visualization components for both live and historical runs

Architecture:
- Uses Streamlit's metric and dataframe components for visualization
- Integrates with view models from `view_models.py` for data transformation
- Provides consistent formatting across different data sources
- Supports both current run results and historical run data

Visualization Components:
    - Summary metrics: Total latency, input cost, output cost, total cost
    - Per-agent breakdown: DataFrame with token counts, latency, and costs
    - Formatted display: Currency formatting, latency formatting, null handling

Usage:
    # Render metrics for current run
    >>> render_cost_latency_section(metrics_view)
    
    # Render metrics for historical run with suffix
    >>> render_cost_latency_section(metrics_view, title_suffix="Stored")

Dependencies:
    - `streamlit`: UI components and rendering
    - `view_models.py`: Data transformation and view model classes
    - `pandas`: DataFrame manipulation for tabular display
"""
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from multi_agent_dashboard.ui.view_models import (
    AgentMetricsView,
    summarize_agent_metrics,
    df_replace_none,
)


def format_cost(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return f"${value:.5f}"


def format_latency(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}s"


def render_cost_latency_section(
    metrics_view: List[AgentMetricsView],
    title_suffix: str = "",
):
    """
    Shared renderer for cost & latency, used for both current and historical runs.
    """
    if not metrics_view:
        st.info("No metrics available.")
        return

    # Convert back into the generic metrics mapping expected by summarize_agent_metrics
    metrics_map: Dict[str, dict] = {
        m.agent_name: {
            "model": m.model,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "latency": m.latency,
            "input_cost": m.input_cost,
            "output_cost": m.output_cost,
            "cost": m.total_cost,
        }
        for m in metrics_view
    }

    summary, df = summarize_agent_metrics(metrics_map)

    header = "Cost & Latency"
    if title_suffix:
        header += f" ({title_suffix})"
    st.subheader(header)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Latency", format_latency(summary["total_latency"]))
    with col2:
        st.metric("Total Input Cost", format_cost(summary["total_input_cost"]))
    with col3:
        st.metric("Total Output Cost", format_cost(summary["total_output_cost"]))
    with col4:
        st.metric("Total Cost", format_cost(summary["total_cost"]))

    st.markdown("---")
    st.subheader("Per-Agent Breakdown")
    st.dataframe(df_replace_none(df), width="stretch")
