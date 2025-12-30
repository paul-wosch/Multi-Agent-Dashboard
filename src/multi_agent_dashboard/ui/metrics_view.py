# ui/metrics_view.py
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
