# ui/graph_view.py
from __future__ import annotations

from typing import Dict, List, Optional

import graphviz
import streamlit as st

from multi_agent_dashboard.config import UI_COLORS
from multi_agent_dashboard.ui.metrics_view import format_cost, format_latency
from multi_agent_dashboard.engine import MultiAgentEngine

# Local defaults (kept in sync with app-level defaults)
DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]


def render_agent_graph(
    steps: List[str],
    agent_metrics: Optional[Dict[str, dict]] = None,
    engine: Optional[MultiAgentEngine] = None,
):
    """
    Generate a Graphviz Digraph representing the pipeline and per-agent metrics.

    Backwards-compatible: if `engine` is None, the function reads the engine from
    st.session_state (same behavior as before). Optionally pass an engine explicitly
    for easier testing.
    """
    dot = graphviz.Digraph()
    # Keep defaults for nodes; individual nodes will override `color`
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#6baed6",
        fillcolor="#deebf7",
    )

    agent_metrics = agent_metrics or {}
    if engine is None:
        if "engine" not in st.session_state:
            raise ValueError("No engine available in session state; provide `engine` argument.")
        engine = st.session_state.engine

    for agent in steps:
        runtime = engine.agents.get(agent)

        # Default label (fallbacks if agent not in engine)
        role = getattr(runtime.spec, "role", None) if runtime else None

        # Use a safe fallback pattern so None or empty values do not leak into labels/colors
        symbol = (
            (getattr(runtime.spec, "symbol", None) or DEFAULT_SYMBOL)
            if runtime
            else DEFAULT_SYMBOL
        )
        color = (
            (getattr(runtime.spec, "color", None) or DEFAULT_COLOR)
            if runtime
            else DEFAULT_COLOR
        )

        # Prefix name with emoji/symbol
        base_label = f"{symbol} {agent}"
        if role:
            label = f"{base_label}\n({role})"
        else:
            label = base_label

        # Optionally annotate node with cost/latency
        m = agent_metrics.get(agent, {}) or {}
        extra = []
        if m.get("latency") is not None:
            extra.append(format_latency(m.get("latency")))
        if m.get("cost") is not None:
            extra.append(format_cost(m.get("cost")))
        if extra:
            label = f"{label}\n" + " | ".join(extra)

        # Use agent-specific color as border color (fallback applied above)
        dot.node(
            agent,
            label=label,
            color=color,        # border color
            fillcolor="#deebf7" # keep shared fill to stay readable
        )

    # Edges: label with downstream agent's metrics
    for i in range(len(steps) - 1):
        src = steps[i]
        dst = steps[i + 1]
        m = agent_metrics.get(dst, {}) or {}
        latency = m.get("latency")
        cost = m.get("cost")
        if latency is not None or cost is not None:
            edge_label = f"{format_latency(latency)} | {format_cost(cost)}"
        else:
            edge_label = "passes state →"
        dot.edge(src, dst, label=edge_label)

    return dot
