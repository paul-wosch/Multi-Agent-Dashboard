# ui/graph_view.py
from __future__ import annotations

from typing import Dict, List, Optional

import graphviz
import streamlit as st

from multi_agent_dashboard.config import UI_COLORS
from multi_agent_dashboard.ui.metrics_view import format_cost, format_latency

# Local defaults (kept in sync with app-level defaults)
DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]


def render_agent_graph(steps: List[str], agent_metrics: Optional[Dict[str, dict]] = None):
    """
    Generate a Graphviz Digraph representing the pipeline and per-agent metrics.

    Kept signature identical to the original to avoid touching callers;
    it reads the engine from st.session_state (same behavior as before).
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
    engine = st.session_state.engine

    for agent in steps:
        runtime = engine.agents.get(agent)

        # Default label (fallbacks if agent not in engine)
        role = runtime.spec.role if runtime else None
        symbol = getattr(runtime.spec, "symbol", DEFAULT_SYMBOL) if runtime else DEFAULT_SYMBOL
        color = getattr(runtime.spec, "color", DEFAULT_COLOR) if runtime else DEFAULT_COLOR

        # Prefix name with emoji/symbol
        base_label = f"{symbol} {agent}"
        if role:
            label = f"{base_label}\n({role})"
        else:
            label = base_label

        # Optionally annotate node with cost/latency
        m = agent_metrics.get(agent, {})
        extra = []
        if m.get("latency") is not None:
            extra.append(format_latency(m.get("latency")))
        if m.get("cost") is not None:
            extra.append(format_cost(m.get("cost")))
        if extra:
            label = f"{label}\n" + " | ".join(extra)

        # Use agent-specific color as border color
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
        m = agent_metrics.get(dst, {})
        latency = m.get("latency")
        cost = m.get("cost")
        if latency is not None or cost is not None:
            edge_label = f"{format_latency(latency)} | {format_cost(cost)}"
        else:
            edge_label = "passes state →"
        dot.edge(src, dst, label=edge_label)

    return dot
