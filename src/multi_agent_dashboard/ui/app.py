# ui/app.py
import json
import logging
import time
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from multi_agent_dashboard.config import (
    DB_FILE_PATH,
    UI_COLORS,
    configure_logging,
)
from multi_agent_dashboard.ui.cache import (
    cached_load_agents,
    cached_load_pipelines,
    cached_load_runs,
    cached_load_run_details,
    cached_load_agent_snapshots,
    cached_load_monthly_costs,
    invalidate_agents,
    invalidate_pipelines,
    invalidate_runs,
    get_agent_service,
    get_pipeline_service,
    get_run_service,
)
from multi_agent_dashboard.ui.bootstrap import app_start, reload_agents_into_engine
from multi_agent_dashboard.ui.logging_ui import LOG_LEVEL_STYLES
from multi_agent_dashboard.ui.metrics_view import format_cost

# ======================================================
# STREAMLIT PAGE CONFIG (must be first Streamlit call)
# ======================================================

st.set_page_config(
    page_title="Multi-Agent Dashboard",
    layout="wide",
)

# ======================================================
# LOGGER
# ======================================================

logger = logging.getLogger(__name__)

DB_PATH = DB_FILE_PATH

# Default agent colors (kept here for backward compat)
DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20 MB
AD_HOC_PIPELINE_LABEL = "<Ad-hoc>"

# ======================================================
# INITIALIZE DB AND PREPARE ENGINE
# ======================================================

configure_logging()

# Only start the app if we don't already have an engine in session state.
# This prevents double initialization during reruns.
if "engine" not in st.session_state:
    app_start()
else:
    # Ensure engine has up-to-date agents after code reloads
    reload_agents_into_engine()

# Initialize Ad-hoc pipeline state (empty on first app start)
if "adhoc_pipeline_steps" not in st.session_state:
    st.session_state.adhoc_pipeline_steps = []

# Also provide the conventional guard so running the script directly will bootstrap.
if __name__ == "__main__":
    if "engine" not in st.session_state:
        app_start()

# ======================================================
# GLOBAL UI STYLES
# ======================================================

st.markdown(
    """
    <style>
    /* Wrap code blocks */
    pre code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: auto !important;
    }

    /* Wrap Streamlit code blocks */
    .stCodeBlock pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap JSON output */
    .stJson pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap text areas */
    textarea {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Prevent horizontal scroll everywhere */
    section.main {
        overflow-x: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# TOP-LEVEL UI
# ======================================================

# Heading and Metrics columns
col_main, col_right = st.columns([7, 2])
with col_main:
    st.title("🧠 Multi-Agent Pipeline Dashboard")
    st.caption("Design pipelines, run tasks, inspect agent behavior, and manage agent snapshots.")

    MODE_RUN = "▶️ Run Pipeline"
    MODE_AGENTS = "🧠 Manage Agents"
    MODE_HISTORY = "📜 History"
    MODE_LOGS = "🪵 Logs"

    mode = st.radio(
        "Mode",
        [MODE_RUN, MODE_AGENTS, MODE_HISTORY, MODE_LOGS],
        horizontal=True,
    )
try:
    with col_right:
        try:
            monthly = cached_load_monthly_costs()
            total_cost = monthly.get("total_cost")
            if total_cost is None:
                st.metric("This month", "–")
            else:
                st.metric("This month", format_cost(total_cost))
        except Exception:
            # Keep UI resilient: if DB/migrations unavailable, do not fail rendering
            try:
                # Fallback small right-hand column so page layout remains stable
                _, col_right = st.columns([8, 1])
                with col_right:
                    st.metric("This month", "–")
            except Exception:
                pass
        # The whole cost UI is inside an expander (default collapsed)
        with st.expander("Advanced cost tracking", expanded=False):
            # Small controls row: period selector + last-N selector
            ctrl_col, limit_col = st.columns([2, 1])
            with ctrl_col:
                period = st.selectbox(
                    "Period",
                    ["daily", "weekly", "monthly", "yearly", "total"],
                    index=2,
                    key="cost_period",
                    format_func=lambda p: p.capitalize(),
                )
            with limit_col:
                limit = st.number_input(
                    "Last N",
                    min_value=2,
                    max_value=52,
                    value=12,
                    step=1,
                    key="cost_limit",
                    help="Number of recent periods to show in the sparkline",
                )

            # Fetch historical totals using the service API prepared server-side.
            # This returns a list ordered by period DESC (most recent first).
            svc = get_run_service()
            try:
                costs = svc.get_costs_by_period(period=period, limit=int(limit))
            except Exception:
                costs = []

            # Friendly labels for the metric header
            LABEL_MAP = {
                "daily": "Today",
                "weekly": "This week",
                "monthly": "This month",
                "yearly": "This year",
                "total": "All time",
            }

            if not costs:
                # Resilient fallback when DB/migrations unavailable
                st.metric(LABEL_MAP.get(period, "Cost"), "–")
            else:
                # Most recent period is first element
                current_total = float(costs[0].get("total_cost", 0.0))
                previous_total = float(costs[1].get("total_cost", 0.0)) if len(costs) > 1 else None

                # Sparkline: provide values oldest->newest (left->right)
                chart_data = [float(r.get("total_cost", 0.0)) for r in costs][::-1]

                # Delta: change vs previous period (if available and applicable)
                delta_display = None
                if previous_total is not None and period != "total":
                    delta_value = current_total - previous_total
                    # Format delta with a leading + for positive changes
                    if delta_value >= 0:
                        delta_display = f"+{format_cost(delta_value)}"
                    else:
                        delta_display = format_cost(delta_value)

                # Show metric with sparkline and delta
                st.metric(
                    LABEL_MAP.get(period, "Cost"),
                    format_cost(current_total),
                    delta=delta_display,
                    chart_data=chart_data,
                )
except Exception:
    # Keep UI resilient: if something goes wrong, fall back to the minimal metric
    try:
        _, col_right = st.columns([8, 1])
        with col_right:
            with st.expander("Cost tracking", expanded=False):
                st.metric("This month", "–")
    except Exception:
        pass



strict_mode = st.sidebar.checkbox(
    "Strict output validation",
    value=False,
    help="Fail fast on agent output mismatches",
)

strict_schema_validation = st.sidebar.checkbox(
    "Strict schema validation",
    value=False,
    help="Fail fast when structured output JSON does not match schema (skips remaining agents).",
)

st.divider()

# Import per-mode renderers (extracted into dedicated modules in Phase 4)
from multi_agent_dashboard.ui.run_mode import render_run_mode
from multi_agent_dashboard.ui.agent_editor_mode import render_agent_editor
from multi_agent_dashboard.ui.history_mode import render_history_mode
from multi_agent_dashboard.ui.logging_ui import render_logs_mode

# ======================================================
# MODE ROUTER
# ======================================================

if mode == MODE_RUN:
    render_run_mode(strict_mode=strict_mode, strict_schema_validation=strict_schema_validation)
elif mode == MODE_AGENTS:
    render_agent_editor()
elif mode == MODE_HISTORY:
    render_history_mode()
elif mode == MODE_LOGS:
    render_logs_mode()
