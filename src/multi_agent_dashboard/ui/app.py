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
    cached_load_prompt_versions,
    invalidate_agents,
    invalidate_pipelines,
    invalidate_runs,
    get_agent_service,
    get_pipeline_service,
    get_run_service,
)
from multi_agent_dashboard.ui.bootstrap import app_start, reload_agents_into_engine
from multi_agent_dashboard.ui.logging_ui import LOG_LEVEL_STYLES

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

st.title("🧠 Multi-Agent Pipeline Dashboard")
st.caption("Design pipelines, run tasks, inspect agent behavior, and manage prompt versions.")

MODE_RUN = "▶️ Run Pipeline"
MODE_AGENTS = "🧠 Manage Agents"
MODE_HISTORY = "📜 History"
MODE_LOGS = "🪵 Logs"

mode = st.radio(
    "Mode",
    [MODE_RUN, MODE_AGENTS, MODE_HISTORY, MODE_LOGS],
    horizontal=True,
)

strict_mode = st.sidebar.checkbox(
    "Strict output validation",
    value=False,
    help="Fail fast on agent output mismatches",
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
    render_run_mode(strict_mode=strict_mode)
elif mode == MODE_AGENTS:
    render_agent_editor()
elif mode == MODE_HISTORY:
    render_history_mode()
elif mode == MODE_LOGS:
    render_logs_mode()
