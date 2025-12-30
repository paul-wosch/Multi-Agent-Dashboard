# ui/logging_ui.py
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Any

import streamlit as st

from multi_agent_dashboard.config import LOG_FILE_PATH, UI_COLORS

# Centralized mapping of log levels to colors and symbols (derived from UI_COLORS)
LOG_LEVEL_STYLES: Dict[str, Dict[str, str]] = {
    "DEBUG": {
        "color": UI_COLORS["grey"]["value"],
        "symbol": UI_COLORS["grey"]["symbol"],
    },
    "INFO": {
        "color": UI_COLORS["green"]["value"],
        "symbol": UI_COLORS["green"]["symbol"],
    },
    "WARNING": {
        "color": UI_COLORS["orange"]["value"],
        "symbol": UI_COLORS["orange"]["symbol"],
    },
    "ERROR": {
        "color": UI_COLORS["red"]["value"],
        "symbol": UI_COLORS["red"]["symbol"],
    },
    "CRITICAL": {
        "color": UI_COLORS["purple"]["value"],
        "symbol": UI_COLORS["purple"]["symbol"],
    },
}


class StreamlitLogHandler(logging.Handler):
    """
    Logging handler that stores recent log records in Streamlit session_state.
    """

    def __init__(self, capacity: int = 500):
        super().__init__()
        self.capacity = capacity

    def emit(self, record: logging.LogRecord):
        try:
            # Only touch Streamlit from the main thread to avoid
            # "missing ScriptRunContext" warnings.
            if threading.current_thread() is not threading.main_thread():
                return

            entry = {
                "time": (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(record.created),
                    )
                    + f",{int(record.msecs):03d}"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Explicitly get and assign the deque to ensure session state stores
            # a deque instance (avoid relying on setdefault semantics).
            logs = st.session_state.get("_log_buffer")
            if logs is None:
                logs = deque(maxlen=self.capacity)
                st.session_state["_log_buffer"] = logs

            # Append the new entry
            logs.append(entry)
        except Exception:
            # Never let logging break the app
            pass


def load_historic_logs_into_buffer(
    log_path: Path,
    capacity: int = 500,
    session_key: str = "_log_buffer",
):
    """
    Load existing log file lines into the Streamlit log buffer on first app start.
    """
    if not log_path.exists():
        return

    # Avoid reloading if buffer already initialized (e.g., rerun)
    if session_key in st.session_state and st.session_state[session_key]:
        return

    buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                # Expected format:
                # "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] logger.name: message"
                time_part = ""
                level = "INFO"
                logger_name = ""
                message = line

                try:
                    ts_and_rest = line.split(" [", 1)
                    time_part = ts_and_rest[0].strip()
                    rest = ts_and_rest[1] if len(ts_and_rest) > 1 else ""

                    level_and_rest = rest.split("]", 1)
                    level = level_and_rest[0].strip("[]") or "INFO"
                    rest2 = level_and_rest[1].strip() if len(level_and_rest) > 1 else ""

                    logger_and_msg = rest2.split(":", 1)
                    logger_name = logger_and_msg[0].strip() if logger_and_msg else ""
                    message = logger_and_msg[1].strip() if len(logger_and_msg) > 1 else rest2
                except Exception:
                    # If parsing fails, just store the raw line as message
                    pass

                entry = {
                    "time": time_part,
                    "level": level,
                    "logger": logger_name,
                    "message": message,
                }
                buf.append(entry)
    except Exception:
        # Never let log loading break the UI
        pass

    st.session_state[session_key] = buf


def attach_streamlit_log_handler(capacity: int = 500):
    """
    Attach a StreamlitLogHandler once per session and load historic logs.
    Safe to call multiple times across reruns.
    """
    if "_log_handler_attached" in st.session_state:
        return

    handler = StreamlitLogHandler(capacity=capacity)
    logging.getLogger().addHandler(handler)
    st.session_state["_log_handler_attached"] = True

    # Load historic logs into buffer on first attachment
    load_historic_logs_into_buffer(LOG_FILE_PATH, capacity=capacity)
