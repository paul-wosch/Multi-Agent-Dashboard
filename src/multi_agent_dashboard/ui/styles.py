# ui/styles.py
from __future__ import annotations

import streamlit as st


def inject_tag_style(scope: str = "global"):
    """
    Shared CSS injector for tag styling.

    scope = "global"  -> all tags
    scope = "sidebar" -> sidebar tags only
    """
    if scope == "sidebar":
        selector = 'section[data-testid="stSidebar"] span[data-baseweb="tag"]'
    else:
        selector = 'span[data-baseweb="tag"]'
    st.markdown(
        f"""
        <style>
        {selector} {{
            background-color: #55575b !important;
            color: white !important;
        }}
        {selector}:hover {{
            background-color: #41454b !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_tag_style_for_sidebar():
    """Backward-compatible wrapper for sidebar tag styling."""
    inject_tag_style("sidebar")


def inject_global_tag_style():
    """Backward-compatible wrapper for global tag styling."""
    inject_tag_style("global")
