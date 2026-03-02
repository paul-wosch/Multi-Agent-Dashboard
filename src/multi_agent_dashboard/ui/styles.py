# ui/styles.py
"""
CSS styling and UI theme components for the Streamlit-based Multi-Agent Dashboard.

This module provides centralized styling functions that enhance the visual appearance
and user experience of the dashboard interface. It handles CSS injection for custom
styling of Streamlit components, particularly focusing on tag styling and theme
consistency across different sections of the application.

Key Responsibilities:
- Inject custom CSS styles into Streamlit application
- Provide consistent tag styling (background colors, hover effects)
- Support scoped styling (global vs sidebar-specific)
- Maintain backward compatibility with existing styling functions

Architecture:
- Uses Streamlit's `st.markdown()` with `unsafe_allow_html=True` for CSS injection
- CSS selectors target specific Streamlit component data attributes
- Modular design with a primary function and backward-compatible wrappers

Usage:
    from multi_agent_dashboard.ui.styles import inject_tag_style, inject_tag_style_for_sidebar

    # Apply global tag styling
    inject_tag_style("global")

    # Apply sidebar-specific tag styling
    inject_tag_style_for_sidebar()

Dependencies:
- streamlit: For CSS injection via markdown components
"""
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
