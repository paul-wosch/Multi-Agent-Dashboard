# ui/utils.py
"""
Shared utility functions for the Multi-Agent Dashboard UI.

This module provides general-purpose utility functions used across multiple
UI modules. It focuses on data parsing, validation, and transformation
operations that don't fit into more specialized modules.

Key responsibilities:
- Safe JSON parsing with graceful error handling
- Data validation and transformation utilities
- Common string and data manipulation functions

Architecture:
- Pure functions with no side effects
- Graceful error handling for malformed data
- Type hints for better IDE support and documentation

Usage:
    # Safely parse JSON fields from database
    >>> tools_config = parse_json_field(db_row.get("tools_json"), {})
    >>> # Returns {} if parsing fails or field is empty
    
    # Handle malformed historical data gracefully
    >>> config = parse_json_field("not valid json", default={})

Note:
    Functions in this module are designed to be resilient to malformed data,
    particularly important for historical runs where schema changes may have
    occurred. Silent fallbacks to defaults prevent UI crashes while logging
    appropriate warnings elsewhere in the application.
"""
from __future__ import annotations

import json
from typing import Any, Optional


def parse_json_field(raw: Optional[str], default: Any):
    """
    Safely parse a JSON-encoded string field.

    On any error (including empty input), return the provided default.
    """
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        # Silent fallback keeps UI resilient to malformed historical data
        return default
