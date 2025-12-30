# ui/utils.py
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
