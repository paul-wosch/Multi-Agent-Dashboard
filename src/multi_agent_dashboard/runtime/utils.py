# utils.py
import logging
from string import Template
from typing import Dict, Any
from multi_agent_dashboard.config import AGENT_INPUT_CAP, AGENT_OUTPUT_CAP

logger = logging.getLogger(__name__)

class SafeTemplate(Template):
    """
    string.Template variant that uses {var} instead of $var
    and safely ignores missing keys.
    """
    delimiter = "{"
    pattern = r"""
    \{(?:
        (?P<escaped>\{) |        # {{ -> {
        (?P<named>[_a-z][_a-z0-9]*)\} |  # {var}
        (?P<braced>[_a-z][_a-z0-9]*)\} |
        (?P<invalid>)
    )
    """
    flags = 0


def safe_format(
    template: str,
    mapping: Dict[str, Any],
    *,
    max_value_len: int = AGENT_INPUT_CAP,
    max_prompt_len: int = AGENT_OUTPUT_CAP,
) -> str:
    """
    Safely formats prompts:
    - No eval / attribute access
    - Missing keys become empty strings
    - Values are truncated
    - Prompt length is capped
    """

    clean: Dict[str, str] = {}

    for k, v in mapping.items():
        s = "" if v is None else str(v)
        if len(s) > max_value_len:
            logger.warning(
                "safe_format: value for key '%s' truncated (%d → %d chars)",
                k,
                len(s),
                max_value_len,
            )
            s = s[:max_value_len]
        clean[k] = s

    rendered = SafeTemplate(template).safe_substitute(clean)

    if len(rendered) > max_prompt_len:
        logger.warning(
            "safe_format: prompt truncated (%d → %d chars)",
            len(rendered),
            max_prompt_len,
        )
        rendered = rendered[:max_prompt_len]

    return rendered
