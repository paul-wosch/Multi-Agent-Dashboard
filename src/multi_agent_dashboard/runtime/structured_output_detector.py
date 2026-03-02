"""
Structured output detection and parsing for agent runtime execution.

This module implements the 4-path detection strategy for extracting structured
output (JSON/dict/list) from LLM responses. It handles multiple response formats
across different providers and instrumentation patterns.

Detection paths (in order of precedence):
1. Direct structured keys in raw response (`structured`, `structured_response`)
2. Instrumentation events with structured payloads
3. Content blocks with structured types (`structured`, `structured_response`,
   `structured_output`, `tool_call` with args)
4. Fallback JSON parsing of raw text output

The module also provides state writeback functionality to update execution
state with parsed structured output while preserving existing state values.

Key functions:
- `detect_structured_output`: Main detection function implementing the 4-path
  strategy with graceful fallbacks
- `writeback_to_state`: Safely merges parsed structured output into execution
  state, handling nested structures and type conflicts
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from ..shared.instrumentation import _structured_from_instrumentation

logger = logging.getLogger(__name__)


def detect_structured_output(
    raw: Dict[str, Any],
    content_blocks: List[Dict[str, Any]],
    raw_output: str,
) -> Optional[Any]:
    """
    Detect structured output from raw metrics, content blocks, and raw text.
    Implements the four-path detection used in AgentRuntime.run.
    Returns parsed dict/list/None.
    """
    parsed = None

    # 1) Structured keys directly on raw
    if isinstance(raw, dict):
        if "structured" in raw:
            parsed = raw.get("structured")
        elif "structured_response" in raw:
            parsed = raw.get("structured_response")
        else:
            # 2) Inspect instrumentation events
            parsed = _structured_from_instrumentation(raw)

        # 3) If still none, look through content blocks for structured payloads
        if parsed is None and isinstance(content_blocks, list):
            for cb in content_blocks:
                if not isinstance(cb, dict):
                    continue
                ctype = (cb.get("type") or "").lower()
                # Typical structured response block names
                if ctype in ("structured", "structured_response", "structured_output"):
                    parsed = cb.get("value") or cb.get("data") or cb.get("json") or cb.get("args") or cb.get("output")
                    break
                # Another pattern: provider returns a tool call with args that represent structured payload
                if ctype in ("tool_call", "server_tool_call") and isinstance(cb.get("args"), dict):
                    parsed = cb.get("args")
                    break

    # 4) Fallback: try best-effort JSON parsing of the textual output
    if parsed is None and isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output) if raw_output.strip() else None
        except Exception:
            parsed = None

    return parsed


def writeback_to_state(
    spec: Any,  # AgentSpec
    state: Dict[str, Any],
    parsed: Optional[Any],
    raw_output: str,
) -> None:
    """
    Write detected structured output (or raw text) into the state dict.
    Mirrors engine.run_seq writeback semantics.
    """
    if spec.output_vars:
        if isinstance(parsed, dict):
            # Warn about unexpected keys (do not raise; standalone runtime mirrors engine's warning behavior)
            for key in parsed:
                if key not in spec.output_vars:
                    logger.warning(
                        "[%s] Unexpected output key '%s' (standalone runtime)", spec.name, key
                    )
            # Populate declared output vars
            for var in spec.output_vars:
                if var in parsed:
                    state[var] = parsed[var]
                else:
                    logger.warning(
                        "[%s] Declared output '%s' missing (standalone runtime)", spec.name, var
                    )
        else:
            # Non-JSON output: if single output var, write the raw text; otherwise store under a raw key
            if len(spec.output_vars) == 1:
                state[spec.output_vars[0]] = raw_output
            else:
                key = f"{spec.name}__raw"
                state[key] = raw_output
                logger.warning(
                    "[%s] Non-JSON output stored as '%s' (standalone runtime)", spec.name, key
                )
    else:
        # No declared output_vars: store whole raw output under agent name
        state[spec.name] = raw_output