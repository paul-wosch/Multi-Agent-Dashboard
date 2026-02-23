# Metrics extraction logic for agent runtime
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..shared.instrumentation import (
    _extract_instrumentation_events,
    _collect_content_blocks,
    _collect_tool_calls,
    _tool_usage_entry_from_payload,
)

logger = logging.getLogger(__name__)


def extract_tokens_from_raw(raw: Dict[str, Any], response: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract input and output token counts from raw response and response object.
    Returns (input_tokens, output_tokens) where each can be None.
    """
    input_tokens = response.input_tokens
    output_tokens = response.output_tokens
    try:
        if (input_tokens is None or output_tokens is None) and isinstance(raw, dict):
            usage = raw.get("usage") or raw.get("usage_metadata") or {}
            if isinstance(usage, dict):
                if input_tokens is None:
                    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count")
                    if input_tokens is None and isinstance(usage.get("token_usage"), dict):
                        input_tokens = usage["token_usage"].get("prompt_tokens") or usage["token_usage"].get("input_tokens")
                if output_tokens is None:
                    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_count")
                    if output_tokens is None and isinstance(usage.get("token_usage"), dict):
                        output_tokens = usage["token_usage"].get("completion_tokens") or usage["token_usage"].get("output_tokens")
    except Exception:
        logger.debug("Token fallback extraction from raw usage failed", exc_info=True)
    
    return input_tokens, output_tokens


def collect_tool_usage(raw: Dict[str, Any], content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool usage entries from raw metrics and content blocks.
    Returns a list of tool usage dicts.
    """
    used_tools: List[Dict[str, Any]] = []
    seen_tool_entries: set[tuple[str, str | None]] = set()

    def _maybe_add_tool_entry(entry: Dict[str, Any] | None) -> None:
        if not entry:
            return
        tool_type = entry.get("tool_type")
        if not tool_type:
            return
        key = (tool_type, entry.get("id"))
        if key in seen_tool_entries:
            return
        seen_tool_entries.add(key)
        used_tools.append(entry)

    TOOL_BLOCK_TYPES = {"tool_call", "server_tool_call", "web_search_call", "web_search", "function_call"}
    for item in content_blocks:
        if not isinstance(item, dict):
            continue
        btype = str(item.get("type") or "").lower()
        if btype and btype not in TOOL_BLOCK_TYPES:
            continue
        _maybe_add_tool_entry(_tool_usage_entry_from_payload(item))

    for call_payload in _collect_tool_calls(raw):
        _maybe_add_tool_entry(_tool_usage_entry_from_payload(call_payload))

    return used_tools


def extract_detected_provider_profile(
    agent_obj_for_invoke: Any,
    raw: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract detected provider profile from agent object or raw metrics.
    Returns the profile dict or None.
    """
    try:
        detected = getattr(agent_obj_for_invoke, "_detected_provider_profile", None)
        if detected is not None:
            return detected
        # In some flows, the LLM client may have included this in response.raw
        resp_detected = raw.get("detected_provider_profile")
        if resp_detected is not None:
            return resp_detected
    except Exception:
        pass
    return None