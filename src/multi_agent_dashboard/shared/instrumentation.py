"""
Instrumentation helper functions for extracting and processing LLM response metadata.

This module provides shared utilities for working with instrumentation events
and metrics collected during LLM agent execution. It handles extraction from
raw response data, content block processing, and structured output detection.

Key functions:
- `_extract_instrumentation_events`: Extract instrumentation events from raw
  metrics with backward compatibility support
- `_collect_content_blocks`: Aggregate content blocks from instrumentation events
- `_structured_from_instrumentation`: Extract structured output from
  instrumentation events
- `_collect_tool_calls`: Collect tool call information from content blocks
- `_tool_usage_entry_from_payload`: Create standardized tool usage entries from
  tool call payloads

These functions are used by both the runtime package (for metrics extraction)
and the engine package (for orchestration and state management).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _extract_instrumentation_events(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    # Primary key used by middleware
    events = raw_metrics.get("_multi_agent_dashboard_events")
    if isinstance(events, list):
        return events
    # Backwards-compatible alias (used by LLMClient.invoke_agent)
    events2 = raw_metrics.get("instrumentation_events")
    if isinstance(events2, list):
        return events2
    return []


def _value_to_dict(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            normalized = value.model_dump()
            if isinstance(normalized, dict):
                return normalized
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            normalized = value.to_dict()
            if isinstance(normalized, dict):
                return normalized
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return None


def _collect_content_blocks(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    blocks: List[Dict[str, Any]] = []
    for event in _extract_instrumentation_events(raw_metrics):
        payload = event.get("content_blocks")
        if isinstance(payload, list):
            blocks.extend(payload)
    direct = raw_metrics.get("content_blocks")
    if isinstance(direct, list):
        blocks.extend(direct)
    # Also inspect messages for content/content_blocks (LangChain agent state)
    messages = raw_metrics.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            msg_dict = _value_to_dict(msg)
            if not isinstance(msg_dict, dict):
                continue
            cb = msg_dict.get("content_blocks")
            if isinstance(cb, list):
                blocks.extend(cb)
            # OpenAI-native content blocks live under "content" as a list of dicts
            content = msg_dict.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                blocks.extend(content)
    elif isinstance(raw_metrics.get("output"), list):
        blocks.extend(raw_metrics["output"])
    return blocks


def _structured_from_instrumentation(raw_metrics: Dict[str, Any] | None) -> Any:
    if not isinstance(raw_metrics, dict):
        return None
    for event in _extract_instrumentation_events(raw_metrics):
        if "structured_response" in event:
            return event["structured_response"]
    return None


def _collect_tool_calls(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    calls: List[Dict[str, Any]] = []

    def _recurse(node: Any) -> None:
        node_dict = _value_to_dict(node)
        if not isinstance(node_dict, dict):
            return
        tool_calls = node_dict.get("tool_calls")
        if isinstance(tool_calls, list):
            for entry in tool_calls:
                entry_dict = _value_to_dict(entry)
                if isinstance(entry_dict, dict):
                    calls.append(entry_dict)
        # Some providers nest tool_calls under additional_kwargs
        additional = node_dict.get("additional_kwargs")
        if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
            for entry in additional.get("tool_calls"):
                entry_dict = _value_to_dict(entry)
                if isinstance(entry_dict, dict):
                    calls.append(entry_dict)
        # Recurse into messages list if present (LangChain agent state)
        messages = node_dict.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                _recurse(msg)
        for key in ("agent_response", "response", "result"):
            _recurse(node_dict.get(key))
        events = node_dict.get("instrumentation_events") or node_dict.get("_multi_agent_dashboard_events")
        if isinstance(events, list):
            for event in events:
                _recurse(event)

    _recurse(raw_metrics)
    return calls


def _tool_usage_entry_from_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    raw_type = payload.get("type") or payload.get("tool_type") or ""
    raw_type_l = str(raw_type).lower()
    # Skip non-tool content blocks (e.g., text/reasoning)
    if raw_type_l in ("text", "reasoning"):
        return None
    tool_type = payload.get("name") or payload.get("tool_type") or payload.get("type") or "unknown"
    if tool_type in ("unknown", "", None) and not payload.get("name"):
        return None
    entry: Dict[str, Any] = {
        "tool_type": tool_type,
        "id": payload.get("id") or payload.get("tool_call_id") or payload.get("tool_use_id"),
    }
    status = payload.get("status") or payload.get("state")
    if status is not None:
        entry["status"] = status
    action = payload.get("args") or payload.get("action") or payload.get("input") or payload.get("result")
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except Exception:
            action = {"raw": action}
    if isinstance(action, dict):
        entry["action"] = action
    elif action is not None:
        entry["action"] = {"raw": action}
    return entry