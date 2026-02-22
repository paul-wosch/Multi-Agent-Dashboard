# src/multi_agent_dashboard/engine/utils.py
# Consolidated helper functions extracted from engine.py and models.py

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from multi_agent_dashboard.config import OPENAI_PRICING, DEEPSEEK_PRICING

logger = logging.getLogger(__name__)


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
        output = node_dict.get("output")
        if isinstance(output, list):
            for entry in output:
                _recurse(entry)
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


def _normalize_content_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    """
    Ensure each content block is a serializable dict (best-effort).
    """
    out_blocks: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return []
    for b in blocks:
        if isinstance(b, dict):
            out_blocks.append(b)
            continue
        try:
            if hasattr(b, "model_dump"):
                out_blocks.append(b.model_dump())
            elif hasattr(b, "to_dict"):
                out_blocks.append(b.to_dict())
            elif hasattr(b, "__dict__"):
                out_blocks.append(dict(b.__dict__))
            else:
                out_blocks.append({"__repr": repr(b)})
        except Exception:
            out_blocks.append({"__repr": repr(b)})
    return out_blocks


def _compute_cost(
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        provider_id: Optional[str] = None,
) -> tuple[float, float, float]:
    """Compute approximate cost for a single LLM call.

    Return (total_cost, input_cost, output_cost).

    Prices are per 1M tokens.

    This helper is provider-aware: for OpenAI-family providers
    (provider_id is None/'' or 'openai' or 'azure_openai') it uses
    OPENAI_PRICING. For other providers we currently return zero so that
    non-OpenAI calls do not get mis-attributed OpenAI prices.
    """
    if input_tokens is None and output_tokens is None:
        return 0.0, 0.0, 0.0

    # Parse provider/model string format
    model_for_pricing = model
    if "/" in model:
        # Split only on first slash
        maybe_provider, model_name = model.split("/", 1)
        model_for_pricing = model_name
        # If provider_id not specified, use extracted provider
        if provider_id is None:
            provider_id = maybe_provider
        # If provider_id specified but differs, log warning but use extracted provider
        elif provider_id != maybe_provider:
            logger.debug(
                f"Provider mismatch in _compute_cost: model suggests '{maybe_provider}', "
                f"but provider_id is '{provider_id}'. Using '{maybe_provider}' for pricing."
            )
            provider_id = maybe_provider

    provider = (provider_id or "").strip().lower()

    # Backwards-compatible default: treat missing provider_id as OpenAI.
    is_openai_family = (not provider) or provider in ("openai", "azure_openai")

    pricing = None
    if is_openai_family:
        pricing = OPENAI_PRICING.get(model_for_pricing)
    elif provider == "deepseek":
        try:
            from multi_agent_dashboard.config import DEEPSEEK_PRICING
            pricing = DEEPSEEK_PRICING.get(model_for_pricing)
        except Exception:
            pricing = None
    if not pricing:
        return 0.0, 0.0, 0.0

    inp = input_tokens or 0
    out = output_tokens or 0

    input_cost = inp / 1_000_000.0 * pricing.get("input", 0.0)
    output_cost = out / 1_000_000.0 * pricing.get("output", 0.0)
    return input_cost + output_cost, input_cost, output_cost


def _extract_provider_features_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a LangChain model 'profile' into a compact provider_features mapping.

    This is intentionally conservative: only expose a few well-known capability hints
    used by the UI (structured_output, tool_calling, reasoning, image_inputs, max_input_tokens).
    """

    features: Dict[str, Any] = {}

    if not isinstance(profile, dict):
        return features

    # Normalize profile keys to handle camelCase, snake_case, and lower-case variants.
    def _normalize_key(k: str) -> str:
        # Convert camelCase / PascalCase to snake_case
        s = re.sub(r'(?<!^)(?=[A-Z])', '_', str(k)).lower()
        s = s.replace('-', '_')
        return s

    normalized: Dict[str, Any] = {}
    for k, v in profile.items():
        try:
            normalized[str(k)] = v
        except Exception:
            normalized[k] = v
        try:
            normalized[str(k).lower()] = v
        except Exception:
            pass
        try:
            nk = _normalize_key(str(k))
            normalized[nk] = v
        except Exception:
            pass

    # Structured output related hints
    if normalized.get("structured_output") or normalized.get("structuredoutput") or normalized.get("reasoning_output") or normalized.get("structured"):
        features["structured_output"] = True

    # Tool calling hints
    if normalized.get("tool_calling") or normalized.get("toolcalling") or normalized.get("tool_calls") or normalized.get("toolcalls") or normalized.get("tool_call"):
        features["tool_calling"] = True

    # Reasoning hints
    if normalized.get("reasoning") or normalized.get("reasoning_output") or normalized.get("reasoningoutput") or normalized.get("supports_reasoning"):
        features["reasoning"] = True

    # Image / multimodal hints
    if "image_inputs" in normalized or "imageinputs" in normalized:
        try:
            features["image_inputs"] = bool(normalized.get("image_inputs") or normalized.get("imageinputs"))
        except Exception:
            features["image_inputs"] = True

    # Max input tokens (context window) — try variants
    max_tokens_candidates = [
        normalized.get("max_input_tokens"),
        normalized.get("maxinputtokens"),
        normalized.get("max_input_token"),
        normalized.get("max_input"),
        normalized.get("maxInputTokens"),
    ]
    for candidate in max_tokens_candidates:
        if candidate is not None:
            try:
                features["max_input_tokens"] = int(candidate)
            except Exception:
                features["max_input_tokens"] = candidate
            break

    # If nothing obvious matched, expose a shallow copy for auditing
    if not features and profile:
        # Keep only a small subset to avoid clobbering DB with huge dicts
        keys_to_copy = ["tool_calling", "structured_output", "reasoning", "image_inputs", "max_input_tokens", "maxInputTokens", "structuredOutput", "toolCalling"]
        for k in keys_to_copy:
            if k in profile:
                features[k if "_" in k else _normalize_key(k)] = profile[k]

    return features