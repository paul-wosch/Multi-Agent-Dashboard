"""
Response Normalizer for standardizing LLM provider responses.

This module normalizes diverse LLM provider responses into consistent,
serializable dictionaries. It handles provider-specific response formats,
extracts tool calls, merges content blocks, and flattens nested agent
response chains for uniform processing by the agent pipeline.

Key normalization tasks:
- Convert provider SDK objects to serializable dictionaries
- Merge multiple content blocks into single text responses
- Extract and structure tool calls from provider-specific formats
- Handle nested agent_response chains (e.g., from LangChain agents)
- Preserve metadata (usage, tokens, costs) across normalization

The normalizer ensures that regardless of the underlying LLM provider,
the engine receives responses in a consistent format for processing.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class ResponseNormalizer:
    """
    Convert SDK/LangChain responses into serializable dicts, merging content blocks,
    extracting tool calls, and flattening nested agent_response chains.
    """

    @staticmethod
    def normalize_to_dict(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except TypeError:
                return value.model_dump()
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {"__repr__": repr(value)}

    @staticmethod
    def to_dict(response: Any) -> Dict[str, Any]:
        """
        Convert SDK/LangChain response into a serializable dict (best-effort),
        avoiding noisy Pydantic serializer warnings from the SDK's internal model graph.
        Ensure `content_blocks` and `usage_metadata` are surfaced when present.
        Also flatten nested 'agent_response' shapes commonly returned by LangChain agents.
        """
        out: Dict[str, Any] = {}
        try:
            # Plain dicts (e.g., agent state returned by create_agent.invoke)
            # should be preserved as-is so that keys like 'structured_response'
            # remain visible to downstream consumers (engine, metrics, etc.).
            if isinstance(response, dict):
                out = dict(response)
            # LangChain AIMessage / Pydantic responses
            elif hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
                try:
                    out = response.model_dump()
                except TypeError:
                    # older versions may accept different args
                    out = response.model_dump()
            elif hasattr(response, "to_dict") and callable(getattr(response, "to_dict")):
                out = response.to_dict()
            elif hasattr(response, "dict"):
                out = response.dict()
            else:
                # Attempt repr capture
                out = {"repr": repr(response)}
        except Exception:
            logger.exception("Failed to convert response to dict")
            try:
                out = {"repr": repr(response)}
            except Exception:
                out = {"repr": "<unserializable response>"}

        # Ensure commonly-used convenience keys are present
        try:
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
            if usage is not None and "usage_metadata" not in out:
                out["usage_metadata"] = usage if isinstance(usage, dict) else {}
        except Exception:
            pass

        # content_blocks: prefer provider-agnostic structured blocks when available
        try:
            cb = getattr(response, "content_blocks", None)
            if cb is not None:
                blocks = []
                for b in cb:
                    if isinstance(b, dict):
                        blocks.append(b)
                    else:
                        try:
                            if hasattr(b, "model_dump"):
                                blocks.append(b.model_dump())
                            elif hasattr(b, "to_dict"):
                                blocks.append(b.to_dict())
                            elif hasattr(b, "__dict__"):
                                blocks.append(dict(b.__dict__))
                            else:
                                blocks.append(repr(b))
                        except Exception:
                            blocks.append(repr(b))
                out.setdefault("content_blocks", blocks)
        except Exception:
            pass

        # Attempt to surface tool_calls for compatibility
        try:
            tc = getattr(response, "tool_calls", None)
            if tc is not None and "tool_calls" not in out:
                out["tool_calls"] = tc
        except Exception:
            pass

        # Propagate any middleware events that might be attached directly on responses
        try:
            # Some agent flows return a dict-like state with our events key; prefer that
            if isinstance(response, dict):
                events = response.get("_multi_agent_dashboard_events") or response.get("instrumentation_events")
                if events is not None and "instrumentation_events" not in out:
                    out["instrumentation_events"] = events
            else:
                events_attr = getattr(response, "_multi_agent_dashboard_events", None) or getattr(response,
                                                                                                  "instrumentation_events",
                                                                                                  None)
                if events_attr is not None and "instrumentation_events" not in out:
                    out["instrumentation_events"] = events_attr
        except Exception:
            pass

        # -------------------------
        # Flatten LangChain agent_response chains (LangChain 2025/2026)
        # -------------------------
        seen_agent_responses = set()

        def _append_list(dest_key: str, value: Any) -> None:
            if value is None:
                return
            existing = out.get(dest_key)
            if existing is None:
                existing = []
                out[dest_key] = existing
            elif not isinstance(existing, list):
                existing = [existing]
                out[dest_key] = existing
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)

        def _ensure_key_from_source(source: dict, src_key: str, dest_key: Optional[str] = None) -> None:
            if dest_key is None:
                dest_key = src_key
            if dest_key in out:
                return
            if src_key in source and source[src_key] is not None:
                out[dest_key] = source[src_key]

        def _prepare_source(candidate: Any) -> Optional[Dict[str, Any]]:
            if candidate is None:
                return None
            if isinstance(candidate, dict):
                return candidate
            normalized = ResponseNormalizer.normalize_to_dict(candidate)
            if isinstance(normalized, dict):
                return normalized
            return None

        def _merge_agent_response(source: Any) -> None:
            src = _prepare_source(source)
            if not src:
                return
            marker = id(src)
            if marker in seen_agent_responses:
                return
            seen_agent_responses.add(marker)

            for key in ("usage", "usage_metadata", "structured_response", "structured", "messages"):
                _ensure_key_from_source(src, key)

            _append_list("tool_calls", src.get("tool_calls"))
            _append_list("instrumentation_events", src.get("instrumentation_events"))
            _append_list("_multi_agent_dashboard_events", src.get("_multi_agent_dashboard_events"))

            if isinstance(src.get("content_blocks"), list):
                _append_list("content_blocks", src.get("content_blocks"))

            output_entries = src.get("output")
            if isinstance(output_entries, list):
                for entry in output_entries:
                    if not isinstance(entry, dict):
                        continue
                    cb_entry = entry.get("content_blocks")
                    if isinstance(cb_entry, list):
                        _append_list("content_blocks", cb_entry)
                    _merge_agent_response(entry.get("response"))
                    _merge_agent_response(entry.get("result"))
                    _merge_agent_response(entry.get("agent_response"))

            nested = src.get("agent_response")
            if nested is not None:
                _merge_agent_response(nested)

        _merge_agent_response(response)
        _merge_agent_response(out.get("agent_response"))
        try:
            attr_agent_resp = getattr(response, "agent_response", None)
        except Exception:
            attr_agent_resp = None
        _merge_agent_response(attr_agent_resp)

        events = out.get("instrumentation_events")
        events_alt = out.get("_multi_agent_dashboard_events")
        if isinstance(events_alt, list) and not isinstance(events, list):
            out["instrumentation_events"] = list(events_alt)
            events = out["instrumentation_events"]
        if isinstance(events, list) and (not isinstance(events_alt, list) or events_alt is not events):
            out["_multi_agent_dashboard_events"] = list(events)

        return out