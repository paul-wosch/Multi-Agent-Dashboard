"""
Response processor for extracting and normalizing LLM agent responses.

This module provides the ResponseProcessor class that extracts metadata,
tool calls, content blocks, and text from LangChain agent responses and
normalizes them into a consistent TextResponse format. It handles the
complexities of different response structures across providers and
LangChain versions.

Key extraction capabilities:
- Usage metadata extraction (tokens, latency, costs)
- Tool call extraction and normalization
- Content block merging and text extraction
- Nested response chain flattening
- Provider-specific response format handling

The response processor ensures that regardless of the underlying LLM
provider or LangChain version, the dashboard receives consistent,
well-structured responses with complete metadata.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..response_normalizer import ResponseNormalizer

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """
    Static methods for processing LLM agent responses.
    
    This class encapsulates the logic previously in LLMClient._process_response,
    extracting usage, tool calls, content blocks, and text from raw agent results.
    """
    

    
    @staticmethod
    def extract_usage_from_messages(messages: Any) -> Dict[str, Any]:
        """
        Extract and accumulate usage metadata from all AIMessage-like objects.
        
        Returns dict with accumulated input_tokens and output_tokens (both may be 0).
        """
        result = {"input_tokens": 0, "output_tokens": 0}
        if not isinstance(messages, list):
            return result
        
        for msg in messages:  # NOT reversed - we want ALL messages
            try:
                # Determine if this is an AI message (where token usage is relevant)
                is_ai = False
                if isinstance(msg, dict):
                    is_ai = msg.get("type") == "ai"
                else:
                    # Check LangChain AIMessage types
                    msg_type = getattr(msg, "type", "")
                    if msg_type == "ai" or hasattr(msg, "usage_metadata") or hasattr(msg, "usage"):
                        is_ai = True
                
                if not is_ai:
                    continue
                
                # Extract usage payload (same logic as before)
                if isinstance(msg, dict):
                    usage_payload = msg.get("usage_metadata") or msg.get("usage") or msg.get("response_metadata")
                else:
                    usage_payload = getattr(msg, "usage_metadata", None) or getattr(msg, "usage", None) or getattr(msg,
                                                                                                                   "response_metadata",
                                                                                                                   None)
                
                if not isinstance(usage_payload, dict):
                    continue
                    
                # Extract token counts with fallback logic (preserve zero values)
                input_tokens = usage_payload.get("input_tokens")
                if input_tokens is None:
                    input_tokens = usage_payload.get("prompt_tokens")
                if input_tokens is None:
                    input_tokens = usage_payload.get("prompt_token_count")
                
                output_tokens = usage_payload.get("output_tokens")
                if output_tokens is None:
                    output_tokens = usage_payload.get("completion_tokens")
                if output_tokens is None:
                    output_tokens = usage_payload.get("completion_token_count")
                
                # Also check nested token_usage field
                token_usage = usage_payload.get("token_usage")
                if isinstance(token_usage, dict):
                    if input_tokens is None:
                        input_tokens = token_usage.get("prompt_tokens")
                        if input_tokens is None:
                            input_tokens = token_usage.get("input_tokens")
                    if output_tokens is None:
                        output_tokens = token_usage.get("completion_tokens")
                        if output_tokens is None:
                            output_tokens = token_usage.get("output_tokens")
                
                # Accumulate (treat None as 0)
                if isinstance(input_tokens, (int, float)):
                    result["input_tokens"] += int(input_tokens)
                if isinstance(output_tokens, (int, float)):
                    result["output_tokens"] += int(output_tokens)
                    
            except Exception:
                continue  # Middleware must not raise
        
        return result
    
    @staticmethod
    def extract_tool_info_from_messages(messages: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract tool_calls and content_blocks from AIMessage-like objects in a messages list.
        
        This method searches through message structures to find tool calls and content blocks,
        handling various formats including dicts, LangChain message objects, and OpenAI-style
        message structures with additional_kwargs.
        
        Args:
            messages: List of message objects or dictionaries to search
            
        Returns:
            Tuple of (tool_calls, content_blocks) where each is a list of dictionaries
            representing the extracted tool calls and content blocks
        """
        if not isinstance(messages, list):
            return [], []

        def _msg_to_dict(m: Any) -> Optional[Dict[str, Any]]:
            if isinstance(m, dict):
                return m
            try:
                if hasattr(m, "model_dump"):
                    out = m.model_dump()
                    return out if isinstance(out, dict) else None
            except Exception:
                pass
            try:
                if hasattr(m, "to_dict"):
                    out = m.to_dict()
                    return out if isinstance(out, dict) else None
            except Exception:
                pass
            try:
                if hasattr(m, "__dict__"):
                    out = dict(m.__dict__)
                    return out if isinstance(out, dict) else None
            except Exception:
                pass
            return None

        tool_calls: list[dict] = []
        content_blocks: list[dict] = []

        for msg in messages:
            msg_dict = _msg_to_dict(msg)
            if not isinstance(msg_dict, dict):
                continue
            tc = msg_dict.get("tool_calls")
            if isinstance(tc, list):
                for entry in tc:
                    e = _msg_to_dict(entry)
                    if isinstance(e, dict):
                        tool_calls.append(e)
            # Some integrations store tool_calls under additional_kwargs
            additional = msg_dict.get("additional_kwargs")
            if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
                for entry in additional.get("tool_calls"):
                    e = _msg_to_dict(entry)
                    if isinstance(e, dict):
                        tool_calls.append(e)
            cb = msg_dict.get("content_blocks")
            if isinstance(cb, list):
                for entry in cb:
                    e = _msg_to_dict(entry)
                    if isinstance(e, dict):
                        content_blocks.append(e)
            # OpenAI-native content blocks are often under "content"
            content = msg_dict.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                for entry in content:
                    e = _msg_to_dict(entry)
                    if isinstance(e, dict):
                        content_blocks.append(e)

        return tool_calls, content_blocks
    
    @staticmethod
    def extract_text_from_messages(result: Any, raw_dict: Dict[str, Any]) -> str:
        """
        Extract textual output from agent result and raw dict.
        
        This method walks through message structures from the end to the beginning,
        preferring assistant/AI messages, then falls back to other fields. It handles
        various message formats including dicts, LangChain message objects, and
        OpenAI-style content blocks.
        
        Args:
            result: Raw agent response object
            raw_dict: Dictionary representation of the result from ResponseNormalizer
            
        Returns:
            Extracted text as string, or empty string if extraction fails
        """
        text_out = None
        try:
            messages = None
            # Prefer actual message objects/structures from the returned result
            if isinstance(result, dict):
                messages = result.get("messages", None)
            if messages is None:
                # Try attribute access (some Agent result objects provide .messages)
                try:
                    messages_attr = getattr(result, "messages", None)
                    if messages_attr is not None:
                        # Coerce to list where possible
                        if isinstance(messages_attr, (list, tuple)):
                            messages = list(messages_attr)
                        else:
                            # Some objects may expose an indexable interface; try converting to list
                            try:
                                messages = list(messages_attr)
                            except Exception:
                                messages = None
                except Exception:
                    messages = None

            # If we have a messages sequence, inspect from last -> first for assistant content
            if isinstance(messages, list) and messages:
                for m in reversed(messages):
                    # m could be dict or message object
                    content_candidate = None
                    role = None
                    try:
                        if isinstance(m, dict):
                            role = (m.get("role") or m.get("type") or "")
                            # Look for typical content fields
                            c = m.get("content")
                            if isinstance(c, str):
                                content_candidate = c
                            elif isinstance(c, list):
                                parts = []
                                for cpart in c:
                                    if isinstance(cpart, dict):
                                        # block form: {'type': 'output_text', 'text': '...'}
                                        if cpart.get("text") is not None:
                                            parts.append(str(cpart.get("text") or ""))
                                        else:
                                            parts.append(str(cpart))
                                    elif isinstance(cpart, str):
                                        parts.append(cpart)
                                if parts:
                                    content_candidate = "".join(parts)
                            elif isinstance(m.get("text"), str):
                                content_candidate = m.get("text")
                            # Some message dicts include 'output' as list
                            elif isinstance(m.get("output"), list):
                                parts = []
                                for o in m.get("output", []):
                                    if isinstance(o, dict) and o.get("type") in ("text", "output_text"):
                                        parts.append(o.get("text") or "")
                                    elif isinstance(o, str):
                                        parts.append(o)
                                if parts:
                                    content_candidate = "".join(parts)
                        else:
                            # Message object: try common attributes
                            if hasattr(m, "content"):
                                mc = getattr(m, "content")
                                if isinstance(mc, str):
                                    content_candidate = mc
                                elif isinstance(mc, list):
                                    parts = []
                                    for p in mc:
                                        if isinstance(p, str):
                                            parts.append(p)
                                        elif isinstance(p, dict) and p.get("text"):
                                            parts.append(p.get("text"))
                                    if parts:
                                        content_candidate = "".join(parts)
                            if content_candidate is None and hasattr(m, "text"):
                                mt = getattr(m, "text")
                                if callable(mt):
                                    try:
                                        mt = mt()
                                    except Exception:
                                        mt = None
                                if isinstance(mt, str):
                                    content_candidate = mt
                            # Some message objects include 'role'
                            if role is None and hasattr(m, "role"):
                                try:
                                    role = getattr(m, "role")
                                except Exception:
                                    role = None
                    except Exception:
                        content_candidate = None
                        role = None

                    # Normalize role to string for comparisons
                    try:
                        role_str = (str(role) if role is not None else "").lower()
                    except Exception:
                        role_str = ""

                    if content_candidate:
                        # Prefer messages that look like assistant responses
                        if role_str in ("assistant", "ai", "bot", "system_assistant", "assistant_response"):
                            text_out = content_candidate
                            break
                        # If role not marked, pick first available candidate as fallback (but continue searching for assistant)
                        if text_out is None:
                            text_out = content_candidate

            # If still None, fall back to common response attributes on the returned result
            if text_out is None:
                text_attr = getattr(result, "text", None)
                if callable(text_attr):
                    try:
                        text_out = text_attr()
                    except Exception:
                        text_out = None
                elif text_attr is not None:
                    text_out = text_attr
                else:
                    content = getattr(result, "content", None) or raw_dict.get("content")
                    if isinstance(content, list):
                        parts = []
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("text", "output_text"):
                                parts.append(c.get("text") or "")
                            elif isinstance(c, str):
                                parts.append(c)
                        if parts:
                            text_out = "".join(parts)
                    elif isinstance(content, str):
                        text_out = content
        except Exception:
            text_out = None

        # Final fallback: stringify the returned result if extraction failed
        if text_out is None:
            try:
                text_out = str(result)
            except Exception:
                text_out = ""

        return text_out
    
    @staticmethod
    def process(result: Any, latency: float, agent: Any = None) -> "TextResponse":
        """
        Process raw agent response into a normalized TextResponse.
        
        This is the main entry point for response processing. It extracts usage metadata,
        tool calls, content blocks, and text from the raw agent response, normalizes them
        into a consistent format, and returns a TextResponse object.
        
        Args:
            result: Raw agent response object from LangChain agent.invoke()
            latency: Execution latency in seconds
            agent: Optional agent instance (currently unused, reserved for future use)
            
        Returns:
            TextResponse containing normalized response data with text, raw dict,
            token counts, and latency
            
        Raises:
            ValueError: If result cannot be processed into a valid TextResponse
        """
        # Convert to serializable dict using ResponseNormalizer
        raw_dict = ResponseNormalizer.to_dict(result)
        
        # Ensure instrumentation events propagate structured response/content blocks.
        events = (
                raw_dict.get("instrumentation_events")
                or raw_dict.get("_multi_agent_dashboard_events")
        )
        if isinstance(events, list):
            cb_list = raw_dict.get("content_blocks")
            if cb_list is None:
                cb_list = []
                raw_dict["content_blocks"] = cb_list
            elif not isinstance(cb_list, list):
                cb_list = [cb_list]
                raw_dict["content_blocks"] = cb_list

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ev_blocks = ev.get("content_blocks")
                if isinstance(ev_blocks, list):
                    cb_list.extend(ev_blocks)
                if "structured_response" in ev and "structured_response" not in raw_dict:
                    raw_dict["structured_response"] = ev["structured_response"]
        
        # Extract usage from top-level result and raw_dict
        input_tokens = None
        output_tokens = None
        usage = (
                getattr(result, "usage_metadata", None)
                or getattr(result, "usage", None)
                or raw_dict.get("usage")
                or raw_dict.get("usage_metadata")
        )
        if isinstance(usage, dict):
            if input_tokens is None:
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get(
                    "prompt_token_count")
            if output_tokens is None:
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get(
                    "completion_token_count")
            token_usage = usage.get("token_usage")
            if isinstance(token_usage, dict):
                if input_tokens is None:
                    input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                if output_tokens is None:
                    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        
        # Promote tool_calls/content_blocks from agent messages into raw_dict (always)
        messages = None
        try:
            if isinstance(result, dict):
                messages = result.get("messages")
            if messages is None and isinstance(raw_dict, dict):
                messages = raw_dict.get("messages")
        except Exception:
            messages = None

        try:
            tool_calls, content_blocks = ResponseProcessor.extract_tool_info_from_messages(messages)
            if isinstance(raw_dict, dict):
                if tool_calls and "tool_calls" not in raw_dict:
                    raw_dict["tool_calls"] = tool_calls
                # Do NOT promote content_blocks into raw_dict here to avoid duplication.
                # _collect_content_blocks() already inspects messages and raw content_blocks.
        except Exception:
            logger.debug("Failed to attach tool info from messages into raw_dict", exc_info=True)
        
        if input_tokens is None or output_tokens is None:
            # Try usage on the last AIMessage from agent state (common in LangChain agents)
            msg_usage = ResponseProcessor.extract_usage_from_messages(messages)
            if isinstance(msg_usage, dict):
                if input_tokens is None:
                    input_tokens = msg_usage.get("input_tokens") or msg_usage.get("prompt_tokens") or msg_usage.get(
                        "prompt_token_count")
                if output_tokens is None:
                    output_tokens = msg_usage.get("output_tokens") or msg_usage.get(
                        "completion_tokens") or msg_usage.get("completion_token_count")
                token_usage = msg_usage.get("token_usage")
                if isinstance(token_usage, dict):
                    if input_tokens is None:
                        input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                    if output_tokens is None:
                        output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                
                # Promote usage to raw_dict for downstream consumers
                try:
                    if isinstance(raw_dict, dict):
                        raw_dict.setdefault("usage", msg_usage)
                        raw_dict.setdefault("usage_metadata", msg_usage)
                except Exception:
                    logger.debug("Failed to attach message usage into raw_dict", exc_info=True)
            
            nested_usage = raw_dict.get("usage") or raw_dict.get("usage_metadata")
            if isinstance(nested_usage, dict):
                if input_tokens is None:
                    input_tokens = nested_usage.get("input_tokens") or nested_usage.get(
                        "prompt_tokens") or nested_usage.get("prompt_token_count")
                if output_tokens is None:
                    output_tokens = nested_usage.get("output_tokens") or nested_usage.get(
                        "completion_tokens") or nested_usage.get("completion_token_count")
                token_usage = nested_usage.get("token_usage")
                if isinstance(token_usage, dict):
                    if input_tokens is None:
                        input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                    if output_tokens is None:
                        output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                
                # Crucial: ensure the detected nested usage is available in the raw dict
                # so downstream consumers (engine/DB) can inspect usage via resp.raw
                try:
                    if isinstance(nested_usage, dict):
                        # Normalize key names conservatively
                        if "usage" not in raw_dict:
                            raw_dict["usage"] = nested_usage
                        if "usage_metadata" not in raw_dict:
                            raw_dict["usage_metadata"] = nested_usage
                except Exception:
                    logger.debug("Failed to attach nested usage into raw_dict", exc_info=True)
        
        # Extract text from messages
        text_out = ResponseProcessor.extract_text_from_messages(result, raw_dict)
        
        # Ensure content_blocks are surfaced in raw_dict if present on result
        try:
            cb = getattr(result, "content_blocks", None)
            if cb is not None and "content_blocks" not in raw_dict:
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
                raw_dict["content_blocks"] = blocks
        except Exception:
            pass
        
        from .client import TextResponse
        return TextResponse(
            text=text_out,
            raw=raw_dict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
        )