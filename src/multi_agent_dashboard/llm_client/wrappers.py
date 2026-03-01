"""
Wrapper classes for structured output and response normalization.

This module provides the StructuredOutputWrapper class that normalizes payloads,
extracts usage/token counts, and wraps structured output results for the agent pipeline.
"""

import json
import logging
from typing import Any, Dict, Optional

# Try to import LangChain AIMessage (optional)
from .availability import get_AIMessage
_AIMessage = get_AIMessage()

logger = logging.getLogger(__name__)


class StructuredOutputWrapper:
    """
    Normalizes payloads, extracts usage/token counts, and wraps structured output
    results for the agent pipeline.
    """

    @staticmethod
    def wrap(model_instance: Any) -> Any:
        """
        Wrap a chat model that returns structured dicts so the agent pipeline
        always receives a BaseMessage (AIMessage) or role/content dict, while
        preserving usage / token metadata (including Ollama prompt_eval_count /
        eval_count) and response_metadata.
        """

        def _normalize_payload(value: Any) -> Optional[Dict[str, Any]]:
            if isinstance(value, dict):
                return value
            if hasattr(value, "model_dump"):
                try:
                    payload = value.model_dump()
                    return payload if isinstance(payload, dict) else None
                except Exception:
                    return None
            if hasattr(value, "dict"):
                try:
                    payload = value.dict()
                    return payload if isinstance(payload, dict) else None
                except Exception:
                    return None
            return None

        def _extract_usage_and_counts(obj: Any) -> tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]:
            if obj is None:
                return None, None, None

            def _maybe(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            usage = None
            for key in ("usage_metadata", "usage"):
                cand = _maybe(obj, key)
                if isinstance(cand, dict):
                    usage = cand
                    break

            pe = None
            ce = None
            resp = _maybe(obj, "response_metadata")
            if isinstance(resp, dict):
                if isinstance(resp.get("usage"), dict):
                    usage = usage or resp["usage"]
                pe = resp.get("prompt_eval_count") if resp.get("prompt_eval_count") is not None else pe
                ce = resp.get("eval_count") if resp.get("eval_count") is not None else ce

            if isinstance(obj, dict):
                pe = obj.get("prompt_eval_count") if obj.get("prompt_eval_count") is not None else pe
                ce = obj.get("eval_count") if obj.get("eval_count") is not None else ce

            return usage, pe, ce

        def _normalize_usage(usage: Optional[Dict[str, Any]], pe: Optional[int], ce: Optional[int]) -> Optional[
            Dict[str, Any]]:
            usage = dict(usage) if isinstance(usage, dict) else {}
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or pe
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or ce
            input_tokens = usage.get("input_tokens") or prompt_tokens
            output_tokens = usage.get("output_tokens") or completion_tokens
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = (input_tokens or 0) + (output_tokens or 0)
                elif input_tokens is not None:
                    total_tokens = input_tokens
                elif output_tokens is not None:
                    total_tokens = output_tokens

            if not any([input_tokens, output_tokens, total_tokens]):
                return None

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        def _wrap_result(result: Any) -> Any:
            raw_msg = None
            parsed_part = None
            if isinstance(result, dict) and ("raw" in result and "parsed" in result):
                raw_msg = result.get("raw")
                parsed_part = result.get("parsed")

            # If there is no parsed structured output but we have a raw message (e.g., tool call),
            # return the raw message directly—it is already an AIMessage or compatible object.
            if parsed_part is None and raw_msg is not None:
                return raw_msg

            payload = _normalize_payload(parsed_part if parsed_part is not None else result)
            if payload is None:
                return result
            if "role" in payload and "content" in payload:
                return payload

            usage_source = raw_msg if raw_msg is not None else result
            usage_res, pe_res, ce_res = _extract_usage_and_counts(usage_source)
            usage_pay, pe_pay, ce_pay = _extract_usage_and_counts(payload)
            pe = pe_res if pe_res is not None else pe_pay
            ce = ce_res if ce_res is not None else ce_pay
            usage_norm = _normalize_usage(usage_res or usage_pay, pe, ce)

            resp_meta: Dict[str, Any] = {}
            try:
                rm = getattr(usage_source, "response_metadata", None)
                if isinstance(rm, dict):
                    resp_meta.update(rm)
            except Exception:
                pass
            if usage_norm:
                resp_meta.setdefault("usage", usage_norm)
            elif pe is not None or ce is not None:
                resp_meta.setdefault(
                    "usage",
                    {
                        "prompt_tokens": pe,
                        "input_tokens": pe,
                        "completion_tokens": ce,
                        "output_tokens": ce,
                        "total_tokens": (pe or 0) + (ce or 0),
                    },
                )

            additional_kwargs: Dict[str, Any] = {}
            try:
                ak = getattr(usage_source, "additional_kwargs", None)
                if isinstance(ak, dict):
                    additional_kwargs.update(ak)
            except Exception:
                pass
            additional_kwargs["structured_response"] = payload

            msg_kwargs: Dict[str, Any] = {"additional_kwargs": additional_kwargs}
            if usage_norm:
                msg_kwargs["usage_metadata"] = usage_norm
            if resp_meta:
                msg_kwargs["response_metadata"] = resp_meta

            content = json.dumps(payload)
            if _AIMessage is not None:
                try:
                    mid = getattr(usage_source, "id", None)
                    if mid is not None:
                        msg_kwargs["id"] = mid
                except Exception:
                    pass
                try:
                    nm = getattr(usage_source, "name", None)
                    if nm is not None:
                        msg_kwargs["name"] = nm
                except Exception:
                    pass
                try:
                    tc = getattr(usage_source, "tool_calls", None)
                    if isinstance(tc, list) and tc:
                        msg_kwargs["tool_calls"] = tc
                except Exception:
                    pass

                msg_kwargs["content"] = content
                try:
                    msg = _AIMessage(**msg_kwargs)
                except Exception:
                    msg = _AIMessage(content=content, additional_kwargs=additional_kwargs)

                try:
                    rm2 = getattr(usage_source, "response_metadata", None)
                    had_counts = isinstance(rm2, dict) and (
                            rm2.get("prompt_eval_count") is not None
                            or rm2.get("eval_count") is not None
                            or isinstance(rm2.get("usage"), dict)
                    )
                    if had_counts and not getattr(msg, "usage_metadata", None):
                        logger.warning(
                            "Structured output wrapper: raw result had usage/counts but wrapped message lacks usage_metadata"
                        )
                except Exception:
                    pass

                return msg

            out = {"role": "assistant", "content": content, "structured_response": payload}
            if usage_norm:
                out["usage_metadata"] = usage_norm
            if resp_meta:
                out["response_metadata"] = resp_meta
            return out

        class _StructuredOutputMessageAdapter:
            def __init__(self, inner: Any):
                self._inner = inner

            def invoke(self, *args, **kwargs):
                return _wrap_result(self._inner.invoke(*args, **kwargs))

            async def ainvoke(self, *args, **kwargs):
                return _wrap_result(await self._inner.ainvoke(*args, **kwargs))

            def stream(self, *args, **kwargs):
                for chunk in self._inner.stream(*args, **kwargs):
                    yield _wrap_result(chunk)

            def bind(self, *args, **kwargs):
                try:
                    bound = self._inner.bind(*args, **kwargs)
                except Exception:
                    bound = self._inner
                return _StructuredOutputMessageAdapter(bound)

            def bind_tools(self, *args, **kwargs):
                try:
                    bound = self._inner.bind_tools(*args, **kwargs)
                except Exception:
                    bound = self._inner
                return _StructuredOutputMessageAdapter(bound)

            def with_config(self, *args, **kwargs):
                try:
                    bound = self._inner.with_config(*args, **kwargs)
                except Exception:
                    bound = self._inner
                return _StructuredOutputMessageAdapter(bound)

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        return _StructuredOutputMessageAdapter(model_instance)