# multi_agent_dashboard/llm_client.py
import time
import logging
import json
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "TextResponse",
    "LLMError",
    "INSTRUMENTATION_MIDDLEWARE",
]

# Try to import LangChain pieces (optional)
_LANGCHAIN_AVAILABLE = False
_init_chat_model = None
_SystemMessage = None
_HumanMessage = None
_create_agent = None
_AgentMiddleware = None

try:
    from langchain.chat_models import init_chat_model  # type: ignore
    from langchain.messages import SystemMessage, HumanMessage  # type: ignore
    from langchain.agents import create_agent  # type: ignore
    from langchain.agents.middleware import AgentMiddleware  # type: ignore

    _LANGCHAIN_AVAILABLE = True
    _init_chat_model = init_chat_model
    _SystemMessage = SystemMessage
    _HumanMessage = HumanMessage
    _create_agent = create_agent
    _AgentMiddleware = AgentMiddleware
except Exception:
    # Keep resilience when LangChain is not installed or partial environments.
    _LANGCHAIN_AVAILABLE = False
    _init_chat_model = None
    _SystemMessage = None
    _HumanMessage = None
    _create_agent = None
    _AgentMiddleware = None

# If AgentMiddleware import failed, provide a minimal fallback so that tests and
# environments without langchain.agents.middleware can still instantiate and use
# the instrumentation middleware class. This fallback intentionally mirrors the
# minimal method signatures used by LangChain's middleware system.
if _AgentMiddleware is None:
    class _FallbackAgentMiddleware:
        """
        Minimal fallback middleware base compatible with LangChain's AgentMiddleware interface.
        This allows tests and environments without langchain.agents.middleware to still
        instantiate and use the instrumentation middleware defined below.
        """
        # Provide multiple common hook names across LangChain minor versions
        def before_model(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        def modify_model_request(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        def after_model(self, state: Dict[str, Any], runtime: Any) -> Any:
            return None

        # Some versions use wrap_model_call semantics (return-through handler)
        def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
            return handler(request)

    _AgentMiddleware = _FallbackAgentMiddleware


# ------------------------
# Instrumentation helpers
# ------------------------

def _normalize_to_dict(value: Any) -> Dict[str, Any]:
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


def _extract_content_blocks_from_message(message: Any) -> List[Dict[str, Any]]:
    normalized = _normalize_to_dict(message)
    blocks = normalized.get("content_blocks")
    if isinstance(blocks, list):
        return blocks
    output = normalized.get("output")
    if isinstance(output, list):
        return output
    return []


_INSTRUMENTATION_MIDDLEWARE: Optional[type] = None

# Always define a dashboard instrumentation middleware class (subclassing the
# real AgentMiddleware when available, otherwise the fallback above).
class _DashboardInstrumentationMiddleware(_AgentMiddleware):  # type: ignore
    name = "MultiAgentInstrumentationMiddleware"

    # Various LangChain versions may call different hooks; implement a superset
    # so the middleware captures the last model message consistently.
    def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        try:
            messages = state.get("messages") or []
            if not messages:
                return None
            event = {
                "content_blocks": _extract_content_blocks_from_message(messages[-1]),
                "structured_response": (
                    state.get("structured_response")
                    or state.get("structured")
                    or _normalize_to_dict(messages[-1]).get("structured_response")
                    or _normalize_to_dict(messages[-1]).get("structured")
                ),
                "text": _normalize_to_dict(messages[-1]).get("text"),
                # attach a monotonic timestamp to aid debugging/auditing
                "ts": time.time(),
            }
            state.setdefault("_multi_agent_dashboard_events", []).append(event)
        except Exception:
            # Middleware must not raise
            logger.debug("Instrumentation middleware after_model failed", exc_info=True)
        return None

    # Also support wrap-style hook names used in some releases
    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        # Call through and capture effects inside after_model when the model returns.
        return handler(request)


_INSTRUMENTATION_MIDDLEWARE = _DashboardInstrumentationMiddleware

INSTRUMENTATION_MIDDLEWARE = _INSTRUMENTATION_MIDDLEWARE


# =========================
# Public data structures
# =========================

@dataclass
class TextResponse:
    """
    Normalized text response returned by LLMClient.
    """
    text: str
    raw: Dict[str, Any]
    # Optional usage metadata
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None  # seconds

class LLMError(RuntimeError):
    """
    Typed exception raised for LLM failures.
    Keeps LLM concerns isolated from orchestration logic.
    """
    pass


# =========================
# Chat model factory (LangChain path)
# =========================

class ChatModelFactory:
    """
    Lightweight factory and cache for LangChain chat models created via init_chat_model.

    Keyed by (model, provider_id, endpoint, use_responses_api, model_class, provider_features_fingerprint, timeout).
    """

    def __init__(self, init_fn: Optional[Callable[..., Any]] = None):
        if init_fn is None and not _LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available; cannot create ChatModelFactory without init function.")
        self._init_fn = init_fn or _init_chat_model
        # include timeout as final component in key tuple (Optional[float])
        self._cache: Dict[Tuple[str, str, Optional[str], bool, Optional[str], str, Optional[float]], Any] = {}


    def _key(
        self,
        model: str,
        provider_id: Optional[str],
        endpoint: Optional[str],
        use_responses_api: bool,
        model_class: Optional[str],
        provider_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[str, str, Optional[str], bool, Optional[str], str, Optional[float]]:
        """
        Build a stable cache key for a chat model, including a fingerprint of provider_features
        and the timeout so that changes to capability hints or per-call timeout cause a fresh
        model instance to be created.
        """
        features_key = ""
        if provider_features:
            try:
                # Stable, order-independent JSON fingerprint
                features_key = json.dumps(
                    provider_features,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                # Fallback: use repr of sorted items when JSON encoding fails
                try:
                    features_key = repr(sorted(provider_features.items()))
                except Exception:
                    features_key = repr(provider_features)
        # Normalize timeout to a float (or None)
        timeout_val: Optional[float] = None
        if timeout is not None:
            try:
                timeout_val = float(timeout)
            except Exception:
                timeout_val = None

        return (
            model or "",
            provider_id or "",
            endpoint or None,
            bool(use_responses_api),
            model_class or "",
            features_key,
            timeout_val,
        )


    def get_model(
        self,
        model: str,
        *,
        provider_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        use_responses_api: bool = None,
        model_class: Optional[str] = None,
        provider_features: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        """
        Return a LangChain chat model instance for the provided metadata.
        Caches instances to avoid repeated init costs.
        """
        # Provider name normalization for provider-specific fallback logic
        provider_norm = (provider_id or "").strip().lower()

        # Normalize endpoint: if user provided a host:port without scheme, add a default scheme.
        if endpoint and "://" not in endpoint:
            endpoint = f"http://{endpoint}"

        key = self._key(
            model,
            provider_id,
            endpoint,
            bool(use_responses_api),
            model_class,
            provider_features,
            timeout=timeout,
        )
        if key in self._cache:
            return self._cache[key]

        init_kwargs: Dict[str, Any] = {}
        # map endpoint to base_url (popular param across providers)
        if endpoint:
            # Some providers expect 'base_url' or 'base_url' like param; init_chat_model passes kwargs to concrete impl
            init_kwargs["base_url"] = endpoint
            # Some integrations (openai) call this 'base_url' or 'api_base' - provider integration will accept 'base_url'
        # Propagate timeout to underlying LangChain model integrations:
        # Use 'request_timeout' (preferred alias used by ChatOpenAI) and set 'timeout' as fallback,
        # so providers reading either name will receive the configured numeric timeout value.
        if timeout is not None:
            try:
                timeout_val = float(timeout)
                init_kwargs["request_timeout"] = timeout_val
                # some integrations accept 'timeout' as the canonical kwarg; ensure it is present as well
                init_kwargs.setdefault("timeout", timeout_val)
            except Exception:
                # Fallback: pass raw value under 'timeout'
                init_kwargs["timeout"] = timeout

        # Convey Responses API preference for providers that support it (e.g., OpenAI)
        if use_responses_api:
            init_kwargs["use_responses_api"] = True
            # Recommend the responses output_version for consistent content_blocks formatting when available
            # many integrations accept 'output_version' (e.g., ChatOpenAI)
            init_kwargs["output_version"] = "responses/v1"

        # Provider profile / features may be passed through where supported
        if provider_features:
            # Some providers accept a 'profile' or 'model_profile' kwarg; pass under 'profile' and let integration ignore unknown keys.
            init_kwargs["profile"] = provider_features

        model_provider = provider_id or None

        # OpenAI API key is sourced from config (.env) and only applied when provider_id == "openai".
        if provider_norm == "openai":
            try:
                from multi_agent_dashboard import config as _cfg_key
                openai_key = getattr(_cfg_key, "OPENAI_API_KEY", None)
                if openai_key:
                    init_kwargs.setdefault("api_key", openai_key)
            except Exception:
                pass

        # Attempt to initialize via the unified helper
        try:
            chat_model = self._init_fn(
                model,
                model_provider=model_provider,
                **init_kwargs,
            )
            # Attach effective_request_timeout attribute for observability
            try:
                eff_to = init_kwargs.get("request_timeout", init_kwargs.get("timeout", None))
                if eff_to is not None:
                    try:
                        setattr(chat_model, "_effective_request_timeout", float(eff_to))
                    except Exception:
                        setattr(chat_model, "_effective_request_timeout", eff_to)
            except Exception:
                logger.debug("Failed to set _effective_request_timeout on chat_model", exc_info=True)

            # cache and return
            self._cache[key] = chat_model
            logger.debug(
                "ChatModelFactory: created model=%s provider=%s endpoint=%s request_timeout=%s",
                model, provider_id, endpoint, getattr(chat_model, "_effective_request_timeout", None)
            )
            return chat_model
        except Exception as e:
            logger.debug(
                "ChatModelFactory: init_chat_model failed for model=%s provider=%s endpoint=%s; error=%s",
                model,
                provider_id,
                endpoint,
                e,
                exc_info=True,
            )
            raise


# =========================
# LLM Client
# =========================

class LLMClient:
    """
    Thin wrapper around LangChain's BaseChatModel factory.

    Responsibilities:
    - Request execution
    - Retry / backoff
    - Response normalization
    """

    def __init__(
        self,
        *,
        timeout: float = 600.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._on_rate_limit = on_rate_limit

        # SDK / LangChain capability detection (best-effort)
        self._langchain_available = _LANGCHAIN_AVAILABLE
        self._init_chat_model = _init_chat_model
        self._SystemMessage = _SystemMessage
        self._HumanMessage = _HumanMessage
        self._create_agent = _create_agent

        # Chat model factory (LangChain path)
        self._model_factory: Optional[ChatModelFactory] = None
        if self._langchain_available and self._init_chat_model:
            try:
                self._model_factory = ChatModelFactory(init_fn=self._init_chat_model)
            except Exception:
                # Should not happen, but keep resilience
                logger.exception("Failed to initialize ChatModelFactory; LangChain path may be disabled")
                self._model_factory = None

        self._capabilities = {"langchain", "stream", "response_format", "tools", "reasoning", "instructions"}

    # -------------------------
    # LangChain agent helpers
    # -------------------------
    def create_agent_for_spec(
        self,
        spec,
        *,
        tools: Optional[List[Any]] = None,
        middleware: Optional[List[Any]] = None,
        response_format: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        """
        Create a LangChain agent bound to the provided AgentSpec-like object.
        """
        if not self._langchain_available or self._model_factory is None or self._create_agent is None:
            raise RuntimeError("LangChain agent creation is not available in this environment")

        model_instance = self._model_factory.get_model(
            spec.model,
            provider_id=getattr(spec, "provider_id", None),
            endpoint=getattr(spec, "endpoint", None),
            use_responses_api=bool(getattr(spec, "use_responses_api", False)),
            model_class=getattr(spec, "model_class", None),
            provider_features=getattr(spec, "provider_features", None),
            timeout=timeout or self._timeout,
        )

        # Normalize middleware list and instantiate classes when provided.
        middleware_list: List[Any] = []
        for mw in (middleware or []):
            try:
                # If a class was passed instead of an instance, try to instantiate.
                if isinstance(mw, type):
                    try:
                        mw_inst = mw()
                        middleware_list.append(mw_inst)
                    except Exception:
                        # Could not instantiate - append the class unchanged (some libs accept classes)
                        middleware_list.append(mw)
                else:
                    middleware_list.append(mw)
            except Exception:
                middleware_list.append(mw)

        # Robust detection helper: accept instances, classes, or subclasses
        def _middleware_includes_instrumentation(mw_list: List[Any]) -> bool:
            if INSTRUMENTATION_MIDDLEWARE is None:
                return False
            for item in mw_list:
                try:
                    # Direct instance of the instrumentation middleware
                    if isinstance(item, INSTRUMENTATION_MIDDLEWARE):
                        return True
                except Exception:
                    # isinstance might fail if types incompatible; continue defensively
                    pass
                try:
                    # If the item is the class itself
                    if item is INSTRUMENTATION_MIDDLEWARE:
                        return True
                except Exception:
                    pass
                try:
                    # If the item is a subclass (class object provided)
                    if isinstance(item, type) and issubclass(item, INSTRUMENTATION_MIDDLEWARE):
                        return True
                except Exception:
                    pass
            return False

        # Detect whether instrumentation middleware already present
        instrumentation_present = False
        try:
            instrumentation_present = _middleware_includes_instrumentation(middleware_list)
        except Exception:
            instrumentation_present = False

        instrumentation_attached = instrumentation_present
        instrumentation_attach_error: Optional[str] = None

        # Try to attach instrumentation middleware safely; if instantiation fails, log and continue without it.
        if not instrumentation_present and INSTRUMENTATION_MIDDLEWARE is not None:
            try:
                mw_instance = None
                try:
                    mw_instance = INSTRUMENTATION_MIDDLEWARE()
                    # Ensure instrumentation is the final element in the middleware list
                    middleware_list.append(mw_instance)
                    instrumentation_attached = True
                except Exception as inst_exc:
                    # As a fallback, some integrations accept middleware classes instead of instances.
                    try:
                        middleware_list.append(INSTRUMENTATION_MIDDLEWARE)
                        instrumentation_attached = True
                        instrumentation_attach_error = f"instantiation_failed:{inst_exc}"
                        logger.debug(
                            "Instrumentation middleware could not be instantiated; appended class instead for agent=%s. instantiation error=%s",
                            getattr(spec, "name", "<unnamed>"),
                            inst_exc,
                            exc_info=True,
                        )
                    except Exception as append_exc:
                        instrumentation_attach_error = f"instantiation_failed:{inst_exc}; append_failed:{append_exc}"
                        logger.warning(
                            "Instrumentation middleware exists but could not be instantiated or appended for agent=%s",
                            getattr(spec, "name", "<unnamed>"),
                        )
                        logger.debug("Instrumentation instantiation/append error: %s / %s", inst_exc, append_exc, exc_info=True)
            except Exception as e:
                instrumentation_attach_error = str(e)
                logger.exception("Failed to instantiate instrumentation middleware: %s", e)

        try:
            agent = self._create_agent(
                model=model_instance,
                tools=tools or [],
                system_prompt=getattr(spec, "system_prompt_template", None),
                middleware=middleware_list,
                response_format=response_format,
            )
            # Annotate agent with instrumentation/profile hints for downstream runtime checks
            try:
                setattr(agent, "_instrumentation_attached", bool(instrumentation_attached))
                if instrumentation_attach_error:
                    setattr(agent, "_instrumentation_attachment_error", instrumentation_attach_error)
            except Exception:
                logger.debug("Unable to set _instrumentation_attached on agent instance", exc_info=True)

            # Propagate effective_request_timeout attribute for observability
            try:
                eff_to = getattr(model_instance, "_effective_request_timeout", None)
                if eff_to is not None:
                    try:
                        setattr(agent, "_effective_request_timeout", float(eff_to))
                    except Exception:
                        setattr(agent, "_effective_request_timeout", eff_to)
                logger.debug("create_agent_for_spec: agent=%s model=%s effective_request_timeout=%s", getattr(spec, "name", "<unnamed>"), spec.model, getattr(agent, "_effective_request_timeout", None))
            except Exception:
                logger.debug("Failed to propagate effective_request_timeout to agent instance", exc_info=True)

            # If instrumentation was not attached, log an explicit warning so operators are aware
            if not instrumentation_attached:
                logger.warning(
                    "create_agent_for_spec: instrumentation middleware was not attached for agent=%s. "
                    "This may prevent collection of content_blocks/tool traces. See logs for details.",
                    getattr(spec, "name", "<unnamed>"),
                )

            return agent
        except Exception as e:
            logger.debug("create_agent_for_spec failed for spec=%s: %s", getattr(spec, "name", "<unnamed>"), e, exc_info=True)
            raise


    def invoke_agent(
        self,
        agent,
        prompt: str,
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> TextResponse:
        """
        Invoke a LangChain agent and normalize its response into TextResponse.
        """
        if not self._langchain_available:
            raise RuntimeError("LangChain invoke_agent not available")

        combined_prompt = str(prompt or "")
        if files:
            for f in files:
                filename = f.get("filename", "file")
                content = f.get("content")
                try:
                    if isinstance(content, (bytes, bytearray)):
                        text = content.decode("utf-8", errors="replace")
                        combined_prompt += f"\n\n--- FILE: {filename} ---\n{text}"
                    else:
                        combined_prompt += f"\n\n--- FILE: {filename} ---\n{str(content)}"
                except Exception:
                    combined_prompt += f"\n\n--- FILE: {filename} (binary not attached) ---\n"

        # Build the state expected by agent.invoke (messages array)
        state = {
            "messages": [
                self._SystemMessage(getattr(agent, "system_prompt", "") or "") if (getattr(agent, "system_prompt", None) and self._SystemMessage) else None,
                self._HumanMessage(combined_prompt) if self._HumanMessage else {"role": "user", "content": combined_prompt},
            ]
        }
        # Clean None if we didn't build a SystemMessage instance
        state["messages"] = [m for m in state["messages"] if m is not None]

        start_ts = time.perf_counter()
        # agent.invoke may accept context parameter in v1 Agents API
        try:
            if context is not None:
                result = agent.invoke(state, context=context)
            else:
                result = agent.invoke(state)
        except Exception as e:
            logger.debug("agent.invoke failed: %s", e, exc_info=True)
            raise

        end_ts = time.perf_counter()
        latency = end_ts - start_ts

        raw_dict = self._to_dict(result)

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

        def _extract_usage_from_candidate(candidate: Any) -> dict | None:
            if not isinstance(candidate, dict):
                return None
            usage_payload = candidate.get("usage") or candidate.get("usage_metadata")
            if isinstance(usage_payload, dict) and usage_payload:
                return usage_payload

            nested = candidate.get("agent_response")
            if isinstance(nested, dict):
                payload = _extract_usage_from_candidate(nested)
                if payload:
                    return payload

            output_entries = candidate.get("output")
            if isinstance(output_entries, list):
                for entry in output_entries:
                    if not isinstance(entry, dict):
                        continue
                    payload = _extract_usage_from_candidate(entry.get("response"))
                    if payload:
                        return payload
                    payload = _extract_usage_from_candidate(entry.get("result"))
                    if payload:
                        return payload
                    payload = _extract_usage_from_candidate(entry.get("agent_response"))
                    if payload:
                        return payload
            return None

        def _extract_usage_from_messages(messages: Any) -> dict | None:
            """
            Pull usage from the last AIMessage-like object in a messages list.
            LangChain commonly attaches usage on AIMessage.usage_metadata or
            AIMessage.response_metadata rather than promoting it to the top-level
            agent state dict.
            """
            if not isinstance(messages, list):
                return None
            for msg in reversed(messages):
                try:
                    # dict-shaped message
                    if isinstance(msg, dict):
                        usage_payload = msg.get("usage_metadata") or msg.get("usage") or msg.get("response_metadata")
                        if isinstance(usage_payload, dict) and usage_payload:
                            return usage_payload
                    # object-shaped message
                    usage_payload = getattr(msg, "usage_metadata", None) or getattr(msg, "usage", None) or getattr(msg, "response_metadata", None)
                    if isinstance(usage_payload, dict) and usage_payload:
                        return usage_payload
                except Exception:
                    continue
            return None

        def _extract_tool_info_from_messages(messages: Any) -> tuple[list[dict], list[dict]]:
            """
            Extract tool_calls and content_blocks from AIMessage-like objects in a messages list.
            Returns (tool_calls, content_blocks).
            """
            if not isinstance(messages, list):
                return [], []

            def _msg_to_dict(m: Any) -> dict | None:
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
            tool_calls, content_blocks = _extract_tool_info_from_messages(messages)
            if isinstance(raw_dict, dict):
                if tool_calls and "tool_calls" not in raw_dict:
                    raw_dict["tool_calls"] = tool_calls
                # Do NOT promote content_blocks into raw_dict here to avoid duplication.
                # _collect_content_blocks() already inspects messages and raw content_blocks.
        except Exception:
            logger.debug("Failed to attach tool info from messages into raw_dict", exc_info=True)

        if input_tokens is None or output_tokens is None:
            # Try usage on the last AIMessage from agent state (common in LangChain agents)
            msg_usage = _extract_usage_from_messages(messages)
            if isinstance(msg_usage, dict):
                if input_tokens is None:
                    input_tokens = msg_usage.get("input_tokens") or msg_usage.get("prompt_tokens") or msg_usage.get("prompt_token_count")
                if output_tokens is None:
                    output_tokens = msg_usage.get("output_tokens") or msg_usage.get("completion_tokens") or msg_usage.get("completion_token_count")
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

            nested_usage = _extract_usage_from_candidate(raw_dict.get("agent_response"))
            if not nested_usage:
                try:
                    nested_usage = _extract_usage_from_candidate(getattr(result, "agent_response", None))
                except Exception:
                    nested_usage = None
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

        # -----------------------------
        # Extract textual output by examining the returned agent state messages.
        # Walk messages from the end, prefer assistant/Ai messages, then fall back to other fields.
        # -----------------------------
        text_out = None
        try:
            messages = None
            # Prefer actual message objects/structures from the returned result
            if isinstance(result, dict):
                messages = result.get("messages") or result.get("messages", None)
                # also consider nested agent_response.messages
                if not messages and "agent_response" in result and isinstance(result["agent_response"], dict):
                    messages = result["agent_response"].get("messages") or None
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

        return TextResponse(
            text=text_out,
            raw=raw_dict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
        )

    # -------------------------
    # Public API
    # -------------------------

    # -------------------------
    # Response normalization
    # -------------------------

    def _to_dict(self, response: Any) -> Dict[str, Any]:
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
                events_attr = getattr(response, "_multi_agent_dashboard_events", None) or getattr(response, "instrumentation_events", None)
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
            normalized = _normalize_to_dict(candidate)
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

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def safe_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON safely; return None on failure.
        """
        try:
            return json.loads(text)
        except Exception:
            return None
