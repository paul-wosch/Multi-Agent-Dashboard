# multi_agent_dashboard/llm_client.py
import io
import time
import logging
import json
import inspect
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

    def _normalize_profile(self, profile: Any) -> Optional[Dict[str, Any]]:
        """
        Convert a LangChain ModelProfile (or similar) into a plain dict for downstream use.
        """
        if profile is None:
            return None
        try:
            if isinstance(profile, dict):
                return profile
            if hasattr(profile, "model_dump"):
                return profile.model_dump()
            if hasattr(profile, "to_dict"):
                return profile.to_dict()
            if hasattr(profile, "__dict__"):
                return dict(profile.__dict__)
            # Fallback: try json loads on repr
            try:
                return json.loads(repr(profile))
            except Exception:
                return {"__repr": repr(profile)}
        except Exception:
            return {"__repr": repr(profile)}

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

        Additionally, attempt to extract a provider/model 'profile' when available and
        attach it as `_detected_provider_profile` on the returned model instance.
        """
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
        if timeout is not None:
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

        # Attempt to initialize via the unified helper
        try:
            chat_model = self._init_fn(
                model,
                model_provider=model_provider,
                **init_kwargs,
            )
            # Attempt to read a model/profile attribute to surface provider capabilities
            try:
                profile = None
                # Common attribute names in LangChain/Integrations
                if hasattr(chat_model, "profile"):
                    profile = getattr(chat_model, "profile")
                elif hasattr(chat_model, "model_profile"):
                    profile = getattr(chat_model, "model_profile")
                elif hasattr(chat_model, "model_profile_"):
                    profile = getattr(chat_model, "model_profile_")

                normalized = self._normalize_profile(profile)
                if normalized is not None:
                    try:
                        setattr(chat_model, "_detected_provider_profile", normalized)
                    except Exception:
                        # If setting attribute fails, swallow but log at debug level
                        logger.debug("Failed to attach _detected_provider_profile to chat_model", exc_info=True)
            except Exception:
                logger.debug("Failed to normalize chat_model profile", exc_info=True)

            # cache and return
            self._cache[key] = chat_model
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
    Thin, reusable wrapper around either:
      - an OpenAI-style SDK client (legacy path), or
      - LangChain's BaseChatModel factory (when langchain is installed).

    Responsibilities:
    - Request execution
    - Retry / backoff
    - Capability detection
    - Response normalization

    This implementation prefers LangChain when available and falls back to the legacy OpenAI SDK behavior for environments that do not have langchain
    or the provider-specific integration packages installed.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        on_rate_limit: Optional[Callable[[int], None]] = None,
    ):
        self._client = client
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

        self._capabilities = self._detect_capabilities()

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

            # Surface detected provider profile (if model exposed one)
            detected = getattr(model_instance, "_detected_provider_profile", None)
            if detected is not None:
                try:
                    setattr(agent, "_detected_provider_profile", detected)
                except Exception:
                    logger.debug("Unable to set _detected_provider_profile on agent instance", exc_info=True)

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

        This mirrors create_text_response's normalization but calls agent.invoke(state, context=...).

        Note: create_agent handles structured output internally via its response_format
        argument and returns structured data (when configured) under the 'structured_response'
        key of the agent state. The optional response_format parameter here is accepted for
        API symmetry but is not currently threaded into create_agent; structured output should
        be configured when the agent is created.
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
        state_events = None
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

        # Convert result into dict
        raw_dict = self._to_dict(result)

        # Merge any middleware-mutated state (e.g., _multi_agent_dashboard_events) into raw_dict.
        try:
            if isinstance(state, dict):
                events = state.get("_multi_agent_dashboard_events") or state.get("_instrumentation_events") or None
                if events:
                    # expose both canonical keys just in case
                    if "instrumentation_events" not in raw_dict:
                        raw_dict["instrumentation_events"] = events
                    if "_multi_agent_dashboard_events" not in raw_dict:
                        raw_dict["_multi_agent_dashboard_events"] = events

                    # Also aggregate any content_blocks found inside events into top-level content_blocks
                    aggregated_cb = raw_dict.get("content_blocks", []) if isinstance(raw_dict.get("content_blocks", []), list) else []
                    for ev in events:
                        if isinstance(ev, dict):
                            cb = ev.get("content_blocks")
                            if isinstance(cb, list):
                                aggregated_cb.extend(cb)
                            # some events carry a structured_response directly on the event
                            if "structured_response" in ev and "structured_response" not in raw_dict:
                                raw_dict["structured_response"] = ev.get("structured_response")
                    if aggregated_cb:
                        # set or extend
                        raw_dict["content_blocks"] = aggregated_cb
        except Exception:
            logger.debug("Failed to merge mutated agent state into raw_dict", exc_info=True)

        # Extract token usage if provided by provider via usage_metadata or usage
        input_tokens = None
        output_tokens = None
        usage = (
            getattr(result, "usage_metadata", None)
            or getattr(result, "usage", None)
            or raw_dict.get("usage")
            or raw_dict.get("usage_metadata")
            or {}
        )
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_count")
            try:
                if input_tokens is None and isinstance(usage.get("token_usage"), dict):
                    input_tokens = usage["token_usage"].get("prompt_tokens") or usage["token_usage"].get("input_tokens")
                if output_tokens is None and isinstance(usage.get("token_usage"), dict):
                    output_tokens = usage["token_usage"].get("completion_tokens") or usage["token_usage"].get("output_tokens")
            except Exception:
                pass

        # Extract textual output
        text_out = None
        try:
            text_attr = getattr(result, "text", None)
            if callable(text_attr):
                text_out = text_attr()
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

        # Attempt to surface detected profile if agent has one (propagated from create_agent_for_spec)
        try:
            detected_profile = getattr(agent, "_detected_provider_profile", None)
            if detected_profile is not None and "detected_provider_profile" not in raw_dict:
                raw_dict["detected_provider_profile"] = detected_profile
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

    def create_text_response(
        self,
        model: str,
        prompt: str,
        *,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        files: Optional[List[Dict[str, Any]]] = None,
        tools_config: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        # provider metadata (optional; new in Phase 2)
        provider_id: Optional[str] = None,
        model_class: Optional[str] = None,
        endpoint: Optional[str] = None,
        use_responses_api: Optional[bool] = None,
        provider_features: Optional[Dict[str, Any]] = None,
    ) -> TextResponse:
        """
        Execute a text-generation request and return a normalized response.

        New provider metadata args allow per-agent choices (OpenAI vs Ollama).
        The function will attempt to use LangChain.init_chat_model when langchain
        and the required provider integration are installed. Otherwise, it falls
        back to the legacy OpenAI SDK call that existed previously.

        Note: The signature was kept backward compatible (all new args optional).
        """
        last_err: Optional[Exception] = None

        # Prefer LangChain path when available and factory initialized
        if self._langchain_available and self._model_factory:
            try:
                # Obtain (cached) chat model instance for this agent's metadata
                chat_model = self._model_factory.get_model(
                    model,
                    provider_id=provider_id,
                    endpoint=endpoint,
                    use_responses_api=bool(use_responses_api),
                    model_class=model_class,
                    provider_features=provider_features,
                    timeout=self._timeout,
                )

                # Build messages. Stick to a conservative string-based embedding for files.
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

                messages = []
                if system_prompt and self._SystemMessage:
                    messages.append(self._SystemMessage(system_prompt))
                messages.append(self._HumanMessage(combined_prompt))

                start_ts = time.perf_counter()

                # If a structured response format is requested, prefer the model.with_structured_output helper
                if response_format is not None:
                    try:
                        # model.with_structured_output may return a wrapped model that yields a structured object
                        structured_model = getattr(chat_model, "with_structured_output", None)
                        if callable(structured_model):
                            wrapped = structured_model(response_format)
                            # Respect binding order: if tools are present, tools should be bound first (docs). We don't bind tools here,
                            # we rely on server-side tool calls surfaced through content_blocks.
                            response_obj = wrapped.invoke(messages)
                            end_ts = time.perf_counter()
                            latency = end_ts - start_ts

                            # If the provider returned a structured python object (Pydantic model / dict), normalize it.
                            if not hasattr(response_obj, "content_blocks"):
                                # structured object returned directly
                                structured_content = response_obj
                                raw_dict = {"structured": structured_content}
                                text_out = json.dumps(structured_content, default=str)
                                input_tokens = None
                                output_tokens = None
                                # Attempt to extract usage metadata if the wrapped model returned it as a tuple-like or has attrs
                                if hasattr(response_obj, "usage_metadata"):
                                    um = getattr(response_obj, "usage_metadata", {}) or {}
                                    input_tokens = um.get("input_tokens")
                                    output_tokens = um.get("output_tokens")
                                return TextResponse(
                                    text=text_out,
                                    raw=raw_dict,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    latency=latency,
                                )
                            else:
                                # If we got an AIMessage-like object, fall through to the common path
                                response_msg = response_obj
                        else:
                            # Wrapped not available; fall back to invoke
                            response_msg = chat_model.invoke(messages)
                            end_ts = time.perf_counter()
                            latency = end_ts - start_ts
                    except Exception as e:
                        # If structured path fails, log and fall back to a normal invocation below
                        logger.debug("Structured output attempt failed for model=%s: %s", model, e, exc_info=True)
                        response_msg = chat_model.invoke(messages)
                        end_ts = time.perf_counter()
                        latency = end_ts - start_ts
                else:
                    # Simple invoke path
                    response_msg = chat_model.invoke(messages)
                    end_ts = time.perf_counter()
                    latency = end_ts - start_ts

                # Normalize raw representation (best-effort) and extract usage/content blocks
                raw_dict = self._to_dict(response_msg)

                # Extract token usage if provided by provider via usage_metadata or usage
                input_tokens = None
                output_tokens = None
                usage = getattr(response_msg, "usage_metadata", None) or getattr(response_msg, "usage", None) or raw_dict.get("usage") or {}
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count")
                    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_count")
                    try:
                        if input_tokens is None and isinstance(usage.get("token_usage"), dict):
                            input_tokens = usage["token_usage"].get("prompt_tokens") or usage["token_usage"].get("input_tokens")
                        if output_tokens is None and isinstance(usage.get("token_usage"), dict):
                            output_tokens = usage["token_usage"].get("completion_tokens") or usage["token_usage"].get("output_tokens")
                    except Exception:
                        pass

                # Extract textual output: prefer message.text property or string content
                text_out = None
                try:
                    # In langchain v1, AIMessage exposes `.text` property
                    text_attr = getattr(response_msg, "text", None)
                    if callable(text_attr):
                        text_out = text_attr()
                    elif text_attr is not None:
                        text_out = text_attr
                    else:
                        # Fallback: check content / content_blocks
                        content = getattr(response_msg, "content", None) or raw_dict.get("content")
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

                if text_out is None:
                    # Final fallback to repr
                    try:
                        text_out = str(response_msg)
                    except Exception:
                        text_out = ""

                # Ensure raw_dict contains explicit content_blocks if available on the object
                try:
                    cb = getattr(response_msg, "content_blocks", None)
                    if cb is not None and "content_blocks" not in raw_dict:
                        # Convert content_blocks to serializable list
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

            except Exception as e:
                logger.debug("LangChain path failed; will attempt legacy client (if available): %s", e, exc_info=True)
                last_err = e
                # Fallthrough to legacy path below

        # Legacy / OpenAI SDK path (unchanged behavior)
        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs = {
                    "model": model,
                }

                # Build user input item(s)
                if files:
                    user_items = self._build_input_with_files(prompt, files)
                    user_input_for_kwargs = user_items  # list
                else:
                    user_item = {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }
                    user_input_for_kwargs = user_item  # single dict (we'll adapt below)

                # Use 'instructions' when available + system_prompt provided
                if system_prompt and "instructions" in self._capabilities:
                    kwargs["instructions"] = system_prompt
                    kwargs["input"] = user_input_for_kwargs
                    logger.debug("LLMClient: using 'instructions' capability to set system prompt.")
                elif system_prompt:
                    system_item = {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                    if isinstance(user_input_for_kwargs, list):
                        kwargs["input"] = [system_item] + user_input_for_kwargs
                    else:
                        kwargs["input"] = [system_item, user_input_for_kwargs]
                    logger.debug("LLMClient: injected system-role input item (fallback to input list).")
                else:
                    kwargs["input"] = user_input_for_kwargs

                if stream and "stream" in self._capabilities:
                    kwargs["stream"] = True

                # response_format (if supported)
                if response_format is not None and "response_format" in self._capabilities:
                    kwargs["response_format"] = response_format

                # tools config (if supported)
                if tools_config:
                    tools = tools_config.get("tools")
                    if tools and "tools" in self._capabilities:
                        kwargs["tools"] = tools
                    tool_choice = tools_config.get("tool_choice")
                    if tool_choice and "tool_choice" in self._capabilities:
                        kwargs["tool_choice"] = tool_choice
                    include = tools_config.get("include")
                    if include and "include" in self._capabilities:
                        kwargs["include"] = include

                # reasoning config (if supported)
                if reasoning_config and "reasoning" in self._capabilities:
                    kwargs["reasoning"] = reasoning_config

                logger.debug(
                    "LLM request (legacy SDK): model=%s, prompt_len=%d, stream=%s, structured=%s, tools=%s, reasoning=%s, system_prompt_set=%s",
                    model,
                    len(prompt),
                    stream,
                    bool(response_format),
                    bool(tools_config),
                    bool(reasoning_config),
                    bool(system_prompt),
                )

                start_ts = time.perf_counter()
                response = self._client.responses.create(**kwargs)
                end_ts = time.perf_counter()
                latency = end_ts - start_ts

                raw_dict = self._to_dict(response)

                # Best-effort usage extraction (OpenAI Responses API)
                input_tokens = None
                output_tokens = None
                try:
                    usage = raw_dict.get("usage") or {}
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                except Exception:
                    logger.debug("No usage metadata found in LLM response", exc_info=True)

                return TextResponse(
                    text=self._extract_text(response),
                    raw=raw_dict,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency=latency,
                )

            except TypeError:
                # Programming / integration error → fail fast
                raise

            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call_failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )

                # Best-effort rate-limit signal
                if self._on_rate_limit and self._looks_like_rate_limit(e):
                    try:
                        self._on_rate_limit(attempt)
                    except Exception:
                        pass  # metrics hooks must never break execution

                if attempt < self._max_retries:
                    time.sleep(self._backoff_base ** attempt)

        raise LLMError("LLM request failed after retries") from last_err

    # -------------------------
    # Capability detection
    # -------------------------

    def _detect_capabilities(self) -> set[str]:
        """
        Introspect the SDK/obj to see which kwargs are supported.
        If langchain is available, we set a lightweight capability set
        describing the LangChain path. Otherwise, inspect the legacy client.
        """
        caps = set()
        if self._langchain_available:
            # Basic indicator that we can use LangChain invoke/path
            caps.add("langchain")
            # We still preserve the old 'response_format'/'tools'/'stream' markers
            caps.add("stream")
            caps.add("response_format")
            caps.add("tools")
            caps.add("reasoning")
            caps.add("instructions")
            return caps

        # Legacy detection for OpenAI SDK (best-effort)
        try:
            fn = getattr(self._client, "responses").create
            sig = inspect.signature(fn)
            caps.update(sig.parameters.keys())
        except Exception:
            # Fall-back: attempt to detect typical attributes on the client
            try:
                if hasattr(self._client, "files") and hasattr(self._client.files, "create"):
                    caps.add("files.create")
            except Exception:
                pass
        return caps

    # -------------------------
    # FILE HANDLING HELPERS
    # -------------------------

    def _upload_file(self, f: Dict[str, Any]) -> str:
        """
        Upload a file to OpenAI and return its file_id.
        (legacy path only)
        """
        file_obj = io.BytesIO(f["content"])
        file_obj.name = f["filename"]  # IMPORTANT: SDK reads filename from here

        uploaded = self._client.files.create(
            file=file_obj,
            purpose="assistants",
        )

        return uploaded.id

    def _build_input_with_files(self, prompt: str, files: List[Dict[str, Any]]) -> list[dict]:
        import base64, mimetypes

        def try_decode_text(b: bytes) -> Optional[str]:
            try:
                text = b.decode("utf-8")
                if "\x00" in text:
                    return None
                return text
            except UnicodeDecodeError:
                try:
                    text = b.decode("utf-8", errors="replace")
                    if "\x00" in text:
                        return None
                    return text
                except Exception:
                    return None

        def detect_mime(filename: str, b: bytes, provided: Optional[str]) -> str:
            # prefer a non-generic provided mime
            if provided and provided != "application/octet-stream":
                return provided
            guess = mimetypes.guess_type(filename)[0]
            if guess:
                return guess
            if try_decode_text(b) is not None:
                return "text/plain"
            if b.startswith(b"%PDF"):
                return "application/pdf"
            if b.startswith(b"\x89PNG\r\n\x1a\n"):
                return "image/png"
            return provided or "application/octet-stream"

        content: list[dict] = [{"type": "input_text", "text": prompt}]

        for f in files:
            file_content = f["content"]
            filename = f.get("filename", "file")
            provided_mime = f.get("mime_type")
            mime = detect_mime(filename, file_content, provided_mime)

            # Inline text
            decoded = try_decode_text(file_content)
            if decoded is not None:
                content.append({
                    "type": "input_text",
                    "text": f"\n\n--- FILE: {filename} ---\n{decoded}",
                })
                continue

            # Images -> input_image (data URI)
            if mime and (mime.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))):
                import base64 as _b64
                b64 = _b64.b64encode(file_content).decode("ascii")
                data_url = f"data:{mime or 'image/png'};base64,{b64}"
                content.append({"type": "input_image", "image_url": data_url})
                continue

            # Try upload (preferred for PDFs/docs)
            try:
                file_obj = io.BytesIO(file_content)
                file_obj.name = filename
                uploaded = self._client.files.create(file=file_obj, purpose="user_data")
                content.append({"type": "input_file", "file_id": uploaded.id})
                continue
            except Exception:
                logger.debug("files.create failed for %s; will try base64 embed", filename, exc_info=True)

            # Fallback: embed raw base64 in file_data (not a data URI)
            try:
                max_embed = 4 * 1024 * 1024
                if len(file_content) <= max_embed:
                    file_data_b64 = base64.b64encode(file_content).decode("ascii")
                    content.append({
                        "type": "input_file",
                        "filename": filename,
                        "file_data": file_data_b64,
                    })
                    continue
                else:
                    logger.warning("Skipping inlined file_data for %s (>%d bytes)", filename, max_embed)
            except Exception:
                logger.exception("Failed to attach %s as file_data", filename)

            # Last resort: mention filename only
            content.append({
                "type": "input_text",
                "text": f"\n\n--- FILE: {filename} (binary not attached) ---\n",
            })

        # Return a list of input items (wrap as a user item)
        return [{"role": "user", "content": content}]

    # -------------------------
    # Response normalization
    # -------------------------

    def _extract_text(self, response: Any) -> str:
        """
        Best-effort text extraction across SDK versions.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            chunks: list[str] = []

            if hasattr(response, "output") and response.output and isinstance(response.output, list):
                for block in response.output:
                    if not isinstance(block, dict):
                        continue
                    content = block.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if (
                                isinstance(c, dict)
                                and c.get("type") == "output_text"
                            ):
                                chunks.append(c.get("text", ""))
        except Exception:
            logger.exception("Failed to extract text from LLM response")
            raise

        if chunks:
            return "".join(chunks)

        if hasattr(response, "text"):
            try:
                # response.text may be callable or property
                t = getattr(response, "text")
                if callable(t):
                    return t()
                return t
            except Exception:
                return str(t)

        return str(response)

    def _to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert SDK/LangChain response into a serializable dict (best-effort),
        avoiding noisy Pydantic serializer warnings from the SDK's internal model graph.
        Ensure `content_blocks` and `usage_metadata` are surfaced when present.
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

        # Attempt to surface tool_calls (legacy name) for compatibility
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

    @staticmethod
    def _looks_like_rate_limit(exc: Exception) -> bool:
        """
        Heuristic check to detect rate-limit-like failures
        without importing provider-specific exception types.
        """
        msg = str(exc).lower()
        return any(
            token in msg
            for token in ("rate limit", "too many requests", "429")
        )
