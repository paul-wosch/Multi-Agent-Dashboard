# multi_agent_dashboard/llm_client.py
import time
import os
import logging
import json
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from multi_agent_dashboard.shared.structured_schemas import resolve_schema_json
from multi_agent_dashboard.tool_integration.provider_tool_adapter import convert_tools_for_provider
from multi_agent_dashboard.tool_integration.registry import get_registry
from multi_agent_dashboard import config
from multi_agent_dashboard.models import AgentSpec
from .provider_adapters import get_adapter
from .chat_model_factory import ChatModelFactory
from .instrumentation import INSTRUMENTATION_MIDDLEWARE, InstrumentationManager
from .tool_binder import ToolBinder
from .structured_output import StructuredOutputBinder
from .wrappers import StructuredOutputWrapper
from .response_normalizer import ResponseNormalizer
# Conditional import for DuckDuckGoSearchTool (may not be available if LangChain missing)
try:
    from multi_agent_dashboard.tool_integration.search import DuckDuckGoSearchTool
    _DUCKDUCKGO_TOOL_AVAILABLE = True
except ImportError:
    DuckDuckGoSearchTool = None
    _DUCKDUCKGO_TOOL_AVAILABLE = False

# Langfuse observability (optional)
try:
    from multi_agent_dashboard.observability import is_langfuse_enabled, get_langfuse_handler
    _LANGFUSE_AVAILABLE = True
except ImportError:
    is_langfuse_enabled = None
    get_langfuse_handler = None
    _LANGFUSE_AVAILABLE = False

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
_AIMessage = None
_create_agent = None
_AgentMiddleware = None

try:
    from langchain.chat_models import init_chat_model  # type: ignore
    from langchain.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
    from langchain.agents import create_agent  # type: ignore
    from langchain.agents.middleware import AgentMiddleware  # type: ignore

    _LANGCHAIN_AVAILABLE = True
    _init_chat_model = init_chat_model
    _SystemMessage = SystemMessage
    _HumanMessage = HumanMessage
    _AIMessage = AIMessage
    _create_agent = create_agent
    _AgentMiddleware = AgentMiddleware
except Exception:
    # Keep resilience when LangChain is not installed or partial environments.
    _LANGCHAIN_AVAILABLE = False
    _init_chat_model = None
    _SystemMessage = None
    _HumanMessage = None
    _AIMessage = None
    _create_agent = None
    _AgentMiddleware = None










class AgentCreationFacade:
    """
    Coordinates agent creation using extracted components (InstrumentationManager,
    ToolBinder, StructuredOutputBinder, ChatModelFactory).
    """

    def __init__(self, client):
        self._client = client

    def create_agent(
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
        # Check LangChain availability
        if not self._client._langchain_available or self._client._model_factory is None or self._client._create_agent is None:
            raise RuntimeError("LangChain agent creation is not available in this environment")

        # Build structured output adapter
        response_format = self._client._build_structured_output_adapter(spec, response_format)
        provider_id = (getattr(spec, "provider_id", None) or "openai").lower()
        logger.debug("Before workaround: provider_id=%s, response_format=%s", provider_id, response_format)

        # Prepare middleware
        middleware_list, instrumentation_attached, instrumentation_attach_error = InstrumentationManager.prepare(middleware, spec)

        # Determine max_tokens from precedence rules (None means no limit)
        max_tokens_val = spec.effective_max_output()
        # Get model instance
        model_instance = self._client._model_factory.get_model(
            spec.model,
            provider_id=getattr(spec, "provider_id", None),
            endpoint=getattr(spec, "endpoint", None),
            use_responses_api=getattr(spec, "use_responses_api", False),
            model_class=getattr(spec, "model_class", None),
            provider_features=getattr(spec, "provider_features", None),
            timeout=timeout or self._client._timeout,
            temperature=getattr(spec, "temperature", None),
            max_tokens=max_tokens_val,
        )

        try:
            # Determine effective response_format: pass only for OpenAI (JSON Schema), others use adapter.
            effective_response_format = response_format if provider_id == "openai" else None
            logger.debug("create_agent_for_spec: provider_id=%s, response_format id=%s value=%s", provider_id,
                         id(response_format), response_format)

            # Convert tool configs to provider-specific format and bind to model
            tool_binder = ToolBinder(self._client)
            model_instance, tools, unified_binding_applied, effective_response_format = tool_binder.process_tools(
                spec, model_instance, response_format, provider_id, tools
            )

            # Provider-specific structured output binding (if unified binding not applied)
            if not unified_binding_applied and response_format is not None:
                structured_binder = StructuredOutputBinder(self._client)
                model_instance, effective_response_format = structured_binder.bind_structured_output(
                    spec, model_instance, response_format, provider_id, spec.model,
                    tools=None, strict=True
                )

            agent = self._client._create_agent(
                model=model_instance,
                tools=tools or [],
                system_prompt=getattr(spec, "system_prompt_template", None),
                middleware=middleware_list,
                response_format=effective_response_format,
            )

            # Set agent name for observability (used in Langfuse metadata)
            try:
                setattr(agent, "_name", getattr(spec, "name", None))
            except Exception:
                logger.debug("Unable to set _name on agent instance", exc_info=True)

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
                logger.debug("create_agent_for_spec: agent=%s model=%s effective_request_timeout=%s",
                             getattr(spec, "name", "<unnamed>"), spec.model,
                             getattr(agent, "_effective_request_timeout", None))
            except Exception:
                logger.debug("Failed to propagate effective_request_timeout to agent instance", exc_info=True)

            # Propagate provider info for multimodal file handling
            try:
                setattr(agent, "_provider_id", getattr(spec, "provider_id", None))
                setattr(agent, "_model", getattr(spec, "model", None))
                setattr(agent, "_provider_features", getattr(spec, "provider_features", None))
            except Exception:
                logger.debug("Unable to set provider info on agent instance", exc_info=True)

            # If instrumentation was not attached, log an explicit warning so operators are aware
            if not instrumentation_attached:
                logger.warning(
                    "create_agent_for_spec: instrumentation middleware was not attached for agent=%s. "
                    "This may prevent collection of content_blocks/tool traces. See logs for details.",
                    getattr(spec, "name", "<unnamed>"),
                )

            return agent
        except Exception as e:
            logger.debug("create_agent_for_spec failed for spec=%s: %s", getattr(spec, "name", "<unnamed>"), e,
                         exc_info=True)
            raise


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
            max_retries: int = 0,
            backoff_base: float = 1.5,
            on_rate_limit: Optional[Callable[[int], None]] = None,
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._on_rate_limit = on_rate_limit

        # SDK / LangChain capability detection (best-effort)
        self._langchain_available = _LANGCHAIN_AVAILABLE



        # Use standard LangChain init_chat_model (may be None if LangChain not available)
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

        # Langfuse observability (optional)
        self._langfuse_enabled = False
        if _LANGFUSE_AVAILABLE and is_langfuse_enabled is not None:
            self._langfuse_enabled = is_langfuse_enabled()

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
        # Delegate to facade that coordinates extracted components
        facade = AgentCreationFacade(self)
        return facade.create_agent(
            spec,
            tools=tools,
            middleware=middleware,
            response_format=response_format,
            timeout=timeout,
        )
    def _wrap_structured_output_model(self, model_instance: Any) -> Any:
        """
        Wrap a chat model that returns structured dicts so the agent pipeline
        always receives a BaseMessage (AIMessage) or role/content dict, while
        preserving usage / token metadata (including Ollama prompt_eval_count /
        eval_count) and response_metadata.
        """
        return StructuredOutputWrapper.wrap(model_instance)

    def _get_structured_output_method(self, provider_id: str, model_name: str) -> str:
        """Return the appropriate method for with_structured_output for given provider."""
        adapter = get_adapter(provider_id)
        return adapter.get_structured_output_method(model_name)

    def _build_structured_output_adapter(self, spec, response_format: Optional[Any]) -> Optional[Any]:
        """
        Centralized provider-agnostic structured output adapter.
        Returns provider-specific response_format payload or schema for Ollama.
        """
        logger.info("_build_structured_output_adapter called with spec=%s",
                    spec.name if hasattr(spec, 'name') else spec)
        if response_format is not None:
            logger.info("Returning existing response_format")
            return response_format
        if not getattr(spec, "structured_output_enabled", False):
            logger.info("structured_output_enabled is False")
            return None

        raw_schema = getattr(spec, "schema_json", None)
        schema = None
        # Accept dicts directly; fall back to resolver for string/json or registry name.
        if isinstance(raw_schema, dict):
            schema = raw_schema
        else:
            schema = resolve_schema_json(
                raw_schema,
                getattr(spec, "schema_name", None),
            )
        if not schema:
            return None
        raw_provider = getattr(spec, "provider_id", None)
        logger.info("_build_structured_output_adapter: raw provider_id=%s", raw_provider)
        provider_id = (raw_provider or "openai").lower()
        schema_name = getattr(spec, "schema_name", None)
        
        adapter = get_adapter(provider_id)
        result = adapter.wrap_schema(schema, schema_name)
        logger.info("Returning %s response_format: %s", provider_id, result)
        return result

    def _prepare_request(self, agent, prompt: str, *, files=None, context=None):
        """
        Build input with files, apply multimodal handling.
        Returns the state dict for agent.invoke.
        """
        if not self._langchain_available:
            raise RuntimeError("LangChain invoke_agent not available")

        combined_prompt = str(prompt or "")
        multimodal_content_parts = None  # If list, use this instead of combined_prompt
        processed_files = []

        if files:
            files_processed = False
            # Try to use multimodal handler regardless of provider
            try:
                from multi_agent_dashboard.llm_client.multimodal import prepare_multimodal_content
            except ImportError:
                logger.warning("multimodal_handler not available, falling back to text concatenation")
                prepare_multimodal_content = None

            if prepare_multimodal_content:
                provider_id = getattr(agent, "_provider_id", None)
                model = getattr(agent, "_model", None)
                provider_features = getattr(agent, "_provider_features", None)
                content, processed_files = prepare_multimodal_content(
                    provider_id=provider_id,
                    model=model,
                    files=files,
                    profile=provider_features,
                    prompt=combined_prompt,
                )
                if isinstance(content, list):
                    multimodal_content_parts = content
                    combined_prompt = ""  # not used
                else:
                    combined_prompt = content  # string
                files_processed = True
            # If prepare_multimodal_content is None, fall through to legacy concatenation

            if not files_processed:
                # Legacy concatenation (original logic)
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
        # Determine user message content based on multimodal handling
        if multimodal_content_parts is not None:
            user_content = multimodal_content_parts
        else:
            user_content = combined_prompt

        state = {
            "messages": [
                self._SystemMessage(getattr(agent, "system_prompt", "") or "") if (
                            getattr(agent, "system_prompt", None) and self._SystemMessage) else None,
                self._HumanMessage(user_content) if self._HumanMessage else {"role": "user", "content": user_content},
            ]
        }
        # Clean None if we didn't build a SystemMessage instance
        state["messages"] = [m for m in state["messages"] if m is not None]
        
        return state
    
    def _execute_with_retry(self, agent, state, context=None):
        """
        Execute agent.invoke with retry/backoff logic.
        Returns (result, latency).
        """
        start_ts = time.perf_counter()

        # Build invocation config with Langfuse callback if enabled
        invoke_config = {}
        if self._langfuse_enabled:
            # Extract metadata from context and agent spec
            pipeline_name = None
            run_id = None
            tags = []
            metadata = {}

            if context is not None:
                pipeline_name = context.get("pipeline_name")
                run_id = context.get("run_id")

            # Always add pipeline tag (pipeline name or "Ad‑Hoc" for ad‑hoc runs)
            pipeline_tag_value = pipeline_name or "Ad‑Hoc"
            tags.append(f"pipeline:{pipeline_tag_value}")
            
            if run_id:
                tags.append(f"run:{run_id}")

            # Agent name (from agent._name, agent.name, or spec)
            agent_name = getattr(agent, "_name", None)
            if not agent_name:
                agent_name = getattr(agent, "name", None)
            if not agent_name:
                # Fallback: try to get from agent spec if available
                agent_spec = getattr(agent, "_agent_spec", None)
                if agent_spec and hasattr(agent_spec, "name"):
                    agent_name = agent_spec.name
            
            # Default agent name if still None
            if not agent_name:
                agent_name = "unknown"
            
            # Add agent tag
            tags.append(f"agent:{agent_name}")

            # Session ID: pipeline name or "Ad‑Hoc" for ad‑hoc runs
            session_id = pipeline_name or "Ad‑Hoc"

            # Create Langfuse handler (no constructor parameters needed)
            if get_langfuse_handler is not None:
                handler = get_langfuse_handler()
                if handler is not None:
                    invoke_config["callbacks"] = [handler]
                    # Trace name (appears as "Name" in Langfuse UI)
                    invoke_config["run_name"] = agent_name
                    # Langfuse‑specific metadata keys
                    invoke_config["metadata"] = {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": "multi_agent_dashboard",
                        "langfuse_tags": tags,
                        **metadata,
                    }

        # agent.invoke may accept context parameter in v1 Agents API
        try:
            if context is not None:
                if invoke_config:
                    result = agent.invoke(state, context=context, config=invoke_config)
                else:
                    result = agent.invoke(state, context=context)
            else:
                if invoke_config:
                    result = agent.invoke(state, config=invoke_config)
                else:
                    result = agent.invoke(state)
        except Exception as e:
            logger.debug("agent.invoke failed: %s", e, exc_info=True)
            raise

        end_ts = time.perf_counter()
        latency = end_ts - start_ts
        
        return result, latency
    
    def _process_response(self, result, latency, agent):
        """
        Process raw response: cost computation, normalization.
        Returns TextResponse.
        """
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
                    usage_payload = getattr(msg, "usage_metadata", None) or getattr(msg, "usage", None) or getattr(msg,
                                                                                                                   "response_metadata",
                                                                                                                   None)
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
        state = self._prepare_request(agent, prompt, files=files, context=context)
        result, latency = self._execute_with_retry(agent, state, context=context)
        return self._process_response(result, latency, agent)
    def _to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert SDK/LangChain response into a serializable dict (best-effort),
        avoiding noisy Pydantic serializer warnings from the SDK's internal model graph.
        Ensure `content_blocks` and `usage_metadata` are surfaced when present.
        Also flatten nested 'agent_response' shapes commonly returned by LangChain agents.
        """
        return ResponseNormalizer.to_dict(response)

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

    def flush_langfuse(self):
        """Flush any pending Langfuse traces (useful for short scripts)."""
        from multi_agent_dashboard.observability import flush_langfuse
        flush_langfuse()
