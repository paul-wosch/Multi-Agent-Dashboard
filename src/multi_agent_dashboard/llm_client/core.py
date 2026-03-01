# multi_agent_dashboard/llm_client/core.py
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
from .response_processor import ResponseProcessor
from .agent_creation import AgentCreationFacade
# Conditional imports now centralized in availability module
from .availability import (
    LANGCHAIN_AVAILABLE,
    LANGFUSE_AVAILABLE,
    DUCKDUCKGO_TOOL_AVAILABLE,
    get_SystemMessage,
    get_HumanMessage,
    get_AIMessage,
    get_init_chat_model,
    get_create_agent,
    get_AgentMiddleware,
    is_langfuse_enabled,
    DuckDuckGoSearchTool,
)
from .observability.langfuse_integration import build_langfuse_config

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "TextResponse",
    "LLMError",
    "INSTRUMENTATION_MIDDLEWARE",
]










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
        self._langchain_available = LANGCHAIN_AVAILABLE



        # Use standard LangChain init_chat_model (may be None if LangChain not available)
        self._init_chat_model = get_init_chat_model()
        self._SystemMessage = get_SystemMessage()
        self._HumanMessage = get_HumanMessage()
        self._create_agent = get_create_agent()

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
        if LANGFUSE_AVAILABLE and is_langfuse_enabled is not None:
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
        invoke_config = build_langfuse_config(agent, context=context, langfuse_enabled=self._langfuse_enabled)

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
        return ResponseProcessor.process(result, latency, agent)
    
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
