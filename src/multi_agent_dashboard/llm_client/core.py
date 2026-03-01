# multi_agent_dashboard/llm_client/core.py
import logging
import json
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from multi_agent_dashboard.shared.structured_schemas import resolve_schema_json
from .provider_adapters import get_adapter
from .chat_model_factory import ChatModelFactory
from .instrumentation import INSTRUMENTATION_MIDDLEWARE
from .wrappers import StructuredOutputWrapper
from .response_normalizer import ResponseNormalizer
from .response_processor import ResponseProcessor
from .agent_creation import AgentCreationFacade
from .request_builder import RequestBuilder
from .execution_engine import ExecutionEngine
# Conditional imports now centralized in availability module
from .availability import (
    LANGCHAIN_AVAILABLE,
    LANGFUSE_AVAILABLE,
    get_SystemMessage,
    get_HumanMessage,
    get_init_chat_model,
    get_create_agent,
    is_langfuse_enabled,
)

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
        
        self._request_builder = RequestBuilder(self._langchain_available, self._SystemMessage, self._HumanMessage)
        self._execution_engine = ExecutionEngine(
            langfuse_enabled=self._langfuse_enabled,
            max_retries=self._max_retries,
            backoff_base=self._backoff_base,
            on_rate_limit=self._on_rate_limit,
        )

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
        return self._request_builder.build(agent, prompt, files=files, context=context)
    
    def _execute_with_retry(self, agent, state, context=None):
        """
        Execute agent.invoke with retry/backoff logic.
        Returns (result, latency).
        """
        return self._execution_engine.execute(agent, state, context=context)
    
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
