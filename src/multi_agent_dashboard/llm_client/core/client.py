# multi_agent_dashboard.llm_client.core.client
import logging
import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from multi_agent_dashboard.shared.structured_schemas import resolve_schema_json
from ..provider_adapters import get_adapter
from ..chat_model_factory import ChatModelFactory
from ..instrumentation import INSTRUMENTATION_MIDDLEWARE
from ..wrappers import StructuredOutputWrapper
from ..response_normalizer import ResponseNormalizer
from .response_processor import ResponseProcessor
from .agent_creation import AgentCreationFacade
from .request_builder import RequestBuilder
from .execution_engine import ExecutionEngine
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


@dataclass
class TextResponse:
    text: str
    raw: Dict[str, Any]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None


class LLMError(RuntimeError):
    pass


class LLMClient:
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
        self._langchain_available = LANGCHAIN_AVAILABLE
        self._init_chat_model = get_init_chat_model()
        self._SystemMessage = get_SystemMessage()
        self._HumanMessage = get_HumanMessage()
        self._create_agent = get_create_agent()
        self._model_factory: Optional[ChatModelFactory] = None
        if self._langchain_available and self._init_chat_model:
            try:
                self._model_factory = ChatModelFactory(init_fn=self._init_chat_model)
            except Exception:
                logger.exception("Failed to initialize ChatModelFactory; LangChain path may be disabled")
                self._model_factory = None
        self._capabilities = {"langchain", "stream", "response_format", "tools", "reasoning", "instructions"}
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

    def create_agent_for_spec(
            self,
            spec,
            *,
            tools: Optional[List[Any]] = None,
            middleware: Optional[List[Any]] = None,
            response_format: Optional[Any] = None,
            timeout: Optional[float] = None,
    ):
        facade = AgentCreationFacade(self)
        return facade.create_agent(
            spec,
            tools=tools,
            middleware=middleware,
            response_format=response_format,
            timeout=timeout,
        )

    def _wrap_structured_output_model(self, model_instance: Any) -> Any:
        return StructuredOutputWrapper.wrap(model_instance)

    def _get_structured_output_method(self, provider_id: str, model_name: str) -> str:
        adapter = get_adapter(provider_id)
        return adapter.get_structured_output_method(model_name)

    def _build_structured_output_adapter(self, spec, response_format: Optional[Any]) -> Optional[Any]:
        logger.info("_build_structured_output_adapter called with spec=%s", spec.name if hasattr(spec, 'name') else spec)
        if response_format is not None:
            logger.info("Returning existing response_format")
            return response_format
        if not getattr(spec, "structured_output_enabled", False):
            logger.info("structured_output_enabled is False")
            return None
        raw_schema = getattr(spec, "schema_json", None)
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
        state = self._request_builder.build(agent, prompt, files=files, context=context)
        result, latency = self._execution_engine.execute(agent, state, context=context)
        return ResponseProcessor.process(result, latency, agent)

    def _to_dict(self, response: Any) -> Dict[str, Any]:
        return ResponseNormalizer.to_dict(response)

    @staticmethod
    def safe_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            return None

    def flush_langfuse(self):
        from multi_agent_dashboard.observability import flush_langfuse
        flush_langfuse()
