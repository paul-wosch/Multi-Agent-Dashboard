"""
Main LLM client class for the Multi-Agent Dashboard.

This module provides the LLMClient class which serves as the main entry point
for LLM interactions in the dashboard. It coordinates agent creation, request
building, execution, and response processing through modular components.

Key features:
- Provider-agnostic LLM integration (OpenAI, DeepSeek, Ollama)
- Structured output with JSON schema validation
- Tool calling with per-agent controls
- Multimodal file handling (images, PDFs, text files)
- Observability integration (Langfuse)
- Token accounting and cost tracking
"""

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
    """
    Container for normalized LLM response data.
    
    This dataclass holds the processed output from an LLM agent invocation,
    including the extracted text, raw response data, token counts, and latency.
    
    Attributes:
        text: Extracted textual content from the response
        raw: Raw response dictionary with all metadata
        input_tokens: Number of input/prompt tokens used (if available)
        output_tokens: Number of output/completion tokens used (if available)
        latency: Execution latency in seconds (if available)
    """
    text: str
    raw: Dict[str, Any]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None


class LLMError(RuntimeError):
    """
    Custom error class for LLM-related failures.
    
    This exception is raised when LLM operations fail due to provider errors,
    configuration issues, or other LLM-specific problems.
    """
    pass


class LLMClient:
    """
    Main client for LLM interactions in the Multi-Agent Dashboard.
    
    This class coordinates the creation and invocation of LLM agents through
    modular components for request building, execution, and response processing.
    It supports multiple providers, structured output, tool calling, and
    observability integration.
    
    The client is designed to be provider-agnostic, with provider-specific
    logic encapsulated in adapter classes.
    """
    
    def __init__(
            self,
            *,
            timeout: float = 600.0,
            max_retries: int = 0,
            backoff_base: float = 1.5,
            on_rate_limit: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize the LLM client with configuration options.
        
        Args:
            timeout: Maximum execution time in seconds for agent invocations
            max_retries: Maximum number of retry attempts (currently unused)
            backoff_base: Exponential backoff base factor (currently unused)
            on_rate_limit: Optional callback for rate-limit events (currently unused)
        """
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
            spec: Any,
            *,
            tools: Optional[List[Any]] = None,
            middleware: Optional[List[Any]] = None,
            response_format: Optional[Any] = None,
            timeout: Optional[float] = None,
    ) -> Any:
        """
        Create a LangChain agent from an AgentSpec configuration.
        
        This method delegates to the AgentCreationFacade which coordinates
        the creation of agents with proper instrumentation, tool binding,
        and structured output configuration.
        
        Args:
            spec: AgentSpec instance containing agent configuration
            tools: Optional list of tools to make available to the agent
            middleware: Optional list of middleware functions to apply
            response_format: Optional response format configuration
            timeout: Optional timeout override for this agent
            
        Returns:
            LangChain agent instance configured according to the spec
            
        Raises:
            RuntimeError: If LangChain is not available in the environment
        """
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
        Wrap a model instance with structured output support.
        
        Args:
            model_instance: Raw model instance to wrap
            
        Returns:
            Wrapped model instance with structured output capabilities
        """
        return StructuredOutputWrapper.wrap(model_instance)

    def _get_structured_output_method(self, provider_id: str, model_name: str) -> str:
        """
        Get the structured output method for a specific provider and model.
        
        Args:
            provider_id: Provider identifier (e.g., 'openai', 'deepseek', 'ollama')
            model_name: Model name to check
            
        Returns:
            String identifier of the structured output method
        """
        adapter = get_adapter(provider_id)
        return adapter.get_structured_output_method(model_name)

    def _build_structured_output_adapter(self, spec: Any, response_format: Optional[Any]) -> Optional[Any]:
        """
        Build a provider-specific structured output adapter from an AgentSpec.
        
        This method creates a response format adapter based on the agent's
        structured output configuration and provider capabilities.
        
        Args:
            spec: AgentSpec instance containing structured output configuration
            response_format: Optional pre-existing response format to use
            
        Returns:
            Provider-specific response format adapter, or None if not needed
        """
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
            agent: Any,
            prompt: str,
            *,
            files: Optional[List[Dict[str, Any]]] = None,
            response_format: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context: Optional[Dict[str, Any]] = None,
    ) -> TextResponse:
        """
        Invoke a LangChain agent with the given prompt and optional files.
        
        This is the main entry point for agent execution. It coordinates
        request building, execution with observability, and response processing.
        
        Args:
            agent: LangChain agent instance to invoke
            prompt: Text prompt to send to the agent
            files: Optional list of file attachments (dicts with filename, content, mime_type)
            response_format: Optional response format configuration (currently unused)
            stream: Whether to stream the response (currently unsupported)
            context: Optional context dictionary with pipeline_name, run_id, etc.
            
        Returns:
            TextResponse containing normalized response data
            
        Raises:
            RuntimeError: If LangChain is not available
            LLMError: If agent invocation fails
        """
        state = self._request_builder.build(agent, prompt, files=files, context=context)
        result, latency = self._execution_engine.execute(agent, state, context=context)
        return ResponseProcessor.process(result, latency, agent)

    def _to_dict(self, response: Any) -> Dict[str, Any]:
        """
        Convert a response object to a normalized dictionary.
        
        This method delegates to the ResponseNormalizer to convert various
        response formats (LangChain messages, raw strings, etc.) into a
        consistent dictionary format.
        
        Args:
            response: Raw response object to normalize
            
        Returns:
            Normalized dictionary with standardized keys
        """
        return ResponseNormalizer.to_dict(response)

    @staticmethod
    def safe_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON text, returning None on failure.
        
        This is a convenience method for parsing JSON responses that
        may be malformed or contain unexpected content.
        
        Args:
            text: JSON string to parse
            
        Returns:
            Parsed dictionary if successful, None otherwise
        """
        try:
            return json.loads(text)
        except Exception:
            return None

    def flush_langfuse(self):
        """
        Flush any pending Langfuse traces to the server.
        
        This ensures that all observability data is sent to Langfuse
        before the application exits or when explicit flushing is needed.
        """
        from multi_agent_dashboard.observability import flush_langfuse
        flush_langfuse()
