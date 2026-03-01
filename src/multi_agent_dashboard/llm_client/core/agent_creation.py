"""
Agent creation facade that coordinates agent creation using extracted components
(InstrumentationManager, ToolBinder, StructuredOutputBinder, ChatModelFactory).
"""

import logging
from typing import Any, List, Optional

from multi_agent_dashboard.models import AgentSpec

logger = logging.getLogger(__name__)

# Internal imports
from ..instrumentation import InstrumentationManager
from ..tool_binder import ToolBinder
from ..structured_output import StructuredOutputBinder


class AgentCreationFacade:
    """
    Coordinates agent creation using extracted components (InstrumentationManager,
    ToolBinder, StructuredOutputBinder, ChatModelFactory).
    """

    def __init__(self, client: Any):
        """
        Initialize the agent creation facade.
        
        Args:
            client: LLMClient instance
        """
        self._client = client

    def create_agent(
        self,
        spec: AgentSpec,
        *,
        tools: Optional[List[Any]] = None,
        middleware: Optional[List[Any]] = None,
        response_format: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Create a LangChain agent bound to the provided AgentSpec.
        
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