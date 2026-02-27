# Agent runtime execution wrapper
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional

from .utils import safe_format
from multi_agent_dashboard.llm_client import LLMError
from ..models import AgentSpec

from ..shared.instrumentation import (
    _extract_instrumentation_events,
    _collect_content_blocks,
    _structured_from_instrumentation,
    _collect_tool_calls,
    _tool_usage_entry_from_payload,
)

from .file_processor import process_files
from .tool_converter import (
    get_allowed_domains,
    build_tools_config,
    build_reasoning_config,
    prepare_tools_for_agent,
)
from .metrics_extractor import extract_tokens_from_raw, collect_tool_usage, extract_detected_provider_profile
from .structured_output_detector import detect_structured_output, writeback_to_state


logger = logging.getLogger(__name__)


@dataclass
class AgentRuntime:
    """
    Execution wrapper.
    Holds runtime-only dependencies (LLM client, memory).
    """
    spec: AgentSpec
    llm_client: Any  # injected, intentionally untyped
    # last LLM call metrics
    last_metrics: Dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        state: Dict[str, Any],
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        # Expose the provided state on the runtime for callers/tests that expect it.
        # This mirrors engine.run_seq's state handling for standalone usage.
        self.state = state

        # -------------------------
        # Determine files to use
        # -------------------------
        all_files = files or state.get("files", [])
        text_files, binary_files = process_files(all_files)

        # -------------------------
        # Build prompt variables
        # -------------------------
        if self.spec.input_vars:
            variables = {k: state.get(k, "") for k in self.spec.input_vars}

            # Inject file info into variables if requested
            if "files" in self.spec.input_vars and all_files:
                variables["files"] = "\n".join(
                    f"- {f['filename']} ({f['mime_type']})"
                    for f in all_files
                )
        else:
            variables = dict(state)

        prompt = safe_format(self.spec.prompt_template, variables)

        # -------------------------
        # Combine inline text files into LLM input
        # -------------------------
        llm_files_payload = []

        for f in text_files:
            llm_files_payload.append({
                "filename": f["filename"],
                "content": f["content"],
                "mime_type": f["mime_type"],
            })

        for f in binary_files:
            llm_files_payload.append(f)  # will be uploaded by LLM client

        # -------------------------
        # Call LLM client (LangChain-only path)
        # -------------------------
        tc = build_tools_config(self.spec, state)
        rc = build_reasoning_config(self.spec)
        # Extract allowed domains for tool instance configuration (runtime-only)
        allowed_domains = get_allowed_domains(self.spec, state)
        logger.debug("Agent %s tools_config=%r reasoning_config=%r", self.spec.name, tc, rc)

        response = None
        used_langchain_agent = False
        agent_obj_for_invoke = None
        langchain_tools = None

        # Extract provider-specific parameters for tool conversion
        provider_id = (getattr(self.spec, "provider_id", None) or "").lower()
        model = getattr(self.spec, "model", "")
        use_responses_api = getattr(self.spec, "use_responses_api", False)
        provider_features = getattr(self.spec, "provider_features", None)
        
        langchain_tools = prepare_tools_for_agent(
            self.spec,
            state,
            provider_id,
            model,
            use_responses_api,
            provider_features,
        )

        # Enforce LangChain-only runtime: create_agent_for_spec + invoke_agent must be available.
        if not getattr(self.llm_client, "_langchain_available", False):
            raise LLMError(
                f"AgentRuntime.run requires LangChain-enabled LLM client for agent '{self.spec.name}'. "
                "Install LangChain and the provider integration (e.g., langchain-openai, langchain-ollama) and retry."
            )

        try:
            # Create a new agent instance per run (simpler and avoids cache invalidation complexity).
            try:
                agent = self.llm_client.create_agent_for_spec(
                    self.spec,
                    tools=langchain_tools,
                    middleware=None,
                    response_format=structured_schema,
                )
            except Exception as e:
                logger.debug("create_agent_for_spec raised while creating agent for %s: %s", self.spec.name, e, exc_info=True)
                raise LLMError(f"LangChain agent creation failed for '{self.spec.name}': {e}") from e

            if agent is None:
                # create_agent_for_spec unexpectedly returned None (should return agent or raise)
                raise LLMError(f"Failed to create LangChain agent for '{self.spec.name}': create_agent_for_spec returned no agent instance.")

            # Context: pass allowed_domains/state if present so middleware / agent may use it.
            context: Dict[str, Any] = {}
            if "allowed_domains_by_agent" in state:
                context["allowed_domains_by_agent"] = state.get("allowed_domains_by_agent")
            if "allowed_domains" in state:
                context["allowed_domains"] = state.get("allowed_domains")

            if tc is not None:
                context["tools_config"] = tc
            if rc is not None:
                context["reasoning_config"] = rc

            # Invoke the agent via the LLMClient helper for consistent normalization
            response = self.llm_client.invoke_agent(
                agent,
                prompt,
                files=llm_files_payload if llm_files_payload else None,
                response_format=structured_schema,
                stream=stream,
                context=context if context else None,
            )
            used_langchain_agent = True
            agent_obj_for_invoke = agent
        except Exception as e:
            # Do not fall back to legacy path; surface a typed error with context.
            logger.debug("LangChain agent path failed for agent '%s'; throwing LLMError", self.spec.name, exc_info=True)
            raise LLMError(
                f"LangChain agent creation/invoke failed for agent '{self.spec.name}': {e}"
            ) from e

        # -------------------------
        # Normalize and store metrics / instrumentation for engine/DB
        # -------------------------
        raw = response.raw or {}

        # Fallback token extraction from raw usage metadata if TextResponse left them None
        input_tokens, output_tokens = extract_tokens_from_raw(raw, response)

        instrumentation_events = _extract_instrumentation_events(raw)
        content_blocks = _collect_content_blocks(raw)

        # Base last_metrics stored for engine consumption
        self.last_metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": response.latency,
            "raw": raw,
            "content_blocks": content_blocks,
            "instrumentation_events": instrumentation_events,
            "tools_config": tc,
            "reasoning_config": rc,
            # indicate whether LangChain agent path was used
            "used_langchain_agent": bool(used_langchain_agent),
            # whether instrumentation middleware was attached (agent instance may carry this flag)
            "instrumentation_attached": bool(getattr(agent_obj_for_invoke, "_instrumentation_attached", False)) if agent_obj_for_invoke is not None else False,
        }

        # surface detected provider profile if available on the agent (propagated from LLMClient.create_agent_for_spec)
        detected = extract_detected_provider_profile(agent_obj_for_invoke, raw)
        if detected is not None:
            self.last_metrics["detected_provider_profile"] = detected

        # -------------------------
        # Tool usage extraction from content_blocks and legacy tool_calls
        # -------------------------
        used_tools = collect_tool_usage(raw, content_blocks)
        if used_tools:
            self.last_metrics["tools"] = used_tools

        # -------------------------
        # Structured output detection & local writeback (standalone runtime behavior)
        # -------------------------
        raw_output = response.text
        parsed = detect_structured_output(raw, content_blocks, raw_output)
        writeback_to_state(self.spec, state, parsed, raw_output)

        return response.text



