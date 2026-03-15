"""
Agent execution logic for the multi-agent engine.

This module contains the AgentExecutor class which encapsulates per‑agent execution
logic extracted from MultiAgentEngine.run_seq(). It handles input validation,
agent invocation, metric extraction, cost computation, configuration snapshotting,
tool‑usage parsing, structured‑output validation, and writeback to pipeline state.

The executor is designed to be stateless with respect to the pipeline; all mutable
state is passed via a PipelineState instance.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Callable

from jsonschema import validate as jsonschema_validate, ValidationError  # type: ignore

from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.runtime import AgentRuntime
from multi_agent_dashboard.shared.structured_schemas import resolve_schema_json
from multi_agent_dashboard.config import RAISE_ON_AGENT_FAIL

from ..shared.instrumentation import (
    _extract_instrumentation_events,

    _structured_from_instrumentation,
)
from .utils import (
    _extract_provider_features_from_profile,
)
from .snapshot_builder import RunSnapshotBuilder
from .schema_validator import SchemaValidator
from .metrics_aggregator import MetricsAggregator
from .types import AgentRunResult, PipelineState, RunMetrics

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes a single agent within a pipeline.

    Encapsulates the per‑agent logic extracted from MultiAgentEngine.run_seq():
    input validation, agent invocation, metric extraction, cost computation,
    configuration snapshotting, tool‑usage parsing, structured‑output validation,
    and writeback to pipeline state.

    The executor is stateless with respect to the pipeline; all mutable state
    is passed via a PipelineState instance.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        strict: bool = False,
        strict_schema_validation: bool = False,
        warning_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            llm_client: The LLM client used for cost computation (provider detection).
            strict: If True, missing input variables raise ValueError.
            strict_schema_validation: If True, schema validation failure triggers early exit.
            warning_callback: Optional function to call when a warning is generated.
        """
        self.llm_client = llm_client
        self.strict = strict
        self.strict_schema_validation = strict_schema_validation
        self.warning_callback = warning_callback

    def _warn(self, message: str, pipeline_state: PipelineState) -> None:
        """Add a warning to pipeline state and optionally call the callback."""
        pipeline_state.warnings.append(message)
        if self.warning_callback:
            self.warning_callback(message)
        else:
            logger.warning(message)

    def _error(self, message: str, pipeline_state: PipelineState) -> None:
        """Add an error to pipeline state."""
        pipeline_state.errors.append(message)
        logger.error(message)

    def execute_agent(
        self,
        agent_name: str,
        agent: AgentRuntime,
        pipeline_state: PipelineState,
    ) -> AgentRunResult:
        """
        Execute a single agent and update the pipeline state.

        Returns an AgentRunResult containing raw output, metrics, parsed output,
        tool usages, configuration snapshot, and cost breakdown.

        Side effects:
          - Updates pipeline_state.state (writeback of output variables).
          - Updates pipeline_state.memory (stores raw output per agent).
          - Appends to pipeline_state.warnings.
          - Adds to pipeline_state.tool_usages[agent_name] if tools were used.
          - Stores agent configuration snapshot in pipeline_state.agent_configs[agent_name].
          - Updates pipeline_state.agent_metrics (optional).
          - May set pipeline_state.strict_schema_exit and
            pipeline_state.agent_schema_validation_failed[agent_name] if validation fails.
        """
        # ---- Input validation ----
        for var in agent.spec.input_vars:
            if var == "files":
                # Special-case: files in input validation
                # files are injected once and may be an empty list; presence is enough
                if "files" not in pipeline_state.state:
                    msg = f"[{agent_name}] Missing input var 'files'"
                    if self.strict:
                        raise ValueError(msg)
                    self._warn(msg, pipeline_state)
                continue

            if var not in pipeline_state.state or pipeline_state.state[var] in ("", None):
                msg = f"[{agent_name}] Missing input var '{var}'"
                if self.strict:
                    raise ValueError(msg)
                self._warn(msg, pipeline_state)

        # ---- Execute agent ----
        try:
            run_kwargs = {}
            if "files" in inspect.signature(agent.run).parameters:
                run_kwargs["files"] = pipeline_state.state.get("files")
            raw_output = agent.run(pipeline_state.state, **run_kwargs)
        except Exception as e:
            if RAISE_ON_AGENT_FAIL:
                logger.exception("Agent '%s' failed: %s", agent_name, e)
                raise
            else:
                logger.error("Agent '%s' failed: %s", agent_name, e)
                self._error(f"Agent '{agent_name}' failed: {e}", pipeline_state)
                raw_output = f"Agent failed: {e}"

        # Retrieve metrics from AgentRuntime.last_metrics
        metrics = getattr(agent, "last_metrics", {}) or {}
        raw_metrics = metrics.get("raw") or {}



        # Extract config used for this agent call
        tc = metrics.get("tools_config")
        rc = metrics.get("reasoning_config")

        # Extract tool usage for this agent (per-call details)
        tools = (metrics.get("tools") or [])
        if tools:
            for t in tools:
                # Keep config on the in-memory tool events for convenience;
                # DB writes now use agent_run_configs instead.
                if tc:
                    t["tools_config"] = tc
                if rc:
                    t["reasoning_config"] = rc
            pipeline_state.tool_usages[agent_name] = tools

        # Build agent configuration snapshot
        snapshot_builder = RunSnapshotBuilder()
        agent_config = snapshot_builder.build(
            agent_name=agent_name,
            agent=agent,
            metrics=metrics,
            raw_metrics=raw_metrics,
            strict_schema_validation=self.strict_schema_validation,
        )
        pipeline_state.agent_configs[agent_name] = agent_config

        # -------------------------
        # Metric extraction and token/cost computation
        # -------------------------
        input_tokens = metrics.get("input_tokens")
        output_tokens = metrics.get("output_tokens")

        # Fallback to raw usage metadata if necessary
        if (input_tokens is None or output_tokens is None) and isinstance(raw_metrics, dict):
            try:
                usage = raw_metrics.get("usage") or raw_metrics.get("usage_metadata") or {}
                if isinstance(usage, dict):
                    if input_tokens is None:
                        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count")
                        if input_tokens is None and isinstance(usage.get("token_usage"), dict):
                            input_tokens = usage["token_usage"].get("prompt_tokens") or usage["token_usage"].get("input_tokens")
                    if output_tokens is None:
                        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_count")
                        if output_tokens is None and isinstance(usage.get("token_usage"), dict):
                            output_tokens = usage["token_usage"].get("completion_tokens") or usage["token_usage"].get("output_tokens")
            except Exception:
                logger.debug("Engine-level token fallback extraction failed for agent=%s", agent_name, exc_info=True)

        latency = metrics.get("latency")

        total_cost, input_cost, output_cost = MetricsAggregator.compute_cost(
            model=agent.spec.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_id=getattr(agent.spec, "provider_id", None),
        )

        run_metrics = RunMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=agent.spec.model,
        )
        pipeline_state.agent_metrics[agent_name] = run_metrics

        pipeline_state.memory[agent_name] = raw_output
        # last_output = raw_output  # Not needed here; engine loop tracks last_output

        # ---- Prefer structured outputs surfaced via LLMClient or LangChain structured response ----
        parsed = None
        # If the LLM client included a canonical 'structured' key
        # (created by LLMClient when with_structured_output returned a parsed object),
        # or when using LangChain agents, a top-level 'structured_response' key from the agent state.
        if isinstance(raw_metrics, dict):
            if "structured" in raw_metrics:
                parsed = raw_metrics.get("structured")
            elif "structured_response" in raw_metrics:
                parsed = raw_metrics.get("structured_response")

        # Fallback: try best-effort JSON parsing of the textual output
        if parsed is None:
            parsed = LLMClient.safe_json(raw_output) if isinstance(raw_output, str) else None

        # ---- Writeback rules ----
        if agent.spec.output_vars:
            if isinstance(parsed, dict):
                for key in parsed:
                    if key not in agent.spec.output_vars:
                        msg = (
                            f"[{agent_name}] Unexpected output key '{key}'"
                        )
                        if self.strict:
                            raise ValueError(msg)
                        self._warn(msg, pipeline_state)

                for var in agent.spec.output_vars:
                    if var in parsed:
                        pipeline_state.state[var] = parsed[var]
                    else:
                        msg = (
                            f"[{agent_name}] Declared output '{var}' missing"
                        )
                        if self.strict:
                            raise ValueError(msg)
                        self._warn(msg, pipeline_state)
            else:
                if len(agent.spec.output_vars) == 1:
                    pipeline_state.state[agent.spec.output_vars[0]] = raw_output
                else:
                    key = f"{agent_name}__raw"
                    pipeline_state.state[key] = raw_output
                    self._warn(
                        f"[{agent_name}] Non-JSON output stored as '{key}'",
                        pipeline_state,
                    )
        else:
            pipeline_state.state[agent_name] = raw_output

        # ---- Structured output validation (per-agent; global strict optionally exits) ----
        validator = SchemaValidator()
        status, reason = validator.validate(
            agent_spec=agent.spec,
            parsed=parsed,
            raw_output=raw_output,
            strict_schema_validation=self.strict_schema_validation,
        )
        validation_failed = False
        if status == "ok":
            pass  # validation succeeded
        elif status in ("missing", "invalid_json", "empty"):
            validation_failed = True
            self._warn(f"[{agent_name}] schema validation skipped: {reason}", pipeline_state)
        elif status == "validation_error":
            validation_failed = True
            self._warn(f"[{agent_name}] schema validation failed: {reason}", pipeline_state)
        else:
            # unexpected status; treat as failure
            validation_failed = True
            self._warn(f"[{agent_name}] schema validation unexpected status: {status}", pipeline_state)

        if validation_failed:
            pipeline_state.agent_schema_validation_failed[agent_name] = True
            if self.strict_schema_validation:
                pipeline_state.strict_schema_exit = True
                logger.error(
                    "[%s] Strict schema validation triggered early exit; output preserved. Reason: %s",
                    agent_name,
                    reason or "schema validation failed",
                )
                # Exit early: skip remaining agents, keep observability/costs
                # The caller (run_seq) will break the loop when pipeline_state.strict_schema_exit is True

        # Build cost breakdown dict for AgentRunResult
        cost_breakdown = {
            "input": input_cost,
            "output": output_cost,
            "total": total_cost,
        }

        return AgentRunResult(
            raw_output=raw_output,
            metrics=run_metrics,
            parsed=parsed,
            tool_usages=tools,  # already collected above
            config_snapshot=agent_config,
            cost_breakdown=cost_breakdown,
        )