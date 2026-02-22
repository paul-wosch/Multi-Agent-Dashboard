"""
AgentExecutor – encapsulates per‑agent execution logic extracted from run_seq().
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Callable

from jsonschema import validate as jsonschema_validate, ValidationError  # type: ignore

from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentRuntime
from multi_agent_dashboard.structured_schemas import resolve_schema_json

from .utils import (
    _extract_instrumentation_events,
    _collect_content_blocks,
    _structured_from_instrumentation,
    _normalize_content_blocks,
    _compute_cost,
    _extract_provider_features_from_profile,
)
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
            logger.exception("Agent '%s' failed with real error:", agent_name)
            raise

        # Retrieve metrics from AgentRuntime.last_metrics
        metrics = getattr(agent, "last_metrics", {}) or {}
        raw_metrics = metrics.get("raw") or {}

        # If LangChain agent path was used but instrumentation appears missing, warn
        try:
            used_langchain = bool(metrics.get("used_langchain_agent"))
            instrumentation_attached = bool(metrics.get("instrumentation_attached"))
            has_content_blocks = bool(metrics.get("content_blocks"))
            has_instrumentation_events = bool(metrics.get("instrumentation_events"))
            # If instrumentation was attached at agent-create time but we still see no content blocks
            if used_langchain and instrumentation_attached and not (has_content_blocks or has_instrumentation_events or metrics.get("detected_provider_profile")):
                # Instrumentation expected for LangChain agents to capture content_blocks/tool traces
                self._warn(
                    f"[{agent_name}] Ran via LangChain with instrumentation attached but produced no content_blocks or instrumentation events. "
                    "Confirm provider supports content_blocks or middleware hooks executed.",
                    pipeline_state,
                )
            # If instrumentation was not attached and LangChain used, warn once
            if used_langchain and not instrumentation_attached:
                self._warn(
                    f"[{agent_name}] Ran via LangChain but instrumentation middleware was not attached. "
                    "Enable instrumentation to capture content_blocks/tool traces.",
                    pipeline_state,
                )
        except Exception:
            logger.debug("Failed to validate instrumentation presence for agent=%s", agent_name, exc_info=True)

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

        # Snapshot this agent's configuration for this particular run.
        # This keeps agent config concerns out of tool_usages rows.
        # Include both user-facing prompt_template and system_prompt_template so
        # stored runs capture both templates used during execution.
        # Also include a compact summary of content_blocks for auditing
        content_blocks = metrics.get("content_blocks")
        if not isinstance(content_blocks, list):
            content_blocks = _collect_content_blocks(raw_metrics)

        def _filter_extra_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # Exclude plain text blocks from extra_config_json to avoid duplicating agent_outputs.output
            out: List[Dict[str, Any]] = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                btype = (b.get("type") or "").lower()
                if btype == "text":
                    continue
                out.append(b)
            return out

        content_blocks_summary = None
        try:
            if isinstance(content_blocks, list) and content_blocks:
                filtered_blocks = _filter_extra_blocks(content_blocks)
                content_blocks_summary = [
                    {
                        "type": (cb.get("type") if isinstance(cb, dict) else None),
                        "name": (cb.get("name") if isinstance(cb, dict) else None),
                        "id": (cb.get("id") if isinstance(cb, dict) else None),
                    }
                    for cb in filtered_blocks
                ]
        except Exception:
            content_blocks_summary = None

        # Normalize full content blocks for DB storage (best-effort)
        content_blocks_full = _normalize_content_blocks(_filter_extra_blocks(content_blocks or []))

        # Provider profile hints detected at runtime (from model or response)
        detected_profile = metrics.get("detected_provider_profile") or raw_metrics.get("detected_provider_profile")
        # Derive a compact provider_features mapping when the AgentSpec didn't provide any
        spec_provider_features = getattr(agent.spec, "provider_features", None) or {}
        provider_features_to_store = dict(spec_provider_features) if isinstance(spec_provider_features, dict) else (spec_provider_features or {})
        if not provider_features_to_store and detected_profile:
            derived = _extract_provider_features_from_profile(detected_profile)
            if derived:
                provider_features_to_store = derived
            else:
                # Keep a trace of the raw detected profile when we cannot derive concise features
                provider_features_to_store = {"detected_profile_present": True}

        # Capture instrumentation events & structured_response for auditing
        instrumentation_events = _extract_instrumentation_events(raw_metrics)
        structured_response = None
        try:
            if isinstance(raw_metrics, dict):
                structured_response = raw_metrics.get("structured_response") or raw_metrics.get("structured")
            if structured_response is None:
                # check instrumentation events for structured payload
                structured_response = _structured_from_instrumentation(raw_metrics)
        except Exception:
            structured_response = None

        # Record whether instrumentation middleware was attached to the agent (if agent runtime set it)
        instrumentation_attached_flag = bool(metrics.get("instrumentation_attached"))

        extra_dict: Dict[str, Any] = {}
        if content_blocks_summary is not None:
            extra_dict["content_blocks_summary"] = content_blocks_summary
        if content_blocks_full:
            extra_dict["content_blocks"] = content_blocks_full
        if detected_profile is not None:
            extra_dict["detected_provider_profile"] = detected_profile
        if instrumentation_events:
            extra_dict["instrumentation_events"] = instrumentation_events
        if structured_response is not None:
            extra_dict["structured_response"] = structured_response
        if instrumentation_attached_flag:
            extra_dict["instrumentation_attached"] = True

        agent_config = {
            "model": agent.spec.model,
            "prompt_template": agent.spec.prompt_template,
            "system_prompt_template": agent.spec.system_prompt_template,
            "role": agent.spec.role,
            "input_vars": list(agent.spec.input_vars),
            "output_vars": list(agent.spec.output_vars),
            # High-level tools overview from AgentSpec.tools
            "tools": agent.spec.tools or {},
            # Low-level tools/reasoning config used in this run
            "tools_config": tc,
            "reasoning_effort": agent.spec.reasoning_effort,
            "reasoning_summary": agent.spec.reasoning_summary,
            "reasoning_config": rc,
            # Reserved for future options such as temperature
            "extra": extra_dict,
            # Provider metadata (ensure runs capture which provider/model was used)
            "provider_id": getattr(agent.spec, "provider_id", None),
            "model_class": getattr(agent.spec, "model_class", None),
            "endpoint": getattr(agent.spec, "endpoint", None),
            "use_responses_api": bool(getattr(agent.spec, "use_responses_api", False)),
            # Persist provider feature hints (explicit OR derived)
            "provider_features": provider_features_to_store,
            # Structured output configuration (provider-agnostic)
            "structured_output_enabled": bool(getattr(agent.spec, "structured_output_enabled", False)),
            "schema_json": getattr(agent.spec, "schema_json", None),
            "schema_name": getattr(agent.spec, "schema_name", None),
            "temperature": getattr(agent.spec, "temperature", None),
            "strict_schema_validation": bool(self.strict_schema_validation),
        }
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

        total_cost, input_cost, output_cost = _compute_cost(
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

        # ---- Prefer structured outputs surfaced via LangChain content_blocks or structured response ----
        parsed = None
        # 1) If the LLM client included a canonical 'structured' key (created by LLMClient when with_structured_output returned a parsed object),
        #    or when using LangChain agents, a top-level 'structured_response' key from the agent state.
        if isinstance(raw_metrics, dict):
            if "structured" in raw_metrics:
                parsed = raw_metrics.get("structured")
            elif "structured_response" in raw_metrics:
                parsed = raw_metrics.get("structured_response")
            else:
                # 2) Look through content_blocks for structured / structured_response / server_tool_result
                cbs = raw_metrics.get("content_blocks") or raw_metrics.get("output") or []
                if isinstance(cbs, list):
                    for cb in cbs:
                        if not isinstance(cb, dict):
                            continue
                        ctype = cb.get("type", "").lower()
                        # Typical structured response block names
                        if ctype in ("structured", "structured_response", "structured_output"):
                            # block may carry its payload under 'value' / 'data' / 'json' / 'args'
                            parsed = cb.get("value") or cb.get("data") or cb.get("json") or cb.get("args") or cb.get("output")
                            break
                        # Another pattern: provider returns a tool call with args that represent structured payload
                        if ctype in ("tool_call", "server_tool_call") and isinstance(cb.get("args"), dict):
                            # If agent declares output_vars with single key, try to use tool args as parsed output
                            parsed = cb.get("args")
                            # do not break here if you prefer more explicit; break for pragmatic mapping
                            break

        # 3) Fallback: try best-effort JSON parsing of the textual output
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
        def _schema_resolution_state() -> Dict[str, Any]:
            cfg_schema_json = getattr(agent.spec, "schema_json", None)
            cfg_schema_name = getattr(agent.spec, "schema_name", None)
            configured = bool(cfg_schema_json) or bool(cfg_schema_name)
            if not configured:
                return {"status": "missing", "schema": None, "error": "Schema not configured"}
            try:
                schema = resolve_schema_json(cfg_schema_json, cfg_schema_name)
            except Exception as e:
                return {"status": "invalid_json", "schema": None, "error": str(e)}
            if schema is None:
                return {"status": "invalid_json", "schema": None, "error": "Schema could not be resolved"}
            if isinstance(schema, dict) and len(schema) == 0:
                return {"status": "empty", "schema": schema, "error": "Schema resolved to empty object"}
            return {"status": "resolved", "schema": schema, "error": None}

        if getattr(agent.spec, "structured_output_enabled", False):
            res = _schema_resolution_state()
            validation_failed = False
            fail_reason = ""

            if res["status"] != "resolved":
                validation_failed = True
                fail_reason = res.get("error") or res["status"]
                self._warn(f"[{agent_name}] schema validation skipped: {fail_reason}", pipeline_state)
            else:
                schema = res["schema"]
                candidate = parsed if isinstance(parsed, dict) else LLMClient.safe_json(raw_output) if isinstance(raw_output, str) else None
                if candidate is None:
                    validation_failed = True
                    fail_reason = "No JSON payload to validate"
                    self._warn(f"[{agent_name}] schema validation failed: {fail_reason}", pipeline_state)
                else:
                    try:
                        jsonschema_validate(candidate, schema)
                    except ValidationError as ve:
                        validation_failed = True
                        fail_reason = str(ve).splitlines()[0]
                        self._warn(f"[{agent_name}] schema validation failed: {fail_reason}", pipeline_state)
                    except Exception as ve:
                        validation_failed = True
                        fail_reason = str(ve)
                        self._warn(f"[{agent_name}] schema validation failed: {fail_reason}", pipeline_state)

            if validation_failed:
                pipeline_state.agent_schema_validation_failed[agent_name] = True
                if self.strict_schema_validation:
                    pipeline_state.strict_schema_exit = True
                    logger.error(
                        "[%s] Strict schema validation triggered early exit; output preserved. Reason: %s",
                        agent_name,
                        fail_reason or "schema validation failed",
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