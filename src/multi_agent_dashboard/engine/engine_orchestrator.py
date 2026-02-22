# engine.py
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from jsonschema import validate as jsonschema_validate, ValidationError  # type: ignore

from multi_agent_dashboard.config import OPENAI_PRICING
from multi_agent_dashboard.models import AgentSpec, AgentRuntime
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.structured_schemas import resolve_schema_json

logger = logging.getLogger(__name__)


# =========================
# Helper Functions
# =========================

from .utils import (
    _extract_instrumentation_events,
    _collect_content_blocks,
    _structured_from_instrumentation,
    _normalize_content_blocks,
    _compute_cost,
    _extract_provider_features_from_profile,
)

from .agent_executor import AgentExecutor
from .types import PipelineState




# =========================
# Engine Result
# =========================

@dataclass
class EngineResult:
    """
    Structured result returned by the engine.
    """
    final_output: Any
    state: Dict[str, Any]
    memory: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    final_agent: Optional[str] = None  # runtime-only
    # per-agent metrics
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # per-agent configuration snapshot for this run (used for DB logging)
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # totals broken down
    total_cost: float = 0.0
    total_latency: float = 0.0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    # per-agent tool usage, as parsed from LLM responses
    tool_usages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # schema validation flags
    strict_schema_exit: bool = False
    agent_schema_validation_failed: Dict[str, bool] = field(default_factory=dict)


# =========================
# MultiAgentEngine
# =========================

class MultiAgentEngine:
    """
    Core orchestration engine.

    - No UI dependencies
    - No global state
    - Safe for CLI, tests, batch jobs
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        on_progress: Optional[Callable[[int, Optional[str]], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ):
        self.llm_client = llm_client
        self.on_progress = on_progress
        self.on_warning = on_warning

        self.agents: Dict[str, AgentRuntime] = {}
        self.state: Dict[str, Any] = {}
        self.memory: Dict[str, Any] = {}
        self._warnings: List[str] = []
        # metrics per agent for last run
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Agent Management
    # -------------------------

    def add_agent(self, spec: AgentSpec) -> None:
        self.agents[spec.name] = AgentRuntime(
            spec=spec,
            llm_client=self.llm_client,
        )

    def remove_agent(self, name: str) -> None:
        self.agents.pop(name, None)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _warn(self, message: str) -> None:
        self._warnings.append(message)
        if self.on_warning:
            self.on_warning(message)
        else:
            logger.warning(message)

    def _progress(self, pct: int, agent_name: Optional[str] = None) -> None:
        if self.on_progress:
            self.on_progress(pct, agent_name)



    # -------------------------
    # Sequential Execution
    # -------------------------

    def run_seq(
        self,
        *,
        steps: List[str],
        initial_input: Any,
        strict: bool = False,
        strict_schema_validation: bool = False,
        last_agent: Optional[str] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        # Can be either:
        #   - List[str]: global allow-list for all agents
        #   - Dict[str, List[str]]: per-agent allow-lists
        allowed_domains: Optional[Any] = None,
    ) -> EngineResult:
        """
        Execute agents sequentially.

        Rules:
        - Shared state dict
        - Explicit input/output contracts
        - Deterministic writeback
        """

        logger.info("Starting pipeline: %s", steps)

        # Reset execution state
        # Create pipeline state container
        pipeline_state = PipelineState(
            state={
                "task": initial_input,
                "input": initial_input,
            },
            memory={},
            warnings=[],
            tool_usages={},
            agent_configs={},
            strict_schema_exit=False,
            agent_schema_validation_failed={},
            agent_metrics={},
        )

        # Store initial files in state so all agents can access
        if files:
            pipeline_state.state["files"] = files

        # Optional domain filters for web_search
        if allowed_domains:
            if isinstance(allowed_domains, dict):
                # Per-agent mapping {agent_name: [domains...]}
                filtered = {
                    k: v for k, v in allowed_domains.items()
                    if isinstance(v, list) and v
                }
                if filtered:
                    pipeline_state.state["allowed_domains_by_agent"] = filtered
            else:
                # Backwards-compatible: single global list
                pipeline_state.state["allowed_domains"] = allowed_domains

        # Create agent executor
        executor = AgentExecutor(
            llm_client=self.llm_client,
            strict=strict,
            strict_schema_validation=strict_schema_validation,
            warning_callback=self._warn,
        )

        last_output: Any = None
        last_agent: Optional[str] = None
        # ---- Progress bar: initialize ----
        num_steps = len(steps)
        total_ticks = max(1, 2 * num_steps)

        for i, agent_name in enumerate(steps):
            # ---- Progress bar: agent start ----
            start_tick = 2 * i + 1
            start_pct = int(100 * start_tick / total_ticks)
            self._progress(start_pct, agent_name)

            agent = self.agents.get(agent_name)
            last_agent = agent_name

            if not agent:
                msg = f"Agent '{agent_name}' is not registered"
                if strict:
                    raise ValueError(msg)
                self._warn(msg)
                pipeline_state.memory[agent_name] = msg
                continue

            try:
                result = executor.execute_agent(agent_name, agent, pipeline_state)
                last_output = result.raw_output
            except Exception as e:
                # The executor already logs the error; propagate
                raise

            # Break early if strict schema validation triggered exit
            if pipeline_state.strict_schema_exit:
                logger.error(
                    "[%s] Strict schema validation triggered early exit; skipping remaining agents",
                    agent_name,
                )
                break

            # ---- Progress bar: agent end ----
            end_tick = 2 * i + 2
            end_pct = int(100 * end_tick / total_ticks)
            self._progress(end_pct, agent_name)

        # After loop, synchronize engine instance attributes for compatibility
        self.state = pipeline_state.state
        self.memory = pipeline_state.memory
        self._warnings = pipeline_state.warnings

        # Convert RunMetrics to dict for EngineResult
        agent_metrics_dict: Dict[str, Dict[str, Any]] = {}
        for agent_name, metrics in pipeline_state.agent_metrics.items():
            agent_metrics_dict[agent_name] = {
                "model": metrics.model,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "latency": metrics.latency,
                "input_cost": metrics.input_cost,
                "output_cost": metrics.output_cost,
                "cost": metrics.total_cost,
            }

        self.agent_metrics = agent_metrics_dict

        final_output = self.state.get("final", last_output)

        total_cost = sum(
            (m.total_cost or 0.0) for m in pipeline_state.agent_metrics.values()
        )
        total_input_cost = sum(
            (m.input_cost or 0.0) for m in pipeline_state.agent_metrics.values()
        )
        total_output_cost = sum(
            (m.output_cost or 0.0) for m in pipeline_state.agent_metrics.values()
        )
        total_latency = sum(
            (m.latency or 0.0) for m in pipeline_state.agent_metrics.values()
        )

        return EngineResult(
            final_output=final_output,
            state=dict(self.state),
            memory=dict(self.memory),
            warnings=list(self._warnings),
            final_agent=(
                                "final" in self.state and last_agent
                        ) or last_agent,
            agent_metrics=dict(self.agent_metrics),
            agent_configs=pipeline_state.agent_configs,
            total_cost=total_cost,
            total_latency=total_latency,
            total_input_cost=total_input_cost,
            total_output_cost=total_output_cost,
            tool_usages=pipeline_state.tool_usages,
            strict_schema_exit=pipeline_state.strict_schema_exit,
            agent_schema_validation_failed=pipeline_state.agent_schema_validation_failed,
        )
