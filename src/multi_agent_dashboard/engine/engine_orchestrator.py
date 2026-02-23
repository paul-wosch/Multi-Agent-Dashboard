# engine.py
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


from multi_agent_dashboard.models import AgentSpec, AgentRuntime
from multi_agent_dashboard.llm_client import LLMClient


logger = logging.getLogger(__name__)


# =========================
# Helper Functions
# =========================



from .metrics_aggregator import MetricsAggregator
from .progress_reporter import ProgressReporter
from .state_manager import StateManager

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
        self.state_manager = StateManager()
        # Backward‑compatibility aliases (properties could be added later if needed)
        self.state = self.state_manager.state
        self.memory = self.state_manager.memory
        self._warnings = self.state_manager.warnings
        self.agent_metrics = self.state_manager.agent_metrics

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
        self.state_manager.add_warning(message)
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
        progress_reporter = ProgressReporter(
            on_progress=self.on_progress,
            num_steps=len(steps),
        )

        for i, agent_name in enumerate(steps):
            # ---- Progress bar: agent start ----
            progress_reporter.start_agent(i, agent_name)

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
            progress_reporter.end_agent(i, agent_name)

        # After loop, synchronize engine instance attributes via StateManager
        self.state_manager.update_from_pipeline_state(pipeline_state)

        final_output = self.state.get("final", last_output)

        total_cost, total_input_cost, total_output_cost, total_latency = MetricsAggregator.aggregate_totals(
            list(pipeline_state.agent_metrics.values())
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
