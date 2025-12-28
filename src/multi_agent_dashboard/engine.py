# engine.py
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from multi_agent_dashboard.config import OPENAI_PRICING
from multi_agent_dashboard.models import AgentSpec, AgentRuntime
from multi_agent_dashboard.llm_client import LLMClient

logger = logging.getLogger(__name__)


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

    def _compute_cost(
            self,
            model: str,
            input_tokens: Optional[int],
            output_tokens: Optional[int],
    ) -> tuple[float, float, float]:
        """Compute approximate cost for a single LLM call.

        Return (total_cost, input_cost, output_cost).
        Prices are per 1M tokens.
        """
        if input_tokens is None and output_tokens is None:
            return 0.0, 0.0, 0.0

        pricing = OPENAI_PRICING.get(model)
        if not pricing:
            return 0.0, 0.0, 0.0

        inp = input_tokens or 0
        out = output_tokens or 0

        input_cost = inp / 1_000_000.0 * pricing.get("input", 0.0)
        output_cost = out / 1_000_000.0 * pricing.get("output", 0.0)
        return input_cost + output_cost, input_cost, output_cost

    # -------------------------
    # Sequential Execution
    # -------------------------

    def run_seq(
        self,
        *,
        steps: List[str],
        initial_input: Any,
        strict: bool = False,
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
        self.state = {
            "task": initial_input,
            "input": initial_input,
        }
        self.memory = {}
        self._warnings = []
        self.agent_metrics = {}
        tool_usages: Dict[str, List[Dict[str, Any]]] = {}
        # Per-agent configuration snapshot filled as agents run
        agent_configs: Dict[str, Dict[str, Any]] = {}

        # Store initial files in state so all agents can access
        if files:
            self.state["files"] = files

        # Optional domain filters for web_search
        if allowed_domains:
            if isinstance(allowed_domains, dict):
                # Per-agent mapping {agent_name: [domains...]}
                filtered = {
                    k: v for k, v in allowed_domains.items()
                    if isinstance(v, list) and v
                }
                if filtered:
                    self.state["allowed_domains_by_agent"] = filtered
            else:
                # Backwards-compatible: single global list
                self.state["allowed_domains"] = allowed_domains

        last_output: Any = None
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
                self.memory[agent_name] = msg
                continue

            # ---- Input validation ----
            for var in agent.spec.input_vars:
                if var == "files":
                    # Special-case: files in input validation
                    # files are injected once and may be an empty list; presence is enough
                    if "files" not in self.state:
                        msg = f"[{agent_name}] Missing input var 'files'"
                        if strict:
                            raise ValueError(msg)
                        self._warn(msg)
                    continue

                if var not in self.state or self.state[var] in ("", None):
                    msg = f"[{agent_name}] Missing input var '{var}'"
                    if strict:
                        raise ValueError(msg)
                    self._warn(msg)

            # ---- Execute agent ----
            try:
                run_kwargs = {}
                if "files" in inspect.signature(agent.run).parameters:
                    run_kwargs["files"] = self.state.get("files")
                raw_output = agent.run(self.state, **run_kwargs)
            except Exception as e:
                # Enabled 'real error' display during development
                # logger.exception("Agent '%s' failed", agent_name)
                # raise RuntimeError(f"Agent '{agent_name}' failed") from e
                logger.exception("Agent '%s' failed with real error:", agent_name)
                raise

            # Retrieve metrics from AgentRuntime.last_metrics
            metrics = getattr(agent, "last_metrics", {}) or {}

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
                tool_usages[agent_name] = tools

            # Snapshot this agent's configuration for this particular run.
            # This keeps agent config concerns out of tool_usages rows.
            agent_configs[agent_name] = {
                "model": agent.spec.model,
                "prompt_template": agent.spec.prompt_template,
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
            }

            input_tokens = metrics.get("input_tokens")
            output_tokens = metrics.get("output_tokens")
            latency = metrics.get("latency")

            total_cost, input_cost, output_cost = self._compute_cost(
                model=agent.spec.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            self.agent_metrics[agent_name] = {
                "model": agent.spec.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cost": total_cost,
            }

            self.memory[agent_name] = raw_output
            last_output = raw_output

            parsed = LLMClient.safe_json(raw_output)

            # ---- Writeback rules ----
            if agent.spec.output_vars:
                if isinstance(parsed, dict):
                    for key in parsed:
                        if key not in agent.spec.output_vars:
                            msg = (
                                f"[{agent_name}] Unexpected output key '{key}'"
                            )
                            if strict:
                                raise ValueError(msg)
                            self._warn(msg)

                    for var in agent.spec.output_vars:
                        if var in parsed:
                            self.state[var] = parsed[var]
                        else:
                            msg = (
                                f"[{agent_name}] Declared output '{var}' missing"
                            )
                            if strict:
                                raise ValueError(msg)
                            self._warn(msg)
                else:
                    if len(agent.spec.output_vars) == 1:
                        self.state[agent.spec.output_vars[0]] = raw_output
                    else:
                        key = f"{agent_name}__raw"
                        self.state[key] = raw_output
                        self._warn(
                            f"[{agent_name}] Non-JSON output stored as '{key}'"
                        )
            else:
                self.state[agent_name] = raw_output

            # ---- Progress bar: agent end ----
            end_tick = 2 * i + 2
            end_pct = int(100 * end_tick / total_ticks)
            self._progress(end_pct, agent_name)

        final_output = self.state.get("final", last_output)

        total_cost = sum(
            (m.get("cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_input_cost = sum(
            (m.get("input_cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_output_cost = sum(
            (m.get("output_cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_latency = sum(
            (m.get("latency") or 0.0) for m in self.agent_metrics.values()
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
            agent_configs=agent_configs,
            total_cost=total_cost,
            total_latency=total_latency,
            total_input_cost=total_input_cost,
            total_output_cost=total_output_cost,
            tool_usages=tool_usages,
        )


# =========================
# File Detection Helper
# =========================

def agent_requires_files(agent_runtime) -> bool:
    try:
        sig = inspect.signature(agent_runtime.run)
        return "files" in sig.parameters
    except Exception:
        return False
