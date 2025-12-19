# engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

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
        last_agent: Optional[str] = None,
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
                if var not in self.state or self.state[var] in ("", None):
                    msg = f"[{agent_name}] Missing input var '{var}'"
                    if strict:
                        raise ValueError(msg)
                    self._warn(msg)

            # ---- Execute agent ----
            try:
                raw_output = agent.run(self.state)
            except Exception as e:
                logger.exception("Agent '%s' failed", agent_name)
                raise RuntimeError(f"Agent '{agent_name}' failed") from e

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

        return EngineResult(
            final_output=final_output,
            state=dict(self.state),
            memory=dict(self.memory),
            warnings=list(self._warnings),
            final_agent=(
                "final" in self.state and last_agent
            ) or last_agent
        )
