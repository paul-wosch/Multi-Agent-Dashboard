"""
StateManager – encapsulates the engine's persistent state between pipeline runs.

Holds:
  - state: shared mutable dict (carries final output, files, allowed_domains, etc.)
  - memory: per‑agent memory dict (stores agent‑specific data across pipeline)
  - warnings: accumulated warning messages
  - agent_metrics: per‑agent metrics from the last run (as dict of dict)

The manager provides a clean interface for resetting and updating from a per‑run
PipelineState instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from .types import PipelineState, RunMetrics


class StateManager:
    """
    Manages the engine's persistent state across multiple pipeline executions.
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.memory: Dict[str, Any] = {}
        self.warnings: List[str] = []
        # Per‑agent metrics from the last run, stored as dict of dict
        # (compatible with the original engine.agent_metrics format)
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear all persistent state."""
        self.state.clear()
        self.memory.clear()
        self.warnings.clear()
        self.agent_metrics.clear()

    def update_from_pipeline_state(self, pipeline_state: PipelineState) -> None:
        """
        Import the state from a finished pipeline run.

        This is called at the end of `run_seq()` to make the per‑run results
        available for inspection via the engine's public attributes.
        """
        self.state.clear()
        self.state.update(pipeline_state.state)
        self.memory.clear()
        self.memory.update(pipeline_state.memory)
        self.warnings.clear()
        self.warnings.extend(pipeline_state.warnings)

        # Convert RunMetrics to dict format for backward compatibility
        self.agent_metrics.clear()
        for agent_name, metrics in pipeline_state.agent_metrics.items():
            self.agent_metrics[agent_name] = {
                "model": metrics.model,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "latency": metrics.latency,
                "input_cost": metrics.input_cost,
                "output_cost": metrics.output_cost,
                "cost": metrics.total_cost,
            }

    def add_warning(self, message: str) -> None:
        """Append a warning message to the internal list."""
        self.warnings.append(message)