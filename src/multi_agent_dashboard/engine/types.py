"""
Data containers for the multi-agent engine.

These dataclasses provide structured interfaces for the engine's internal data flow,
enabling modularization of the orchestration logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunMetrics:
    """Per-agent token and cost metrics."""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_cost: Optional[float] = None
    model: Optional[str] = None  # optional, for reference


@dataclass
class AgentRunResult:
    """Result of executing a single agent within a pipeline."""
    raw_output: Any
    metrics: RunMetrics
    parsed: Optional[Any] = None
    tool_usages: List[Dict[str, Any]] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)  # keys: "input", "output", "total"


@dataclass
class PipelineState:
    """Mutable state shared across all agents in a pipeline."""
    state: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    tool_usages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strict_schema_exit: bool = False
    agent_schema_validation_failed: Dict[str, bool] = field(default_factory=dict)
    # Per-agent metrics accumulated across the run (optional, can be stored separately)
    agent_metrics: Dict[str, RunMetrics] = field(default_factory=dict)