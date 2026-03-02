# src/multi_agent_dashboard/models.py
"""
Core data classes for the Multi-Agent Dashboard system.

This module defines the immutable data structures that represent agents and pipelines
in the system. These classes are designed to be safe for serialization, storage,
comparison, and testing. They serve as the canonical representation of agent
configurations and pipeline definitions throughout the application.

Key Classes:
- `AgentSpec`: Immutable agent definition with provider configuration, tool settings,
  structured output configuration, and UI metadata.
- `PipelineSpec`: Immutable pipeline definition with ordered agent steps and metadata.

Design Principles:
- **Immutability**: All classes are frozen dataclasses to prevent accidental mutation
- **Serialization Safety**: All fields are JSON-serializable for database storage
- **Provider Agnosticism**: Agent specifications work with any supported LLM provider
- **Extensibility**: New fields can be added without breaking existing serialized data

Usage:
    from multi_agent_dashboard.models import AgentSpec, PipelineSpec
    
    # Create an agent specification
    agent = AgentSpec(
        name="researcher",
        model="gpt-4",
        prompt_template="Research topic: {topic}",
        provider_id="openai",
        tools={"enabled": True, "tools": ["web_search"]},
        structured_output_enabled=True,
        schema_json='{"type": "object", "properties": {"summary": {"type": "string"}}}'
    )
    
    # Create a pipeline specification
    pipeline = PipelineSpec(
        name="research_pipeline",
        steps=["researcher", "writer", "reviewer"],
        metadata={"description": "Research and writing workflow"}
    )

Token Limit Precedence:
The `AgentSpec.effective_max_output()` method implements token limit precedence rules:
1. If `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE=True`: Use `AGENT_OUTPUT_TOKEN_CAP` if > 0
2. Otherwise: Use smallest non-zero value among `AGENT_OUTPUT_TOKEN_CAP` and agent's `max_output`
3. `0` means no limit (treated as `None`)

Dependencies:
- dataclasses: For immutable data class definitions
- typing: For type hints and optional fields

See Also:
- `multi_agent_dashboard.db.agents`: Database operations for AgentSpec persistence
- `multi_agent_dashboard.runtime.AgentRuntime`: Execution engine using AgentSpec
- `multi_agent_dashboard.engine.MultiAgentEngine`: Pipeline orchestration using PipelineSpec
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
# -------------------------
# Agent domain models
# -------------------------

@dataclass(frozen=True)
class AgentSpec:
    """
    Immutable agent definition.
    Safe to serialize, store, diff, and test.
    """
    name: str
    model: str
    prompt_template: str
    role: str = ""
    input_vars: List[str] = field(default_factory=list)
    output_vars: List[str] = field(default_factory=list)
    # UI metadata
    color: str | None = None
    symbol: str | None = None
    # Tool configuration (backed by agents.tools_json)
    # Example: {"enabled": True, "tools": ["web_search"]}
    tools: Dict[str, Any] = field(default_factory=dict)
    # Reasoning configuration (per agent)
    # effort: "none" | "low" | "medium" | "high" | "xhigh"
    # summary: "auto" | "concise" | "detailed" | "none"
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    # Explicit system (developer) prompt
    system_prompt_template: Optional[str] = None
    # Provider metadata
    provider_id: Optional[str] = None       # e.g. 'openai', 'ollama', 'custom'
    model_class: Optional[str] = None       # provider-specific class / family hint
    endpoint: Optional[str] = None          # optional host/URL override
    use_responses_api: bool = False         # whether to prefer Responses API or equivalent
    provider_features: Dict[str, Any] = field(default_factory=dict)  # capability hints
    # Structured output configuration (provider-agnostic)
    structured_output_enabled: bool = False
    schema_json: Optional[str] = None
    schema_name: Optional[str] = None
    temperature: Optional[float] = None
    max_output: int = 0

    def effective_max_output(self) -> int | None:
        """
        Compute the effective max output token limit following precedence rules:
        - If STRICT_OUTPUT_TOKEN_CAP_OVERRIDE is True: use AGENT_OUTPUT_TOKEN_CAP if > 0 else None
        - Otherwise: smallest non-zero among AGENT_OUTPUT_TOKEN_CAP and self.max_output
        - 0 means no limit → treat as None
        """
        from multi_agent_dashboard import config
        env_cap = config.AGENT_OUTPUT_TOKEN_CAP
        agent_cap = self.max_output
        
        if config.STRICT_OUTPUT_TOKEN_CAP_OVERRIDE:
            return env_cap if env_cap > 0 else None
        
        candidates = [c for c in (env_cap, agent_cap) if c > 0]
        if not candidates:
            return None
        return min(candidates)

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
