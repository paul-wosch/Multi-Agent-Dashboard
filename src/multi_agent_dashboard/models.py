# src/multi_agent_dashboard/models.py
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
