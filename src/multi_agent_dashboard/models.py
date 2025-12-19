# models.py
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


@dataclass
class AgentRuntime:
    """
    Execution wrapper.
    Holds runtime-only dependencies (LLM client, memory).
    """
    spec: AgentSpec
    llm_client: Any  # injected, intentionally untyped

    def run(
        self,
        state: Dict[str, Any],
        *,
        structured_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        from utils import safe_format  # local import to keep domain clean

        # Build variable map
        if self.spec.input_vars:
            variables = {k: state.get(k, "") for k in self.spec.input_vars}
        else:
            variables = dict(state)

        prompt = safe_format(self.spec.prompt_template, variables)

        response = self.llm_client.create_text_response(
            model=self.spec.model,
            prompt=prompt,
            response_format=structured_schema,
            stream=stream,
        )

        return response.text


# -------------------------
# Pipeline domain model
# -------------------------

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
