# models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from multi_agent_dashboard.utils import safe_format

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
    # last LLM call metrics
    last_metrics: Dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        state: Dict[str, Any],
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        # -------------------------
        # Determine files to use
        # -------------------------
        all_files = files or state.get("files", [])
        text_files: List[Dict[str, Any]] = []
        binary_files: List[Dict[str, Any]] = []

        for f in all_files:
            mime = f.get("mime_type", "")
            if mime in {"text/plain", "text/markdown", "text/csv", "application/json"}:
                # Inline small text files
                try:
                    content = f["content"].decode("utf-8", errors="replace")
                except Exception:
                    content = ""
                text_files.append({
                    "filename": f["filename"],
                    "mime_type": mime,
                    "content": content.encode("utf-8"),
                })
            else:
                # Treat as binary (PDF, images, audio)
                binary_files.append(f)

        # -------------------------
        # Build prompt variables
        # -------------------------
        if self.spec.input_vars:
            variables = {k: state.get(k, "") for k in self.spec.input_vars}

            # Inject file info into variables if requested
            if "files" in self.spec.input_vars and all_files:
                variables["files"] = "\n".join(
                    f"- {f['filename']} ({f['mime_type']})"
                    for f in all_files
                )
        else:
            variables = dict(state)

        prompt = safe_format(self.spec.prompt_template, variables)

        # -------------------------
        # Combine inline text files into LLM input
        # -------------------------
        llm_files_payload = []

        for f in text_files:
            llm_files_payload.append({
                "filename": f["filename"],
                "content": f["content"],
                "mime_type": f["mime_type"],
            })

        for f in binary_files:
            llm_files_payload.append(f)  # will be uploaded by LLM client

        # -------------------------
        # Call LLM client
        # -------------------------
        response = self.llm_client.create_text_response(
            model=self.spec.model,
            prompt=prompt,
            response_format=structured_schema,
            stream=stream,
            files=llm_files_payload if llm_files_payload else None,
        )

        # Save metrics for engine to retrieve
        self.last_metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency": response.latency,
            "raw": response.raw,
        }

        return response.text


# -------------------------
# Pipeline domain model
# -------------------------

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
