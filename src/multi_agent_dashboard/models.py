# models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging
from multi_agent_dashboard.utils import safe_format

logger = logging.getLogger(__name__)

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
    # NEW: tool configuration (backed by agents.tools_json)
    # Example: {"enabled": True, "tools": ["web_search"]}
    tools: Dict[str, Any] = field(default_factory=dict)
    # NEW: reasoning configuration (per agent)
    # effort: "none" | "low" | "medium" | "high" | "xhigh"
    # summary: "auto" | "concise" | "detailed" | "none"
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


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
        tc = self._build_tools_config(state)
        rc = self._build_reasoning_config()
        logger.debug("Agent %s tools_config=%r reasoning_config=%r", self.spec.name, tc, rc)

        response = self.llm_client.create_text_response(
            model=self.spec.model,
            prompt=prompt,
            response_format=structured_schema,
            stream=stream,
            files=llm_files_payload if llm_files_payload else None,
            tools_config=self._build_tools_config(state),
            reasoning_config=self._build_reasoning_config(),
        )

        # Save metrics for engine to retrieve
        self.last_metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency": response.latency,
            "raw": response.raw,
            "tools_config": tc,
            "reasoning_config": rc,
        }

        # If there was tool usage (web_search), store for engine / run logging
        # We'll store high-level info into last_metrics["tools"]
        tool_events = response.raw.get("output", [])
        used_tools: List[Dict[str, Any]] = []

        for item in tool_events:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "web_search_call":
                # Basic call info
                usage_entry = {
                    "tool_type": "web_search",
                    "id": item.get("id"),
                    "status": item.get("status"),
                }
                action = item.get("action") or {}
                if isinstance(action, dict):
                    usage_entry["action"] = action
                used_tools.append(usage_entry)

        if used_tools:
            self.last_metrics["tools"] = used_tools

        return response.text

    def _build_tools_config(self, state: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Build tools/tool_choice/include args for OpenAI Responses API
        based on agent spec tools and state (allowed domains).
        Supports:
          - state["allowed_domains_by_agent"][agent_name]
          - state["allowed_domains"] as a global fallback.
        """
        tools_cfg = self.spec.tools or {}
        if not tools_cfg.get("enabled"):
            return None

        enabled_tools = tools_cfg.get("tools") or []
        tools_array: List[Dict[str, Any]] = []

        # Here we only support web_search for now, but structure supports more later
        if "web_search" in enabled_tools:
            tool_obj: Dict[str, Any] = {"type": "web_search"}

            # 1) Per-agent domains, if provided
            allowed_domains = None
            per_agent = state.get("allowed_domains_by_agent")
            if isinstance(per_agent, dict):
                maybe = per_agent.get(self.spec.name)
                if isinstance(maybe, list) and maybe:
                    allowed_domains = maybe

            # 2) Global domains as fallback
            if allowed_domains is None:
                global_domains = state.get("allowed_domains")
                if isinstance(global_domains, list) and global_domains:
                    allowed_domains = global_domains

            if allowed_domains:
                tool_obj["filters"] = {"allowed_domains": allowed_domains}

            # Keep default: external_web_access = True unless we want an option later
            tools_array.append(tool_obj)

        if not tools_array:
            return None

        return {
            "tools": tools_array,
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
        }

    def _build_reasoning_config(self) -> Dict[str, Any] | None:
        effort = self.spec.reasoning_effort
        summary = self.spec.reasoning_summary

        if not effort and not summary:
            return None

        reasoning: Dict[str, Any] = {}
        if effort and effort != "none":
            reasoning["effort"] = effort
        # For summary, "none" means do not request it
        if summary and summary != "none":
            reasoning["summary"] = summary

        if not reasoning:
            return None
        return reasoning

# -------------------------
# Pipeline domain model
# -------------------------

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
