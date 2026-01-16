# src/multi_agent_dashboard/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging
import threading
from multi_agent_dashboard.utils import safe_format

logger = logging.getLogger(__name__)


# =========================
# Helper Functions
# =========================

def _extract_instrumentation_events(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    events = raw_metrics.get("_multi_agent_dashboard_events")
    if isinstance(events, list):
        return events
    return []


def _collect_content_blocks(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    blocks: List[Dict[str, Any]] = []
    for event in _extract_instrumentation_events(raw_metrics):
        payload = event.get("content_blocks")
        if isinstance(payload, list):
            blocks.extend(payload)
    direct = raw_metrics.get("content_blocks")
    if isinstance(direct, list):
        blocks.extend(direct)
    elif isinstance(raw_metrics.get("output"), list):
        blocks.extend(raw_metrics["output"])
    return blocks


def _structured_from_instrumentation(raw_metrics: Dict[str, Any] | None) -> Any:
    if not isinstance(raw_metrics, dict):
        return None
    for event in _extract_instrumentation_events(raw_metrics):
        if "structured_response" in event:
            return event["structured_response"]
    return None


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


@dataclass
class AgentRuntime:
    """
    Execution wrapper.
    Holds runtime-only dependencies (LLM client, memory).

    Note: this class now caches a LangChain 'agent' instance (when the LLM client
    provides the LangChain agent path) to avoid re-creating agents on every call,
    which in turn avoids repeatedly instantiating/attaching middleware that can
    generate noisy warnings during long runs.
    """
    spec: AgentSpec
    llm_client: Any  # injected, intentionally untyped
    # last LLM call metrics
    last_metrics: Dict[str, Any] = field(default_factory=dict)

    # Internal cache for LangChain agent instances (not part of dataclass equality / init)
    _cached_agent: Any = field(default=None, init=False, repr=False)
    # Response-format fingerprint used to decide whether cached agent is compatible.
    _cached_agent_response_format: Optional[str] = field(default=None, init=False, repr=False)
    # Lock to make cache operations thread-safe if engine is used in threaded contexts
    _cached_agent_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def run(
        self,
        state: Dict[str, Any],
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        structured_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        # Expose the provided state on the runtime for callers/tests that expect it.
        # This mirrors engine.run_seq's state handling for standalone usage.
        self.state = state

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

        # Prefer LangChain agent path when available. This uses create_agent + agent.invoke
        response = None
        used_langchain_agent = False
        agent_obj_for_invoke = None

        # Helper: fingerprint structured_schema to decide cache compatibility.
        def _fingerprint_schema(sch: Optional[Any]) -> Optional[str]:
            if sch is None:
                return None
            try:
                return json.dumps(sch, sort_keys=True, separators=(",", ":"), default=str)
            except Exception:
                return repr(sch)

        requested_rf = _fingerprint_schema(structured_schema)

        if getattr(self.llm_client, "_langchain_available", False):
            try:
                agent = None
                # Attempt to reuse a cached agent when available and compatible with requested response format.
                try:
                    with self._cached_agent_lock:
                        if self._cached_agent is not None and self._cached_agent_response_format == requested_rf:
                            agent = self._cached_agent
                            logger.debug("Reusing cached LangChain agent for %s (rf=%s)", self.spec.name, requested_rf)
                        else:
                            # Create and cache a new agent instance.
                            new_agent = None
                            try:
                                new_agent = self.llm_client.create_agent_for_spec(
                                    self.spec,
                                    tools=None,
                                    middleware=None,
                                    response_format=structured_schema,
                                )
                            except Exception:
                                new_agent = None

                            if new_agent is not None:
                                self._cached_agent = new_agent
                                self._cached_agent_response_format = requested_rf
                                agent = new_agent
                                logger.debug("Created and cached new LangChain agent for %s (rf=%s)", self.spec.name, requested_rf)
                except Exception:
                    # Cache logic must be resilient; fall back to creating agent without caching.
                    logger.debug("Agent cache logic failed; attempting non-cached create for %s", self.spec.name, exc_info=True)
                    try:
                        agent = self.llm_client.create_agent_for_spec(
                            self.spec,
                            tools=None,
                            middleware=None,
                            response_format=structured_schema,
                        )
                    except Exception:
                        agent = None

                if agent is not None:
                    # Context: pass allowed_domains/state if present so middleware / agent may use it.
                    context: Dict[str, Any] = {}
                    if "allowed_domains_by_agent" in state:
                        context["allowed_domains_by_agent"] = state.get("allowed_domains_by_agent")
                    if "allowed_domains" in state:
                        context["allowed_domains"] = state.get("allowed_domains")

                    if tc is not None:
                        context["tools_config"] = tc
                    if rc is not None:
                        context["reasoning_config"] = rc

                    # Invoke the agent via the LLMClient helper for consistent normalization
                    response = self.llm_client.invoke_agent(
                        agent,
                        prompt,
                        files=llm_files_payload if llm_files_payload else None,
                        response_format=structured_schema,
                        stream=stream,
                        context=context if context else None,
                    )
                    used_langchain_agent = True
                    agent_obj_for_invoke = agent
            except Exception:
                logger.debug("LangChain agent path failed for agent '%s'; falling back to create_text_response", self.spec.name, exc_info=True)
                response = None

        # Fallback to existing path (works for legacy OpenAI SDK and also LangChain chat-model-only path)
        if response is None:
            response = self.llm_client.create_text_response(
                model=self.spec.model,
                prompt=prompt,
                response_format=structured_schema,
                stream=stream,
                files=llm_files_payload if llm_files_payload else None,
                tools_config=tc,
                reasoning_config=rc,
                system_prompt=self.spec.system_prompt_template,
                # Provider metadata
                provider_id=self.spec.provider_id,
                model_class=self.spec.model_class,
                endpoint=self.spec.endpoint,
                use_responses_api=self.spec.use_responses_api,
                provider_features=self.spec.provider_features,
            )

        # -------------------------
        # Normalize and store metrics / instrumentation for engine/DB
        # -------------------------
        raw = response.raw or {}
        instrumentation_events = _extract_instrumentation_events(raw)
        content_blocks = _collect_content_blocks(raw)

        # Base last_metrics stored for engine consumption
        self.last_metrics = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency": response.latency,
            "raw": raw,
            "content_blocks": content_blocks,
            "instrumentation_events": instrumentation_events,
            "tools_config": tc,
            "reasoning_config": rc,
            # indicate whether LangChain agent path was used
            "used_langchain_agent": bool(used_langchain_agent),
            # whether instrumentation middleware was attached (agent instance may carry this flag)
            "instrumentation_attached": bool(getattr(agent_obj_for_invoke, "_instrumentation_attached", False)) if agent_obj_for_invoke is not None else False,
        }

        # surface detected provider profile if available on the agent (propagated from LLMClient.create_agent_for_spec)
        try:
            detected = getattr(agent_obj_for_invoke, "_detected_provider_profile", None)
            if detected is not None:
                self.last_metrics["detected_provider_profile"] = detected
            else:
                # In some flows, the create_text_response path may populate this in response.raw
                resp_detected = raw.get("detected_provider_profile")
                if resp_detected is not None:
                    self.last_metrics["detected_provider_profile"] = resp_detected
        except Exception:
            pass

        # If there was tool usage (web_search), store for engine / run logging
        # We'll store high-level info into last_metrics["tools"]
        used_tools: List[Dict[str, Any]] = []
        # Normalize different tool-call/ server-tool-call block types into a tool usage entry
        for item in content_blocks:
            if not isinstance(item, dict):
                continue
            btype = item.get("type", "").lower()
            if btype in ("server_tool_call", "tool_call", "web_search_call", "web_search"):
                usage_entry = {
                    "tool_type": item.get("name") or item.get("tool_type") or "web_search",
                    "id": item.get("id") or item.get("tool_call_id"),
                    "status": item.get("status") or item.get("state") or None,
                }
                action = item.get("args") or item.get("action") or item.get("input") or item.get("args_json")
                if isinstance(action, dict):
                    usage_entry["action"] = action
                elif isinstance(action, str):
                    try:
                        usage_entry["action"] = json.loads(action)
                    except Exception:
                        usage_entry["action"] = {"raw": action}
                used_tools.append(usage_entry)

        # If still empty, check for legacy 'tool_calls' metadata
        if not used_tools and isinstance(raw, dict):
            legacy_calls = raw.get("tool_calls") or raw.get("tool_calls", None)
            if isinstance(legacy_calls, list):
                for lc in legacy_calls:
                    if not isinstance(lc, dict):
                        continue
                    usage_entry = {
                        "tool_type": lc.get("tool_type") or lc.get("name") or "unknown",
                        "id": lc.get("id") or lc.get("tool_call_id") or None,
                        "status": lc.get("status"),
                    }
                    action = lc.get("args") or lc.get("action") or lc.get("args_json")
                    if isinstance(action, dict):
                        usage_entry["action"] = action
                    used_tools.append(usage_entry)

        if used_tools:
            # Attach the used tools list into last_metrics for engine persistence
            self.last_metrics["tools"] = used_tools

        # -------------------------
        # Structured output detection & local writeback (standalone runtime behavior)
        # -------------------------
        parsed = None

        # 1) Structured keys directly on raw
        if isinstance(raw, dict):
            if "structured" in raw:
                parsed = raw.get("structured")
            elif "structured_response" in raw:
                parsed = raw.get("structured_response")
            else:
                # 2) Inspect instrumentation events
                parsed = _structured_from_instrumentation(raw)

            # 3) If still none, look through content blocks for structured payloads
            if parsed is None and isinstance(content_blocks, list):
                for cb in content_blocks:
                    if not isinstance(cb, dict):
                        continue
                    ctype = cb.get("type", "").lower()
                    # Typical structured response block names
                    if ctype in ("structured", "structured_response", "structured_output"):
                        parsed = cb.get("value") or cb.get("data") or cb.get("json") or cb.get("args") or cb.get("output")
                        break
                    # Another pattern: provider returns a tool call with args that represent structured payload
                    if ctype in ("tool_call", "server_tool_call") and isinstance(cb.get("args"), dict):
                        parsed = cb.get("args")
                        break

        # 4) Fallback: try best-effort JSON parsing of the textual output
        raw_output = response.text
        if parsed is None:
            try:
                parsed = json.loads(raw_output) if isinstance(raw_output, str) and raw_output.strip() else None
            except Exception:
                parsed = None

        # Local writeback to the passed-in state dict (mirrors engine.run_seq writeback semantics)
        if self.spec.output_vars:
            if isinstance(parsed, dict):
                # Warn about unexpected keys (do not raise; standalone runtime mirrors engine's warning behavior)
                for key in parsed:
                    if key not in self.spec.output_vars:
                        logger.warning(
                            "[%s] Unexpected output key '%s' (standalone runtime)", self.spec.name, key
                        )
                # Populate declared output vars
                for var in self.spec.output_vars:
                    if var in parsed:
                        state[var] = parsed[var]
                    else:
                        logger.warning(
                            "[%s] Declared output '%s' missing (standalone runtime)", self.spec.name, var
                        )
            else:
                # Non-JSON output: if single output var, write the raw text; otherwise store under a raw key
                if len(self.spec.output_vars) == 1:
                    state[self.spec.output_vars[0]] = raw_output
                else:
                    key = f"{self.spec.name}__raw"
                    state[key] = raw_output
                    logger.warning(
                        "[%s] Non-JSON output stored as '%s' (standalone runtime)", self.spec.name, key
                    )
        else:
            # No declared output_vars: store whole raw output under agent name
            state[self.spec.name] = raw_output

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
            "tool_choice": "required",
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

    # Helper to allow explicit cache invalidation if callers need it (e.g., spec was mutated)
    def invalidate_cached_agent(self) -> None:
        """
        Clear the cached LangChain agent instance, forcing a recreate on next run.
        """
        try:
            with self._cached_agent_lock:
                self._cached_agent = None
                self._cached_agent_response_format = None
                logger.debug("Invalidated cached LangChain agent for %s", self.spec.name)
        except Exception:
            logger.debug("Failed to invalidate cached agent for %s", self.spec.name, exc_info=True)


# -------------------------
# Pipeline domain model
# -------------------------

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
