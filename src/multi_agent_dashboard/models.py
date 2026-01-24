# src/multi_agent_dashboard/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging
from multi_agent_dashboard.utils import safe_format
from multi_agent_dashboard.llm_client import LLMError

logger = logging.getLogger(__name__)


# =========================
# Helper Functions
# =========================

def _extract_instrumentation_events(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    # Primary key used by middleware
    events = raw_metrics.get("_multi_agent_dashboard_events")
    if isinstance(events, list):
        return events
    # Backwards-compatible alias (used by LLMClient.invoke_agent)
    events2 = raw_metrics.get("instrumentation_events")
    if isinstance(events2, list):
        return events2
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
    # Also inspect messages for content/content_blocks (LangChain agent state)
    messages = raw_metrics.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            msg_dict = _value_to_dict(msg)
            if not isinstance(msg_dict, dict):
                continue
            cb = msg_dict.get("content_blocks")
            if isinstance(cb, list):
                blocks.extend(cb)
            # OpenAI-native content blocks live under "content" as a list of dicts
            content = msg_dict.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                blocks.extend(content)
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


def _value_to_dict(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            normalized = value.model_dump()
            if isinstance(normalized, dict):
                return normalized
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            normalized = value.to_dict()
            if isinstance(normalized, dict):
                return normalized
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return None


def _collect_tool_calls(raw_metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, dict):
        return []
    calls: List[Dict[str, Any]] = []

    def _recurse(node: Any) -> None:
        node_dict = _value_to_dict(node)
        if not isinstance(node_dict, dict):
            return
        tool_calls = node_dict.get("tool_calls")
        if isinstance(tool_calls, list):
            for entry in tool_calls:
                entry_dict = _value_to_dict(entry)
                if isinstance(entry_dict, dict):
                    calls.append(entry_dict)
        # Some providers nest tool_calls under additional_kwargs
        additional = node_dict.get("additional_kwargs")
        if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
            for entry in additional.get("tool_calls"):
                entry_dict = _value_to_dict(entry)
                if isinstance(entry_dict, dict):
                    calls.append(entry_dict)
        # Recurse into messages list if present (LangChain agent state)
        messages = node_dict.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                _recurse(msg)
        for key in ("agent_response", "response", "result"):
            _recurse(node_dict.get(key))
        output = node_dict.get("output")
        if isinstance(output, list):
            for entry in output:
                _recurse(entry)
        events = node_dict.get("instrumentation_events") or node_dict.get("_multi_agent_dashboard_events")
        if isinstance(events, list):
            for event in events:
                _recurse(event)

    _recurse(raw_metrics)
    return calls


def _tool_usage_entry_from_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    raw_type = payload.get("type") or payload.get("tool_type") or ""
    raw_type_l = str(raw_type).lower()
    # Skip non-tool content blocks (e.g., text/reasoning)
    if raw_type_l in ("text", "reasoning"):
        return None
    tool_type = payload.get("name") or payload.get("tool_type") or payload.get("type") or "unknown"
    if tool_type in ("unknown", "", None) and not payload.get("name"):
        return None
    entry: Dict[str, Any] = {
        "tool_type": tool_type,
        "id": payload.get("id") or payload.get("tool_call_id") or payload.get("tool_use_id"),
    }
    status = payload.get("status") or payload.get("state")
    if status is not None:
        entry["status"] = status
    action = payload.get("args") or payload.get("action") or payload.get("input") or payload.get("result")
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except Exception:
            action = {"raw": action}
    if isinstance(action, dict):
        entry["action"] = action
    elif action is not None:
        entry["action"] = {"raw": action}
    return entry


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
        # Call LLM client (LangChain-only path)
        # -------------------------
        tc = self._build_tools_config(state)
        rc = self._build_reasoning_config()
        logger.debug("Agent %s tools_config=%r reasoning_config=%r", self.spec.name, tc, rc)

        response = None
        used_langchain_agent = False
        agent_obj_for_invoke = None
        langchain_tools = None

        # Map configured tools to LangChain-compatible tool specs (OpenAI built-ins)
        try:
            if (getattr(self.spec, "provider_id", None) or "").lower() == "openai":
                tools_cfg = self.spec.tools or {}
                if tools_cfg.get("enabled"):
                    enabled_tools = tools_cfg.get("tools") or []
                    if "web_search" in enabled_tools:
                        tool_spec: Dict[str, Any] = {"type": "web_search"}
                        # Pass allowed_domains filters when available
                        if isinstance(tc, dict):
                            tools_arr = tc.get("tools")
                            if isinstance(tools_arr, list):
                                for t in tools_arr:
                                    if isinstance(t, dict) and t.get("type") == "web_search":
                                        filters = t.get("filters")
                                        if filters:
                                            tool_spec["filters"] = filters
                        langchain_tools = [tool_spec]
        except Exception:
            langchain_tools = None

        # Enforce LangChain-only runtime: create_agent_for_spec + invoke_agent must be available.
        if not getattr(self.llm_client, "_langchain_available", False):
            raise LLMError(
                f"AgentRuntime.run requires LangChain-enabled LLM client for agent '{self.spec.name}'. "
                "Install LangChain and the provider integration (e.g., langchain-openai, langchain-ollama) and retry."
            )

        try:
            # Create a new agent instance per run (simpler and avoids cache invalidation complexity).
            try:
                agent = self.llm_client.create_agent_for_spec(
                    self.spec,
                    tools=langchain_tools,
                    middleware=None,
                    response_format=structured_schema,
                )
            except Exception as e:
                logger.debug("create_agent_for_spec raised while creating agent for %s: %s", self.spec.name, e, exc_info=True)
                raise LLMError(f"LangChain agent creation failed for '{self.spec.name}': {e}") from e

            if agent is None:
                # create_agent_for_spec unexpectedly returned None (should return agent or raise)
                raise LLMError(f"Failed to create LangChain agent for '{self.spec.name}': create_agent_for_spec returned no agent instance.")

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
        except Exception as e:
            # Do not fall back to legacy path; surface a typed error with context.
            logger.debug("LangChain agent path failed for agent '%s'; throwing LLMError", self.spec.name, exc_info=True)
            raise LLMError(
                f"LangChain agent creation/invoke failed for agent '{self.spec.name}': {e}"
            ) from e

        # -------------------------
        # Normalize and store metrics / instrumentation for engine/DB
        # -------------------------
        raw = response.raw or {}

        # Fallback token extraction from raw usage metadata if TextResponse left them None
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        try:
            if (input_tokens is None or output_tokens is None) and isinstance(raw, dict):
                usage = raw.get("usage") or raw.get("usage_metadata") or {}
                if isinstance(usage, dict):
                    if input_tokens is None:
                        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count")
                        if input_tokens is None and isinstance(usage.get("token_usage"), dict):
                            input_tokens = usage["token_usage"].get("prompt_tokens") or usage["token_usage"].get("input_tokens")
                    if output_tokens is None:
                        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_count")
                        if output_tokens is None and isinstance(usage.get("token_usage"), dict):
                            output_tokens = usage["token_usage"].get("completion_tokens") or usage["token_usage"].get("output_tokens")
        except Exception:
            logger.debug("Token fallback extraction from raw usage failed for agent=%s", self.spec.name, exc_info=True)

        instrumentation_events = _extract_instrumentation_events(raw)
        content_blocks = _collect_content_blocks(raw)

        # Base last_metrics stored for engine consumption
        self.last_metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
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
                # In some flows, the LLM client may have included this in response.raw
                resp_detected = raw.get("detected_provider_profile")
                if resp_detected is not None:
                    self.last_metrics["detected_provider_profile"] = resp_detected
        except Exception:
            pass

        # -------------------------
        # Tool usage extraction from content_blocks and legacy tool_calls
        # -------------------------
        used_tools: List[Dict[str, Any]] = []
        seen_tool_entries: set[tuple[str, str | None]] = set()

        def _maybe_add_tool_entry(entry: Dict[str, Any] | None) -> None:
            if not entry:
                return
            tool_type = entry.get("tool_type")
            if not tool_type:
                return
            key = (tool_type, entry.get("id"))
            if key in seen_tool_entries:
                return
            seen_tool_entries.add(key)
            used_tools.append(entry)

        TOOL_BLOCK_TYPES = {"tool_call", "server_tool_call", "web_search_call", "web_search", "function_call"}
        for item in content_blocks:
            if not isinstance(item, dict):
                continue
            btype = str(item.get("type") or "").lower()
            if btype and btype not in TOOL_BLOCK_TYPES:
                continue
            _maybe_add_tool_entry(_tool_usage_entry_from_payload(item))

        for call_payload in _collect_tool_calls(raw):
            _maybe_add_tool_entry(_tool_usage_entry_from_payload(call_payload))

        if used_tools:
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
                    ctype = (cb.get("type") or "").lower()
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
        if parsed is None and isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output) if raw_output.strip() else None
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


# -------------------------
# Pipeline domain model
# -------------------------

@dataclass(frozen=True)
class PipelineSpec:
    name: str
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
