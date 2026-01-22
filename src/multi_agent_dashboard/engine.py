# engine.py
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from multi_agent_dashboard.config import OPENAI_PRICING
from multi_agent_dashboard.models import AgentSpec, AgentRuntime
from multi_agent_dashboard.llm_client import LLMClient

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
    # fallback to a standardized key if present
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


def _normalize_content_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    """
    Ensure each content block is a serializable dict (best-effort).
    """
    out_blocks: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return []
    for b in blocks:
        if isinstance(b, dict):
            out_blocks.append(b)
            continue
        try:
            if hasattr(b, "model_dump"):
                out_blocks.append(b.model_dump())
            elif hasattr(b, "to_dict"):
                out_blocks.append(b.to_dict())
            elif hasattr(b, "__dict__"):
                out_blocks.append(dict(b.__dict__))
            else:
                out_blocks.append({"__repr": repr(b)})
        except Exception:
            out_blocks.append({"__repr": repr(b)})
    return out_blocks


def _extract_provider_features_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a LangChain model 'profile' into a compact provider_features mapping.

    This is intentionally conservative: only expose a few well-known capability hints
    used by the UI (structured_output, tool_calling, reasoning, image_inputs, max_input_tokens).
    """

    features: Dict[str, Any] = {}

    if not isinstance(profile, dict):
        return features

    # Normalize profile keys to handle camelCase, snake_case, and lower-case variants.
    def _normalize_key(k: str) -> str:
        # Convert camelCase / PascalCase to snake_case
        s = re.sub(r'(?<!^)(?=[A-Z])', '_', str(k)).lower()
        s = s.replace('-', '_')
        return s

    normalized: Dict[str, Any] = {}
    for k, v in profile.items():
        try:
            normalized[str(k)] = v
        except Exception:
            normalized[k] = v
        try:
            normalized[str(k).lower()] = v
        except Exception:
            pass
        try:
            nk = _normalize_key(str(k))
            normalized[nk] = v
        except Exception:
            pass

    # Structured output related hints
    if normalized.get("structured_output") or normalized.get("structuredoutput") or normalized.get("reasoning_output") or normalized.get("structured"):
        features["structured_output"] = True

    # Tool calling hints
    if normalized.get("tool_calling") or normalized.get("toolcalling") or normalized.get("tool_calls") or normalized.get("toolcalls") or normalized.get("tool_call"):
        features["tool_calling"] = True

    # Reasoning hints
    if normalized.get("reasoning") or normalized.get("reasoning_output") or normalized.get("reasoningoutput") or normalized.get("supports_reasoning"):
        features["reasoning"] = True

    # Image / multimodal hints
    if "image_inputs" in normalized or "imageinputs" in normalized:
        try:
            features["image_inputs"] = bool(normalized.get("image_inputs") or normalized.get("imageinputs"))
        except Exception:
            features["image_inputs"] = True

    # Max input tokens (context window) — try variants
    max_tokens_candidates = [
        normalized.get("max_input_tokens"),
        normalized.get("maxinputtokens"),
        normalized.get("max_input_token"),
        normalized.get("max_input"),
        normalized.get("maxInputTokens"),
    ]
    for candidate in max_tokens_candidates:
        if candidate is not None:
            try:
                features["max_input_tokens"] = int(candidate)
            except Exception:
                features["max_input_tokens"] = candidate
            break

    # If nothing obvious matched, expose a shallow copy for auditing
    if not features and profile:
        # Keep only a small subset to avoid clobbering DB with huge dicts
        keys_to_copy = ["tool_calling", "structured_output", "reasoning", "image_inputs", "max_input_tokens", "maxInputTokens", "structuredOutput", "toolCalling"]
        for k in keys_to_copy:
            if k in profile:
                features[k if "_" in k else _normalize_key(k)] = profile[k]

    return features


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
    # per-agent metrics
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # per-agent configuration snapshot for this run (used for DB logging)
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # totals broken down
    total_cost: float = 0.0
    total_latency: float = 0.0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    # per-agent tool usage, as parsed from LLM responses
    tool_usages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


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
        # metrics per agent for last run
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}

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

    def _compute_cost(
            self,
            model: str,
            input_tokens: Optional[int],
            output_tokens: Optional[int],
            provider_id: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """Compute approximate cost for a single LLM call.

        Return (total_cost, input_cost, output_cost).

        Prices are per 1M tokens.

        This helper is provider-aware: for OpenAI-family providers
        (provider_id is None/'' or 'openai' or 'azure_openai') it uses
        OPENAI_PRICING. For other providers we currently return zero so that
        non-OpenAI calls do not get mis-attributed OpenAI prices.
        """
        if input_tokens is None and output_tokens is None:
            return 0.0, 0.0, 0.0

        provider = (provider_id or "").strip().lower()

        # Backwards-compatible default: treat missing provider_id as OpenAI.
        is_openai_family = (not provider) or provider in ("openai", "azure_openai")

        pricing = OPENAI_PRICING.get(model) if is_openai_family else None
        if not pricing:
            return 0.0, 0.0, 0.0

        inp = input_tokens or 0
        out = output_tokens or 0

        input_cost = inp / 1_000_000.0 * pricing.get("input", 0.0)
        output_cost = out / 1_000_000.0 * pricing.get("output", 0.0)
        return input_cost + output_cost, input_cost, output_cost

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
        files: Optional[List[Dict[str, Any]]] = None,
        # Can be either:
        #   - List[str]: global allow-list for all agents
        #   - Dict[str, List[str]]: per-agent allow-lists
        allowed_domains: Optional[Any] = None,
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
        self.agent_metrics = {}
        tool_usages: Dict[str, List[Dict[str, Any]]] = {}
        # Per-agent configuration snapshot filled as agents run
        agent_configs: Dict[str, Dict[str, Any]] = {}

        # Store initial files in state so all agents can access
        if files:
            self.state["files"] = files

        # Optional domain filters for web_search
        if allowed_domains:
            if isinstance(allowed_domains, dict):
                # Per-agent mapping {agent_name: [domains...]}
                filtered = {
                    k: v for k, v in allowed_domains.items()
                    if isinstance(v, list) and v
                }
                if filtered:
                    self.state["allowed_domains_by_agent"] = filtered
            else:
                # Backwards-compatible: single global list
                self.state["allowed_domains"] = allowed_domains

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
                if var == "files":
                    # Special-case: files in input validation
                    # files are injected once and may be an empty list; presence is enough
                    if "files" not in self.state:
                        msg = f"[{agent_name}] Missing input var 'files'"
                        if strict:
                            raise ValueError(msg)
                        self._warn(msg)
                    continue

                if var not in self.state or self.state[var] in ("", None):
                    msg = f"[{agent_name}] Missing input var '{var}'"
                    if strict:
                        raise ValueError(msg)
                    self._warn(msg)

            # ---- Execute agent ----
            try:
                run_kwargs = {}
                if "files" in inspect.signature(agent.run).parameters:
                    run_kwargs["files"] = self.state.get("files")
                raw_output = agent.run(self.state, **run_kwargs)
            except Exception as e:
                # Enabled 'real error' display during development
                # logger.exception("Agent '%s' failed", agent_name)
                # raise RuntimeError(f"Agent '{agent_name}' failed") from e
                logger.exception("Agent '%s' failed with real error:", agent_name)
                raise

            # Retrieve metrics from AgentRuntime.last_metrics
            metrics = getattr(agent, "last_metrics", {}) or {}
            raw_metrics = metrics.get("raw") or {}

            # If LangChain agent path was used but instrumentation appears missing, warn
            try:
                used_langchain = bool(metrics.get("used_langchain_agent"))
                instrumentation_attached = bool(metrics.get("instrumentation_attached"))
                has_content_blocks = bool(metrics.get("content_blocks"))
                has_instrumentation_events = bool(metrics.get("instrumentation_events"))
                # If instrumentation was attached at agent-create time but we still see no content blocks
                if used_langchain and instrumentation_attached and not (has_content_blocks or has_instrumentation_events or metrics.get("detected_provider_profile")):
                    # Instrumentation expected for LangChain agents to capture content_blocks/tool traces
                    self._warn(
                        f"[{agent_name}] Ran via LangChain with instrumentation attached but produced no content_blocks or instrumentation events. "
                        "Confirm provider supports content_blocks or middleware hooks executed."
                    )
                # If instrumentation was not attached and LangChain used, warn once
                if used_langchain and not instrumentation_attached:
                    self._warn(
                        f"[{agent_name}] Ran via LangChain but instrumentation middleware was not attached. "
                        "Enable instrumentation to capture content_blocks/tool traces."
                    )
            except Exception:
                logger.debug("Failed to validate instrumentation presence for agent=%s", agent_name, exc_info=True)

            # Extract config used for this agent call
            tc = metrics.get("tools_config")
            rc = metrics.get("reasoning_config")

            # Extract tool usage for this agent (per-call details)
            tools = (metrics.get("tools") or [])
            if tools:
                for t in tools:
                    # Keep config on the in-memory tool events for convenience;
                    # DB writes now use agent_run_configs instead.
                    if tc:
                        t["tools_config"] = tc
                    if rc:
                        t["reasoning_config"] = rc
                tool_usages[agent_name] = tools

            # Snapshot this agent's configuration for this particular run.
            # This keeps agent config concerns out of tool_usages rows.
            # Include both user-facing prompt_template and system_prompt_template so
            # stored runs capture both templates used during execution.
            # Also include a compact summary of content_blocks for auditing
            content_blocks = metrics.get("content_blocks")
            if not isinstance(content_blocks, list):
                content_blocks = _collect_content_blocks(raw_metrics)

            def _filter_extra_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                # Exclude plain text blocks from extra_config_json to avoid duplicating agent_outputs.output
                out: List[Dict[str, Any]] = []
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    btype = (b.get("type") or "").lower()
                    if btype == "text":
                        continue
                    out.append(b)
                return out

            content_blocks_summary = None
            try:
                if isinstance(content_blocks, list) and content_blocks:
                    filtered_blocks = _filter_extra_blocks(content_blocks)
                    content_blocks_summary = [
                        {
                            "type": (cb.get("type") if isinstance(cb, dict) else None),
                            "name": (cb.get("name") if isinstance(cb, dict) else None),
                            "id": (cb.get("id") if isinstance(cb, dict) else None),
                        }
                        for cb in filtered_blocks
                    ]
            except Exception:
                content_blocks_summary = None

            # Normalize full content blocks for DB storage (best-effort)
            content_blocks_full = _normalize_content_blocks(_filter_extra_blocks(content_blocks or []))

            # Provider profile hints detected at runtime (from model or response)
            detected_profile = metrics.get("detected_provider_profile") or raw_metrics.get("detected_provider_profile")
            # Derive a compact provider_features mapping when the AgentSpec didn't provide any
            spec_provider_features = getattr(agent.spec, "provider_features", None) or {}
            provider_features_to_store = dict(spec_provider_features) if isinstance(spec_provider_features, dict) else (spec_provider_features or {})
            if not provider_features_to_store and detected_profile:
                derived = _extract_provider_features_from_profile(detected_profile)
                if derived:
                    provider_features_to_store = derived
                else:
                    # Keep a trace of the raw detected profile when we cannot derive concise features
                    provider_features_to_store = {"detected_profile_present": True}

            # Capture instrumentation events & structured_response for auditing
            instrumentation_events = _extract_instrumentation_events(raw_metrics)
            structured_response = None
            try:
                if isinstance(raw_metrics, dict):
                    structured_response = raw_metrics.get("structured_response") or raw_metrics.get("structured")
                if structured_response is None:
                    # check instrumentation events for structured payload
                    structured_response = _structured_from_instrumentation(raw_metrics)
            except Exception:
                structured_response = None

            # Record whether instrumentation middleware was attached to the agent (if agent runtime set it)
            instrumentation_attached_flag = bool(metrics.get("instrumentation_attached"))

            extra_dict: Dict[str, Any] = {}
            if content_blocks_summary is not None:
                extra_dict["content_blocks_summary"] = content_blocks_summary
            if content_blocks_full:
                extra_dict["content_blocks"] = content_blocks_full
            if detected_profile is not None:
                extra_dict["detected_provider_profile"] = detected_profile
            if instrumentation_events:
                extra_dict["instrumentation_events"] = instrumentation_events
            if structured_response is not None:
                extra_dict["structured_response"] = structured_response
            if instrumentation_attached_flag:
                extra_dict["instrumentation_attached"] = True

            agent_configs[agent_name] = {
                "model": agent.spec.model,
                "prompt_template": agent.spec.prompt_template,
                "system_prompt_template": agent.spec.system_prompt_template,
                "role": agent.spec.role,
                "input_vars": list(agent.spec.input_vars),
                "output_vars": list(agent.spec.output_vars),
                # High-level tools overview from AgentSpec.tools
                "tools": agent.spec.tools or {},
                # Low-level tools/reasoning config used in this run
                "tools_config": tc,
                "reasoning_effort": agent.spec.reasoning_effort,
                "reasoning_summary": agent.spec.reasoning_summary,
                "reasoning_config": rc,
                # Reserved for future options such as temperature
                "extra": extra_dict,
                # Provider metadata (ensure runs capture which provider/model was used)
                "provider_id": getattr(agent.spec, "provider_id", None),
                "model_class": getattr(agent.spec, "model_class", None),
                "endpoint": getattr(agent.spec, "endpoint", None),
                "use_responses_api": bool(getattr(agent.spec, "use_responses_api", False)),
                # Persist provider feature hints (explicit OR derived)
                "provider_features": provider_features_to_store,
            }

            # -------------------------
            # Metric extraction and token/cost computation
            # -------------------------
            input_tokens = metrics.get("input_tokens")
            output_tokens = metrics.get("output_tokens")

            # Fallback to raw usage metadata if necessary
            if (input_tokens is None or output_tokens is None) and isinstance(raw_metrics, dict):
                try:
                    usage = raw_metrics.get("usage") or raw_metrics.get("usage_metadata") or {}
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
                    logger.debug("Engine-level token fallback extraction failed for agent=%s", agent_name, exc_info=True)

            latency = metrics.get("latency")

            total_cost, input_cost, output_cost = self._compute_cost(
                model=agent.spec.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id=getattr(agent.spec, "provider_id", None),
            )

            self.agent_metrics[agent_name] = {
                "model": agent.spec.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cost": total_cost,
            }

            self.memory[agent_name] = raw_output
            last_output = raw_output

            # ---- Prefer structured outputs surfaced via LangChain content_blocks or structured response ----
            parsed = None
            # 1) If the LLM client included a canonical 'structured' key (created by LLMClient when with_structured_output returned a parsed object),
            #    or when using LangChain agents, a top-level 'structured_response' key from the agent state.
            if isinstance(raw_metrics, dict):
                if "structured" in raw_metrics:
                    parsed = raw_metrics.get("structured")
                elif "structured_response" in raw_metrics:
                    parsed = raw_metrics.get("structured_response")
                else:
                    # 2) Look through content_blocks for structured / structured_response / server_tool_result
                    cbs = raw_metrics.get("content_blocks") or raw_metrics.get("output") or []
                    if isinstance(cbs, list):
                        for cb in cbs:
                            if not isinstance(cb, dict):
                                continue
                            ctype = cb.get("type", "").lower()
                            # Typical structured response block names
                            if ctype in ("structured", "structured_response", "structured_output"):
                                # block may carry its payload under 'value' / 'data' / 'json' / 'args'
                                parsed = cb.get("value") or cb.get("data") or cb.get("json") or cb.get("args") or cb.get("output")
                                break
                            # Another pattern: provider returns a tool call with args that represent structured payload
                            if ctype in ("tool_call", "server_tool_call") and isinstance(cb.get("args"), dict):
                                # If agent declares output_vars with single key, try to use tool args as parsed output
                                parsed = cb.get("args")
                                # do not break here if you prefer more explicit; break for pragmatic mapping
                                break

            # 3) Fallback: try best-effort JSON parsing of the textual output
            if parsed is None:
                parsed = LLMClient.safe_json(raw_output) if isinstance(raw_output, str) else None

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

        total_cost = sum(
            (m.get("cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_input_cost = sum(
            (m.get("input_cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_output_cost = sum(
            (m.get("output_cost") or 0.0) for m in self.agent_metrics.values()
        )
        total_latency = sum(
            (m.get("latency") or 0.0) for m in self.agent_metrics.values()
        )

        return EngineResult(
            final_output=final_output,
            state=dict(self.state),
            memory=dict(self.memory),
            warnings=list(self._warnings),
            final_agent=(
                                "final" in self.state and last_agent
                        ) or last_agent,
            agent_metrics=dict(self.agent_metrics),
            agent_configs=agent_configs,
            total_cost=total_cost,
            total_latency=total_latency,
            total_input_cost=total_input_cost,
            total_output_cost=total_output_cost,
            tool_usages=tool_usages,
        )
