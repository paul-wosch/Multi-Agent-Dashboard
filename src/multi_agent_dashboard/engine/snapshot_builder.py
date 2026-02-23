"""
RunSnapshotBuilder – creates agent configuration snapshots for run logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..shared.instrumentation import (
    _collect_content_blocks,
    _extract_instrumentation_events,
    _structured_from_instrumentation,
)
from .utils import (
    _normalize_content_blocks,
    _extract_provider_features_from_profile,
)
from multi_agent_dashboard.models import AgentRuntime

logger = logging.getLogger(__name__)


class RunSnapshotBuilder:
    """
    Builds a configuration snapshot for a single agent execution.

    The snapshot includes:
      - Agent specification (model, templates, role, input/output vars, tools, etc.)
      - Runtime configuration (tools_config, reasoning_config, provider_features)
      - Observability data (content_blocks summary, instrumentation events, structured_response)
      - Provider metadata (provider_id, model_class, endpoint, use_responses_api)
      - Structured output configuration
      - Strict schema validation flag
    """

    def build(
        self,
        agent_name: str,
        agent: AgentRuntime,
        metrics: Dict[str, Any],
        raw_metrics: Dict[str, Any],
        strict_schema_validation: bool,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary suitable for pipeline_state.agent_configs[agent_name].

        The snapshot is used for DB logging and later inspection; it must be JSON‑serializable.
        """
        # Extract config used for this agent call
        tc = metrics.get("tools_config")
        rc = metrics.get("reasoning_config")

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

        agent_config = {
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
            # Structured output configuration (provider-agnostic)
            "structured_output_enabled": bool(getattr(agent.spec, "structured_output_enabled", False)),
            "schema_json": getattr(agent.spec, "schema_json", None),
            "schema_name": getattr(agent.spec, "schema_name", None),
            "temperature": getattr(agent.spec, "temperature", None),
            "strict_schema_validation": bool(strict_schema_validation),
        }
        return agent_config