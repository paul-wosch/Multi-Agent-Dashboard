# ui/agent_editor_mode.py
from __future__ import annotations

from datetime import datetime
import json
import logging
import time
from typing import List, Dict, Any
from urllib.parse import urlparse

import streamlit as st

from multi_agent_dashboard.config import UI_COLORS, AGENT_SNAPSHOT_PRUNE_KEEP
from multi_agent_dashboard.ui.cache import (
    cached_load_agents,
    cached_load_agent_snapshots,
    get_agent_service,
    invalidate_agents,
)
from multi_agent_dashboard.ui.bootstrap import reload_agents_into_engine

logger = logging.getLogger(__name__)

DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]


def _parse_endpoint_to_host_port(endpoint: str) -> tuple[str | None, int | None]:
    """
    Parse an endpoint to extract host and port if present.
    Returns (host, port) or (None, None) when not parseable.
    """
    if not endpoint:
        return None, None
    try:
        p = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
        host = p.hostname
        port = p.port
        return host, port
    except Exception:
        return None, None


def _compose_endpoint_from_host_port(host: str | None, port: str | None) -> str | None:
    """
    Build an endpoint string from host and port values.
    If host already includes a scheme, preserve it; otherwise default to http://.
    """
    if not host:
        return None
    host = host.strip().rstrip("/")
    if not host:
        return None

    # If host contains scheme, keep it
    if host.startswith("http://") or host.startswith("https://"):
        base = host
    else:
        base = f"http://{host}"

    if port:
        port = port.strip()
        if port:
            # Avoid double port if user included it
            h, existing_port = _parse_endpoint_to_host_port(base)
            if existing_port is None:
                # append port
                base = base.rstrip("/") + f":{port}"
    return base


def render_agent_editor():
    st.header("🧠 Agent Editor")

    # -------------------------
    # Load agents from DB
    # -------------------------
    agents_raw = cached_load_agents()
    agents = [
        {
            "name": a["agent_name"],
            "model": a["model"],
            "role": a["role"],
            "prompt": a["prompt_template"],
            "input_vars": a["input_vars"],
            "output_vars": a["output_vars"],
            "color": a.get("color") or DEFAULT_COLOR,
            "symbol": a.get("symbol") or DEFAULT_SYMBOL,
            "tools": a.get("tools") or {},
            "reasoning_effort": a.get("reasoning_effort"),
            "reasoning_summary": a.get("reasoning_summary"),
            "system_prompt": a.get("system_prompt_template"),
            # Provider metadata
            "provider_id": a.get("provider_id"),
            "model_class": a.get("model_class"),
            "endpoint": a.get("endpoint"),
            "use_responses_api": a.get("use_responses_api"),
            "provider_features": a.get("provider_features") or {},
            "structured_output_enabled": bool(a.get("structured_output_enabled")),
            "schema_json": a.get("schema_json") or "",
            "schema_name": a.get("schema_name") or "",
            "temperature": a.get("temperature"),
        }
        for a in agents_raw
    ]
    names = [a["name"] for a in agents]

    # Map agent name -> symbol (with defaults)
    def get_agent_symbol_map() -> Dict[str, str]:
        symbol_map: Dict[str, str] = {}
        engine = st.session_state.get("engine")
        if engine and getattr(engine, "agents", None):
            for name, runtime in engine.agents.items():
                sym = getattr(runtime.spec, "symbol", None) or DEFAULT_SYMBOL
                symbol_map[name] = sym
            return symbol_map
        try:
            for a in cached_load_agents():
                symbol_map[a["agent_name"]] = a.get("symbol") or DEFAULT_SYMBOL
        except Exception:
            pass
        return symbol_map

    agent_symbols = get_agent_symbol_map()

    def format_agent_label(name: str) -> str:
        if name == "<New Agent>":
            return name
        symbol = agent_symbols.get(name, DEFAULT_SYMBOL)
        return f"{symbol} {name}"

    # -------------------------
    # Editor internal state
    # -------------------------
    if "agent_editor_state" not in st.session_state:
        st.session_state.agent_editor_state = {
            "selected_name": "<New Agent>",
            "name": "",
            "model": "gpt-4.1-nano",
            "role": "",
            "prompt": "",
            "system_prompt": "",
            "input_vars": [],
            "output_vars": [],
            "color": DEFAULT_COLOR,
            "symbol": DEFAULT_SYMBOL,
            "tools": {"enabled": False, "tools": []},
            "reasoning_effort": None,
            "reasoning_summary": None,
            # provider defaults
            "provider_id": "openai",
            "model_class": None,
            "endpoint": None,
            "endpoint_host": None,
            "endpoint_port": None,
            "use_responses_api": True,
            "provider_features": {},
            "structured_output_enabled": False,
            "schema_json": "",
            "schema_name": "",
            "temperature": None,
        }
    state = st.session_state.agent_editor_state

    # Persistent flag to survive st.rerun
    if "agent_editor_state_changed_this_run" not in st.session_state:
        st.session_state.agent_editor_state_changed_this_run = False

    state_changed_this_run = st.session_state.agent_editor_state_changed_this_run

    # -------------------------
    # Agent selection
    # -------------------------
    options = ["<New Agent>"] + names

    current_selected_name = state.get("selected_name", "<New Agent>")
    if current_selected_name not in options:
        current_selected_name = "<New Agent>"
    current_index = options.index(current_selected_name)

    selected = st.selectbox(
        "Agent",
        options,
        index=current_index,
        key="agent_editor_selected_agent",
        format_func=format_agent_label,
    )

    # If user changed selection via the selectbox, populate editor fields
    if selected != state.get("selected_name"):
        state["selected_name"] = selected
        if selected == "<New Agent>":
            base_agent = {
                "name": "",
                "model": "gpt-4.1-nano",
                "role": "",
                "prompt": "",
                "system_prompt": "",
                "input_vars": [],
                "output_vars": [],
                "color": DEFAULT_COLOR,
                "symbol": DEFAULT_SYMBOL,
                "tools": {"enabled": False, "tools": []},
                "reasoning_effort": None,
                "reasoning_summary": None,
                "provider_id": "openai",
                "model_class": None,
                "endpoint": None,
                "endpoint_host": None,
                "endpoint_port": None,
                "use_responses_api": True,
                "provider_features": {},
                "structured_output_enabled": False,
                "schema_json": "",
                "schema_name": "",
                "temperature": None,
            }
        else:
            base_agent = next(a for a in agents if a["name"] == selected)

        # Parse endpoint to host/port for convenience in the editor
        endpoint_val = base_agent.get("endpoint")
        host, port = _parse_endpoint_to_host_port(endpoint_val or "")

        state.update(
            {
                "name": base_agent["name"],
                "model": base_agent["model"],
                "role": base_agent["role"],
                "prompt": base_agent["prompt"],
                "system_prompt": base_agent.get("system_prompt", "") or "",
                "input_vars": base_agent["input_vars"],
                "output_vars": base_agent["output_vars"],
                "color": base_agent.get("color", DEFAULT_COLOR) or DEFAULT_COLOR,
                "symbol": base_agent.get("symbol", DEFAULT_SYMBOL)
                or DEFAULT_SYMBOL,
                "tools": base_agent.get("tools")
                or {"enabled": False, "tools": []},
                "reasoning_effort": base_agent.get("reasoning_effort"),
                "reasoning_summary": base_agent.get("reasoning_summary"),
                "provider_id": base_agent.get("provider_id") or "openai",
                "model_class": base_agent.get("model_class"),
                "endpoint": endpoint_val,
                "endpoint_host": host,
                "endpoint_port": str(port) if port is not None else None,
                "use_responses_api": bool(base_agent.get("use_responses_api")),
                "provider_features": base_agent.get("provider_features") or {},
                "structured_output_enabled": bool(base_agent.get("structured_output_enabled")),
                "schema_json": base_agent.get("schema_json") or "",
                "schema_name": base_agent.get("schema_name") or "",
                "temperature": base_agent.get("temperature"),
            }
        )

        state_changed_this_run = True
        st.session_state.agent_editor_state_changed_this_run = True

    is_new = state.get("selected_name") == "<New Agent>"

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(
        [
            "1️⃣ Basics",
            "2️⃣ Prompt",
            "3️⃣ Inputs / Outputs",
            "⚙️ Advanced",
            "📚 Snapshots",
        ]
    )

    basics_tab, prompt_tab, io_tab, adv_tab, snapshots_tab = tabs

    # ----- Basics tab -----
    with basics_tab:
        name_val = st.text_input(
            "Name",
            value=state["name"],
        )
        model_val = st.text_input(
            "Model",
            value=state["model"],
        )
        role_val = st.text_input(
            "Role",
            value=state["role"],
        )

        # Provider selection
        st.markdown("### Provider (per-agent)")
        provider_options = ["openai", "ollama", "custom"]
        provider_display = {"openai": "OpenAI", "ollama": "Ollama (local)", "custom": "Custom"}
        provider_index = provider_options.index(state.get("provider_id") or "openai") if (state.get("provider_id") or "openai") in provider_options else 0
        provider_choice = st.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            format_func=lambda v: provider_display.get(v, v),
            help="Select which provider this agent should use. This only records metadata for later runtime wiring.",
        )

        # Endpoint / host override for custom or local hosts
        endpoint_val = st.text_input(
            "Endpoint / Host (optional)",
            value=state.get("endpoint") or "",
            help="Optional host or endpoint used to reach the model provider (e.g., http://localhost:11434 for a local Ollama)."
        )

        # Show Host / Port fields when provider is local / custom to make it convenient
        show_host_port = provider_choice in ("ollama", "custom")
        host_val = state.get("endpoint_host") or ""
        port_val = state.get("endpoint_port") or ""

        if show_host_port:
            col_h, col_p = st.columns([3, 1])
            with col_h:
                host_val = st.text_input(
                    "Host (e.g., localhost)",
                    value=host_val,
                    help="Host part of endpoint. If left empty, the 'Endpoint' text above is used as-is.",
                )
            with col_p:
                port_val = st.text_input(
                    "Port (e.g., 11434)",
                    value=port_val or "",
                    help="Optional port. If provided, will be combined with Host when saving.",
                )

        use_responses_api_val = st.checkbox(
            "Use provider Responses-style API when available",
            value=bool(state.get("use_responses_api")),
            help="Enable when provider supports a Responses-style structured API (e.g., OpenAI Responses).",
        )

        st.markdown("### Appearance")

        # Prepare color options
        color_keys = list(UI_COLORS.keys())
        color_labels = {
            key: f"{UI_COLORS[key]['symbol']} {key.capitalize()}"
            for key in color_keys
        }

        def infer_color_key_from_hex(hex_value: str) -> str:
            for k, v in UI_COLORS.items():
                if v["value"].lower() == (hex_value or "").lower():
                    return k
            return "default"

        current_color_key = infer_color_key_from_hex(
            state.get("color", DEFAULT_COLOR)
        )
        if current_color_key not in color_labels:
            current_color_key = "default"

        prev_palette_key = state.get("_palette_key", current_color_key)

        selected_color_key = st.selectbox(
            "Base Color",
            options=color_keys,
            index=color_keys.index(current_color_key),
            format_func=lambda k: color_labels[k],
            help=(
                "Choose a base color; this sets both the default color and "
                "symbol for the agent. You can still override them below."
            ),
            key="agent_editor_base_color",
        )

        selected_palette = UI_COLORS[selected_color_key]
        palette_color_value = selected_palette["value"]
        palette_symbol_value = selected_palette["symbol"]

        # If user changed the base palette, immediately update color/symbol
        if selected_color_key != prev_palette_key:
            state["color"] = palette_color_value
            state["symbol"] = palette_symbol_value
            state["_palette_key"] = selected_color_key
        else:
            state["_palette_key"] = prev_palette_key

        color_val = st.color_picker(
            "Agent Color",
            value=state.get("color", palette_color_value) or palette_color_value,
            help="Override the color for this agent.",
        )

        symbol_val = st.text_input(
            "Agent Symbol (emoji or short text)",
            value=state.get("symbol", palette_symbol_value) or palette_symbol_value,
            help="Override the symbol for this agent.",
            max_chars=8,
        )

        # Contextual provider capability hint (short)
        st.markdown("### Provider capability hints")
        prov_feats_preview = state.get("provider_features") or {}
        # Determine a friendly summary
        def _friendly_caps_summary(pf: dict) -> str:
            parts = []
            if pf.get("structured_output"):
                parts.append("structured output")
            if pf.get("tool_calling"):
                parts.append("tool calling")
            if pf.get("reasoning"):
                parts.append("reasoning")
            if not parts:
                return "No explicit capability hints provided. Edit provider features or use the Advanced tab to set them."
            return "Detected: " + ", ".join(parts) + "."

        st.caption(_friendly_caps_summary(prov_feats_preview))

        # Link to LangChain content_blocks docs for more info
        st.markdown(
            """
            Providers that expose structured outputs and tool calls usually surface them via LangChain's standardized content blocks (`content_blocks`),
            which the engine uses to detect tool calls, reasoning traces, and structured responses. See the LangChain standard message content docs in the Advanced tab for details.
            """
        )

    # ----- Prompt tab -----
    with prompt_tab:
        system_prompt_val = st.text_area(
            "System Prompt (optional)",
            height=200,
            value=state.get("system_prompt", "") or "",
            help=(
                "Authoritative instructions for the agent (developer/system role). "
                "These instructions will be sent separately from the user/task input when supported "
                "by the LLM client. Keep them concise and avoid inserting untrusted user data here."
            ),
        )

        prompt_val = st.text_area(
            "Prompt Template",
            height=200,
            value=state["prompt"],
        )

    # ----- Inputs / Outputs tab -----
    with io_tab:
        col1, col2 = st.columns(2)
        with col1:
            input_vars_val = st.text_area(
                "Input Variables (one per line)",
                value="\n".join(state["input_vars"]),
            )
        with col2:
            output_vars_val = st.text_area(
                "Output Variables (one per line)",
                value="\n".join(state["output_vars"]),
            )

    # ----- Snapshots tab -----
    with snapshots_tab:
        st.markdown("### Snapshots")

        # Optional note for a snapshot
        snapshot_note_key = f"agent_snapshot_note_{state['name']}"
        if snapshot_note_key not in st.session_state:
            st.session_state[snapshot_note_key] = ""

        st.text_input(
            "Snapshot note (optional)",
            key=snapshot_note_key,
            placeholder="Short note about this snapshot",
        )

        if st.button("📸 Create Snapshot", key=f"create_snapshot_{state['name']}"):
            # Build snapshot dict based on current editor state
            snapshot = {
                "model": state["model"],
                "prompt_template": state["prompt"],
                "system_prompt_template": state.get("system_prompt", "") or None,
                "role": state["role"],
                "input_vars": state["input_vars"],
                "output_vars": state["output_vars"],
                "color": state["color"],
                "symbol": state["symbol"],
                "tools": state["tools"],
                "reasoning_effort": state.get("reasoning_effort"),
                "reasoning_summary": state.get("reasoning_summary"),
                "provider_id": state.get("provider_id"),
                "model_class": state.get("model_class"),
                "endpoint": state.get("endpoint"),
                "use_responses_api": state.get("use_responses_api"),
                "provider_features": state.get("provider_features"),
            }
            note = st.session_state.get(snapshot_note_key) or ""
            try:
                get_agent_service().save_snapshot(
                    state["name"],
                    snapshot,
                    metadata={"note": note} if note else {},
                    is_auto=False,
                )
                invalidate_agents()
                st.success("Snapshot saved")
                st.rerun()
            except Exception:
                logger.exception("Failed to save snapshot")
                st.error("Failed to save snapshot to database")

        # Manual prune controls
        st.markdown("### Prune old snapshots")
        st.caption(
            "Remove older snapshots and keep the most recent N per agent. Changes are permanent."
        )
        keep_key = f"prune_keep_{state['name']}"
        # initialize session_state default for the number input to avoid changing on every rerun
        if keep_key not in st.session_state:
            st.session_state[keep_key] = AGENT_SNAPSHOT_PRUNE_KEEP

        # Create the number input using the session state's value as the authoritative initial value.
        # Important: do not pass the `value=` parameter when the session state key has been initialized,
        # because Streamlit throws a warning if a widget is created with a default value while the
        # Session State API already set the key. Using only key=keep_key follows Streamlit guidance.
        keep_val = st.number_input(
            "Keep latest N snapshots per agent",
            min_value=0,
            step=1,
            key=keep_key,
        )

        col_prune_a, col_prune_b = st.columns(2)
        with col_prune_a:
            if st.button("🧹 Prune snapshots for this agent", key=f"prune_agent_{state['name']}"):
                try:
                    deleted = get_agent_service().prune_snapshots(agent_name=state["name"], keep=int(keep_val))
                    invalidate_agents()
                    st.success(f"Pruned {deleted} snapshots for {state['name']}")
                    st.rerun()
                except Exception:
                    logger.exception("Failed to prune snapshots for %s", state["name"])
                    st.error("Failed to prune snapshots; check logs for details")
        with col_prune_b:
            if st.button("🧹 Prune snapshots for all agents", key=f"prune_all_{state['name']}"):
                try:
                    deleted = get_agent_service().prune_snapshots(agent_name=None, keep=int(keep_val))
                    invalidate_agents()
                    st.success(f"Pruned {deleted} snapshots across all agents")
                    st.rerun()
                except Exception:
                    logger.exception("Failed to prune snapshots for all agents")
                    st.error("Failed to prune snapshots; check logs for details")

        snapshots = cached_load_agent_snapshots(state["name"])
        if not snapshots:
            st.info("No snapshots available.")
        for snap in snapshots:
            with st.expander(f"Snapshot v{snap['version']} — {snap['created_at']}", expanded=False):
                st.subheader("Snapshot")
                st.json(snap["snapshot"])
                if snap.get("metadata"):
                    st.subheader("Metadata")
                    st.json(snap["metadata"])

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Revert", key=f"revert_snap_{snap['id']}"):
                        # Load snapshot into editor state (do not auto-save)
                        s = snap.get("snapshot") or {}
                        # Update the editor state before widgets are re-created on rerun
                        # Parse endpoint into host/port again for convenience
                        host_s, port_s = _parse_endpoint_to_host_port(s.get("endpoint") or "")
                        st.session_state.agent_editor_state.update(
                            {
                                "name": state.get("name", "") or "",
                                "model": s.get("model", "gpt-4.1-nano"),
                                "role": s.get("role", ""),
                                "prompt": s.get("prompt_template", "") or "",
                                "system_prompt": s.get("system_prompt_template", "") or "",
                                "input_vars": s.get("input_vars", []) or [],
                                "output_vars": s.get("output_vars", []) or [],
                                "color": s.get("color", DEFAULT_COLOR) or DEFAULT_COLOR,
                                "symbol": s.get("symbol", DEFAULT_SYMBOL) or DEFAULT_SYMBOL,
                                "tools": s.get("tools") or {"enabled": False, "tools": []},
                                "reasoning_effort": s.get("reasoning_effort"),
                                "reasoning_summary": s.get("reasoning_summary"),
                                "provider_id": s.get("provider_id", "openai"),
                                "model_class": s.get("model_class"),
                                "endpoint": s.get("endpoint"),
                                "endpoint_host": host_s,
                                "endpoint_port": str(port_s) if port_s is not None else None,
                                "use_responses_api": s.get("use_responses_api", True),
                                "provider_features": s.get("provider_features") or {},
                            }
                        )
                        st.session_state.agent_editor_state_changed_this_run = True
                        st.success(f"Snapshot v{snap['version']} loaded into editor. Click Save to persist.")
                        st.rerun()
                with col_b:
                    export_payload = json.dumps({"snapshot": snap.get("snapshot"), "metadata": snap.get("metadata")}, indent=2)
                    st.download_button(
                        "Export JSON",
                        data=export_payload,
                        file_name=f"{state['name']}_snapshot_v{snap['version']}.json",
                        mime="application/json",
                        key=f"export_snap_{snap['id']}",
                    )

    # ----- Advanced tab -----
    with adv_tab:
        st.markdown("### Provider features (optional)")
        st.caption("Provide a small JSON blob that hints at provider capabilities (e.g. {\"structured_output\": true, \"tool_calling\": true}). You can also toggle common capability flags below.")

        # Parse current provider_features JSON into pf_val for use in the UI
        pf_default = json.dumps(state.get("provider_features") or {}, indent=2)
        # Use a unique session key so editing multiple agents in the same session doesn't collide badly
        pf_session_key = f"agent_editor_pf_text_{state.get('name','__unknown')}"
        if pf_session_key not in st.session_state:
            st.session_state[pf_session_key] = pf_default

        pf_text = st.text_area("Provider Features (JSON)", value=st.session_state[pf_session_key], height=120, key=pf_session_key)
        try:
            pf_val = json.loads(pf_text) if pf_text and pf_text.strip() else {}
        except Exception:
            pf_val = {}
            st.warning("Provider features JSON invalid; saving will store an empty object.")

        # Capability toggles: prefill from provider_features if present, else provider defaults
        # Default heuristics (conservative):
        default_structured = bool(pf_val.get("structured_output")) if "structured_output" in pf_val else (provider_choice == "openai")
        default_tool_calling = bool(pf_val.get("tool_calling")) if "tool_calling" in pf_val else (provider_choice in ("openai", "ollama"))
        default_reasoning = bool(pf_val.get("reasoning")) if "reasoning" in pf_val else True  # default to True for backward compatibility

        col_s, col_t, col_r = st.columns(3)
        with col_s:
            structured_checkbox_key = f"agent_editor_pf_structured_{state.get('name','__unknown')}"
            structured_checked = st.checkbox(
                "Structured output (content_blocks)",
                value=st.session_state.get(structured_checkbox_key, default_structured),
                help="Whether the provider can produce structured outputs via LangChain content_blocks (e.g., for schema'd responses).",
                key=structured_checkbox_key,
            )
        with col_t:
            tool_calling_checkbox_key = f"agent_editor_pf_tool_calling_{state.get('name','__unknown')}"
            tool_calling_checked = st.checkbox(
                "Tool calling",
                value=st.session_state.get(tool_calling_checkbox_key, default_tool_calling),
                help="Provider supports server-side tool calls or function/tool invocation exposed through content blocks.",
                key=tool_calling_checkbox_key,
            )
        with col_r:
            reasoning_checkbox_key = f"agent_editor_pf_reasoning_{state.get('name','__unknown')}"
            reasoning_checked = st.checkbox(
                "Reasoning traces",
                value=st.session_state.get(reasoning_checkbox_key, default_reasoning),
                help="Provider emits structured reasoning / chain-of-thought blocks exposed through content_blocks.",
                key=reasoning_checkbox_key,
            )

        st.markdown("---")

        # Compute the effective provider_feats used to gate UI controls
        provider_feats = dict(pf_val or {})
        # Merge the checkbox preferences (checkboxes are authoritative in the UI; they will be persisted into provider_features on Save)
        provider_feats["structured_output"] = bool(structured_checked)
        provider_feats["tool_calling"] = bool(tool_calling_checked)
        provider_feats["reasoning"] = bool(reasoning_checked)

        st.markdown("#### Structured output settings")
        st.caption("Provider-agnostic controls for schema-based output enforcement.")

        structured_enabled = st.checkbox(
            "Enable structured output enforcement",
            value=bool(state.get("structured_output_enabled")),
            help="When enabled, the runtime will attempt provider-specific structured output enforcement.",
        )

        schema_name_val = st.text_input(
            "Schema name (optional)",
            value=state.get("schema_name") or "",
            help="Optional registry key for a reusable schema (Pydantic/Zod).",
        )

        schema_json_val = st.text_area(
            "Schema JSON (optional)",
            value=state.get("schema_json") or "",
            height=140,
            help="JSON Schema used to validate/enforce structured output.",
        )

        temperature_val = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(state.get("temperature") or 0.0),
            step=0.1,
            help="Lower values (e.g., 0) improve structured output reliability.",
        )

        state["structured_output_enabled"] = bool(structured_enabled)
        state["schema_name"] = schema_name_val or ""
        state["schema_json"] = schema_json_val or ""
        state["temperature"] = float(temperature_val)

        st.markdown("#### Capability summary")
        caps = []
        if provider_feats.get("structured_output"):
            caps.append("structured output")
        if provider_feats.get("tool_calling"):
            caps.append("tool calling")
        if provider_feats.get("reasoning"):
            caps.append("reasoning traces")
        if caps:
            st.markdown(f"- Detected / selected capabilities: **{', '.join(caps)}**")
        else:
            st.markdown("- No capabilities selected.")

        # Small contextual help about content_blocks and where to learn more.
        st.info(
            "LangChain standardized message content ('content_blocks') normalizes text, reasoning, tool calls and multimodal outputs across providers. "
            "When providers expose content_blocks, the dashboard can more reliably detect structured outputs and tool invocations."
        )
        st.caption(
            "Learn more: LangChain standard message content and Messages docs (content_blocks)."
        )
        st.markdown(
            "• LangChain blog: [Standard message content](https://www.blog.langchain.com/standard-message-content/?utm_source=openai).  \n"
            "• LangChain docs: [Messages / content_blocks](https://docs.langchain.com/oss/python/langchain/messages?utm_source=openai)."
        )

        # --- Ollama endpoint health check indicator ---
        # Only show when provider_choice indicates local Ollama and some endpoint info is present
        try:
            endpoint_candidate = None
            # prefer Endpoint field if present
            if endpoint_val and endpoint_val.strip():
                endpoint_candidate = endpoint_val.strip()
            # else consider host+port composition
            elif show_host_port and host_val:
                endpoint_candidate = _compose_endpoint_from_host_port(host_val, port_val)

            if provider_choice == "ollama" and endpoint_candidate:
                # If an LLM client was initialized at app bootstrap, prefer its cached check
                llm_client = st.session_state.get("llm_client")
                health = None
                try:
                    if llm_client and hasattr(llm_client, "check_ollama_endpoint"):
                        # Use a small timeout for UI snappiness; the check is cached on the factory if present.
                        health = llm_client.check_ollama_endpoint(endpoint_candidate, timeout=0.9)
                    else:
                        # Fallback: attempt a minimal probe (very short timeout)
                        import urllib.request, urllib.error
                        try:
                            req = urllib.request.Request(endpoint_candidate, method="GET",
                                                         headers={"User-Agent": "multi-agent-dashboard/0.1"})
                            with urllib.request.urlopen(req, timeout=0.9) as resp:
                                code = getattr(resp, "status", None) or getattr(resp, "getcode", lambda: None)()
                                health = {"endpoint": endpoint_candidate, "reachable": True,
                                          "status": int(code) if code else None,
                                          "message": f"HTTP {code} @ {endpoint_candidate}", "checked_at": time.time()}
                        except urllib.error.HTTPError as he:
                            health = {"endpoint": endpoint_candidate, "reachable": True,
                                      "status": int(getattr(he, "code", None)),
                                      "message": f"HTTP {getattr(he, 'code', None)} @ {endpoint_candidate}",
                                      "checked_at": time.time()}
                        except Exception as e:
                            health = {"endpoint": endpoint_candidate, "reachable": False, "status": None,
                                      "message": str(e), "checked_at": time.time()}
                except Exception:
                    health = {"endpoint": endpoint_candidate, "reachable": False, "status": None,
                              "message": "health check failed", "checked_at": time.time()}

                # Display a succinct status hint
                if health:
                    if health.get("reachable"):
                        st.success(f"Ollama reachable ({health.get('message')})")
                    else:
                        st.error(f"Ollama unreachable: {health.get('message')}")
        except Exception:
            # Keep the editor resilient: ignore health-check failures
            logger.debug("Agent editor endpoint health check failed", exc_info=True)

        # Tools section (respects provider_feat.tool_calling)
        st.markdown("### Tools")

        tools_state = state.get("tools") or {}
        tools_enabled_default = bool(tools_state.get("enabled"))
        selected_tools_default = tools_state.get("tools") or []

        # Disable tools enabling if provider clearly doesn't support tool_calling
        tools_supported = bool(provider_feats.get("tool_calling", False))

        tools_enabled = st.checkbox(
            "Enable tool calling",
            value=tools_enabled_default,
            help="Allow this agent to call tools such as web search.",
            disabled=not tools_supported,
        )

        available_tools = ["web_search"]
        selected_tools: List[str] = []
        if tools_enabled:
            selected_tools = st.multiselect(
                "Allowed tools",
                options=available_tools,
                default=[
                    t for t in selected_tools_default if t in available_tools
                ],
                help="For now only web_search is available.",
            )

        st.markdown("### Reasoning")

        effort_options = ["none", "low", "medium", "high", "xhigh"]
        current_effort = state.get("reasoning_effort") or "medium"

        # Reasoning support derived above in provider_feats
        reasoning_supported = bool(provider_feats.get("reasoning", True))

        reasoning_effort_val = st.selectbox(
            "Reasoning Effort",
            options=effort_options,
            index=effort_options.index(current_effort)
            if current_effort in effort_options
            else effort_options.index("medium"),
            help="Controls depth & cost. 'none' disables special reasoning behavior.",
            disabled=not reasoning_supported,
        )

        summary_options = ["auto", "concise", "detailed", "none"]
        current_summary = state.get("reasoning_summary") or "none"
        reasoning_summary_val = st.selectbox(
            "Reasoning Summary",
            options=summary_options,
            index=summary_options.index(current_summary)
            if current_summary in summary_options
            else summary_options.index("none"),
            help="Configure reasoning summaries returned by reasoning models.",
            disabled=not reasoning_supported,
        )

    st.divider()

    # -------------------------
    # Reflect edits into state
    # -------------------------
    if not state_changed_this_run:
        # Decide which endpoint value to persist:
        # 1) If Endpoint text input provided, use it.
        # 2) Else if Host provided, compose host[:port] into endpoint.
        endpoint_final = endpoint_val.strip() if endpoint_val and endpoint_val.strip() else None
        if not endpoint_final and show_host_port and host_val:
            endpoint_final = _compose_endpoint_from_host_port(host_val, port_val)

        # Merged provider_features for state: prefer pf_val (parsed JSON), then overlay checkbox toggles
        merged_pf = dict(pf_val or {})
        merged_pf["structured_output"] = bool(structured_checked)
        merged_pf["tool_calling"] = bool(tool_calling_checked)
        merged_pf["reasoning"] = bool(reasoning_checked)

        state.update(
            {
                "name": name_val.strip(),
                "model": model_val.strip() or "gpt-4.1-nano",
                "role": role_val.strip(),
                "prompt": prompt_val,
                "system_prompt": system_prompt_val or "",
                "input_vars": [
                    v.strip() for v in input_vars_val.splitlines() if v.strip()
                ],
                "output_vars": [
                    v.strip() for v in output_vars_val.splitlines() if v.strip()
                ],
                "color": color_val or DEFAULT_COLOR,
                "symbol": symbol_val or UI_COLORS[selected_color_key]["symbol"],
                "tools": {
                    "enabled": tools_enabled,
                    "tools": selected_tools if tools_enabled else [],
                },
                "reasoning_effort": reasoning_effort_val,
                "reasoning_summary": reasoning_summary_val,
                "provider_id": provider_choice,
                "endpoint": endpoint_final,
                "endpoint_host": host_val or None,
                "endpoint_port": port_val or None,
                "use_responses_api": bool(use_responses_api_val),
                "provider_features": merged_pf or {},
            }
        )

    # Reset the persistent flag at the end of the render
    st.session_state.agent_editor_state_changed_this_run = False

    col_a, col_b, col_c, col_d = st.columns(4)

    # -------------------------
    # Save button
    # -------------------------
    with col_a:
        if st.button("💾 Save"):
            old_name = state["selected_name"] if not is_new else ""
            new_name = state["name"].strip()

            if not new_name:
                st.error("Name cannot be empty.")
                return

            # Rename agent if needed
            if not is_new and old_name != new_name:
                try:
                    get_agent_service().rename_agent_atomic(old_name, new_name)
                except Exception:
                    logger.exception("Failed to rename agent")
                    st.error("Rename completed but failed to save to database")
                invalidate_agents()

            previous_prompt = ""
            if not is_new:
                previous_agent = next(a for a in agents if a["name"] == old_name)
                previous_prompt = previous_agent["prompt"]

            # Persist provider_features: ensure the checkboxes are merged into JSON
            provider_features_to_save = state.get("provider_features") or {}
            # (The state was already updated above, but merge again to be safe)
            provider_features_to_save["structured_output"] = bool(st.session_state.get(f"agent_editor_pf_structured_{state.get('name','__unknown')}", provider_features_to_save.get("structured_output", False)))
            provider_features_to_save["tool_calling"] = bool(st.session_state.get(f"agent_editor_pf_tool_calling_{state.get('name','__unknown')}", provider_features_to_save.get("tool_calling", False)))
            provider_features_to_save["reasoning"] = bool(st.session_state.get(f"agent_editor_pf_reasoning_{state.get('name','__unknown')}", provider_features_to_save.get("reasoning", False)))

            # If host/port were provided and endpoint empty, ensure endpoint stored from host/port (last-resort)
            ep = state.get("endpoint")
            if not ep and show_host_port and state.get("endpoint_host"):
                ep = _compose_endpoint_from_host_port(state.get("endpoint_host"), state.get("endpoint_port"))
            try:
                get_agent_service().save_agent_atomic(
                    new_name,
                    state["model"],
                    state["prompt"],
                    state["role"],
                    state["input_vars"],
                    state["output_vars"],
                    color=state.get("color") or DEFAULT_COLOR,
                    symbol=state.get("symbol") or DEFAULT_SYMBOL,
                    save_prompt_version=(state["prompt"] != previous_prompt),
                    metadata={},
                    tools=state.get("tools")
                    or {"enabled": False, "tools": []},
                    reasoning_effort=state.get("reasoning_effort"),
                    reasoning_summary=state.get("reasoning_summary"),
                    system_prompt=state.get("system_prompt") or None,
                    provider_id=state.get("provider_id"),
                    model_class=state.get("model_class"),
                    endpoint=ep,
                    use_responses_api=bool(state.get("use_responses_api")),
                    provider_features=provider_features_to_save,
                    structured_output_enabled=bool(state.get("structured_output_enabled")),
                    schema_json=state.get("schema_json") or None,
                    schema_name=state.get("schema_name") or None,
                    temperature=state.get("temperature"),
                )
            except Exception:
                logger.exception("Failed to save agent")
                st.error("Failed to save agent to database")
                return

            invalidate_agents()
            reload_agents_into_engine()
            st.success("Agent saved")
            st.rerun()

    # -------------------------
    # Duplicate button
    # -------------------------
    with col_b:
        if not is_new and st.button("📄 Duplicate"):
            try:
                get_agent_service().save_agent(
                    f"{state['name']}_copy",
                    state["model"],
                    state["prompt"],
                    state["role"],
                    state["input_vars"],
                    state["output_vars"],
                    color=state.get("color") or DEFAULT_COLOR,
                    symbol=state.get("symbol") or DEFAULT_SYMBOL,
                    tools=state.get("tools")
                    or {"enabled": False, "tools": []},
                    reasoning_effort=state.get("reasoning_effort"),
                    reasoning_summary=state.get("reasoning_summary"),
                    system_prompt=state.get("system_prompt") or None,
                    provider_id=state.get("provider_id"),
                    model_class=state.get("model_class"),
                    endpoint=state.get("endpoint"),
                    use_responses_api=bool(state.get("use_responses_api")),
                    provider_features=state.get("provider_features"),
                    structured_output_enabled=bool(state.get("structured_output_enabled")),
                    schema_json=state.get("schema_json") or None,
                    schema_name=state.get("schema_name") or None,
                    temperature=state.get("temperature"),
                )
            except Exception:
                logger.exception("Failed to duplicate agent")
                st.error("Agent duplicated but failed to save to database")
            invalidate_agents()
            reload_agents_into_engine()
            st.success("Duplicated")
            st.rerun()

    # -------------------------
    # Delete button
    # -------------------------
    with col_c:
        if not is_new and st.button("🗑 Delete"):
            try:
                get_agent_service().delete_agent_atomic(state["name"])
            except Exception:
                logger.exception("Failed to delete agent")
                st.error("Failed to delete agent from database")
            invalidate_agents()
            reload_agents_into_engine()
            st.warning("Deleted")
            st.rerun()

    # -------------------------
    # Load from Template (no DB persistence)
    # -------------------------
    if is_new:
        with col_d:
            st.markdown("**Load from Template**")
            uploaded = st.file_uploader(
                "Upload agent template JSON",
                type=["json"],
                key="agent_template_file",
                help="Upload a JSON file containing one or more agents",
            )

            template_agents: List[dict] = []
            if uploaded is not None:
                try:
                    tpl_data = json.loads(uploaded.read().decode("utf-8"))
                    st.session_state["agent_template_data"] = tpl_data
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")
                    st.session_state["agent_template_data"] = None

            tpl_data = st.session_state.get("agent_template_data")
            if tpl_data:
                if isinstance(tpl_data, dict) and "agents" in tpl_data:
                    template_agents = tpl_data["agents"]
                elif isinstance(tpl_data, dict):
                    template_agents = [tpl_data]
                elif isinstance(tpl_data, list):
                    template_agents = tpl_data

                template_agents = [
                    a
                    for a in template_agents
                    if isinstance(a, dict)
                    and all(
                        k
                        in a
                        for k in (
                            "name",
                            "model",
                            "prompt_template",
                            "input_vars",
                            "output_vars",
                        )
                    )
                ]

            chosen_agent = None
            if template_agents:
                if len(template_agents) == 1:
                    chosen_agent = template_agents[0]
                else:
                    names_tpl = [a["name"] for a in template_agents]
                    chosen_name = st.selectbox(
                        "Choose agent from template",
                        names_tpl,
                        key="agent_template_choose_name",
                    )
                    chosen_agent = next(
                        a for a in template_agents if a["name"] == chosen_name
                    )

            if chosen_agent and st.button(
                "Apply Template", key="agent_template_apply"
            ):
                original_name = str(chosen_agent.get("name", "")).strip() or "imported_agent"
                suffix = datetime.now().strftime("_%y%m%d-%H%M_imported")
                imported_name = f"{original_name}{suffix}"

                state["selected_name"] = "<New Agent>"

                # Parse endpoint if template provided
                tpl_endpoint = chosen_agent.get("endpoint")
                host_tpl, port_tpl = _parse_endpoint_to_host_port(tpl_endpoint or "")

                state.update(
                    {
                        "name": imported_name,
                        "model": chosen_agent.get("model", "gpt-4.1-nano"),
                        "role": chosen_agent.get("role", ""),
                        "prompt": chosen_agent.get("prompt_template", ""),
                        "system_prompt": chosen_agent.get("system_prompt_template", "") or "",
                        "input_vars": chosen_agent.get("input_vars", []),
                        "output_vars": chosen_agent.get("output_vars", []),
                        "color": DEFAULT_COLOR,
                        "symbol": DEFAULT_SYMBOL,
                        "provider_id": chosen_agent.get("provider_id", "openai"),
                        "model_class": chosen_agent.get("model_class"),
                        "endpoint": tpl_endpoint,
                        "endpoint_host": host_tpl,
                        "endpoint_port": str(port_tpl) if port_tpl is not None else None,
                        "use_responses_api": chosen_agent.get("use_responses_api", True),
                        "provider_features": chosen_agent.get("provider_features") or {},
                    }
                )

                st.session_state.agent_editor_state_changed_this_run = True

                st.success(
                    "Template loaded into editor. Review and click Save to store the agent."
                )
                st.rerun()
