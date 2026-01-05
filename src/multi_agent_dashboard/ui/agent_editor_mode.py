# ui/agent_editor_mode.py
from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import List, Dict, Any

import streamlit as st

from multi_agent_dashboard.config import UI_COLORS, AGENT_SNAPSHOT_PRUNE_KEEP
from multi_agent_dashboard.ui.cache import (
    cached_load_agents,
    cached_load_prompt_versions,
    cached_load_agent_snapshots,
    get_agent_service,
    invalidate_agents,
)
from multi_agent_dashboard.ui.bootstrap import reload_agents_into_engine

logger = logging.getLogger(__name__)

DEFAULT_COLOR = UI_COLORS["default"]["value"]
DEFAULT_SYMBOL = UI_COLORS["default"]["symbol"]


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
            }
        else:
            base_agent = next(a for a in agents if a["name"] == selected)

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
            "📚 Versions",
        ]
    )

    basics_tab, prompt_tab, io_tab, adv_tab, versions_tab = tabs

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

    # ----- Versions tab -----
    with versions_tab:
        if not is_new:
            versions = cached_load_prompt_versions(state["name"])
            for v in versions:
                with st.expander(
                    f"Version {v['version']} — {v['created_at']}",
                ):
                    st.code(v["prompt"])
                    if v.get("metadata"):
                        st.json(v["metadata"])

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
        st.markdown("### Tools")

        tools_state = state.get("tools") or {}
        tools_enabled_default = bool(tools_state.get("enabled"))
        selected_tools_default = tools_state.get("tools") or []

        tools_enabled = st.checkbox(
            "Enable tool calling",
            value=tools_enabled_default,
            help="Allow this agent to call tools such as web search.",
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
        reasoning_effort_val = st.selectbox(
            "Reasoning Effort",
            options=effort_options,
            index=effort_options.index(current_effort)
            if current_effort in effort_options
            else effort_options.index("medium"),
            help="Controls depth & cost. 'none' disables special reasoning behavior.",
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
        )

    st.divider()

    # -------------------------
    # Reflect edits into state
    # -------------------------
    if not state_changed_this_run:
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
            )

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
                    }
                )

                st.session_state.agent_editor_state_changed_this_run = True

                st.success(
                    "Template loaded into editor. Review and click Save to store the agent."
                )
                st.rerun()
