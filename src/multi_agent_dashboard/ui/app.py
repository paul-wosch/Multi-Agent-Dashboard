# ui/app.py
import graphviz
import streamlit as st
from openai import OpenAI  # type/factory only
import difflib
import json
import inspect
from dataclasses import asdict
from datetime import datetime, UTC
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from multi_agent_dashboard.config import OPENAI_API_KEY, DB_FILE_PATH, configure_logging

from multi_agent_dashboard.db.db import init_db
from multi_agent_dashboard.db.runs import RunDAO
from multi_agent_dashboard.db.agents import AgentDAO
from multi_agent_dashboard.db.pipelines import PipelineDAO
from multi_agent_dashboard.db.services import RunService, AgentService, PipelineService

from multi_agent_dashboard.engine import MultiAgentEngine, EngineResult
from multi_agent_dashboard.llm_client import LLMClient
from multi_agent_dashboard.models import AgentSpec

logger = logging.getLogger(__name__)

# =======================
# Configuration
# =======================
DB_PATH = DB_FILE_PATH

pipeline_svc = PipelineService(DB_PATH)
agent_svc = AgentService(DB_PATH)
run_svc = RunService(DB_PATH)

LOG_LEVEL_COLORS = {
    "DEBUG": "#6c757d",     # gray
    "INFO": "#198754",      # green
    "WARNING": "#fd7e14",   # orange
    "ERROR": "#dc3545",     # red
    "CRITICAL": "#842029",  # dark red
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20 MB

# =======================
# DEFAULT AGENTS (Backward-compatible examples)
# Each entry now includes metadata fields to show examples of self-description.
# =======================
default_agents = {
    "planner": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Planner Agent.
Refrain from producing the actual solution.
Only clarify the task and produce steps.

Task:
{task}

Output:
- Clarified Task
- Plan
""",
        "role": "planner",  # meta label (informational only)
        "input_vars": ["task"],
        "output_vars": ["plan"]
    },

    "solver": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Solver Agent.
Use the plan to produce an answer.

Plan:
{plan}

Output:
- Answer
""",
        "role": "solver",
        "input_vars": ["plan"],
        "output_vars": ["answer"]
    },

    "critic": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Critic Agent.
Evaluate the answer.

Answer:
{answer}

Output:
- Issues
- Improvements
""",
        "role": "critic",
        "input_vars": ["answer"],
        "output_vars": ["critique"]
    },

    "finalizer": {
        "model": "gpt-4.1-nano",
        "prompt_template": """
You are the Finalizer Agent.
You must revise the original answer using the critique.

Original Answer:
{answer}

Critique:
{critique}

Your task:
Return the improved final answer only.
""",
        "role": "finalizer",
        "input_vars": ["answer", "critique"],
        "output_vars": ["final"]
    },
}


# =======================
# CACHED WRAPPERS
# =======================

@st.cache_data(ttl=60)
def cached_load_agents():
    return agent_svc.list_agents()

@st.cache_data(ttl=60)
def cached_load_pipelines():
    return pipeline_svc.list_pipelines()

@st.cache_data(ttl=30)
def cached_load_runs():
    return run_svc.list_runs()

@st.cache_data(ttl=30)
def cached_load_run_details(run_id: int):
    return run_svc.get_run_details(run_id)

@st.cache_data(ttl=60)
def cached_load_prompt_versions(agent_name: str):
    return agent_svc.load_prompt_versions(agent_name)

# =======================
# LOG HANDLER CLASS
# =======================

class StreamlitLogHandler(logging.Handler):
    """
    Logging handler that stores recent log records in Streamlit session_state.
    """

    def __init__(self, capacity: int = 500):
        super().__init__()
        self.capacity = capacity

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "time": (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(record.created)
                        )
                        + f",{int(record.msecs):03d}"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            logs = st.session_state.setdefault(
                "_log_buffer",
                deque(maxlen=self.capacity),
            )
            logs.append(entry)
        except Exception:
            # Never let logging break the app
            pass


# =======================
# CACHE INVALIDATION HELPER
# =======================

def invalidate_caches(*names: str):
    """
    Centralized cache invalidation helper.
    Allowed names:
      - agents
      - prompt_versions
      - pipelines
      - runs
      - run_details
      - all
    """
    if "all" in names:
        cached_load_agents.clear()
        cached_load_prompt_versions.clear()
        cached_load_pipelines.clear()
        cached_load_runs.clear()
        cached_load_run_details.clear()
        return

    for name in names:
        if name == "agents":
            cached_load_agents.clear()
        elif name == "prompt_versions":
            cached_load_prompt_versions.clear()
        elif name == "pipelines":
            cached_load_pipelines.clear()
        elif name == "runs":
            cached_load_runs.clear()
        elif name == "run_details":
            cached_load_run_details.clear()


def invalidate_agent_state():
    invalidate_caches("agents", "prompt_versions", "pipelines")


def invalidate_run_state():
    invalidate_caches("runs", "run_details")


# =======================
# OpenAI client factory
# =======================
def create_openai_client(api_key: str):
    """Factory to create an OpenAI client. Allows tests to replace this factory or pass fake client."""
    return OpenAI(api_key=api_key)


# ======================================================
# SHARED HELPERS (USED BY APP_START AND UI)
# ======================================================

def pipeline_requires_files(engine, steps) -> bool:
    for name in steps:
        agent = engine.agents.get(name)
        if not agent:
            continue
        if "files" in agent.spec.input_vars:
            return True
    return False


def render_agent_graph(steps: list[str]):
    dot = graphviz.Digraph()
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#6baed6",
        fillcolor="#deebf7"
    )

    for agent in steps:
        if agent in st.session_state.engine.agents:
            role = st.session_state.engine.agents[agent].spec.role
            label = f"{agent}\n({role})" if role else agent
        else:
            label = agent

        dot.node(agent, label)

    for i in range(len(steps) - 1):
        dot.edge(steps[i], steps[i + 1], label="passes state →")

    return dot


def reload_agents_into_engine():
    "Helper to reload agents into the engine."
    engine = st.session_state.engine
    engine.agents.clear()

    stored_agents = cached_load_agents()
    for a in stored_agents:
        spec = AgentSpec(
            name=a["agent_name"],
            model=a["model"],
            prompt_template=a["prompt_template"],
            role=a["role"],
            input_vars=a["input_vars"],
            output_vars=a["output_vars"],
        )
        engine.add_agent(spec)


def export_pipeline_agents_as_json(pipeline_name: str, steps: List[str]) -> str:
    """
    Export all agents used in a pipeline as a JSON string.
    """
    engine = st.session_state.engine

    agents_payload = []
    for agent_name in steps:
        agent = engine.agents.get(agent_name)
        if not agent:
            continue
        agents_payload.append(asdict(agent.spec))

    export = {
        "pipeline": pipeline_name,
        "exported_at": datetime.now(UTC).isoformat(),
        "agents": agents_payload,
    }

    return json.dumps(export, indent=2)


# ======================================================
# INITIALIZE DB AND PREPARE GUI
# ======================================================

# Initialize DB + bootstrap defaults if agents table empty and populate Streamlit session with engine
def bootstrap_default_agents(defaults: Dict[str, dict]):
    existing = cached_load_agents()
    if existing:
        return
    for name, data in defaults.items():
        agent_svc.save_agent(
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", [])
        )
        # also save a versioned prompt snapshot
        agent_svc.save_prompt_version(
            name,
            data.get("prompt_template", ""),
            metadata={"role": data.get("role", "")}
        )


def app_start():
    """
    Explicit application bootstrap. Call this once when running the Streamlit app.
    - initializes DB and migrations
    - creates an OpenAI client via factory
    - creates the MultiAgentEngine with the injected client
    - bootstraps default agents if DB empty
    - loads agents from DB into the engine
    - stores engine into st.session_state
    """
    # Attach Streamlit log handler once per session
    if "_log_handler_attached" not in st.session_state:
        handler = StreamlitLogHandler(capacity=500)
        logging.getLogger().addHandler(handler)
        st.session_state["_log_handler_attached"] = True

    # Initialize DB and apply migrations
    init_db(DB_PATH)

    # create OpenAI client (factory)
    openai_client = create_openai_client(OPENAI_API_KEY)
    llm_client = LLMClient(openai_client)

    # create engine with injected client
    engine = MultiAgentEngine(
        llm_client=llm_client
    )
    st.session_state.engine = engine

    # Ensure default agents exist in DB (only if table empty)
    bootstrap_default_agents(default_agents)

    # Load agents from DB into engine
    stored_agents = cached_load_agents()
    for a in stored_agents:
        spec = AgentSpec(
            name=a["agent_name"],
            model=a["model"],
            prompt_template=a["prompt_template"],
            role=a["role"],
            input_vars=a["input_vars"],
            output_vars=a["output_vars"],
        )
        engine.add_agent(spec)

    # Save engine into session state
    st.session_state.engine = engine
    # Optionally keep client in session state for custom use
    st.session_state.llm_client = llm_client


configure_logging()

# Only start the app if we don't already have an engine in session state.
# This prevents double initialization during reruns.
if "engine" not in st.session_state:
    # For Streamlit invocation, app_start() should be called here so side-effects
    # happen only when we actually run the app (not at import time by a test runner).
    if "engine" not in st.session_state:
        app_start()
    else:
        reload_agents_into_engine()

# Initialize Ad-hoc pipeline state (empty on first app start)
if "adhoc_pipeline_steps" not in st.session_state:
    st.session_state.adhoc_pipeline_steps = []

# Also provide the conventional guard so running the script directly will bootstrap.
if __name__ == "__main__":
    # If run as a script, ensure app_start was executed (idempotent).
    if "engine" not in st.session_state:
        app_start()


# ======================================================
# GUI START (REFACTORED UI ONLY)
# ======================================================

st.set_page_config(
    page_title="Multi-Agent Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Wrap code blocks */
    pre code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: auto !important;
    }

    /* Wrap Streamlit code blocks */
    .stCodeBlock pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap JSON output */
    .stJson pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Wrap text areas */
    textarea {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }

    /* Prevent horizontal scroll everywhere */
    section.main {
        overflow-x: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Multi-Agent Pipeline Dashboard")
st.caption("Design pipelines, run tasks, inspect agent behavior, and manage prompt versions.")

# ======================================================
# MODES
# ======================================================

MODE_RUN = "▶️ Run Pipeline"
MODE_AGENTS = "🧠 Manage Agents"
MODE_HISTORY = "📜 History"
MODE_LOGS = "🪵 Logs"

mode = st.radio(
    "Mode",
    [MODE_RUN, MODE_AGENTS, MODE_HISTORY, MODE_LOGS],
    horizontal=True
)

strict_mode = st.sidebar.checkbox(
    "Strict output validation",
    value=False,
    help="Fail fast on agent output mismatches"
)

st.divider()

# ======================================================
# ---------- RUN MODE ----------
# ======================================================

def render_run_sidebar():
    st.sidebar.header("Run Configuration")

    # -------------------------
    # Load pipelines
    # -------------------------
    pipelines = cached_load_pipelines()
    pipeline_names = [p["pipeline_name"] for p in pipelines]

    selected_pipeline = st.sidebar.selectbox(
        "Pipeline",
        ["<Ad-hoc>"] + pipeline_names
    )

    # -------------------------
    # Task input
    # -------------------------
    task = st.sidebar.text_area(
        "Task",
        placeholder="Describe the task you want the agents to solve…",
        height=120
    )

    # -------------------------
    # Resolve base steps
    # -------------------------
    engine = st.session_state.engine
    available_agents = list(engine.agents.keys())

    if selected_pipeline != "<Ad-hoc>":
        base_steps = next(
            p["steps"] for p in pipelines
            if p["pipeline_name"] == selected_pipeline
        )
    else:
        # Use stored Ad-hoc pipeline steps from session state
        base_steps = st.session_state.get("adhoc_pipeline_steps", [])

    # -------------------------
    # Agent selection (SOURCE OF TRUTH)
    # -------------------------
    selected_steps = st.sidebar.multiselect(
        "Agents (execution order)",
        available_agents,
        default=base_steps
    )

    # Persist Ad-hoc steps so they survive pipeline switching & reruns
    if selected_pipeline == "<Ad-hoc>":
        st.session_state.adhoc_pipeline_steps = selected_steps

    # -------------------------
    # Detect file requirement (from selected steps!)
    # -------------------------
    requires_files = pipeline_requires_files(
        engine,
        selected_steps,
    )

    # -------------------------
    # File attachment section
    # -------------------------
    files_payload = None

    if requires_files:
        with st.sidebar.expander("📎 Attach files", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload files",
                accept_multiple_files=True,
                type=["txt", "pdf", "csv", "md", "json", "log", "py"],
            )

            files_payload = []
            total_size = 0

            for f in uploaded_files or []:
                if f.size > MAX_FILE_SIZE:
                    st.error(f"{f.name} exceeds 5MB limit")
                    return

                total_size += f.size
                if total_size > MAX_TOTAL_SIZE:
                    st.error("Total file size exceeds 20MB")
                    return

                files_payload.append({
                    "filename": f.name,
                    "content": f.read(),
                    "mime_type": f.type,
                })

    # -------------------------
    # Run button
    # -------------------------
    run_clicked = st.sidebar.button(
        "🚀 Run Pipeline",
        use_container_width=True
    )

    # -------------------------
    # Advanced: Pipeline editing
    # -------------------------
    with st.sidebar.expander("Advanced", expanded=False):
        """
        available_agents = list(st.session_state.engine.agents.keys())

        if selected_pipeline != "<Ad-hoc>":
            steps = next(
                p["steps"] for p in pipelines
                if p["pipeline_name"] == selected_pipeline
            )
        else:
            steps = available_agents

        selected_steps = st.multiselect(
            "Agents (execution order)",
            available_agents,
            default=active_steps,
        )
        """
        name = st.text_input("Save as Pipeline")

        if st.button("💾 Save Pipeline"):
            if name.strip():
                try:
                    pipeline_svc.save_pipeline(name.strip(), selected_steps)
                except Exception:
                    logger.exception("Failed to persist pipeline")
                    st.error("Failed to save pipeline to database")
                invalidate_agent_state()
                st.success("Pipeline saved")
                st.rerun()

        st.divider()

        if selected_pipeline != "<Ad-hoc>" and selected_steps:
            agents_json = export_pipeline_agents_as_json(
                selected_pipeline,
                selected_steps
            )

            st.download_button(
                label="⬇️ Download Pipeline Agents (JSON)",
                data=agents_json,
                file_name=f"{selected_pipeline}_agents.json",
                mime="application/json",
                use_container_width=True,
            )

    # -------------------------
    # Return values
    # -------------------------
    return (
        selected_pipeline,
        selected_steps,
        task,
        run_clicked,
        files_payload,
    )


def render_warnings(result: EngineResult):
    if not result.warnings:
        return

    st.warning("⚠️ Pipeline Warnings")
    for w in result.warnings:
        st.markdown(f"- {w}")


def render_final_output(result: EngineResult):
    if not result.final_output:
        st.info("No output yet.")
        return

    st.subheader("Final Output")

    try:
        parsed = json.loads(result.final_output)
        st.json(parsed)
    except Exception:
        # JSON detection failed – enable markdown/code toggle
        view = st.radio(
            "View as",
            ["Markdown", "Code"],
            horizontal=True,
            key="run_final_view"
        )
        if view == "Markdown":
            st.markdown(result.final_output)
        else:
            st.code(result.final_output)



def render_agent_outputs(result: EngineResult, steps):
    for agent in steps:
        with st.expander(f"🔹 {agent}"):
            out = result.memory.get(agent, "")
            try:
                parsed = json.loads(out)
                st.json(parsed)
            except Exception:
                # JSON detection failed – enable markdown/code toggle
                view = st.radio(
                    "View as",
                    ["Markdown", "Code"],
                    horizontal=True,
                    key=f"run_agent_{agent}_view"
                )
                if view == "Markdown":
                    st.markdown(out)
                else:
                    st.code(out)


def render_graph_tab(steps):
    if not steps:
        st.info("No agents selected.")
        return
    st.graphviz_chart(render_agent_graph(steps))


def render_compare_tab(result: EngineResult, steps):
    col1, col2 = st.columns(2)

    with col1:
        a1 = st.selectbox("Agent A", steps, key="cmp_a")

    with col2:
        a2 = st.selectbox("Agent B", steps, key="cmp_b")

    if a1 != a2:
        out1 = str(result.memory.get(a1, ""))
        out2 = str(result.memory.get(a2, ""))

        diff = difflib.unified_diff(
            out1.splitlines(),
            out2.splitlines(),
            fromfile=a1,
            tofile=a2,
            lineterm=""
        )

        st.code("\n".join(diff), language="diff")


def render_run_results(result: EngineResult, steps):
    tabs = st.tabs([
        "🟢 Final Output",
        "⚠️ Warnings",
        "📁 Agent Outputs",
        "🧩 Graph",
        "🔍 Compare"
    ])

    with tabs[0]:
        render_final_output(result)

    with tabs[1]:
        render_warnings(result)

    with tabs[2]:
        render_agent_outputs(result, steps)

    with tabs[3]:
        render_graph_tab(steps)

    with tabs[4]:
        render_compare_tab(result, steps)


def render_run_mode():
    pipeline, steps, task, run_clicked, files_payload = render_run_sidebar()

    if run_clicked:
        # Create progress UI *only when running*
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(pct: int, agent_name: str | None = None):
            progress_bar.progress(pct)
            if agent_name:
                progress_text.caption(f"Running {agent_name} — {pct}%")
            else:
                progress_text.caption(f"Pipeline progress: {pct}%")

        engine = st.session_state.engine
        engine.on_progress = update_progress

        with st.spinner("Running pipeline…"):
            result: EngineResult = engine.run_seq(
                steps=steps,
                initial_input=task,
                strict=strict_mode,
                files=files_payload if files_payload else None,
            )

            st.session_state.last_result = result
            st.session_state.last_steps = steps

        progress_bar.progress(100)
        progress_text.caption("Pipeline completed ✅")

        agent_models = {
            name: runtime.spec.model
            for name, runtime in engine.agents.items()
        }

        final_model = (
            agent_models.get(result.final_agent)
            if result.final_agent
            else None
        )

        try:
            run_svc.save_run(
                task,
                result.final_output,
                result.memory,
                agent_models=agent_models,
                final_model=final_model,
            )
        except Exception:
            logger.exception("Failed to persist run")
            st.error("Run completed but failed to save to database")
        invalidate_run_state()

        st.success("Pipeline completed!")

        # Inline warning banner
        if "__warnings__" in st.session_state.engine.memory:
            st.warning(
                f"{len(st.session_state.engine.memory['__warnings__'])} warning(s) occurred during execution."
            )

    if "last_result" in st.session_state:
        render_run_results(
            st.session_state.last_result,
            st.session_state.last_steps,
        )


# ======================================================
# ---------- AGENT EDITOR MODE ----------
# ======================================================

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
        }
        for a in agents_raw
    ]
    names = [a["name"] for a in agents]

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
            "input_vars": [],
            "output_vars": [],
        }
    state = st.session_state.agent_editor_state

    # Persistent flag to survive st.rerun
    if "agent_editor_state_changed_this_run" not in st.session_state:
        st.session_state.agent_editor_state_changed_this_run = False

    # Track if we loaded a different agent/template in this run
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
                "input_vars": [],
                "output_vars": [],
            }
        else:
            base_agent = next(a for a in agents if a["name"] == selected)

        state.update({
            "name": base_agent["name"],
            "model": base_agent["model"],
            "role": base_agent["role"],
            "prompt": base_agent["prompt"],
            "input_vars": base_agent["input_vars"],
            "output_vars": base_agent["output_vars"],
        })

        # mark as changed in both local and session state
        state_changed_this_run = True
        st.session_state.agent_editor_state_changed_this_run = True

    is_new = (state.get("selected_name") == "<New Agent>")

    # -------------------------
    # Tabs: Basics / Prompt / IO / Versions
    # -------------------------
    tabs = st.tabs([
        "1️⃣ Basics",
        "2️⃣ Prompt",
        "3️⃣ Inputs / Outputs",
        "📚 Versions"
    ])

    # ----- Basics tab -----
    with tabs[0]:
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

    # ----- Prompt tab -----
    with tabs[1]:
        prompt_val = st.text_area(
            "Prompt Template",
            height=400,
            value=state["prompt"],
        )

    # ----- Inputs / Outputs tab -----
    with tabs[2]:
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
    with tabs[3]:
        if not is_new:
            versions = cached_load_prompt_versions(state["name"])
            for v in versions:
                with st.expander(f"Version {v['version']} — {v['created_at']}"):
                    st.code(v["prompt"])
                    if v.get("metadata"):
                        st.json(v["metadata"])

    st.divider()

    # -------------------------
    # Reflect edits into state
    # -------------------------
    # Only write back widget values if we did NOT just load a different agent/template.
    if not state_changed_this_run:
        state.update({
            "name": name_val.strip(),
            "model": model_val.strip() or "gpt-4.1-nano",
            "role": role_val.strip(),
            "prompt": prompt_val,
            "input_vars": [
                v.strip() for v in input_vars_val.splitlines() if v.strip()
            ],
            "output_vars": [
                v.strip() for v in output_vars_val.splitlines() if v.strip()
            ],
        })

    # IMPORTANT: reset the persistent flag at the end of the render
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
                    agent_svc.rename_agent_atomic(old_name, new_name)
                except Exception:
                    logger.exception("Failed to rename agent")
                    st.error("Rename completed but failed to save to database")
                invalidate_agent_state()

            # Save agent
            previous_prompt = ""
            if not is_new:
                previous_agent = next(a for a in agents if a["name"] == old_name)
                previous_prompt = previous_agent["prompt"]

            agent_svc.save_agent_atomic(
                new_name,
                state["model"],
                state["prompt"],
                state["role"],
                state["input_vars"],
                state["output_vars"],
                save_prompt_version=(state["prompt"] != previous_prompt),
            )

            invalidate_agent_state()
            reload_agents_into_engine()
            st.success("Agent saved")
            st.rerun()

    # -------------------------
    # Duplicate button
    # -------------------------
    with col_b:
        if not is_new and st.button("📄 Duplicate"):
            try:
                agent_svc.save_agent(
                    f"{state['name']}_copy",
                    state["model"],
                    state["prompt"],
                    state["role"],
                    state["input_vars"],
                    state["output_vars"],
                )
            except Exception:
                logger.exception("Failed to duplicate agent")
                st.error("Agent duplicated but failed to save to database")
            invalidate_agent_state()
            reload_agents_into_engine()
            st.success("Duplicated")
            st.rerun()

    # -------------------------
    # Delete button
    # -------------------------
    with col_c:
        if not is_new and st.button("🗑 Delete"):
            try:
                agent_svc.delete_agent_atomic(state["name"])
            except Exception:
                logger.exception("Failed to delete agent")
                st.error("Failed to delete agent from database")
            invalidate_agent_state()
            reload_agents_into_engine()
            st.warning("Deleted")
            st.rerun()

    # -------------------------
    # Load from Template (no DB persistence)
    # -------------------------
    # Show this only when creating a new agent
    if is_new:
        with col_d:
            st.markdown("**Load from Template**")
            uploaded = st.file_uploader(
                "Upload agent template JSON",
                type=["json"],
                key="agent_template_file",
                help="Upload a JSON file containing one or more agents"
            )

            template_agents = []
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
                    a for a in template_agents
                    if isinstance(a, dict)
                    and all(k in a for k in ("name", "model", "prompt_template", "input_vars", "output_vars"))
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
                    chosen_agent = next(a for a in template_agents if a["name"] == chosen_name)

            if chosen_agent and st.button("Apply Template", key="agent_template_apply"):
                original_name = str(chosen_agent.get("name", "")).strip() or "imported_agent"
                suffix = datetime.now().strftime("_%y%m%d-%H%M_imported")
                imported_name = f"{original_name}{suffix}"

                # Treat as a new agent in the editor (not yet saved)
                state["selected_name"] = "<New Agent>"
                # THIS WON'T WORK: keep selectbox in sync so it doesn't reload another agent
                # st.session_state.agent_editor_selected_agent = "<New Agent>"

                state.update({
                    "name": imported_name,
                    "model": chosen_agent.get("model", "gpt-4.1-nano"),
                    "role": chosen_agent.get("role", ""),
                    "prompt": chosen_agent.get("prompt_template", ""),
                    "input_vars": chosen_agent.get("input_vars", []),
                    "output_vars": chosen_agent.get("output_vars", []),
                })

                # Mark change in persistent flag so next run skips write-back
                st.session_state.agent_editor_state_changed_this_run = True

                st.success("Template loaded into editor. Review and click Save to store the agent.")
                st.rerun()


# ======================================================
# ---------- HISTORY MODE ----------
# ======================================================

def render_history_mode():
    st.header("📜 Past Runs")

    runs = cached_load_runs()

    options = {
        f"Run {r['id']} — {r['timestamp']}": r["id"]
        for r in runs
    }

    selected = st.selectbox(
        "Select Run",
        ["None"] + list(options.keys())
    )

    if selected == "None":
        return

    run_id = options[selected]
    run, agents = cached_load_run_details(run_id)

    ts = run["timestamp"]
    task = run["task_input"]
    final = run["final_output"]
    final_is_json = run["final_is_json"]
    final_model = run["final_model"]

    st.subheader(f"Run {run_id}")
    st.code(task)

    header = "Final Output"
    if final_model:
        header += f" · {final_model}"

    with st.expander(header):
        if final_is_json:
            try:
                st.json(json.loads(final))
            except Exception:
                st.warning("⚠️ Final output marked as JSON but failed to parse")
                # JSON detection failed – enable markdown/code toggle
                view = st.radio(
                    "View as",
                    ["Markdown", "Code"],
                    horizontal=True,
                    key=f"hist_run_{run_id}_final_view"
                )
                if view == "Markdown":
                    st.markdown(final)
                else:
                    st.code(final)
        else:
            # Not JSON – enable markdown/code toggle
            view = st.radio(
                "View as",
                ["Markdown", "Code"],
                horizontal=True,
                key=f"hist_run_{run_id}_final_view"
            )
            if view == "Markdown":
                st.markdown(final)
            else:
                st.code(final)

    for a in agents:
        name = a["agent_name"]
        output = a["output"]
        is_json = a["is_json"]
        model = a["model"]
        header = f"{name}"
        if model:
            header += f" · {model}"

        with st.expander(header):
            if is_json:
                try:
                    st.json(json.loads(output))
                except Exception:
                    st.warning("⚠️ Output marked as JSON but failed to parse")
                    # JSON detection failed – enable markdown/code toggle
                    view = st.radio(
                        "View as",
                        ["Markdown", "Code"],
                        horizontal=True,
                        key=f"hist_run_{run_id}_{name}_view"
                    )
                    if view == "Markdown":
                        st.markdown(output)
                    else:
                        st.code(output)
            else:
                # Not JSON – enable markdown/code toggle
                view = st.radio(
                    "View as",
                    ["Markdown", "Code"],
                    horizontal=True,
                    key=f"hist_run_{run_id}_{name}_view"
                )
                if view == "Markdown":
                    st.markdown(output)
                else:
                    st.code(output)

    export = {
        "run_id": run_id,
        "timestamp": ts,
        "task_input": task,
        "final_output": {
            "output": final,
            "is_json": bool(final_is_json),
            "model": final_model,
            },
        "agents": {
            a["agent_name"]: {
                "output": a["output"],
                "is_json": bool(a["is_json"]),
                "model": a["model"],
            }
            for a in agents
        }
    }

    st.download_button(
        "Download Run as JSON",
        data=json.dumps(export, indent=2),
        file_name=f"run_{run_id}.json",
        mime="application/json"
    )


# ======================================================
# ---------- LOGS MODE ----------
# ======================================================

def render_logs_mode():
    st.header("🪵 Application Logs")

    logs = st.session_state.get("_log_buffer", [])

    if not logs:
        st.info("No logs yet.")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        level_filter = st.multiselect(
            "Levels",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        )

        def build_log_lines(logs, level_filter, search):
            lines = []
            for entry in logs:
                if entry["level"] not in level_filter:
                    continue
                if search and search.lower() not in entry["message"].lower():
                    continue

                lines.append(
                    f"{entry['time']} "
                    f"[{entry['level']}] "
                    f"{entry['logger']}: "
                    f"{entry['message']}"
                )
            return "\n".join(lines)

        search = st.text_input("Search")

        if st.button("🧹 Clear logs"):
            logs.clear()
            st.rerun()

        export_text = build_log_lines(
            logs=list(logs),
            level_filter=level_filter,
            search=search,
        )

        st.download_button(
            label="⬇️ Download logs",
            data=export_text,
            file_name="application.log",
            mime="text/plain",
            disabled=not bool(export_text),
        )

    with col1:
        for entry in reversed(logs):
            if entry["level"] not in level_filter:
                continue
            if search and search.lower() not in entry["message"].lower():
                continue

            level = entry["level"]
            color = LOG_LEVEL_COLORS.get(level, "#000000")

            prefix = f"{entry['time']} "
            level_token = f"[{level}]"
            suffix = f" {entry['logger']}: {entry['message']}"

            st.markdown(
                f"""
            <pre style="
                margin: 0;
                padding: 6px 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
            ">
            {prefix}<span style="color:{color}; font-weight:bold;">{level_token}</span>{suffix}
            </pre>
            """,
                unsafe_allow_html=True,
            )


# ======================================================
# DEV-TIME SAFETY CHECKS
# ======================================================

for fn in [
    cached_load_runs,
    cached_load_run_details,
    render_agent_graph,
    reload_agents_into_engine,
]:
    assert callable(fn), f"{fn.__name__} is not defined"

# ======================================================
# MODE ROUTER
# ======================================================

if mode == MODE_RUN:
    render_run_mode()

elif mode == MODE_AGENTS:
    render_agent_editor()

elif mode == MODE_HISTORY:
    render_history_mode()

elif mode == MODE_LOGS:
    render_logs_mode()
