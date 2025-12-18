import graphviz
import streamlit as st
from openai import OpenAI  # still imported for type / factory use
import difflib
import json
from dataclasses import asdict
from datetime import datetime, UTC
import time
from config import OPENAI_API_KEY, DB_FILE_PATH, MIGRATIONS_PATH, configure_logging
from db.db import (
    init_db,
    load_agents_from_db,
    load_pipelines_from_db,
    load_runs,
    load_run_details,
    load_prompt_versions,
    save_run_to_db,
    save_agent_to_db,
    save_prompt_version,
    save_pipeline_to_db,
    delete_pipeline_from_db,
    rename_agent_in_db,
    delete_agent
)
# from db.migrations import apply_migrations
from utils import safe_format
from llm_client import LLMClient, TextResponse
from models import AgentSpec, AgentRuntime
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

# =======================
# Configuration
# =======================
DB_PATH = DB_FILE_PATH
# NOTE: removed global OpenAI client creation and removed DB init/migration at import time.
# Use app_start() to perform initialization when running the app.

LOG_LEVEL_COLORS = {
    "DEBUG": "#6c757d",     # gray
    "INFO": "#198754",      # green
    "WARNING": "#fd7e14",   # orange
    "ERROR": "#dc3545",     # red
    "CRITICAL": "#842029",  # dark red
}

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
        load_agents_from_db.clear()
        load_prompt_versions.clear()
        load_pipelines_from_db.clear()
        load_runs.clear()
        load_run_details.clear()
        return

    for name in names:
        if name == "agents":
            load_agents_from_db.clear()
        elif name == "prompt_versions":
            load_prompt_versions.clear()
        elif name == "pipelines":
            load_pipelines_from_db.clear()
        elif name == "runs":
            load_runs.clear()
        elif name == "run_details":
            load_run_details.clear()


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


# =======================
# MultiAgentEngine with client injection
# =======================
class MultiAgentEngine:
    def __init__(self, llm_client: LLMClient):
        # agents: name -> Agent
        self.agents: Dict[str, AgentRuntime] = {}
        # shared structured state used by pipelines
        self.state: Dict[str, Any] = {}
        # memory: agent_name -> raw agent output (string or parsed)
        self.memory: Dict[str, Any] = {}
        # injected OpenAI client instance
        self.llm_client = llm_client

    def add_agent(self, spec: AgentSpec):
        self.agents[spec.name] = AgentRuntime(
            spec=spec,
            llm_client=self.llm_client,
        )

    def remove_agent(self, name: str):
        self.agents.pop(name, None)

    def update_agent_prompt(self, name: str, new_prompt: str):
        if name in self.agents:
            self.agents[name].spec.prompt_template = new_prompt

    def _log_or_raise(self, strict: bool, agent_name: str, message: str):
        """
        Internal helper for handling writeback contract violations.
        """
        if strict:
            raise ValueError(message)
        self.memory.setdefault("__warnings__", []).append(
            f"[{agent_name}] {message}"
        )

    def run_seq(
            self,
            steps: List[str],
            initial_input: Any,
            *,
            on_progress: Optional[callable] = None,
            strict: bool = False,
    ) -> Any:
        """
        Sequential pipeline executor with explicit, safe state writeback rules.

        - initialize shared state with initial_input available under 'task' and 'input' keys
        - for each agent in steps, get declared input_vars, run agent with subset of state,
          then write back outputs to state using the agent's declared output_vars.
        - memory stores raw outputs keyed by agent name.
        - final output returned is state.get('final') if present, else the last agent's output.

        Writeback contract:
        - If agent.spec.output_vars is declared:
            * JSON dict → keys must match output_vars
            * Non-JSON:
                - len(output_vars) == 1 → assign raw text
                - len(output_vars) > 1 → assign `{agent.name}__raw`
        - If agent.spec.output_vars is empty:
            * Output written to state[agent.name]
        """
        logger.info("Starting pipeline: %s", steps)

        # Initialize shared state dictionary
        self.state = {}
        # Always keep the original user input under 'task' and 'input' for backward compatibility
        self.state['task'] = initial_input
        self.state['input'] = initial_input

        self.memory = {}

        last_output = None
        num_agents = len(steps)
        base = 100 / (2 * num_agents) if num_agents else 100

        for i, agent_name in enumerate(steps):
            logger.debug("Pipeline step start: %s", agent_name)
            # Progress: agent start
            if on_progress:
                on_progress(
                    int(base * (2 * i + 1)),
                    agent_name=agent_name,
                )

            agent = self.agents.get(agent_name)
            if not agent:
                # skip unknown agent but record a warning in memory
                msg = f"Agent '{agent_name}' not registered"
                self.memory[agent_name] = msg
                self._log_or_raise(strict, agent_name, msg)
                continue

            # === INPUT CONTRACT VALIDATION ===
            if agent.spec.input_vars:
                for var in agent.spec.input_vars:
                    if var not in self.state or self.state.get(var) in ("", None):
                        self._log_or_raise(
                            strict,
                            agent_name,
                            f"Input var '{var}' is missing or empty when '{agent_name}' ran"
                        )

            # Run agent with generic state (Agent will pick only input_vars if it declares them)
            try:
                raw_output = agent.run(self.state)
            except Exception:
                logger.exception("Agent %s execution failed", agent_name)
                raise
            # Store raw output in memory
            self.memory[agent_name] = raw_output
            last_output = raw_output

            # Attempt JSON parse
            try:
                parsed = LLMClient.safe_json(raw_output)
            except Exception:
                parsed = None

            # === WRITEBACK RULES ===

            if agent.spec.output_vars:
                # Case 1: structured JSON output
                if isinstance(parsed, dict):
                    logger.debug(
                        "Agent %s returned JSON keys: %s",
                        agent_name,
                        list(parsed.keys())
                    )
                    for key in parsed.keys():
                        if key not in agent.spec.output_vars:
                            self._log_or_raise(
                                strict,
                                agent_name,
                                f"Unexpected output key '{key}' not declared in output_vars"
                            )

                    for var in agent.spec.output_vars:
                        if var in parsed:
                            self.state[var] = parsed[var]
                        else:
                            self._log_or_raise(
                                strict,
                                agent_name,
                                f"Declared output_var '{var}' missing from JSON output"
                            )

                # Case 2: non-JSON output
                else:
                    if len(agent.spec.output_vars) == 1:
                        self.state[agent.spec.output_vars[0]] = raw_output
                    else:
                        raw_key = f"{agent.spec.name}__raw"
                        self.state[raw_key] = raw_output
                        self._log_or_raise(
                            strict,
                            agent_name,
                            f"Non-JSON output with multiple output_vars; stored under '{raw_key}'"
                        )

            # Case 3: no declared output_vars
            else:
                self.state[agent.spec.name] = raw_output

            # Progress: agent end
            if on_progress:
                on_progress(
                    int(base * (2 * i + 2)),
                    agent_name=agent_name,
                )

        # Safety: guarantee 100% even if rounding skipped it
        if on_progress:
            on_progress(100, agent_name=None)

        return self.state.get("final", last_output)

    def get_output(self, agent_name: str) -> Any:
        return self.memory.get(agent_name, "")


# ======================================================
# SHARED HELPERS (USED BY APP_START AND UI)
# ======================================================

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

    stored_agents = load_agents_from_db(DB_PATH)
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
    existing = load_agents_from_db(DB_PATH)
    if existing:
        return
    for name, data in defaults.items():
        save_agent_to_db(
            DB_PATH,
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", [])
        )
        # also save a versioned prompt snapshot
        save_prompt_version(DB_PATH, name, data.get("prompt_template", ""), metadata={"role": data.get("role", "")})


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
    engine = MultiAgentEngine(llm_client=llm_client)

    # Ensure default agents exist in DB (only if table empty)
    bootstrap_default_agents(default_agents)

    # Load agents from DB into engine
    stored_agents = load_agents_from_db(DB_PATH)
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

    pipelines = load_pipelines_from_db(DB_PATH)
    pipeline_names = [p["pipeline_name"] for p in pipelines]

    selected_pipeline = st.sidebar.selectbox(
        "Pipeline",
        ["<Ad-hoc>"] + pipeline_names
    )

    task = st.sidebar.text_area(
        "Task",
        placeholder="Describe the task you want the agents to solve…",
        height=120
    )

    run_clicked = st.sidebar.button(
        "🚀 Run Pipeline",
        use_container_width=True
    )

    with st.sidebar.expander("Advanced: Pipeline Editing", expanded=True):
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
            default=steps
        )

        name = st.text_input("Save as Pipeline")

        if st.button("💾 Save Pipeline"):
            if name.strip():
                try:
                    save_pipeline_to_db(DB_PATH, name.strip(), selected_steps)
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

    return selected_pipeline, selected_steps, task, run_clicked


def render_warnings(engine):
    warnings = engine.memory.get("__warnings__", [])
    if not warnings:
        return

    st.warning("⚠️ Pipeline Warnings")
    for w in warnings:
        st.markdown(f"- {w}")


def render_final_output(engine):
    final = engine.state.get("final")

    if not final:
        st.info("No output yet.")
        return

    st.subheader("Final Output")

    try:
        parsed = json.loads(final)
        st.json(parsed)
    except Exception:
        st.markdown(final)


def render_agent_outputs(engine, steps):
    for agent in steps:
        with st.expander(f"🔹 {agent}"):
            out = engine.get_output(agent)
            try:
                parsed = json.loads(out)
                st.json(parsed)
            except Exception:
                st.code(out)


def render_graph_tab(steps):
    if not steps:
        st.info("No agents selected.")
        return
    st.graphviz_chart(render_agent_graph(steps))


def render_compare_tab(engine, steps):
    col1, col2 = st.columns(2)

    with col1:
        a1 = st.selectbox("Agent A", steps, key="cmp_a")

    with col2:
        a2 = st.selectbox("Agent B", steps, key="cmp_b")

    if a1 != a2:
        out1 = str(engine.get_output(a1) or "")
        out2 = str(engine.get_output(a2) or "")

        diff = difflib.unified_diff(
            out1.splitlines(),
            out2.splitlines(),
            fromfile=a1,
            tofile=a2,
            lineterm=""
        )

        st.code("\n".join(diff), language="diff")


def render_run_results(engine, steps):
    tabs = st.tabs([
        "🟢 Final Output",
        "⚠️ Warnings",
        "📁 Agent Outputs",
        "🧩 Graph",
        "🔍 Compare"
    ])

    with tabs[0]:
        render_final_output(engine)

    with tabs[1]:
        render_warnings(engine)

    with tabs[2]:
        render_agent_outputs(engine, steps)

    with tabs[3]:
        render_graph_tab(steps)

    with tabs[4]:
        render_compare_tab(engine, steps)


def render_run_mode():
    pipeline, steps, task, run_clicked = render_run_sidebar()

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

        with st.spinner("Running pipeline…"):
            final = st.session_state.engine.run_seq(
                steps=steps,
                initial_input=task,
                on_progress=update_progress,
                strict=strict_mode,
            )

        progress_bar.progress(100)
        progress_text.caption("Pipeline completed ✅")

        try:
            save_run_to_db(
                DB_PATH,
                task,
                final if isinstance(final, str) else json.dumps(final),
                st.session_state.engine.memory
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

    if st.session_state.engine.memory:
        render_run_results(st.session_state.engine, steps)


# ======================================================
# ---------- AGENT EDITOR MODE ----------
# ======================================================

def render_agent_editor():
    st.header("🧠 Agent Editor")

    agents_raw = load_agents_from_db(DB_PATH)
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

    selected = st.selectbox(
        "Agent",
        ["<New Agent>"] + names
    )

    is_new = selected == "<New Agent>"
    agent = (
        {"name": "", "model": "gpt-4.1-nano", "role": "", "prompt": "", "input_vars": [], "output_vars": []}
        if is_new
        else next(a for a in agents if a["name"] == selected)
    )

    tabs = st.tabs([
        "1️⃣ Basics",
        "2️⃣ Prompt",
        "3️⃣ Inputs / Outputs",
        "📚 Versions"
    ])

    with tabs[0]:
        name = st.text_input("Name", agent["name"])
        model = st.text_input("Model", agent["model"])
        role = st.text_input("Role", agent["role"])

    with tabs[1]:
        prompt = st.text_area(
            "Prompt Template",
            agent["prompt"],
            height=400
        )

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            input_vars = st.text_area(
                "Input Variables (one per line)",
                "\n".join(agent["input_vars"])
            )
        with col2:
            output_vars = st.text_area(
                "Output Variables (one per line)",
                "\n".join(agent["output_vars"])
            )

    with tabs[3]:
        if not is_new:
            versions = load_prompt_versions(DB_PATH, agent["name"])
            for v in versions:
                with st.expander(f"Version {v['version']} — {v['created_at']}"):
                    st.code(v["prompt"])
                    if v["metadata"]:
                        st.json(v["metadata"])

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("💾 Save"):
            old_name = agent["name"]

            if not is_new and old_name and old_name != name:
                try:
                    rename_agent_in_db(DB_PATH, old_name, name)
                except Exception:
                    logger.exception("Failed to persist rename")
                    st.error("Rename completed but failed to save to database")
                invalidate_agent_state()

            save_agent_to_db(
                DB_PATH,
                name,
                model,
                prompt,
                role,
                [v.strip() for v in input_vars.splitlines() if v.strip()],
                [v.strip() for v in output_vars.splitlines() if v.strip()],
            )

            # Only save new prompt version when prompt was actually changed
            # Ensures agent meta-data changes don't spam the prompt history
            if prompt != agent["prompt"]:
                try:
                    save_prompt_version(DB_PATH, name, prompt)
                except Exception:
                    logger.exception("Failed to persist new prompt version")
                    st.error("New prompt version created but failed to save to database")

            invalidate_agent_state()
            reload_agents_into_engine()
            st.success("Agent saved")
            st.rerun()

    with col_b:
        if not is_new and st.button("📄 Duplicate"):
            try:
                save_agent_to_db(
                    DB_PATH,
                    f"{name}_copy",
                    model,
                    prompt,
                    role,
                    agent["input_vars"],
                    agent["output_vars"]
                )
            except Exception:
                logger.exception("Failed to persist duplicated agent")
                st.error("Agent duplicated but failed to save to database")
            invalidate_agent_state()
            reload_agents_into_engine()
            st.success("Duplicated")
            st.rerun()

    with col_c:
        if not is_new and st.button("🗑 Delete"):
            try:
                delete_agent(DB_PATH, name)
            except Exception:
                logger.exception("Failed to persist agent delete")
                st.error("Failed to delete agent from database")
            invalidate_agent_state()
            reload_agents_into_engine()
            st.warning("Deleted")
            st.rerun()


# ======================================================
# ---------- HISTORY MODE ----------
# ======================================================

def render_history_mode():
    st.header("📜 Past Runs")

    runs = load_runs(DB_PATH)

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
    run, agents = load_run_details(DB_PATH, run_id)

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
                st.code(final)
        else:
            st.markdown(final)

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
                    st.code(output)
            else:
                st.markdown(output)

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
    load_runs,
    load_run_details,
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
