import graphviz
import streamlit as st
from openai import OpenAI  # still imported for type / factory use
import difflib
import json
from datetime import datetime
from config import OPENAI_API_KEY, DB_FILE_PATH, MIGRATIONS_PATH
from db.db import get_conn
from db.migrations import apply_migrations
from llm_client import LLMClient, TextResponse
from typing import List, Dict, Any, Tuple, Optional

# =======================
# Configuration
# =======================
DB_PATH = DB_FILE_PATH
# NOTE: removed global OpenAI client creation and removed DB init/migration at import time.
# Use app_start() to perform initialization when running the app.

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
# Persistence Helpers
# =======================

def save_run_to_db(task_input: str, final_output: str, memory_dict: Dict[str, Any]) -> int:
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        ts = datetime.utcnow().isoformat()

        c.execute("""
            INSERT INTO runs (timestamp, task_input, final_output)
            VALUES (?, ?, ?)
        """, (ts, task_input, final_output))

        run_id = c.lastrowid

        for agent, output in memory_dict.items():
            # store as text (string)
            c.execute("""
                INSERT INTO agent_outputs (run_id, agent_name, output)
                VALUES (?, ?, ?)
            """, (run_id, agent, json.dumps(output) if not isinstance(output, str) else output))

    return run_id


def load_agents_from_db() -> List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]:
    """Returns rows: agent_name, model, prompt_template, role, input_vars_json, output_vars_json"""
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT agent_name, model, prompt_template, role, input_vars, output_vars FROM agents")
        rows = c.fetchall()
        return rows


def save_agent_to_db(agent_name: str, model: str, prompt_template: str,
                     role: str = "", input_vars: Optional[List[str]] = None,
                     output_vars: Optional[List[str]] = None):
    """Saves agent metadata. input_vars/output_vars are stored as JSON arrays (strings) for flexibility."""

    input_json = json.dumps(input_vars or [])
    output_json = json.dumps(output_vars or [])

    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO agents (agent_name, model, prompt_template, role, input_vars, output_vars)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (agent_name, model, prompt_template, role, input_json, output_json))


def load_prompt_versions(agent_name: str):
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        c.execute("""
            SELECT id, version, prompt, metadata_json, timestamp
            FROM agent_prompt_versions
            WHERE agent_name = ?
            ORDER BY version DESC
        """, (agent_name,))

        rows = c.fetchall()
        return rows


def save_prompt_version(agent_name: str, prompt_text: str, metadata: Optional[dict] = None) -> int:
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        c.execute("SELECT MAX(version) FROM agent_prompt_versions WHERE agent_name = ?", (agent_name,))
        result = c.fetchone()[0]
        new_version = 1 if result is None else result + 1

        ts = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {})

        c.execute("""
            INSERT INTO agent_prompt_versions (agent_name, version, prompt, metadata_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_name, new_version, prompt_text, metadata_json, ts))

    return new_version


# =======================
# Pipeline persistence (save/load)
# =======================
def save_pipeline_to_db(pipeline_name: str, steps: List[str], metadata: Optional[dict] = None):
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        ts = datetime.utcnow().isoformat()
        steps_json = json.dumps(steps)
        metadata_json = json.dumps(metadata or {})

        c.execute("""
            INSERT OR REPLACE INTO pipelines (pipeline_name, steps_json, metadata_json, timestamp)
            VALUES (?, ?, ?, ?)
        """, (pipeline_name, steps_json, metadata_json, ts))


def load_pipelines_from_db() -> List[Tuple[str, List[str], dict, str]]:
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT pipeline_name, steps_json, metadata_json, timestamp FROM pipelines ORDER BY pipeline_name")
        rows = c.fetchall()

    result = []
    for name, steps_json, metadata_json, ts in rows:
        steps = json.loads(steps_json) if steps_json else []
        metadata = json.loads(metadata_json) if metadata_json else {}
        result.append((name, steps, metadata, ts))
    return result


def delete_pipeline_from_db(pipeline_name: str):
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM pipelines WHERE pipeline_name = ?", (pipeline_name,))


# =======================
# OpenAI client factory
# =======================
def create_openai_client(api_key: str):
    """Factory to create an OpenAI client. Allows tests to replace this factory or pass fake client."""
    return OpenAI(api_key=api_key)


# =======================
# Agent class (self-describing) with injected client
# =======================
class Agent:
    def __init__(
        self,
        name: str,
        prompt_template: str,
        model: str = "gpt-4.1-nano",
        role: str = "",
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.model = model
        self.role = role or ""
        self.input_vars = input_vars or []
        self.output_vars = output_vars or []
        self.llm_client = llm_client

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "input_vars": list(self.input_vars),
            "output_vars": list(self.output_vars),
            "prompt_template": self.prompt_template
        }

    def run(
            self,
            state: Dict[str, Any],
            *,
            structured_schema: Optional[Dict[str, Any]] = None,
            stream: bool = False,
    ) -> str:
        if not self.llm_client:
            raise RuntimeError("Agent has no LLMClient injected")

        # Build variable map
        if self.input_vars:
            variables = {k: state.get(k, "") for k in self.input_vars}
        else:
            variables = dict(state)

        # Safe prompt formatting
        try:
            prompt = self.prompt_template.format(**variables)
        except KeyError:
            class SafeDict(dict):
                def __missing__(self, key): return ""

            prompt = self.prompt_template.format_map(SafeDict(**variables))

        response: TextResponse = self.llm_client.create_text_response(
            model=self.model,
            prompt=prompt,
            response_format=structured_schema,
            stream=stream,
        )

        return response.text


# =======================
# MultiAgentEngine with client injection
# =======================
class MultiAgentEngine:
    def __init__(self, llm_client: LLMClient):
        # agents: name -> Agent
        self.agents: Dict[str, Agent] = {}
        # shared structured state used by pipelines
        self.state: Dict[str, Any] = {}
        # memory: agent_name -> raw agent output (string or parsed)
        self.memory: Dict[str, Any] = {}
        # injected OpenAI client instance
        self.llm_client = llm_client

    def add_agent(
            self,
            name: str,
            prompt_template: str,
            model: str = "gpt-4.1-nano",
            role: str = "",
            input_vars: Optional[List[str]] = None,
            output_vars: Optional[List[str]] = None,
    ):
        self.agents[name] = Agent(
            name=name,
            prompt_template=prompt_template,
            model=model,
            role=role,
            input_vars=input_vars,
            output_vars=output_vars,
            llm_client=self.llm_client,
        )

    def remove_agent(self, name: str):
        self.agents.pop(name, None)

    def update_agent_prompt(self, name: str, new_prompt: str):
        if name in self.agents:
            self.agents[name].prompt_template = new_prompt

    def run_seq(self, steps: List[str], initial_input: Any) -> Any:
        """
        Generic pipeline executor:
        - initialize shared state with initial_input available under 'task' and 'input' keys
        - for each agent in steps, get declared input_vars, run agent with subset of state,
          then write back outputs to state using the agent's declared output_vars.
        - memory stores raw outputs keyed by agent name.
        - final output returned is state.get('final') if present, else the last agent's output.
        """

        # Initialize shared state dictionary (CORE PRINCIPLE 2)
        self.state = {}
        # Always keep the original user input under 'task' and 'input' for backward compatibility
        self.state['task'] = initial_input
        self.state['input'] = initial_input

        self.memory = {}

        last_output = None

        for agent_name in steps:
            if agent_name not in self.agents:
                # skip unknown agent but record a warning in memory
                self.memory[agent_name] = f"[ERROR] Agent '{agent_name}' not registered."
                continue

            agent = self.agents[agent_name]

            # Run agent with generic state (Agent will pick only input_vars if it declares them)
            raw_output = agent.run(self.state)

            # Try to parse JSON if agent produced JSON and multiple outputs are expected
            parsed_output = None
            try:
                parsed_output = LLMClient.safe_json(raw_output)
            except Exception:
                parsed_output = None

            # Store raw output in memory
            self.memory[agent_name] = raw_output

            # Map outputs into shared state using output_vars
            if agent.output_vars:
                # If parsed JSON dict and keys match output_vars -> map them
                if parsed_output and isinstance(parsed_output, dict):
                    for k in agent.output_vars:
                        if k in parsed_output:
                            self.state[k] = parsed_output[k]
                        else:
                            # fallback: if parsed_output has only one item and output_vars length == 1
                            pass
                else:
                    # No JSON mapping: if single output var, assign the whole raw text to it.
                    if len(agent.output_vars) == 1:
                        self.state[agent.output_vars[0]] = raw_output
                    else:
                        # multiple output vars but raw_text not JSON: put raw text under agent_name + first var
                        # and also keep raw text in a generic key to not lose info.
                        self.state[agent.output_vars[0]] = raw_output
                        # Optionally: place prefixed keys to indicate raw content
                        for idx, v in enumerate(agent.output_vars[1:], start=1):
                            # leave subsequent variables empty to be filled by downstream steps
                            if v not in self.state:
                                self.state[v] = ""
            else:
                # Agent didn't declare output_vars: store under agent name key
                self.state[agent.name] = raw_output

            last_output = raw_output

        # Return final output (prefer 'final' key if present)
        return self.state.get('final', last_output)


    def get_output(self, agent_name: str) -> Any:
        return self.memory.get(agent_name, "")


# ======================================================
# SHARED HELPERS (USED BY APP_START AND UI)
# ======================================================

def parse_agent_row(row):
    """
    Normalize agent DB row into a dict.
    UI helper — no backend behavior change.
    """
    name, model, prompt, role, input_json, output_json = row
    return {
        "name": name,
        "model": model,
        "prompt": prompt,
        "role": role or "",
        "input_vars": json.loads(input_json) if input_json else [],
        "output_vars": json.loads(output_json) if output_json else []
    }


def load_runs():
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, timestamp, task_input FROM runs ORDER BY id DESC"
        )
        return c.fetchall()


def load_run_details(run_id: int):
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        c.execute(
            "SELECT timestamp, task_input, final_output FROM runs WHERE id = ?",
            (run_id,)
        )
        run = c.fetchone()

        c.execute(
            "SELECT agent_name, output FROM agent_outputs WHERE run_id = ?",
            (run_id,)
        )
        agents = c.fetchall()

    return run, agents


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
            role = st.session_state.engine.agents[agent].role
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

    stored_agents = load_agents_from_db()
    for name, model, prompt, role, input_json, output_json in stored_agents:
        input_vars = json.loads(input_json) if input_json else []
        output_vars = json.loads(output_json) if output_json else []
        engine.add_agent(name, prompt, model, role, input_vars, output_vars)


# ======================================================
# INITIALIZE DB AND PREPARE GUI
# ======================================================

# Initialize DB + bootstrap defaults if agents table empty and populate Streamlit session with engine
def bootstrap_default_agents(defaults: Dict[str, dict]):
    existing = load_agents_from_db()
    if existing:
        return
    for name, data in defaults.items():
        save_agent_to_db(
            name,
            data.get("model", "gpt-4.1-nano"),
            data.get("prompt_template", ""),
            data.get("role", ""),
            data.get("input_vars", []),
            data.get("output_vars", [])
        )
        # also save a versioned prompt snapshot
        save_prompt_version(name, data.get("prompt_template", ""), metadata={"role": data.get("role", "")})


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
    # DB migrations
    with get_conn(DB_PATH) as conn:
        apply_migrations(conn, migrations_dir=MIGRATIONS_PATH)

    # create OpenAI client (factory)
    openai_client = create_openai_client(OPENAI_API_KEY)
    llm_client = LLMClient(openai_client)

    # create engine with injected client
    engine = MultiAgentEngine(llm_client=llm_client)

    # Ensure default agents exist in DB (only if table empty)
    bootstrap_default_agents(default_agents)

    # Load agents from DB into engine
    stored_agents = load_agents_from_db()
    for name, model, prompt, role, input_json, output_json in stored_agents:
        engine.add_agent(
            name,
            prompt,
            model,
            role,
            json.loads(input_json or "[]"),
            json.loads(output_json or "[]"),
        )

    # Save engine into session state
    st.session_state.engine = engine
    # Optionally keep client in session state for custom use
    st.session_state.llm_client = llm_client


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

mode = st.radio(
    "Mode",
    [MODE_RUN, MODE_AGENTS, MODE_HISTORY],
    horizontal=True
)

st.divider()

# ======================================================
# ---------- RUN MODE ----------
# ======================================================

def render_run_sidebar():
    st.sidebar.header("Run Configuration")

    pipelines = load_pipelines_from_db()
    pipeline_names = [p[0] for p in pipelines]

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
            steps = next(p[1] for p in pipelines if p[0] == selected_pipeline)
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
                save_pipeline_to_db(name.strip(), selected_steps)
                st.success("Pipeline saved")
                st.rerun()

    return selected_pipeline, selected_steps, task, run_clicked


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
                st.markdown(out)


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
        "📁 Agent Outputs",
        "🧩 Graph",
        "🔍 Compare"
    ])

    with tabs[0]:
        render_final_output(engine)

    with tabs[1]:
        render_agent_outputs(engine, steps)

    with tabs[2]:
        render_graph_tab(steps)

    with tabs[3]:
        render_compare_tab(engine, steps)


def render_run_mode():
    pipeline, steps, task, run_clicked = render_run_sidebar()

    if run_clicked:
        with st.spinner("Running pipeline…"):
            final = st.session_state.engine.run_seq(
                steps=steps,
                initial_input=task
            )

        save_run_to_db(
            task,
            final if isinstance(final, str) else json.dumps(final),
            st.session_state.engine.memory
        )

        st.success("Pipeline completed!")

    if st.session_state.engine.memory:
        render_run_results(st.session_state.engine, steps)


# ======================================================
# ---------- AGENT EDITOR MODE ----------
# ======================================================

def render_agent_editor():
    st.header("🧠 Agent Editor")

    agents_raw = load_agents_from_db()
    agents = [parse_agent_row(a) for a in agents_raw]
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
            versions = load_prompt_versions(agent["name"])
            for _, vnum, vprompt, meta, ts in versions:
                with st.expander(f"Version {vnum} — {ts}"):
                    st.code(vprompt)
                    st.json(json.loads(meta or "{}"))

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("💾 Save"):
            save_agent_to_db(
                name,
                model,
                prompt,
                role,
                [v.strip() for v in input_vars.splitlines() if v.strip()],
                [v.strip() for v in output_vars.splitlines() if v.strip()]
            )
            save_prompt_version(name, prompt)
            reload_agents_into_engine()
            st.success("Agent saved")
            st.rerun()

    with col_b:
        if not is_new and st.button("📄 Duplicate"):
            save_agent_to_db(
                f"{name}_copy",
                model,
                prompt,
                role,
                agent["input_vars"],
                agent["output_vars"]
            )
            reload_agents_into_engine()
            st.success("Duplicated")
            st.rerun()

    with col_c:
        if not is_new and st.button("🗑 Delete"):
            with get_conn(DB_PATH) as conn:
                conn.execute("DELETE FROM agents WHERE agent_name = ?", (name,))
            reload_agents_into_engine()
            st.warning("Deleted")
            st.rerun()


# ======================================================
# ---------- HISTORY MODE ----------
# ======================================================

def render_history_mode():
    st.header("📜 Past Runs")

    runs = load_runs()
    options = {f"Run {r[0]} — {r[1]}": r[0] for r in runs}

    selected = st.selectbox(
        "Select Run",
        ["None"] + list(options.keys())
    )

    if selected == "None":
        return

    run_id = options[selected]
    run, agents = load_run_details(run_id)

    ts, task, final = run

    st.subheader(f"Run {run_id}")
    st.code(task)

    with st.expander("Final Output"):
        st.markdown(final)

    for name, out in agents:
        with st.expander(name):
            st.code(out)


# ======================================================
# DEV-TIME SAFETY CHECKS
# ======================================================

for fn in [
    parse_agent_row,
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
