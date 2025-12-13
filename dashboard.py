import graphviz
import streamlit as st
from openai import OpenAI  # still imported for type / factory use
import difflib
import json
import sqlite3
from datetime import datetime
from config import OPENAI_API_KEY, DB_FILE_PATH
from db import get_conn
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


# ======================================================
# DB MIGRATION & HELPERS
# (unchanged; safe to call from app_start())
# ======================================================

def init_db():
    """Create base tables if missing and migrate agents table to include metadata columns.
    Steps:
    - create runs, agent_outputs, agent_prompt_versions, agents, pipelines tables if missing
    - if agents table exists but lacks new columns, add them (role, input_vars, output_vars)
    - keep prompt_version compatible
    """
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()

        # Runs table
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                task_input TEXT,
                final_output TEXT
            )
        """)

        # Agent outputs per run
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                agent_name TEXT,
                output TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
        """)

        # Versioned agent prompts (augment to include metadata JSON for full snapshot)
        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                version INTEGER,
                prompt TEXT,
                metadata_json TEXT,
                timestamp TEXT
            )
        """)

        # Agents table (persistent registry)
        # We'll create with the new schema if not exists.
        c.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_name TEXT PRIMARY KEY,
                model TEXT,
                prompt_template TEXT,
                role TEXT,
                input_vars TEXT,   -- JSON array
                output_vars TEXT   -- JSON array
            )
        """)

        # Pipelines table (saved pipelines)
        c.execute("""
            CREATE TABLE IF NOT EXISTS pipelines (
                pipeline_name TEXT PRIMARY KEY,
                steps_json TEXT,     -- JSON list of agent names in order
                metadata_json TEXT,  -- optional pipeline metadata
                timestamp TEXT
            )
        """)


def _get_table_columns(table: str) -> List[str]:
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table})")
        rows = c.fetchall()
    return [r[1] for r in rows]  # name column


def migrate_agents_table():
    """If someone has an older agents table without the role/input/output columns,
    create them. We used CREATE TABLE IF NOT EXISTS with the new schema above, so
    this function is mainly defensive if older DB structure exists.
    """
    cols = _get_table_columns("agents")
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        if "role" not in cols:
            # can't use IF NOT EXISTS for ALTER in SQLite; do add columns safely
            c.execute("ALTER TABLE agents ADD COLUMN role TEXT")
        if "input_vars" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN input_vars TEXT")
        if "output_vars" not in cols:
            c.execute("ALTER TABLE agents ADD COLUMN output_vars TEXT")


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
    def __init__(self,
                 name: str,
                 prompt_template: str,
                 model: str = "gpt-4.1-nano",
                 role: str = "",
                 input_vars: Optional[List[str]] = None,
                 output_vars: Optional[List[str]] = None,
                 client: Optional[Any] = None):
        self.name = name
        self.prompt_template = prompt_template
        self.model = model
        self.role = role or ""
        self.input_vars = input_vars or []
        self.output_vars = output_vars or []
        self.client = client  # injected OpenAI client

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "input_vars": list(self.input_vars),
            "output_vars": list(self.output_vars),
            "prompt_template": self.prompt_template
        }

    def run(self, state: Dict[str, Any]) -> Any:
        """
        Generic run method:
          - builds a variables dict from `state` using input_vars.
          - formats prompt_template with those variables (missing variables replaced with empty strings).
          - calls LLM and returns raw text output.
        """
        # Build local variables mapping: prefer explicit input_vars; if none declared, pass full state.
        if not self.input_vars:
            variables = dict(state)  # role-agnostic: agent can see entire shared state if it chooses
        else:
            variables = {}
            for k in self.input_vars:
                variables[k] = state.get(k, "")

        # Safe formatting: if prompt_template references keys not present, .format will fail.
        # We'll use a fallback that ensures all placeholders replaced (by empty string).
        try:
            prompt = self.prompt_template.format(**variables)
        except KeyError:
            # Fill missing keys with empty strings by constructing a SafeDict
            class SafeDict(dict):
                def __missing__(self, key):
                    return ""
            prompt = self.prompt_template.format_map(SafeDict(**variables))

        # Call LLM using injected client
        client = self.client
        if client is None:
            raise RuntimeError("No OpenAI client injected into Agent. Provide a client via MultiAgentEngine or Agent constructor.")

        response = client.responses.create(
            model=self.model,
            input=prompt
        )

        # The OpenAI python client used in your base code returned response.output_text previously.
        # Here we attempt to get a reasonable textual representation.
        raw_text = getattr(response, "output_text", None)
        if raw_text is None:
            # Some response shapes may have 'output' structured content; join if necessary
            try:
                # Try to extract textual output from response.output[0].content (best-effort)
                raw_text = ""
                if hasattr(response, "output") and isinstance(response.output, (list, tuple)):
                    for block in response.output:
                        if isinstance(block, dict) and 'content' in block:
                            # content may be a list of dicts with 'text' or 'type'
                            content = block['content']
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get('type') == 'output_text':
                                        raw_text += c.get('text', '')
                            elif isinstance(content, str):
                                raw_text += content
                elif hasattr(response, "text"):
                    raw_text = response.text
                else:
                    raw_text = str(response)
            except Exception:
                raw_text = str(response)

        return raw_text


# =======================
# MultiAgentEngine with client injection
# =======================
class MultiAgentEngine:
    def __init__(self, client: Any):
        # agents: name -> Agent
        self.agents: Dict[str, Agent] = {}
        # shared structured state used by pipelines
        self.state: Dict[str, Any] = {}
        # memory: agent_name -> raw agent output (string or parsed)
        self.memory: Dict[str, Any] = {}
        # injected OpenAI client instance
        self.client = client

    def add_agent(self, name: str, prompt_template: str, model: str = "gpt-4.1-nano",
                  role: str = "", input_vars: Optional[List[str]] = None,
                  output_vars: Optional[List[str]] = None):
        # ensure Agent has the engine's client
        self.agents[name] = Agent(name, prompt_template, model, role, input_vars, output_vars, client=self.client)

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
                parsed_output = json.loads(raw_output)
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
# GUI START (Streamlit)
# ======================================================

st.set_page_config(page_title="Multi-Agent Dynamic Dashboard", layout="wide")
st.title("🧠 Multi-Agent Pipeline Dashboard — Dynamic Agents & Pipelines")
st.caption("Agents are self-describing. Pipelines run against a shared state dict. Versioned prompts and DB persistence included.")


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
    init_db()
    migrate_agents_table()

    # create OpenAI client (factory)
    client = create_openai_client(OPENAI_API_KEY)

    # create engine with injected client
    engine = MultiAgentEngine(client=client)

    # Ensure default agents exist in DB (only if table empty)
    bootstrap_default_agents(default_agents)

    # Load agents from DB into engine
    stored_agents = load_agents_from_db()
    for name, model, prompt, role, input_json, output_json in stored_agents:
        try:
            input_vars = json.loads(input_json) if input_json else []
        except Exception:
            input_vars = []
        try:
            output_vars = json.loads(output_json) if output_json else []
        except Exception:
            output_vars = []
        engine.add_agent(name, prompt, model, role, input_vars, output_vars)

    # Save engine into session state
    st.session_state.engine = engine
    # Optionally keep client in session state for custom use
    st.session_state.openai_client = client


def reload_agents_into_engine():
    "Helper to reload agents into the engine."
    engine = st.session_state.engine
    engine.agents.clear()

    stored_agents = load_agents_from_db()
    for name, model, prompt, role, input_json, output_json in stored_agents:
        input_vars = json.loads(input_json) if input_json else []
        output_vars = json.loads(output_json) if output_json else []
        engine.add_agent(name, prompt, model, role, input_vars, output_vars)


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


# -------------------------
# SIDEBAR — PIPELINE CONFIG
# -------------------------
st.sidebar.header("Pipeline Configuration")

# load pipelines from DB for selection
pipelines = load_pipelines_from_db()
pipeline_names = [p[0] for p in pipelines]

selected_pipeline_name = st.sidebar.selectbox("Saved Pipeline", ["<None>"] + pipeline_names)

if selected_pipeline_name and selected_pipeline_name != "<None>":
    # load steps for this pipeline
    pipeline_steps = next((p[1] for p in pipelines if p[0] == selected_pipeline_name), [])
else:
    pipeline_steps = []

# Fallback: let user pick agents if no saved pipeline selected
available_agents = list(st.session_state.engine.agents.keys())

# Allow creating an ad-hoc pipeline if no saved pipeline chosen:
default_steps = pipeline_steps or available_agents

selected_steps = st.sidebar.multiselect(
    "Or select Agents to run (order is preserved by selection)",
    available_agents,
    default=default_steps
)

# Text input
task_input = st.sidebar.text_area("Task Input", placeholder="Enter your task here...", key="task_input_sidebar")

run_button = st.sidebar.button("Run Pipeline")

# Pipeline save controls
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Pipeline")
new_pipeline_name = st.sidebar.text_input("Pipeline name (to save)", key="new_pipeline_name")
if st.sidebar.button("Save Pipeline"):
    if not new_pipeline_name.strip():
        st.sidebar.error("Pipeline name cannot be empty.")
    else:
        save_pipeline_to_db(new_pipeline_name.strip(), selected_steps)
        st.sidebar.success(f"Pipeline '{new_pipeline_name}' saved.")
        st.rerun()

if selected_pipeline_name and selected_pipeline_name != "<None>":
    if st.sidebar.button("Delete Selected Pipeline"):
        delete_pipeline_from_db(selected_pipeline_name)
        st.sidebar.success(f"Deleted pipeline '{selected_pipeline_name}'.")
        st.rerun()


# ======================================================
# AGENT INTERACTION GRAPH
# ======================================================
st.write("---")
st.header("🧩 Agent Interaction Graph")

def render_agent_graph(steps: List[str]):
    dot = graphviz.Digraph()
    dot.attr("node", shape="box", style="rounded,filled", color="#6baed6", fillcolor="#deebf7")
    for agent in steps:
        # Use role label to annotate node if available
        role = st.session_state.engine.agents.get(agent).role if agent in st.session_state.engine.agents else ""
        label = f"{agent}\n({role})" if role else agent
        dot.node(agent, label)
    for i in range(len(steps) - 1):
        dot.edge(steps[i], steps[i + 1], label="passes state →")
    return dot

if selected_steps:
    st.graphviz_chart(render_agent_graph(selected_steps))
else:
    st.info("Select agents in the pipeline to generate the graph.")


# ======================================================
# EXECUTE PIPELINE
# ======================================================
if run_button:
    st.subheader("🚀 Pipeline Execution")
    with st.spinner("Running agents…"):
        final_output = st.session_state.engine.run_seq(
            steps=selected_steps,
            initial_input=task_input
        )

    st.success("Pipeline completed!")
    st.write("### 🟢 Final Output")
    # If final_output is JSON-like, pretty print
    try:
        parsed = json.loads(final_output) if isinstance(final_output, str) else None
    except Exception:
        parsed = None

    if parsed:
        st.json(parsed)
    else:
        st.code(final_output, language="markdown")

    run_id = save_run_to_db(
        task_input=task_input,
        final_output=final_output if isinstance(final_output, str) else json.dumps(final_output),
        memory_dict=st.session_state.engine.memory
    )
    st.info(f"Run saved to DB with ID: {run_id}")


# ======================================================
# VERSIONED PROMPT EDITOR
# ======================================================
st.write("---")
st.header("✏️ Versioned Agent Prompt Editor")

agent_to_edit = st.selectbox(
    "Select Agent",
    list(st.session_state.engine.agents.keys()),
    key="versioned_editor_select"
)

agent_obj = st.session_state.engine.agents[agent_to_edit]
current_prompt = agent_obj.prompt_template

col_meta, col_prompt = st.columns([1, 3])

with col_meta:
    st.subheader("Metadata")
    edited_name = st.text_input("Agent Name", agent_obj.name, key="editor_name")
    edited_model = st.text_input("Model", agent_obj.model, key="editor_model")
    edited_role = st.text_input("Role (freeform)", agent_obj.role, key="editor_role")
    edited_input_vars = st.text_input("Input vars (comma separated)", ", ".join(agent_obj.input_vars), key="editor_input_vars")
    edited_output_vars = st.text_input("Output vars (comma separated)", ", ".join(agent_obj.output_vars), key="editor_output_vars")

with col_prompt:
    st.subheader("Prompt Template")
    new_prompt = st.text_area("Prompt Template", current_prompt, height=300, key="editor_prompt")

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save New Prompt Version"):
        # Parse var lists
        input_vars = [v.strip() for v in (edited_input_vars or "").split(",") if v.strip()]
        output_vars = [v.strip() for v in (edited_output_vars or "").split(",") if v.strip()]

        # Save versioned prompt snapshot with metadata
        metadata_snapshot = {
            "role": edited_role,
            "input_vars": input_vars,
            "output_vars": output_vars,
            "model": edited_model
        }
        version = save_prompt_version(agent_to_edit, new_prompt, metadata=metadata_snapshot)

        # Update DB entry for agent (if name changed we need to handle renaming)
        save_agent_to_db(edited_name, edited_model, new_prompt, edited_role, input_vars, output_vars)

        # If name changed and differs from currently loaded agent, update engine registry
        if edited_name != agent_obj.name:
            # remove old entry in engine and add new
            st.session_state.engine.remove_agent(agent_obj.name)
        st.session_state.engine.add_agent(edited_name, new_prompt, edited_model, edited_role, input_vars, output_vars)

        st.success(f"Saved as version {version}")
        st.rerun()

with col2:
    if st.button("♻️ Revert to Default"):
        if agent_to_edit in default_agents:
            default = default_agents[agent_to_edit]
            save_agent_to_db(agent_to_edit, default.get("model", ""), default.get("prompt_template", ""),
                             default.get("role", ""), default.get("input_vars", []), default.get("output_vars", []))
            st.session_state.engine.add_agent(agent_to_edit,
                                              default.get("prompt_template", ""),
                                              default.get("model", ""),
                                              default.get("role", ""),
                                              default.get("input_vars", []),
                                              default.get("output_vars", []))
            st.success("Reverted to default prompt and metadata.")
            st.rerun()
        else:
            st.error("No default available for this agent.")


# Version history
st.subheader("📚 Prompt Version History")
versions = load_prompt_versions(agent_to_edit)
if versions:
    for vid, vnum, vprompt, vmeta_json, ts in versions:
        try:
            vmeta = json.loads(vmeta_json) if vmeta_json else {}
        except Exception:
            vmeta = {}
        with st.expander(f"Version {vnum} — {ts} — metadata: {vmeta}"):
            st.code(vprompt)
            if st.button(f"Load Version {vnum}", key=f"load_{vid}"):
                # on load, replace agent state in engine and db
                metadata_inputs = vmeta.get("input_vars", [])
                metadata_outputs = vmeta.get("output_vars", [])
                metadata_model = vmeta.get("model", agent_obj.model)
                save_agent_to_db(agent_to_edit, metadata_model, vprompt, vmeta.get("role", agent_obj.role),
                                 metadata_inputs, metadata_outputs)
                st.session_state.engine.add_agent(agent_to_edit, vprompt, metadata_model, vmeta.get("role", agent_obj.role),
                                                  metadata_inputs, metadata_outputs)
                st.success(f"Loaded version {vnum}!")
                st.rerun()
else:
    st.info("No versions found for this agent.")


# ======================================================
# AGENT OUTPUTS
# ======================================================
st.write("---")
st.header("📁 Agent Outputs")

if st.session_state.engine.memory:
    # Use the pipeline selection that was run (selected_steps)
    for agent_name in selected_steps:
        st.subheader(f"🔹 {agent_name}")
        out = st.session_state.engine.get_output(agent_name)
        # try show parsed JSON nicely
        try:
            parsed = json.loads(out) if isinstance(out, str) else None
        except Exception:
            parsed = None
        if parsed:
            st.json(parsed)
        else:
            st.code(out)
else:
    st.info("Run a pipeline to see outputs.")


# ======================================================
# COMPARISON TOOLS
# ======================================================
st.write("---")
st.header("🔍 Compare Agent Outputs")

a1 = st.selectbox("Agent A", ["None"] + selected_steps, key="compare_a")
a2 = st.selectbox("Agent B", ["None"] + selected_steps, key="compare_b")

if a1 != "None" and a2 != "None" and a1 != a2:
    out1 = str(st.session_state.engine.get_output(a1) or "")
    out2 = str(st.session_state.engine.get_output(a2) or "")

    diff = difflib.unified_diff(
        out1.splitlines(),
        out2.splitlines(),
        fromfile=a1,
        tofile=a2,
        lineterm=""
    )

    st.code("\n".join(diff), language="diff")


# ======================================================
# PAST RUNS VIEWER
# ======================================================
def load_runs():
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, timestamp, task_input FROM runs ORDER BY id DESC")
        rows = c.fetchall()
        return rows

def load_run_details(run_id: int):
    with get_conn(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT timestamp, task_input, final_output FROM runs WHERE id = ?", (run_id,))
        run = c.fetchone()
        c.execute("SELECT agent_name, output FROM agent_outputs WHERE run_id = ?", (run_id,))
        agents = c.fetchall()
        return run, agents

st.write("---")
st.header("📜 Past Runs")

runs = load_runs()
run_options = {f"Run {r[0]} — {r[1]}": r[0] for r in runs}

sel_run = st.selectbox("Select Past Run", ["None"] + list(run_options.keys()))

if sel_run != "None":
    run_id = run_options[sel_run]
    run, agents = load_run_details(run_id)
    ts, task, final = run

    st.subheader(f"🗂 Run {run_id} — {ts}")
    st.write("### Task Input")
    st.code(task)
    st.write("### Final Output")
    st.code(final)

    for agent_name, output in agents:
        st.write(f"#### 🔸 {agent_name}")
        # attempt to pretty print JSON if present
        try:
            parsed = json.loads(output)
            st.json(parsed)
        except Exception:
            st.code(output)

    export = {
        "run_id": run_id,
        "timestamp": ts,
        "task_input": task,
        "final_output": final,
        "agents": {a[0]: a[1] for a in agents}
    }

    st.download_button(
        "Download Run as JSON",
        data=json.dumps(export, indent=2),
        file_name=f"run_{run_id}.json",
        mime="application/json"
    )


def parse_agent_row(row):
    """Normalize agent DB row into a dict."""
    name, model, prompt, role, input_json, output_json = row
    return {
        "name": name,
        "model": model,
        "prompt": prompt,
        "role": role or "",
        "input_vars": json.loads(input_json) if input_json else [],
        "output_vars": json.loads(output_json) if output_json else []
    }


# ======================================================
# SHARED HELPERS (CRUD)
# ======================================================

def reload_agents_into_engine():
    engine = st.session_state.engine
    engine.agents.clear()

    for row in load_agents_from_db():
        a = parse_agent_row(row)
        engine.add_agent(
            a["name"],
            a["prompt"],
            a["model"],
            a["role"],
            a["input_vars"],
            a["output_vars"]
        )


# ======================================================
# AGENT MANAGEMENT PANEL (CRUD)
# ======================================================

st.write("---")
st.header("🛠️ Agent Management (CRUD)")

agents_raw = load_agents_from_db()
agents = [parse_agent_row(a) for a in agents_raw]
agent_names = [a["name"] for a in agents]

tabs = st.tabs([
    "📋 List",
    "➕ Create",
    "✏️ Edit",
    "🗑️ Delete",
    "📄 Duplicate"
])

# ------------------------------------------------------
# LIST AGENTS
# ------------------------------------------------------
with tabs[0]:
    if not agents:
        st.info("No agents registered.")
    else:
        for a in agents:
            with st.expander(f"🤖 {a['name']} — {a['model']} — {a['role']}"):
                st.write("**Input vars:**", a["input_vars"])
                st.write("**Output vars:**", a["output_vars"])
                st.code(a["prompt"], language="markdown")

# ------------------------------------------------------
# CREATE AGENT
# ------------------------------------------------------
with tabs[1]:
    st.subheader("Create Agent")

    name = st.text_input("Agent Name")
    model = st.text_input("Model", "gpt-4.1-nano")
    role = st.text_input("Role")
    input_vars = st.text_input("Input vars (comma separated)")
    output_vars = st.text_input("Output vars (comma separated)")
    prompt = st.text_area("Prompt Template", height=200)

    if st.button("Create Agent"):
        if not name.strip():
            st.error("Agent name required.")
        else:
            save_agent_to_db(
                name.strip(),
                model,
                prompt,
                role,
                [v.strip() for v in input_vars.split(",") if v.strip()],
                [v.strip() for v in output_vars.split(",") if v.strip()]
            )
            reload_agents_into_engine()
            st.success("Agent created.")
            st.rerun()

# ------------------------------------------------------
# EDIT AGENT
# ------------------------------------------------------
with tabs[2]:
    if not agents:
        st.info("No agents to edit.")
    else:
        selected = st.selectbox(
            "Select Agent",
            agent_names,
            key="crud_edit_select"
        )

        agent = next(a for a in agents if a["name"] == selected)

        st.subheader(f"Editing: {selected}")

        name = st.text_input(
            "Agent Name",
            agent["name"],
            key=f"edit_name_{selected}"
        )
        model = st.text_input(
            "Model",
            agent["model"],
            key=f"edit_model_{selected}"
        )
        role = st.text_input(
            "Role",
            agent["role"],
            key=f"edit_role_{selected}"
        )
        input_vars = st.text_input(
            "Input vars (comma separated)",
            ", ".join(agent["input_vars"]),
            key=f"edit_input_{selected}"
        )
        output_vars = st.text_input(
            "Output vars (comma separated)",
            ", ".join(agent["output_vars"]),
            key=f"edit_output_{selected}"
        )
        prompt = st.text_area(
            "Prompt Template",
            agent["prompt"],
            height=200,
            key=f"edit_prompt_{selected}"
        )

        if st.button("Save Changes"):
            save_agent_to_db(
                name.strip(),
                model,
                prompt,
                role,
                [v.strip() for v in input_vars.split(",") if v.strip()],
                [v.strip() for v in output_vars.split(",") if v.strip()]
            )

            if name != selected:
                with get_conn(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute(
                        "DELETE FROM agents WHERE agent_name = ?",
                        (selected,)
                    )

            reload_agents_into_engine()
            st.success("Agent updated.")
            st.rerun()

# ------------------------------------------------------
# DELETE AGENT
# ------------------------------------------------------
with tabs[3]:
    if not agent_names:
        st.info("No agents to delete.")
    else:
        target = st.selectbox(
            "Agent to delete",
            agent_names,
            key="crud_delete_select"
        )

        if st.button("Delete Agent"):
            with get_conn(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("DELETE FROM agents WHERE agent_name = ?", (target,))

            reload_agents_into_engine()
            st.warning(f"Deleted agent '{target}'.")
            st.rerun()

# ------------------------------------------------------
# DUPLICATE AGENT
# ------------------------------------------------------
with tabs[4]:
    if not agents:
        st.info("No agents to duplicate.")
    else:
        source_name = st.selectbox(
            "Source Agent",
            agent_names,
            key="crud_duplicate_select"
        )
        source = next(a for a in agents if a["name"] == source_name)

        base = f"{source_name}_copy"
        new_name = base
        i = 2
        while new_name in agent_names:
            new_name = f"{base}{i}"
            i += 1

        st.write(f"New agent name: **{new_name}**")

        if st.button("Duplicate"):
            save_agent_to_db(
                new_name,
                source["model"],
                source["prompt"],
                source["role"],
                source["input_vars"],
                source["output_vars"]
            )
            reload_agents_into_engine()
            st.success("Agent duplicated.")
            st.rerun()

