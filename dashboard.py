import graphviz
import streamlit as st
from openai import OpenAI  # still imported for type / factory use
import difflib
import json
import sqlite3
from datetime import datetime
from config import OPENAI_API_KEY, DB_FILE_PATH
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
    conn = sqlite3.connect(DB_PATH)
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

    conn.commit()
    conn.close()


def _get_table_columns(table: str) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table})")
    rows = c.fetchall()
    conn.close()
    return [r[1] for r in rows]  # name column


def migrate_agents_table():
    """If someone has an older agents table without the role/input/output columns,
    create them. We used CREATE TABLE IF NOT EXISTS with the new schema above, so
    this function is mainly defensive if older DB structure exists.
    """
    cols = _get_table_columns("agents")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if "role" not in cols:
        # can't use IF NOT EXISTS for ALTER in SQLite; do add columns safely
        c.execute("ALTER TABLE agents ADD COLUMN role TEXT")
    if "input_vars" not in cols:
        c.execute("ALTER TABLE agents ADD COLUMN input_vars TEXT")
    if "output_vars" not in cols:
        c.execute("ALTER TABLE agents ADD COLUMN output_vars TEXT")
    conn.commit()
    conn.close()


# =======================
# Persistence Helpers
# =======================

def save_run_to_db(task_input: str, final_output: str, memory_dict: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
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

    conn.commit()
    conn.close()
    return run_id


def load_agents_from_db() -> List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]:
    """Returns rows: agent_name, model, prompt_template, role, input_vars_json, output_vars_json"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT agent_name, model, prompt_template, role, input_vars, output_vars FROM agents")
    rows = c.fetchall()
    conn.close()
    return rows


def save_agent_to_db(agent_name: str, model: str, prompt_template: str,
                     role: str = "", input_vars: Optional[List[str]] = None,
                     output_vars: Optional[List[str]] = None):
    """Saves agent metadata. input_vars/output_vars are stored as JSON arrays (strings) for flexibility."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    input_json = json.dumps(input_vars or [])
    output_json = json.dumps(output_vars or [])

    c.execute("""
        INSERT OR REPLACE INTO agents (agent_name, model, prompt_template, role, input_vars, output_vars)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (agent_name, model, prompt_template, role, input_json, output_json))

    conn.commit()
    conn.close()


def load_prompt_versions(agent_name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT id, version, prompt, metadata_json, timestamp
        FROM agent_prompt_versions
        WHERE agent_name = ?
        ORDER BY version DESC
    """, (agent_name,))

    rows = c.fetchall()
    conn.close()
    return rows


def save_prompt_version(agent_name: str, prompt_text: str, metadata: Optional[dict] = None) -> int:
    conn = sqlite3.connect(DB_PATH)
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

    conn.commit()
    conn.close()
    return new_version


# =======================
# Pipeline persistence (save/load)
# =======================
def save_pipeline_to_db(pipeline_name: str, steps: List[str], metadata: Optional[dict] = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    ts = datetime.utcnow().isoformat()
    steps_json = json.dumps(steps)
    metadata_json = json.dumps(metadata or {})

    c.execute("""
        INSERT OR REPLACE INTO pipelines (pipeline_name, steps_json, metadata_json, timestamp)
        VALUES (?, ?, ?, ?)
    """, (pipeline_name, steps_json, metadata_json, ts))
    conn.commit()
    conn.close()


def load_pipelines_from_db() -> List[Tuple[str, List[str], dict, str]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT pipeline_name, steps_json, metadata_json, timestamp FROM pipelines ORDER BY pipeline_name")
    rows = c.fetchall()
    conn.close()
    result = []
    for name, steps_json, metadata_json, ts in rows:
        steps = json.loads(steps_json) if steps_json else []
        metadata = json.loads(metadata_json) if metadata_json else {}
        result.append((name, steps, metadata, ts))
    return result


def delete_pipeline_from_db(pipeline_name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM pipelines WHERE pipeline_name = ?", (pipeline_name,))
    conn.commit()
    conn.close()


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


# Only start the app if we don't already have an engine in session state.
# This prevents double initialization during reruns.
if "engine" not in st.session_state:
    # For Streamlit invocation, app_start() should be called here so side-effects
    # happen only when we actually run the app (not at import time by a test runner).
    app_start()


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
selected_steps = st.sidebar.multiselect(
    "Or select Agents to run (order is preserved by selection)",
    available_agents,
    default=pipeline_steps or list(default_agents.keys())
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

agent_to_edit = st.selectbox("Select Agent", list(st.session_state.engine.agents.keys()))

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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, task_input FROM runs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def load_run_details(run_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, task_input, final_output FROM runs WHERE id = ?", (run_id,))
    run = c.fetchone()
    c.execute("SELECT agent_name, output FROM agent_outputs WHERE run_id = ?", (run_id,))
    agents = c.fetchall()
    conn.close()
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


# ======================================================
# AGENT MANAGEMENT PANEL (CRUD)
# ======================================================

st.write("---")
st.header("🛠️ Agent Management (CRUD)")

# Load agents
agents = load_agents_from_db()
agent_names = [a[0] for a in agents]

crud_tabs = st.tabs([
    "📋 List Agents",
    "➕ Create Agent",
    "✏️ Edit Agent",
    "🗑️ Delete Agent",
    "📄 Duplicate Agent"
])

# ------------------------------------------------------
# LIST AGENTS
# ------------------------------------------------------
with crud_tabs[0]:
    st.subheader("📋 Registered Agents")
    if not agents:
        st.info("No agents registered.")
    else:
        for name, model, prompt, role, input_json, output_json in agents:
            input_vars = json.loads(input_json) if input_json else []
            output_vars = json.loads(output_json) if output_json else []
            with st.expander(f"🤖 {name} — Model: {model} — Role: {role}"):
                st.write("**Input vars:**", input_vars)
                st.write("**Output vars:**", output_vars)
                st.code(prompt, language="markdown")

# ------------------------------------------------------
# CREATE AGENT
# ------------------------------------------------------
with crud_tabs[1]:
    st.subheader("➕ Create New Agent")

    new_name = st.text_input("Agent Name", "", key="crud_new_name")
    new_model = st.text_input("Model", "gpt-4.1-nano", key="crud_new_model")
    new_role = st.text_input("Role (freeform)", "", key="crud_new_role")
    new_input_vars = st.text_input("Input vars (comma separated)", "", key="crud_new_input_vars")
    new_output_vars = st.text_input("Output vars (comma separated)", "", key="crud_new_output_vars")
    new_prompt = st.text_area("Prompt Template", "", height=200, key="crud_new_prompt")

    if st.button("💾 Create Agent"):
        if not new_name.strip():
            st.error("Agent name cannot be empty.")
        else:
            input_vars = [v.strip() for v in (new_input_vars or "").split(",") if v.strip()]
            output_vars = [v.strip() for v in (new_output_vars or "").split(",") if v.strip()]
            save_agent_to_db(new_name, new_model, new_prompt, new_role, input_vars, output_vars)

            # Register in engine
            st.session_state.engine.add_agent(new_name, new_prompt, new_model, new_role, input_vars, output_vars)

            st.success(f"Agent '{new_name}' created successfully!")
            st.rerun()

# ------------------------------------------------------
# EDIT AGENT
# ------------------------------------------------------
with crud_tabs[2]:
    st.subheader("✏️ Edit Existing Agent")

    if not agent_names:
        st.info("No agents to edit.")
    else:
        agent_to_edit = st.selectbox("Select Agent", agent_names, key="crud_edit_agent_select")

        # Load current agent data
        for name, model, prompt, role, input_json, output_json in agents:
            if name == agent_to_edit:
                current_model = model
                current_prompt = prompt
                current_role = role or ""
                current_input_vars = ", ".join(json.loads(input_json) if input_json else [])
                current_output_vars = ", ".join(json.loads(output_json) if output_json else [])
                break

        updated_name = st.text_input("Agent Name", agent_to_edit, key="crud_edit_name")
        updated_model = st.text_input("Model", current_model, key="crud_edit_model")
        updated_role = st.text_input("Role (freeform)", current_role, key="crud_edit_role")
        updated_input_vars = st.text_input("Input vars (comma separated)", current_input_vars, key="crud_edit_input_vars")
        updated_output_vars = st.text_input("Output vars (comma separated)", current_output_vars, key="crud_edit_output_vars")
        updated_prompt = st.text_area("Prompt Template", current_prompt, height=200, key="crud_edit_prompt")

        if st.button("💾 Save Changes"):
            input_vars = [v.strip() for v in (updated_input_vars or "").split(",") if v.strip()]
            output_vars = [v.strip() for v in (updated_output_vars or "").split(",") if v.strip()]

            # Update DB
            save_agent_to_db(updated_name, updated_model, updated_prompt, updated_role, input_vars, output_vars)

            # Update engine: remove old if renamed
            if updated_name != agent_to_edit:
                st.session_state.engine.remove_agent(agent_to_edit)

            st.session_state.engine.add_agent(updated_name, updated_prompt, updated_model, updated_role, input_vars, output_vars)

            st.success("Agent updated!")
            st.rerun()

# ------------------------------------------------------
# DELETE AGENT
# ------------------------------------------------------
with crud_tabs[3]:
    st.subheader("🗑️ Delete Agent")

    if not agent_names:
        st.info("No agents to delete.")
    else:
        agent_to_delete = st.selectbox("Select Agent to Delete", agent_names, key="crud_delete_agent_select")

        if st.button("🗑️ Delete Agent"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM agents WHERE agent_name = ?", (agent_to_delete,))
            conn.commit()
            conn.close()

            # Remove from engine
            st.session_state.engine.remove_agent(agent_to_delete)

            st.warning(f"Agent '{agent_to_delete}' deleted.")
            st.rerun()

# ------------------------------------------------------
# DUPLICATE AGENT
# ------------------------------------------------------
with crud_tabs[4]:
    st.subheader("📄 Duplicate Agent")

    if not agent_names:
        st.info("No agents to duplicate.")
    else:
        source_agent = st.selectbox(
            "Select Agent to Duplicate",
            agent_names,
            key="crud_duplicate_select"
        )

        # Load existing agent details
        for name, model, prompt, role, input_json, output_json in agents:
            if name == source_agent:
                source_model = model
                source_prompt = prompt
                source_role = role or ""
                source_input_vars = json.loads(input_json) if input_json else []
                source_output_vars = json.loads(output_json) if output_json else []
                break

        # Auto-generate a unique duplicate name
        base_name = f"{source_agent}_copy"
        new_name = base_name
        counter = 2
        existing_names = set(agent_names)
        while new_name in existing_names:
            new_name = f"{base_name}{counter}"
            counter += 1

        st.write(f"New agent will be named: **{new_name}**")

        if st.button("📄 Duplicate Agent Now", key="crud_duplicate_button"):
            # Save new entry to DB
            save_agent_to_db(new_name, source_model, source_prompt, source_role, source_input_vars, source_output_vars)

            # Add to engine
            st.session_state.engine.add_agent(new_name, source_prompt, source_model, source_role, source_input_vars, source_output_vars)

            st.success(f"Agent '{source_agent}' duplicated as '{new_name}'!")
            st.rerun()
