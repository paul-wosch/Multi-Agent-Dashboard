import graphviz
import streamlit as st
from openai import OpenAI
import difflib
import json
import sqlite3
from datetime import datetime
from config import OPENAI_API_KEY, DB_FILE_PATH

# ======================================================
# DEFAULT AGENTS (STRUCTURED STATE)
# ======================================================

default_agents = {
    "planner": """
You are the Planner Agent.
Refrain from producing the actual solution.
Only clarify the task and produce steps.


Task:
{task}

Output:
- Clarified Task
- Plan
""",

    "solver": """
You are the Solver Agent.
Use the plan to produce an answer.

Plan:
{plan}

Output:
- Answer
""",

    "critic": """
You are the Critic Agent.
Evaluate the answer.

Answer:
{answer}

Output:
- Issues
- Improvements
""",

    "finalizer": """
You are the Finalizer Agent.
You must revise the original answer using the critique.

Original Answer:
{answer}

Critique:
{critique}

Your task:
Return the improved final answer only.
""",
}

# ======================================================
# DATABASE SETUP
# ======================================================

DB_PATH = DB_FILE_PATH

def init_db():
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

    # Agent outputs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            agent_name TEXT,
            output TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
    """)

    # Versioned agent prompts
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_prompt_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            version INTEGER,
            prompt TEXT,
            timestamp TEXT
        )
    """)

    # Agents table (persistent registry)
    c.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_name TEXT PRIMARY KEY,
            model TEXT,
            prompt_template TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_run_to_db(task_input, final_output, memory_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    ts = datetime.utcnow().isoformat()

    c.execute("""
        INSERT INTO runs (timestamp, task_input, final_output)
        VALUES (?, ?, ?)
    """, (ts, task_input, final_output))

    run_id = c.lastrowid

    for agent, output in memory_dict.items():
        c.execute("""
            INSERT INTO agent_outputs (run_id, agent_name, output)
            VALUES (?, ?, ?)
        """, (run_id, agent, output))

    conn.commit()
    conn.close()
    return run_id


# ======================================================
# AGENT PERSISTENCE
# ======================================================

def load_agents_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT agent_name, model, prompt_template FROM agents")
    rows = c.fetchall()
    conn.close()
    return rows


def save_agent_to_db(agent_name, model, prompt_template):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO agents (agent_name, model, prompt_template)
        VALUES (?, ?, ?)
    """, (agent_name, model, prompt_template))

    conn.commit()
    conn.close()


def bootstrap_default_agents(defaults):
    """Runs once if agents table is empty."""
    existing = load_agents_from_db()
    if existing:
        return  # already bootstrapped

    for name, prompt in defaults.items():
        save_agent_to_db(name, "gpt-4.1-nano", prompt)


# ======================================================
# VERSIONED PROMPT STORAGE
# ======================================================

def save_prompt_version(agent_name, prompt_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT MAX(version) FROM agent_prompt_versions WHERE agent_name = ?", (agent_name,))
    result = c.fetchone()[0]
    new_version = 1 if result is None else result + 1

    ts = datetime.utcnow().isoformat()

    c.execute("""
        INSERT INTO agent_prompt_versions (agent_name, version, prompt, timestamp)
        VALUES (?, ?, ?, ?)
    """, (agent_name, new_version, prompt_text, ts))

    conn.commit()
    conn.close()

    return new_version


def load_prompt_versions(agent_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT id, version, prompt, timestamp
        FROM agent_prompt_versions
        WHERE agent_name = ?
        ORDER BY version DESC
    """, (agent_name,))

    rows = c.fetchall()
    conn.close()
    return rows


def load_prompt_version_by_id(version_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT agent_name, version, prompt FROM agent_prompt_versions WHERE id = ?
    """, (version_id,))

    result = c.fetchone()
    conn.close()
    return result


# ======================================================
# MULTI-AGENT ENGINE (MERGED VERSION)
# ======================================================

client = OpenAI(api_key=OPENAI_API_KEY)

class Agent:
    def __init__(self, name, prompt_template, model="gpt-4.1-nano"):
        self.name = name
        self.prompt_template = prompt_template
        self.model = model

    def run(self, variables: dict):
        prompt = self.prompt_template.format(**variables)
        response = client.responses.create(
            model=self.model,
            input=prompt
        )
        return response.output_text


class MultiAgentEngine:
    def __init__(self):
        self.agents = {}
        self.state = {}
        self.memory = {}

    def add_agent(self, name, prompt_template, model="gpt-4.1-nano"):
        self.agents[name] = Agent(name, prompt_template, model)

    def update_agent_prompt(self, name, new_prompt):
        self.agents[name].prompt_template = new_prompt

    def run_seq(self, steps, initial_input):

        # Structured variable state
        self.state = {
            "task": initial_input,
            "plan": "",
            "answer": "",
            "critique": "",
            "final": ""
        }

        self.memory = {}

        for agent_name in steps:
            agent = self.agents[agent_name]

            output = agent.run(self.state)

            if agent_name == "planner":
                self.state["plan"] = output
            elif agent_name == "solver":
                self.state["answer"] = output
            elif agent_name == "critic":
                self.state["critique"] = output
            elif agent_name == "finalizer":
                self.state["final"] = output

            self.memory[agent_name] = output

        return self.state.get("final", "")

    def get_output(self, agent_name):
        return self.memory.get(agent_name, "")


# ======================================================
# GUI START
# ======================================================

st.set_page_config(page_title="Multi-Agent Debug Dashboard", layout="wide")
st.title("🧠 Multi-Agent Pipeline Debug Dashboard (Structured + Prompt Versioning)")
st.caption("Inspect agent outputs, version agent prompts, run pipelines, and store history in SQLite.")

init_db()
bootstrap_default_agents(default_agents)

# First-time initialization (now loaded from DB)
if "engine" not in st.session_state:
    st.session_state.engine = MultiAgentEngine()

    stored_agents = load_agents_from_db()
    for name, model, prompt in stored_agents:
        st.session_state.engine.add_agent(name, prompt_template=prompt, model=model)


# ======================================================
# SIDEBAR — PIPELINE CONFIG
# ======================================================

st.sidebar.header("Pipeline Configuration")

pipeline_steps = st.sidebar.multiselect(
    "Select Agents to Run in Sequence",
    list(st.session_state.engine.agents.keys()),
    default=list(default_agents.keys())
)

task_input = st.sidebar.text_area("Task Input", placeholder="Enter your task here...")

run_button = st.sidebar.button("Run Pipeline")


# ======================================================
# AGENT INTERACTION GRAPH
# ======================================================

st.write("---")
st.header("🧩 Agent Interaction Graph")


def render_agent_graph(steps):
    AGENT_COLORS = {
        "planner": "#c7e9c0",
        "solver": "#fdd0a2",
        "critic": "#fcbba1",
        "finalizer": "#c6dbef"
    }

    dot = graphviz.Digraph()

    # Style (optional)
    dot.attr("node", shape="box", style="rounded,filled", color="#6baed6", fillcolor="#deebf7")

    # Add nodes
    for agent in steps:
        color = AGENT_COLORS.get(agent, "#deebf7")
        dot.node(agent, agent, fillcolor=color)
        # dot.node(agent, agent)

    # Add edges
    for i in range(len(steps) - 1):
        dot.edge(steps[i], steps[i + 1], label="passes output →")

    return dot

if pipeline_steps:
    st.graphviz_chart(render_agent_graph(pipeline_steps))
else:
    st.info("Select agents in the pipeline to generate the graph.")


# ======================================================
# EXECUTE PIPELINE
# ======================================================

if run_button:
    st.subheader("🚀 Pipeline Execution")
    with st.spinner("Running agents…"):
        final_output = st.session_state.engine.run_seq(
            steps=pipeline_steps,
            initial_input=task_input
        )

    st.success("Pipeline completed!")
    st.write("### 🟢 Final Output")
    st.code(final_output, language="markdown")

    run_id = save_run_to_db(
        task_input=task_input,
        final_output=final_output,
        memory_dict=st.session_state.engine.memory
    )
    st.info(f"Run saved to DB with ID: {run_id}")


# ======================================================
# VERSIONED PROMPT EDITOR
# ======================================================

st.write("---")
st.header("✏️ Versioned Agent Prompt Editor")

agent_to_edit = st.selectbox("Select Agent", list(st.session_state.engine.agents.keys()))

current_prompt = st.session_state.engine.agents[agent_to_edit].prompt_template
new_prompt = st.text_area("Prompt Template", current_prompt, height=200)

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save New Prompt Version"):
        version = save_prompt_version(agent_to_edit, new_prompt)
        st.session_state.engine.update_agent_prompt(agent_to_edit, new_prompt)
        save_agent_to_db(agent_to_edit,
                         st.session_state.engine.agents[agent_to_edit].model,
                         new_prompt)
        st.success(f"Saved as version {version}")

with col2:
    if st.button("♻️ Revert to Default"):
        default = default_agents[agent_to_edit]
        st.session_state.engine.update_agent_prompt(agent_to_edit, default)
        save_agent_to_db(agent_to_edit,
                         st.session_state.engine.agents[agent_to_edit].model,
                         default)
        st.success("Reverted to default prompt")


# Version history
st.subheader("📚 Prompt Version History")
versions = load_prompt_versions(agent_to_edit)

for vid, vnum, vprompt, ts in versions:
    with st.expander(f"Version {vnum} — {ts}"):
        st.code(vprompt)
        if st.button(f"Load Version {vnum}", key=f"load_{vid}"):
            st.session_state.engine.update_agent_prompt(agent_to_edit, vprompt)
            st.success(f"Loaded version {vnum}!")


# ======================================================
# AGENT OUTPUTS
# ======================================================

st.write("---")
st.header("📁 Agent Outputs")

if st.session_state.engine.memory:
    for agent_name in pipeline_steps:
        st.subheader(f"🔹 {agent_name.upper()} Output")
        st.code(st.session_state.engine.get_output(agent_name))
else:
    st.info("Run a pipeline to see outputs.")


# ======================================================
# COMPARISON TOOLS
# ======================================================

st.write("---")
st.header("🔍 Compare Agent Outputs")

a1 = st.selectbox("Agent A", ["None"] + pipeline_steps)
a2 = st.selectbox("Agent B", ["None"] + pipeline_steps)

if a1 != "None" and a2 != "None" and a1 != a2:
    out1 = st.session_state.engine.get_output(a1)
    out2 = st.session_state.engine.get_output(a2)

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

def load_run_details(run_id):
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
# 🛠️ AGENT MANAGEMENT PANEL (CRUD)
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
# 📋 LIST AGENTS
# ------------------------------------------------------
with crud_tabs[0]:
    st.subheader("📋 Registered Agents")
    if not agents:
        st.info("No agents registered.")
    else:
        for name, model, prompt in agents:
            with st.expander(f"🤖 {name} — Model: {model}"):
                st.code(prompt, language="markdown")


# ------------------------------------------------------
# ➕ CREATE AGENT
# ------------------------------------------------------
with crud_tabs[1]:
    st.subheader("➕ Create New Agent")

    new_name = st.text_input("Agent Name", "", key="crud_new_name")
    new_model = st.text_input("Model", "gpt-4.1-nano", key="crud_new_model")
    new_prompt = st.text_area("Prompt Template", "", height=200, key="crud_new_prompt")

    if st.button("💾 Create Agent"):
        if not new_name.strip():
            st.error("Agent name cannot be empty.")
        else:
            save_agent_to_db(new_name, new_model, new_prompt)

            # Register in engine
            st.session_state.engine.add_agent(new_name, new_prompt, new_model)

            st.success(f"Agent '{new_name}' created successfully!")
            st.rerun()


# ------------------------------------------------------
# ✏️ EDIT AGENT
# ------------------------------------------------------
with crud_tabs[2]:
    st.subheader("✏️ Edit Existing Agent")

    if not agent_names:
        st.info("No agents to edit.")
    else:
        agent_to_edit = st.selectbox("Select Agent", agent_names, key="crud_edit_agent_select")

        # Load current agent data
        for name, model, prompt in agents:
            if name == agent_to_edit:
                current_model = model
                current_prompt = prompt
                break

        updated_name = st.text_input("Agent Name", agent_to_edit, key="crud_edit_name")
        updated_model = st.text_input("Model", current_model, key="crud_edit_model")
        updated_prompt = st.text_area("Prompt Template", current_prompt, height=200, key="crud_edit_prompt")

        if st.button("💾 Save Changes"):
            # Update DB
            save_agent_to_db(updated_name, updated_model, updated_prompt)

            # Update engine
            # If name changed, remove old entry
            if updated_name != agent_to_edit:
                st.session_state.engine.agents.pop(agent_to_edit, None)

            st.session_state.engine.add_agent(updated_name, updated_prompt, updated_model)

            st.success("Agent updated!")
            st.rerun()


# ------------------------------------------------------
# 🗑️ DELETE AGENT
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
            st.session_state.engine.agents.pop(agent_to_delete, None)

            st.warning(f"Agent '{agent_to_delete}' deleted.")
            st.rerun()


# ------------------------------------------------------
# 📄 DUPLICATE AGENT
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
        for name, model, prompt in agents:
            if name == source_agent:
                source_model = model
                source_prompt = prompt
                break

        # Auto-generate a unique duplicate name
        base_name = f"{source_agent}_copy"
        new_name = base_name
        counter = 2

        existing_names = set(agent_names)

        # Ensure unique name
        while new_name in existing_names:
            new_name = f"{base_name}{counter}"
            counter += 1

        st.write(f"New agent will be named: **{new_name}**")

        if st.button("📄 Duplicate Agent Now", key="crud_duplicate_button"):
            # Save new entry to DB
            save_agent_to_db(new_name, source_model, source_prompt)

            # Add to engine
            st.session_state.engine.add_agent(new_name, source_prompt, source_model)

            st.success(f"Agent '{source_agent}' duplicated as '{new_name}'!")
            st.rerun()
