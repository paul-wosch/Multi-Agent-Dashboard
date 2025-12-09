import streamlit as st
from openai import OpenAI
import difflib
import json
import sqlite3
from datetime import datetime
from config import OPENAI_API_KEY, DB_FILE_PATH

# ======================================================
# DATABASE SETUP
# ======================================================

DB_PATH = DB_FILE_PATH

def init_db():
    conn = sqlite3.connect(DB_FILE_PATH)
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

    conn.commit()
    conn.close()


def save_run_to_db(task_input, final_output, memory_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    ts = datetime.utcnow().isoformat()

    # Store run entry
    c.execute("""
        INSERT INTO runs (timestamp, task_input, final_output)
        VALUES (?, ?, ?)
    """, (ts, task_input, final_output))

    run_id = c.lastrowid

    # Store agent outputs
    for agent, output in memory_dict.items():
        c.execute("""
            INSERT INTO agent_outputs (run_id, agent_name, output)
            VALUES (?, ?, ?)
        """, (run_id, agent, output))

    conn.commit()
    conn.close()
    return run_id


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


# ======================================================
# MULTI-AGENT ENGINE (FIXED VERSION)
# ======================================================

client = OpenAI(api_key=OPENAI_API_KEY)

class Agent:
    def __init__(self, name, prompt_template, model="gpt-4.1-nano"):
        self.name = name
        self.prompt_template = prompt_template
        self.model = model

    def run(self, variables: dict):
        """
        variables = {"task": "...", "plan": "...", "answer": "...", "critique": "..."}
        Any variable used inside the prompt template will be filled.
        """

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

    def run_seq(self, steps, initial_input):

        # structured state used internally
        self.state = {
            "task": initial_input,
            "plan": "",
            "answer": "",
            "critique": "",
            "final": ""
        }

        # memory used for UI (clean agent-output mapping)
        self.memory = {}

        for agent_name in steps:
            agent = self.agents[agent_name]

            output = agent.run(self.state)

            # Update structured state
            if agent_name == "planner":
                self.state["plan"] = output
            elif agent_name == "solver":
                self.state["answer"] = output
            elif agent_name == "critic":
                self.state["critique"] = output
            elif agent_name == "finalizer":
                self.state["final"] = output

            # Update memory for the dashboard
            self.memory[agent_name] = output

        return self.state.get("final", "")

    def get_output(self, agent_name):
        return self.memory.get(agent_name, "")


# ======================================================
# GUI START
# ======================================================

st.set_page_config(page_title="Multi-Agent Debug Dashboard", layout="wide")
st.title("🧠 Multi-Agent Pipeline Debug Dashboard")
st.caption("Inspect agent outputs, compare steps, visualize pipeline flows, and store runs in SQLite.")


# Initialize DB
init_db()

# Sidebar
st.sidebar.header("Pipeline Configuration")

# ======================================================
# DEFAULT AGENTS (UPDATED TO MATCH STRUCTURED STATE)
# ======================================================

default_agents = {
    "planner": """
You are the Planner Agent.
Clarify the task and produce steps.

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


# Session
if "engine" not in st.session_state:
    st.session_state.engine = MultiAgentEngine()
    for name, tmpl in default_agents.items():
        st.session_state.engine.add_agent(name, tmpl)

pipeline_steps = st.sidebar.multiselect(
    "Select Agents to Run in Sequence",
    list(st.session_state.engine.agents.keys()),
    default=list(default_agents.keys())
)

task_input = st.sidebar.text_area("Task Input", placeholder="Enter your task...")

run_button = st.sidebar.button("Run Pipeline")


# ======================================================
# EXECUTE PIPELINE
# ======================================================

if run_button:
    st.subheader("🚀 Pipeline Execution")
    with st.spinner("Running agents..."):
        final_output = st.session_state.engine.run_seq(
            steps=pipeline_steps,
            initial_input=task_input
        )

    st.success("Pipeline completed!")
    st.write("### 🟢 Final Output")
    st.code(final_output, language="markdown")

    # Save run to DB
    run_id = save_run_to_db(
        task_input=task_input,
        final_output=final_output,
        memory_dict=st.session_state.engine.memory
    )
    st.info(f"Run saved to DB with ID: {run_id}")


# ======================================================
# AGENT OUTPUT PANEL
# ======================================================

st.write("---")
st.header("📁 Agent Outputs")

if st.session_state.engine.memory:
    for agent_name in pipeline_steps:
        st.subheader(f"🔹 {agent_name.upper()} Output")
        st.code(st.session_state.engine.get_output(agent_name), language="markdown")
else:
    st.info("Run the pipeline to inspect agent outputs.")


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
# DATABASE HISTORY VIEWER
# ======================================================

st.write("---")
st.header("📜 Past Runs (Saved in Database)")

runs = load_runs()

run_options = {f"Run {r[0]} — {r[1]}": r[0] for r in runs}

selected_run = st.selectbox("Select a past run", ["None"] + list(run_options.keys()))

if selected_run != "None":
    run_id = run_options[selected_run]
    run, agents = load_run_details(run_id)

    timestamp, task, final = run

    st.subheader(f"🗂 Run {run_id} — {timestamp}")
    st.write("### Task Input")
    st.code(task)

    st.write("### Final Output")
    st.code(final)

    st.write("### Agent Outputs")
    for agent_name, output in agents:
        st.markdown(f"#### 🔸 {agent_name}")
        st.code(output)

    # Export JSON
    export = {
        "run_id": run_id,
        "timestamp": timestamp,
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
