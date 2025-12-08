import streamlit as st
from openai import OpenAI
import difflib
import json
from config import OPENAI_API_KEY

# --------------------------------------
# Multi-agent engine (drop-in compatible)
# --------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)


class Agent:
    def __init__(self, name, prompt_template, model="gpt-4.1-nano"):
        self.name = name
        self.prompt_template = prompt_template
        self.model = model

    def run(self, input):
        prompt = self.prompt_template.format(input=input)
        response = client.responses.create(
            model=self.model,
            input=prompt
        )
        return response.output_text


class MultiAgentEngine:
    def __init__(self):
        self.agents = {}
        self.memory = {}

    def add_agent(self, name, prompt_template, model="gpt-4.1-nano"):
        self.agents[name] = Agent(name, prompt_template, model)

    def run_seq(self, steps, initial_input):
        current = initial_input
        for agent_name in steps:
            agent = self.agents[agent_name]
            current = agent.run(current)
            self.memory[agent_name] = current
        return current

    def get_output(self, agent_name):
        return self.memory.get(agent_name, "")


# --------------------------------------
#  GUI Dashboard
# --------------------------------------

st.set_page_config(page_title="Multi-Agent Debug Dashboard", layout="wide")

st.title("🧠 Multi-Agent Pipeline Debug Dashboard")
st.caption("Inspect agent outputs, compare steps, and visualize pipeline flows.")

# --------------------------------------
# Sidebar: Pipeline configuration
# --------------------------------------

st.sidebar.header("Pipeline Configuration")

default_agents = {
    "planner": """
You are the Planner Agent.
Clarify the task and produce steps.
Task: {input}
Output:
- Clarified Task
- Plan
""",
    "solver": """
You are the Solver Agent.
Use the plan to produce an answer.
Plan:
{input}
Output:
- Answer
""",
    "critic": """
You are the Critic Agent.
Evaluate the answer.
Answer:
{input}
Output:
- Issues
- Improvements
""",
    "finalizer": """
You are the Finalizer Agent.
Apply improvements and produce final answer.
Input:
{input}
Output:
- Final Answer
"""
}

# Session state initialization
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

# --------------------------------------
# Main: Execute pipeline
# --------------------------------------

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

# --------------------------------------
# Display agent-by-agent outputs
# --------------------------------------

st.write("---")
st.header("📁 Agent Outputs")

if st.session_state.engine.memory:
    for agent_name in pipeline_steps:
        st.subheader(f"**🔹 {agent_name.upper()} Output**")
        st.code(st.session_state.engine.get_output(agent_name), language="markdown")
else:
    st.info("Run a pipeline to inspect agent outputs.")

# --------------------------------------
# Comparison Tools
# --------------------------------------

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

elif a1 != "None" and a1 == a2:
    st.warning("Select two different agents.")

# --------------------------------------
# Memory & Export
# --------------------------------------

st.write("---")
st.header("📦 Memory & Export Tools")

if st.session_state.engine.memory:

    if st.button("Export Memory as JSON"):
        json_data = json.dumps(st.session_state.engine.memory, indent=2)
        st.download_button(
            "Download Memory.json",
            data=json_data,
            file_name="agent_memory.json",
            mime="application/json"
        )

    st.json(st.session_state.engine.memory)
else:
    st.info("Run the pipeline to populate memory.")