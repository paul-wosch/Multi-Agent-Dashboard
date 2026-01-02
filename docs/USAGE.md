# 🧭 Usage (expanded)

This document expands the brief usage summary in the README with more detail on what you can do in each UI mode.

## 👤 Agent Management

Manage reusable agent definitions:

- 🎛️ Create, edit, duplicate, delete, rename, and import agents
- 🧩 Configure:
  - Model, role, and system prompts
  - Inputs & outputs (with contract validation)
  - Tools (including web search) and reasoning behavior
  - Allowed domains per agent for web tools (configured per run)
  - Color and symbol used in graphs and selectors
- 🕒 Inspect versioned prompt history per agent

Agents are persisted to SQLite and safely versioned, so you can inspect older prompts and configurations.

## 🔗 Pipelines (Run Mode)

Build and execute multi-agent pipelines:

- 🔀 Construct pipelines from agents, including:
  - Named, reusable pipelines
  - Ad-hoc pipelines using the current session state
- ▶️ Run pipelines and inspect:
  - Per-agent inputs and outputs
  - Tool calls and reasoning traces
  - Execution warnings and contract violations
- 📤 Export:
  - Pipeline definitions and associated agents as JSON (pipeline-agent export)
  - Final and intermediate outputs for offline analysis

## 🕒 History

Review and analyze previous runs:

- 📚 Browse historical runs stored in SQLite with rich metadata:
  - Models, agent configs, JSON/markdown flags, timing
- 👀 Inspect per-run and per-agent outputs
- 🔍 Compare outputs between agents using a unified diff tool
- 📊 View cost & latency metrics:
  - Per-run and per-agent cost breakdowns
  - Separate input/output token costs stored alongside totals
- 📦 Export full run records (including agent configs and metrics) as JSON

## 📁 File Attachments

Augment runs with files:

- 📎 Attach files to agents and runs
- 🧬 Automatic MIME-type detection & size limits enforced centrally
- Supports mixed text/binary LLM calls where supported by the model / API

## 📜 Logs & Observability

Monitor and debug live behavior:

- 📚 Built-in log viewer reads from a rotating log file
- 🎨 Color-coded log levels with search, filters, download, and live updates
- 🧠 Logs are written via centralized configuration in `config.py` to both stdout and `data/logs/`
