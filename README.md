# 🤖 Multi-Agent Dashboard

![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen) ![Python ≥3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue) ![Streamlit](https://img.shields.io/badge/streamlit-ready-orange)

*Local Streamlit playground to build, run and inspect multi-agent LLM pipelines.*

---

## 📖 Overview

**Multi-Agent Dashboard** is a Streamlit-based Python application for building, managing, and running multi-agent LLM pipelines with:

- 🧠 Reusable, UI-agnostic execution engine
- 💾 Persistent SQLite storage
- 👀 Rich observability (cost, latency, logs, history)
- 🧰 Tool calling (incl. web search) with per-agent controls

Use it to prototype agent workflows, compare models and prompts, inspect tool usage, and keep a detailed history of runs.

---

## 🕵️‍♀️ Who this is for (and who it's not)

- ✅ Intended users:
  
  - LLM/agent engineers and researchers who want a local experimentation playground.
  - Internal tooling teams exploring multi-agent orchestration patterns.
  - Developers prototyping agent prompt/versioning and cost/latency monitoring.

- ❌ Non-goals:
  
  - Not a managed production orchestration SaaS (e.g., not intended as a full replacement for production workflow orchestrators).
  - Not a turn-key chatbot hosting service for external users (it's an experimentation & observability tool).

---

## ⚡ Try it in 5 minutes (fast path)

Try it in 5 minutes — copy these commands into your terminal and open the shown localhost URL to see the app.

```bash
# 1) Clone the repo & enter the directory
git clone https://github.com/paul-wosch/Multi-Agent-Dashboard.git
cd Multi-Agent-Dashboard

# 2) Create & activate a venv (macOS / Linux)
python -m venv .venv && source .venv/bin/activate

# 3) Install project (editable mode) and write a minimal .env
pip install -e . && echo 'OPENAI_API_KEY=your_key_here' > .env

# 4) Run the Streamlit app
streamlit run src/multi_agent_dashboard/ui/app.py
```

### 💻 Windows PowerShell (quick)

```powershell
git clone https://github.com/paul-wosch/Multi-Agent-Dashboard.git
cd Multi-Agent-Dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
'OPENAI_API_KEY=your_key_here' > .env
streamlit run src/multi_agent_dashboard/ui/app.py---
```

## 🧩 System Requirements

- 🐍 **Python**: >=3.11 (tested with CPython 3.14)  
- 💻 **OS**: Tested on macOS; should work on Linux and Windows with appropriate environment setup  
- 🌐 **Network**: Outbound HTTPS access to OpenAI’s APIs  
- 🔑 **Credentials**: Valid `OPENAI_API_KEY` in `.env`

Note: pandas (and numpy) are installed as direct dependencies of Streamlit, so you generally won't need to pip install `pandas` separately for the UI/analytics workflows that rely on Streamlit.

---

## ⚙️ Configuration (essential)

Most configuration is centralized in `config.py` and `.env`. See [docs/CONFIG.md](docs/CONFIG.md)  for the full reference.

### 🌱 Environment Variables (`.env` at project root)

| Name             | Required | Default | Description                                                |
| ---------------- | -------- | ------- | ---------------------------------------------------------- |
| `OPENAI_API_KEY` | ✅        | None    | OpenAI API key used by the LLM client                      |
| `LOG_LEVEL`      | ❌        | `INFO`  | Global logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🚀 Getting started — fast

1. Clone the repo (see top).  
2. Create & activate a Python 3.11+ virtualenv.  
3. `pip install -e .`
4. Copy `.env.template` → `.env` (or create `.env`) and set `OPENAI_API_KEY`.  
5. `streamlit run src/multi_agent_dashboard/ui/app.py`
6. Open [http://localhost:8501](http://localhost:8501)

### 🩺 First run behavior (what the app does on first start)

On first run, the app will:

- Ensure `data/`, `data/db/`, `data/logs/`, and `data/migrations/` exist
- Create the SQLite DB at: `data/db/multi_agent_runs.db`
- Apply SQL migrations from `data/migrations/`
- Seed default agents (planner/solver/critic/finalizer-style roles) if the `agents` table is empty
- Initialize a rotating log file under `data/logs/` (see [docs/CONFIG.md](docs/CONFIG.md) for logging details)

---

## 🧾 Commands cheat sheet

```bash
# Setup (macOS / Linux)
python -m venv .venv && source .venv/bin/activate

# Install for development
pip install -e .

# Start the UI
streamlit run src/multi_agent_dashboard/ui/app.py

# One-click
./scripts/quick_start.sh        # macOS / Linux
.\scripts\quick_start.ps1       # Windows PowerShell

# Migration helpers (after pip install -e .)
python -m multi_agent_dashboard.db.infra.generate_migration add_feature --dry-run
python -m multi_agent_dashboard.db.infra.generate_migration add_feature

# Safe sqlite rebuild (when migrations require rebuild)
python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --all-with-diffs data/db/multi_agent_runs.db
```

---

## 🧭 Usage — UI modes at a glance

- 🎛️ Agent Management: create/edit agents, manage prompt versions, configure tools & domains, set colors/symbols.  
- ▶️ Pipelines / Run Mode: compose pipelines from agents, run them, and inspect per-agent inputs/outputs, tool calls, and execution traces.  
- 🕒 History: browse persisted runs with cost/latency metrics, export full run JSON.  
- 📚 Logs: live/rotating log viewer with filters and download.  
- 📎 File Attachments: attach files to runs/agents (with MIME detection and size caps). 

For a detailed breakdown of each UI mode and what you can inspect/configure, see [docs/USAGE.md](docs/USAGE.md).

---

## ✨ Key Features

### 🧠 Engine & Contracts

- 🤖 Multi-agent pipeline execution with a unified dashboard UI  
- 🧩 Dynamic agent configuration (model, role, inputs/outputs, tools, reasoning behavior, colors, symbols)  
- 🧠 Strict vs permissive execution modes:
  - Strict mode with explicit input/output contracts and writeback behavior
  - Permissive mode for easier experimentation
- 🧾 Rich output metadata:
  - JSON vs markdown flags
  - Model identifiers and execution context

### 🔌 Tools & Reasoning

- 🛠 Tool calling:
  - Optional web search tools
  - Per-agent domain restrictions
  - Tool-call traces and agent configurations persisted with each run
- 🧠 Reasoning controls:
  - Configurable reasoning effort / style per agent where supported
  - Persisted reasoning and tool-usage configuration per run

### 💾 Database & Migrations

- 💾 SQLite-backed persistence for agents, prompts, pipelines, runs, and metrics
- 🧱 Centralized migration system (canonical schema + generator + ordered SQL migrations)
- 🔁 Tools & guidance for safe FK/constraint changes and rebuilds (see docs/MIGRATIONS.md)

### 📊 Monitoring & Metrics

- Per-run and per-agent cost & latency metrics (input/output token breakdown)
- Pipeline graph visualization with per-agent colors & symbols
- Run exports including configs, outputs, metrics, and tool usage

---

## 🏗️ Architecture — short

- 🎛️ UI (Streamlit): `src/multi_agent_dashboard/ui/` — presentation, graph view, run mode, history, logs.  
- 🧰 Services: `src/multi_agent_dashboard/db/services.py` — transactional APIs for UI and scripts.  
- 🗃️ Persistence / DAOs: `src/multi_agent_dashboard/db/*.py` — agents, pipelines, runs.  
- 🧱 DB infra & migrations: `src/multi_agent_dashboard/db/infra/`.  
- 🧠 Engine & LLM client: `src/multi_agent_dashboard/engine.py`, `src/multi_agent_dashboard/llm_client.py`.

For a deeper architecture walkthrough and the repository layout, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🧬 Migrations & DB — summary

- DB file (auto-created): `data/db/multi_agent_runs.db`  
- Migrations live in `data/migrations/` (ordered SQL files).  
- Canonical schema: `src/multi_agent_dashboard/db/infra/schema.py`.  
- To generate migrations (recommended after `pip install -e .`):
  - `python -m multi_agent_dashboard.db.infra.generate_migration my_change --dry-run`
  - `python -m multi_agent_dashboard.db.infra.generate_migration my_change`
- For migrations tagged `_REQUIRES_REBUILD` on non-empty DBs, use the sqlite_rebuild tool (see [docs/MIGRATIONS.md](docs/MIGRATIONS.md) for full instructions and caveats).

---

## 🔎 Troubleshooting — short

- UI opens but model calls fail → check `OPENAI_API_KEY` in `.env`.
- Permission errors when writing `data/` → ensure write permissions.
- Python syntax errors → use Python >= 3.11 (this repo is tested with CPython 3.14).   
- Graphviz export requires system `dot` binary when exporting images; in-browser Streamlit `graphviz_chart` generally works without it.

For a longer troubleshooting guide and FAQs, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## 🤝 Contributing

To keep the project healthy:

- ✅ Keep UI changes confined to `multi_agent_dashboard/ui`
- 🚫 Avoid introducing `sys.path` hacks
- 🗃️ Use DAOs and `services.py` instead of direct SQLite access
- 🧱 Always use the migration system for schema changes (see Migration: Safe Workflow)
- 🧠 Preserve engine/UI separation; keep the engine free of Streamlit dependencies
- 🧪 Add or extend tests for new engine, DB, or migration behavior where applicable

Developer checklist (quick):

- Create a feature branch
- Update `schema.py` for DB changes (if any)
- Run `generate_migration.py --dry-run`, review diffs, then run without `--dry-run`
- Start the app or run scripts that call `init_db` to apply migrations
- If `_REQUIRES_REBUILD` appears and DB is non-empty, run `sqlite_rebuild.py` with backups
- Add tests for new behavior and include them under `tests/`
- Submit a PR with a clear description and migration notes

---

## 🔧 Status & Known Gaps

- Unit tests: not yet implemented.  
- CHANGELOG: not currently maintained — a `CHANGELOG.md` would be helpful.  
- CI: add checks for linting and tests once a test suite exists.

---

## 📝 Project History

The project evolved from a single-file Streamlit script into a modular, package-based system featuring:

- 🧠 A decoupled multi-agent execution engine with structured results and hooks
- 🕒 Versioned prompt management with atomic agent operations
- 💾 Persistent execution history with rich metadata and FK-aware migrations
- 🧱 A clean `src/`-based layout for long-term maintainability
- 🗃️ A DAO + service-based database layer, decoupled from the UI
- 👀 First-class observability:
  - Log viewer panel
  - Pipeline warnings
  - Input/output contracts
  - Cost & latency metrics
- 🧩 Advanced UX features:
  - File attachments
  - JSON import/export for pipelines and agent templates
  - Ad-hoc pipelines and improved run selection
- 🛠 Recent enhancements:
  - Tool-calling and reasoning controls
  - Per-agent colors and symbols reused across graph and selectors
  - Persisted, color-coded logs with search & filters
  - Finer-grained input/output cost tracking
  - Safer migration tooling for foreign-key changes with explicit rebuild helpers

Use this dashboard as both a day-to-day multi-agent playground and a reference architecture for building robust, observable LLM workflows.

---

## 📚 Deep reference / full docs

For deep, step-by-step instructions (install, CLI reference, migrations, architecture, troubleshooting), see the `docs/` files:  

- [docs/INSTALL.md](docs/INSTALL.md)
- [docs/MIGRATIONS.md](docs/MIGRATIONS.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/CONFIG.md](docs/CONFIG.md)
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/USAGE.md](docs/USAGE.md)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).  
See the LICENSE file for full details.