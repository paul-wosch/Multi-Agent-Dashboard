# 🤖 Multi-Agent Dashboard

## 📖 Overview

**Multi-Agent Dashboard** is a Streamlit-based Python application for building, managing, and running multi-agent LLM pipelines with:

- 🧠 Reusable, UI-agnostic execution engine
- 💾 Persistent SQLite storage
- 👀 Rich observability (cost, latency, logs, history)
- 🧰 Tool calling (incl. web search) with per-agent controls

Use it to prototype agent workflows, compare models and prompts, inspect tool usage, and keep a detailed history of runs.

The project uses a **standard `src/` Python package layout** for clean imports and long-term maintainability.

---

## ⚡ Quick Start (TL;DR)

1. **Clone & enter repo**

   ```bash
   git clone <your-repo-url>
   cd <repo-root>
   ```

2. **Create & activate a virtualenv (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   # .venv\Scripts\activate         # Windows
   ```

3. **Install in editable mode**

   ```bash
   pip install -e .
   ```

4. **Configure environment**

   Create `.env` in the project root:

   ```text
   OPENAI_API_KEY=your_api_key_here
   LOG_LEVEL=INFO
   ```

5. **Run the dashboard**

   From the project root:

   ```bash
   streamlit run src/multi_agent_dashboard/ui/app.py
   ```

6. **Open the app**

   👉 [http://localhost:8501](http://localhost:8501)

On first run, the app will:

- Create `data/` and subdirectories (if missing)
- Initialize SQLite at `data/db/multi_agent_runs.db`
- Apply SQL migrations from `data/migrations/`
- Seed default agents if the DB is empty

---

## 🧩 System Requirements

- 🐍 **Python**: 3.10+ (uses modern typing & standard libraries)
- 💻 **OS**: Tested on macOS and Linux; should work on Windows with appropriate environment setup
- 🌐 **Network**: Outbound HTTPS access to OpenAI’s APIs
- 🔑 **Credentials**: Valid `OPENAI_API_KEY` in `.env`

---

## ⚙️ Configuration Reference

Most configuration is centralized in `config.py` and `.env`.

### 🌱 Environment Variables (`.env` at project root)

| Name            | Required | Default | Description                                      |
|-----------------|----------|---------|--------------------------------------------------|
| `OPENAI_API_KEY`| ✅       | None    | OpenAI API key used by the LLM client           |
| `LOG_LEVEL`     | ❌       | `INFO`  | Global logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

If `OPENAI_API_KEY` is missing or invalid, LLM calls will fail at runtime; the UI may load but requests to the model will error.

### 🧱 Core Paths & Caps (from `config.py`)

| Setting              | Default                          | Description                                      |
|----------------------|----------------------------------|--------------------------------------------------|
| `PROJECT_ROOT`       | Repo root                       | Auto-detected project root                       |
| `DATA_PATH`          | `PROJECT_ROOT / "data"`         | Root for data and artifacts                      |
| `DB_PATH`            | `DATA_PATH / "db"`              | Directory for SQLite databases                   |
| `DB_FILE_PATH`       | `data/db/multi_agent_runs.db`   | Main SQLite DB file (auto-created)              |
| `MIGRATIONS_PATH`    | `data/migrations`               | Ordered SQL migrations                           |
| `LOGS_PATH`          | `data/logs`                     | Log directory for rotating app logs              |
| `AGENT_INPUT_CAP`    | defined in `config.py`          | Max characters per formatted input segment       |
| `AGENT_OUTPUT_CAP`   | defined in `config.py`          | Max characters per rendered prompt / output      |

Prompt formatting and outputs are passed through `utils.safe_format` using these caps to avoid unbounded prompt sizes.

---

## 🚀 Getting Started (Detailed)

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo-root>
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3️⃣ Install the project in editable mode

This is **required** when using a `src/` layout so that `multi_agent_dashboard` is importable:

```bash
pip install -e .
```

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

You can adjust the log level (e.g. `DEBUG`) while developing.

### 5️⃣ Run the dashboard

From the project root:

```bash
streamlit run src/multi_agent_dashboard/ui/app.py
```

Then open:

👉 [http://localhost:8501](http://localhost:8501)

### 6️⃣ First Run Behavior

On first successful run, the app will:

- Ensure `data/`, `data/db/`, `data/logs/`, and `data/migrations/` exist
- Create the SQLite DB at:

  ```text
  data/db/multi_agent_runs.db
  ```

- Apply all SQL migrations from `data/migrations/` using the centralized migration system
- Seed default agents (planner/solver/critic/finalizer-style roles) if the `agents` table is empty
- Initialize a rotating log file under `data/logs/`

### 🩺 Troubleshooting First Run

- ❗ **No API key or invalid key**
  - Symptom: UI loads, but model calls fail with authentication/authorization errors.
  - Fix: Set `OPENAI_API_KEY` in `.env`, restart Streamlit.

- ❗ **Permission issues**
  - Symptom: Errors writing to `data/db` or `data/logs`.
  - Fix: Ensure you have write permissions in the repo directory; adjust Docker/container volume mounts if applicable.

- ❗ **Python version errors**
  - Symptom: Syntax errors on union types (`str | None`) or similar.
  - Fix: Upgrade to Python 3.10+.

---

## 🧭 Usage

The UI is organized into distinct modes (tabs/pages inside the Streamlit app):

### 👤 Agent Management

Manage reusable agent definitions:

- 🎛️ Create, edit, duplicate, delete, rename, and import agents
- 🧩 Configure:
  - Model, role, and system prompts
  - Inputs & outputs (with contract validation)
  - Tools (including web search) and reasoning behavior
  - Allowed domains per agent for web tools
  - Color and symbol used in graphs and selectors
- 🕒 Inspect versioned prompt history per agent

Agents are persisted to SQLite and safely versioned, so you can inspect older prompts and configurations.

### 🔗 Pipelines (Run Mode)

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

### 🕒 History

Review and analyze previous runs:

- 📚 Browse historical runs stored in SQLite with rich metadata:
  - Models, agent configs, JSON/markdown flags, timing
- 👀 Inspect per-run and per-agent outputs
- 🔍 Compare outputs between agents using a unified diff tool
- 📊 View cost & latency metrics:
  - Per-run and per-agent cost breakdowns
  - Separate input/output token costs stored alongside totals
- 📦 Export full run records (including agent configs and metrics) as JSON

### 📁 File Attachments

Augment runs with files:

- 📎 Attach files to agents and runs
- 🧬 Automatic MIME-type detection & size limits enforced centrally
- Supports mixed text/binary LLM calls where supported by the model / API

### 📜 Logs & Observability

Monitor and debug live behavior:

- 📚 Built-in log viewer reads from a rotating log file
- 🎨 Color-coded log levels with search, filters, download, and live updates
- 🧠 Logs are written via centralized configuration in `config.py` to both stdout and `data/logs/`

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

- 💾 SQLite-backed persistence:
  - Agents and versioned prompts
  - Pipelines and pipeline steps
  - Runs, per-agent outputs, and metrics
- 🧱 Centralized migration system:
  - Canonical schema in `db/infra/schema.py`
  - Migration generator (`generate_migration.py`) with FK-aware diffing
  - SQL migrations under `data/migrations/` (e.g. `000_*.sql`, `001_*.sql`, …)
- 🔁 Foreign key & constraint management:
  - Migrations that require constraint rebuild are tagged with `_REQUIRES_REBUILD`
  - On **fresh (empty) databases**, relevant tables are auto-rebuilt when those migrations run
  - On existing (non-empty) databases, you must run `sqlite_rebuild.py` explicitly for safe rebuilds

### 📊 Monitoring & Metrics

- 📊 Cost & latency profiling:
  - Per-run and per-agent metrics with:
    - Input vs output token/cost breakdown
    - Aggregated per-run cost/latency summaries
  - Metrics are persisted and included in JSON exports
- 👀 Pipeline visualization:
  - Agent graph view with per-agent colors and symbols
  - Performance and cost overlays

### 📤 Import/Export

- 📥 Import agent definitions via JSON templates in the UI
- 📤 Export:
  - Pipelines + their agents (pipeline-agent export)
  - Per-run history including agent configs, outputs, metrics, and tool usage
- 🧭 Enhanced run selector:
  - Displays run ID, timestamp, agent execution order, and abbreviated task

---

## 🏗️ Architecture Overview

The project is organized into clear layers:

- 🎛️ **UI Layer (`ui/`)**
  - `ui/app.py`: Streamlit app
  - Handles presentation and user interaction
  - Talks only to services, engine, and models — not directly to DB internals

- 🧰 **Service Layer (`db/services.py`)**
  - `AgentService`, `RunService`, `PipelineService`
  - Transactional APIs for the UI and scripts
  - Orchestrates DAOs and engine operations

- 🗃️ **Persistence Layer (`db/*.py`)**
  - DAOs:
    - `agents.py` – agent persistence & prompt versions
    - `pipelines.py` – pipeline specs and steps
    - `runs.py` – run records, outputs, metrics
  - `db.py` – backwards-compatible DB entry points, re-exporting DAOs and providing `init_db`

- 🧱 **DB Infrastructure (`db/infra/`)**
  - `core.py` – connection and migrations bootstrap
  - `schema.py` – canonical schema
  - `migrations.py` – migration application logic
  - `generate_migration.py` – CLI to generate migrations from `schema.py`
  - `schema_diff.py`, `schema_diff_constraints.py` – schema diff tools
  - `sqlite_rebuild.py` – safe table rebuild helpers & CLI

- 🧠 **Engine & LLM Client**
  - `engine.py` – core multi-agent orchestration engine (UI-agnostic)
  - `llm_client.py` – OpenAI client abstraction with:
    - Typed errors
    - Retries & backoff
    - Tool-calling support
    - Normalized responses for the engine

- 🧩 **Models & Utilities**
  - `models.py` – domain models:
    - `AgentSpec`, `AgentRuntime`, `PipelineSpec`, run result types, etc.
  - `utils.py` – shared helpers, including `safe_format` with centralized caps
  - `config.py` – global configuration (paths, pricing tables, logging, caps, colors, symbols)

A typical flow:

```text
UI (Streamlit)
  → Services (AgentService / RunService / PipelineService)
    → DAOs (agents / pipelines / runs)
      → DB (SQLite)
  → Engine (orchestrates agents & LLM calls)
    → LLMClient (OpenAI Responses API + tools)
```

---

## 🗂️ Repository Structure

```text
repo_root/
├── .env                         # Environment variables (API keys, log level; not committed)
├── .gitignore                   # Ignore sensitive/generated files
├── LICENSE                      # Project license
├── pyproject.toml               # Project metadata, dependencies, packaging config
├── README.md                    # Project documentation
├── src/
│   └── multi_agent_dashboard/   # Main Python package (src layout)
│       ├── __init__.py
│       ├── config.py            # Paths, logging, pricing, UI theming, caps
│       ├── engine.py            # Core multi-agent orchestration engine
│       ├── llm_client.py        # LLM client abstraction (retries, tools, backoff, errors)
│       ├── models.py            # Domain models (AgentSpec, PipelineSpec, etc.)
│       ├── utils.py             # Shared utilities (safe_format, helpers)
│       ├── db/
│       │   ├── __init__.py
│       │   ├── agents.py        # Agent DAOs and helpers
│       │   ├── db.py            # Backwards-compatible DB entry points + init_db
│       │   ├── pipelines.py     # Pipeline DAOs and helpers
│       │   ├── runs.py          # Run/agent output DAOs, metrics persistence
│       │   ├── services.py      # AgentService, RunService, PipelineService
│       │   └── infra/
│       │       ├── __init__.py
│       │       ├── core.py                    # Connection, migrations bootstrap
│       │       ├── generate_migration.py      # Migration generator CLI
│       │       ├── migrations.py              # Migration application logic
│       │       ├── schema.py                  # Canonical schema definition
│       │       ├── schema_diff.py             # Column-level diffing
│       │       ├── schema_diff_constraints.py # FK / constraints diffing
│       │       └── sqlite_rebuild.py          # Safe table rebuild helpers & CLI
│       └── ui/
│           ├── __init__.py
│           ├── agent_editor_mode.py  # Agent CRUD + prompt versioning UI
│           ├── app.py                # Streamlit app entrypoint & mode routing
│           ├── bootstrap.py          # App setup: DB, clients, engine, defaults
│           ├── cache.py              # Streamlit caching helpers
│           ├── exports.py            # Pipeline/run export helpers
│           ├── graph_view.py         # Pipeline graph visualization
│           ├── history_mode.py       # Past runs viewer & export UI
│           ├── logging_ui.py         # Log viewer & Streamlit log handler
│           ├── metrics_view.py       # Cost & latency metrics UI
│           ├── run_mode.py           # Run configuration, execution, results UI
│           ├── styles.py             # Streamlit CSS helpers
│           ├── tools_view.py         # Tool usage & per-agent tool UI
│           ├── utils.py              # UI utility helpers
│           └── view_models.py        # UI view-model transformations
├── data/
│   ├── db/
│   │   └── multi_agent_runs.db  # Auto-created SQLite database (not tracked)
│   ├── logs/                    # Rotating log files
│   └── migrations/
│       ├── 000_*.sql
│       ├── 001_*.sql
│       ├── 002_*.sql
│       └── ...                  # Future migrations
└── tests/                       # Tests (may be empty / future expansion)
```

> 📝 Note: The `data/` directory and its contents are typically created automatically at runtime. The exact set of migration files will evolve over time; see `data/migrations/` in your clone.

---

## 🧬 Database & Migration Workflow

Database schema management is centralized in `db/infra`.

### 🔧 Initialization

- `init_db(db_path: Path)` (from `db/db.py` and infra) is the canonical way to:
  - Open a connection
  - Ensure migration tracking is set up
  - Apply all pending migrations from `data/migrations/` against the DB file

The Streamlit UI calls this automatically when you run `app.py`.

### 🧱 Schema & Migrations

- **Canonical schema**: `db/infra/schema.py`
- **Migrations**: Ordered SQL files in `data/migrations/` (e.g. `000_create_base_tables.sql`, `001_add_agent_output_metadata.sql`, …)
- **Migration application**: `db/infra/migrations.py`:
  - Tracks applied migrations in a dedicated table
  - Applies new migrations in order

### 🧮 FK-Aware Changes & Rebuilds

Some schema changes (especially foreign keys and constraints) require table rebuilds. The tooling handles this in two ways:

- `generate_migration.py`:
  - Diffs the current DB schema vs `schema.py`
  - Emits SQL migrations and, when necessary:
    - Attaches `_REQUIRES_REBUILD` to the migration filename
    - Inserts comments/instructions regarding affected tables

- `_REQUIRES_REBUILD` migrations:
  - On a **fresh (empty) DB**:
    - Relevant tables can be auto-rebuilt to match `schema.py` when applying the migration
  - On a **non-empty DB**:
    - The migration is applied but constraints may not fully match `schema.py` until you explicitly run `sqlite_rebuild.py`
    - This design makes destructive rebuilds **opt-in** for non-empty databases

### 🛠 CLI Tools

From the project root (using `python -m ...` style):

#### 1. Generate a Migration

```bash
# Preview diffs (no files written)
python -m multi_agent_dashboard.db.infra.generate_migration add_new_feature --dry-run

# Generate migration files under data/migrations/
python -m multi_agent_dashboard.db.infra.generate_migration add_new_feature
```

Key options:

- `name`: required suffix for the migration name
- `--dry-run`: show diffs without writing files
- `--disable-constraints`: ignore constraint diffs if desired

Typical workflow to add a column or table:

1. Edit `schema.py` to describe the target schema
2. Run `generate_migration.py` (with `--dry-run` first)
3. Run again without `--dry-run` to write the SQL
4. Start the app (or run a dedicated script) to apply migrations via `init_db`

#### 2. Rebuild Tables Safely

```bash
# Rebuild a single table in-place (with backup)
python -m multi_agent_dashboard.db.infra.sqlite_rebuild agents

# Rebuild all tables with pending FK/constraint diffs (recommended after *_REQUIRES_REBUILD on non-empty DBs)
python -m multi_agent_dashboard.db.infra.sqlite_rebuild --all-with-diffs
```

Options include `--dry-run` to preview planned rebuilds. The tool creates backups and carefully migrates data to match the canonical schema.

---

## 🧪 Development Notes

- 🧱 **`src/` layout**:
  - Avoids accidental imports from the working directory
  - Always develop with `pip install -e .`

- 🖼️ **Engine/UI separation**:
  - Keep Streamlit-specific code in `multi_agent_dashboard/ui`
  - Engine (`engine.py`) and services (`services.py`) should remain UI-agnostic for reuse in scripts/tests

- 🗃️ **DB access layering**:
  - `db/infra`: low-level connections, schema, migrations
  - `db/*.py`: DAOs for agents, pipelines, runs, metrics
  - `db/services.py`: higher-level transactional APIs used by the UI and other frontends

- 🔐 **Prompt safety & caps**:
  - Use `utils.safe_format` for prompt formatting
  - Respect centralized caps (`AGENT_INPUT_CAP`, `AGENT_OUTPUT_CAP`) to avoid unbounded prompts

- 🎨 **UI theming & symbols**:
  - Shared color and emoji schemes live in `config.py`
  - Avoid hardcoding colors/symbols in UI components

### 🔁 Typical Schema-Change Flow

To add a new column or table:

1. Update `schema.py` (canonical schema)
2. Run:

   ```bash
   python -m multi_agent_dashboard.db.infra.generate_migration add_new_field --dry-run
   python -m multi_agent_dashboard.db.infra.generate_migration add_new_field
   ```

3. Review the generated SQL in `data/migrations/`
4. Run the app (or a script calling `init_db`) to apply migrations
5. If migration files end with `_REQUIRES_REBUILD` and you’re on a non-empty DB:
   - Run `sqlite_rebuild.py` with `--all-with-diffs` (or per-table) after taking backups

---

## 🤝 Contributing

Contributions are welcome. To keep the project healthy:

- ✅ Keep UI changes confined to `multi_agent_dashboard/ui`
- 🚫 Avoid introducing `sys.path` hacks
- 🗃️ Use DAOs and `services.py` instead of direct SQLite access
- 🧱 Always use the migration system for schema changes
- 🧠 Preserve engine/UI separation; keep the engine free of Streamlit dependencies
- 🧪 Add or extend tests for new engine, DB, or migration behavior where applicable

---

## 📄 License

This project is licensed under the terms described in the `LICENSE` file in this repository.

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