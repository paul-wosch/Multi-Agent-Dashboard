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

   Note: Installing in editable mode is recommended so the `multi_agent_dashboard` package is importable in your environment. See the "How to run commands" note below if you prefer not to install.

   Important: module-style commands shown elsewhere in this README (for example, `python -m multi_agent_dashboard.db.infra.generate_migration ...`) require that the package be importable in your environment — e.g. after `pip install -e .` or with `PYTHONPATH` including `src/`. If you prefer not to install, run the helper scripts directly from the repository root, e.g.:

   ```bash
   python src/multi_agent_dashboard/db/infra/generate_migration.py ...
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

Quick note — How to run CLI scripts
- Module-style (recommended for examples in this README):
  - `python -m multi_agent_dashboard.db.infra.generate_migration ...`
  - Requires the package to be importable, i.e. `pip install -e .` (or appropriate PYTHONPATH).
- Direct script invocation (works without `pip install -e .`):
  - Run the helper script from the repository top-level, e.g.:
    - `python src/multi_agent_dashboard/db/infra/generate_migration.py ...`
  - This can be handy for quick one-off runs when you don't want to install the package.

---

## 🧩 System Requirements

- 🐍 **Python**: >=3.10 (tested with CPython 3.14)
- 💻 **OS**: Tested on macOS; should work on Linux and Windows with appropriate environment setup
- 🌐 **Network**: Outbound HTTPS access to OpenAI’s APIs
- 🔑 **Credentials**: Valid `OPENAI_API_KEY` in `.env`

Note: pandas (and numpy) are installed as direct dependencies of Streamlit, so you generally won't need to pip install `pandas` separately for the UI/analytics workflows that rely on Streamlit. (See Streamlit docs.) ([docs.streamlit.io](https://docs.streamlit.io/deploy/concepts/dependencies?utm_source=openai))

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

(If you prefer not to install, see the Quick Start note above for how to invoke helper scripts directly.)

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
- Initialize a rotating log file under `data/logs/` (see Logging below)

### 🩺 Troubleshooting First Run

- ❗ **No API key or invalid key**
  - Symptom: UI loads, but model calls fail with authentication/authorization errors.
  - Fix: Set `OPENAI_API_KEY` in `.env`, restart Streamlit.

- ❗ **Permission issues**
  - Symptom: Errors writing to `data/db` or `data/logs`.
  - Fix: Ensure you have write permissions in the repo directory; adjust Docker/container volume mounts if applicable.

- ❗ **Python version errors**
  - Symptom: Syntax errors on union types (`str | None`) or similar.
  - Fix: Use Python >=3.10; this project is tested with CPython 3.14 (upgrade your interpreter if you encounter syntax incompatibilities).

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

### 🧭 Recommended safe workflow (concise)

Follow this checklist to reduce errors when changing schema:

1. Update the canonical schema:
   - Edit `src/multi_agent_dashboard/db/infra/schema.py` to model the intended change.
2. Preview changes (dry-run):
   - After installing the package (recommended), run:
     ```bash
     python -m multi_agent_dashboard.db.infra.generate_migration my_change --dry-run
     ```
   - Or run the script directly from repo top if not installed:
     ```bash
     python src/multi_agent_dashboard/db/infra/generate_migration.py my_change --dry-run
     ```
   - Review the diff output carefully.
3. Generate migration SQL files:
   ```bash
   python -m multi_agent_dashboard.db.infra.generate_migration my_change
   ```
   (or run the script directly if you didn't install the package)
   - This writes SQL under `data/migrations/` (e.g. `000_...`, `001_...`).
4. Apply migrations:
   - Start the app (or run a script which calls `init_db`) to apply migrations; `init_db` will apply new migrations automatically.
5. Handle `_REQUIRES_REBUILD` migrations:
   - If the migration filename includes `_REQUIRES_REBUILD` and your DB is non-empty, run the safe rebuild tool:
     ```bash
     python -m multi_agent_dashboard.db.infra.sqlite_rebuild --all-with-diffs data/db/multi_agent_runs.db
     ```
     (This tool creates backups before destructive operations. Use `--dry-run` to preview.)
   - If your DB is fresh (no user tables or all user tables empty), the migration system may auto-rebuild those tables for you during init; otherwise, use `sqlite_rebuild.py` to make the rebuild explicit and safe.

Note about "fresh DB" heuristic:
- The system treats a DB as "fresh" when no user-created tables exist, or when existing user tables are empty. In that case, rebuilds required by a migration may be applied automatically. For any non-empty DB you should run `sqlite_rebuild.py` explicitly to avoid unexpected destructive changes and to ensure data is preserved/migrated.

### 🛠 CLI Tools (examples)

From the project root, two ways to run tools:

- Module-style (recommended after `pip install -e .`):

  ```bash
  # Preview diffs (no files written)
  python -m multi_agent_dashboard.db.infra.generate_migration add_new_feature --dry-run

  # Generate migration files under data/migrations/
  python -m multi_agent_dashboard.db.infra.generate_migration add_new_feature
  ```

- Direct script invocation (no install required; run from repo root):

  ```bash
  python src/multi_agent_dashboard/db/infra/generate_migration.py add_new_feature --dry-run
  python src/multi_agent_dashboard/db/infra/generate_migration.py add_new_feature
  ```

Rebuild examples:

```bash
# Rebuild a single table in-place (with backup)
python -m multi_agent_dashboard.db.infra.sqlite_rebuild agents

# Rebuild all tables with pending FK/constraint diffs
python -m multi_agent_dashboard.db.infra.sqlite_rebuild --all-with-diffs data/db/multi_agent_runs.db
```

Use `--dry-run` to preview rebuild plans before executing. The rebuild tool creates backups by default.

---

## 🧪 Development Notes

- 🧱 **`src/` layout**:
  - Avoids accidental imports from the working directory
  - Always develop with `pip install -e .` for module-style runs and imports

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

- 🧪 Tests & CI:
  - Unit tests are not yet implemented (help wanted). See "Status & Known Gaps" below for details and how to contribute.

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

## 🧾 Logging

- Log directory and files:
  - Log path (rotating file): `data/logs/application.log`
  - The app uses a RotatingFileHandler with these parameters:
    - maxBytes = 5 * 1024 * 1024 (5 MB) per file
    - backupCount = 3 (keeps up to 3 rotated backups)
- Logs also stream to stdout for easy Streamlit viewing.

---

## 🔎 Troubleshooting & FAQ

- Missing `OPENAI_API_KEY`:
  - Symptom: UI appears but LLM calls fail with authentication errors.
  - Fix: Add `OPENAI_API_KEY` to `.env` and restart the app.

- Permission errors writing to `data/`:
  - Symptom: PermissionError when creating DB/logs.
  - Fix: Ensure your user or container has write access to the project directory. Adjust mount options in Docker or CI.

- Python version errors:
  - Symptom: Syntax errors for newer syntax constructs.
  - Fix: Use the project-tested interpreter (this repo is tested with CPython 3.14). Ensure your environment uses a compatible Python >=3.10 if 3.14 is not available.

- Graphviz rendering/export confusion:
  - The Python `graphviz` package is included as a Python dependency in pyproject. A system-level Graphviz installation (the `dot` binary) is only required if you plan to render/export graphs to image/PDF files locally using the Graphviz toolchain. Typical in-browser Streamlit `graphviz_chart` usage does not require the system `dot` binary, but exporting to files (e.g., `graphviz.Source(...).render(...)`) may require installing Graphviz on your OS (e.g., `apt install graphviz` or `brew install graphviz`).

- Migrations showing `_REQUIRES_REBUILD`:
  - Symptom: Migration file name includes `_REQUIRES_REBUILD`.
  - Fix: Read the migration comments, back up your DB, and run:
    ```bash
    python -m multi_agent_dashboard.db.infra.sqlite_rebuild --all-with-diffs data/db/multi_agent_runs.db
    ```
    Use `--dry-run` first to preview.

- Tests / CI:
  - Symptom: You expect tests to run but the `tests/` folder is empty or minimal.
  - Fix: Unit tests are currently not implemented.

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
- CHANGELOG: not currently maintained — a `CHANGELOG.md` would be a helpful addition for releases.
- CI: add checks for linting and tests once a test suite exists.

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