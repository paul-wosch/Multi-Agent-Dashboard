# 🤖 Multi-Agent Dashboard

## 📖 Description

**Multi-Agent Dashboard** is a Streamlit-based Python application for building, managing, and running multi-agent pipelines with persistent storage and rich observability. It provides a UI for configuring agents, executing pipelines, inspecting outputs, comparing results, reviewing historical runs stored in SQLite, and analyzing cost, latency, and logs.

The project is structured as a **standard Python package using a `src/` layout**, ensuring clean imports, testability, and long-term maintainability.

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo-root>
```

---

### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install the project in editable mode

This step is **required** when using a `src/` layout.

```bash
pip install -e .
```

This makes the `multi_agent_dashboard` package importable in Streamlit, tests, and scripts.

---

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional:

```text
LOG_LEVEL=INFO
```

---

### 5️⃣ Run the dashboard

From the project root:

```bash
streamlit run src/multi_agent_dashboard/ui/app.py
```

Then open your browser at:

👉 [http://localhost:8501](http://localhost:8501)

---

## 🧭 Usage

* 🎛️ Create, edit, duplicate, delete, rename, and revert agents via the UI
* 🔗 Build and execute multi-agent pipelines (including ad-hoc pipelines with session-persisted state)
* 👀 Inspect agent outputs, intermediate state, and pipeline warnings
* 🔍 Compare outputs between agents with a unified diff tool
* 🕒 Review historical runs stored in SQLite with rich metadata (models, JSON flags, metrics)
* 📦 Export runs, agent outputs, and pipeline definitions as JSON
* 📁 Attach files to agents and runs for mixed text/file LLM calls
* 📊 Analyze per-run and per-agent **cost & latency** via dedicated metrics views
* 📜 Inspect application logs live from the UI with filtering and download

---

## 🛠️ Tech Stack

* **Python** – core application logic
* **Streamlit** – web-based UI
* **SQLite** – persistent storage for agents, prompts, runs and metrics
* **OpenAI API** – LLM execution via an abstracted client
* **Graphviz** – agent pipeline visualization with metrics annotations

---

## ✨ Features

* 🤖 Multi-agent pipeline execution with a unified dashboard UI
* 🧩 Dynamic agent configuration (model, role, inputs, outputs)
* 💾 SQLite-backed persistence for agents, prompt versions, runs, and output metadata
* 🕒 Versioned agent prompts with history, revert, and inspection
* 🗺️ Agent graph visualization for pipeline flow clarity (with performance and cost overlays)
* 🧪 Output comparison and diff tooling
* 🧱 Centralized SQLite migration system with FK-aware migration generation and rebuild helpers
* 🔌 Decoupled LLM client abstraction with typed errors, retries, normalization, and rate-limit hooks
* 📤 JSON export for pipelines, agents, and historical runs (including metadata)
* 🧠 Strict vs permissive execution modes with explicit writeback and contract validation
* ⚠️ Surface pipeline warnings and input contract violations directly in the dashboard
* 🧾 Rich output metadata (JSON vs markdown, model used) for agents and final pipeline outputs
* 🔐 Safe prompt formatting via `safe_format` to avoid unsafe `.format` usage
* 📊 Cost & Latency profiling per run and per agent, with a dedicated tab and breakdown views
* 📁 File attachment support for agents with automatic MIME type detection and size limits
* 🔀 Import/export agent configurations via JSON templates and pipeline-agent export
* 🧭 Enhanced run selector showing run ID, timestamp, agent names, and an abbreviated task
* 📚 Streamlit log viewer with search, filters, download, and live updates
* 🧰 Service-layer DB architecture (DAOs + services) decoupled from Streamlit

---

## 🗂️ Repository Structure

```
repo_root/
├── .env                         # Environment variables (API keys, log level; not committed)
├── .gitignore                   # Ignore sensitive/generated files
├── LICENSE                      # Project license
├── pyproject.toml               # Project metadata, dependencies, packaging config
├── README.md                    # Project documentation
├── src/
│   └── multi_agent_dashboard/   # Main Python package (src layout)
│       ├── __init__.py          # Package marker
│       ├── config.py            # Global configuration, paths, logging setup
│       ├── engine.py            # Core multi-agent orchestration engine (UI-agnostic)
│       ├── llm_client.py        # LLM client abstraction (retries, normalization, backoff, errors)
│       ├── models.py            # Domain models (AgentSpec, AgentRuntime, PipelineSpec, etc.)
│       ├── utils.py             # Shared utilities (safe prompt formatting, helpers)
│       ├── db/
│       │   ├── __init__.py      # Database subpackage
│       │   ├── agents.py        # Agent DAOs and helpers
│       │   ├── db.py            # Backwards-compatible DB entry points
│       │   ├── infra
│       │   │   ├── __init__.py
│       │   │   ├── core.py                    # Connection, migrations bootstrap
│       │   │   ├── generate_migration.py      # Migration generator CLI
│       │   │   ├── migrations.py              # Migration application logic
│       │   │   ├── schema.py                  # Canonical schema definition
│       │   │   ├── schema_diff.py             # Column-level diffing
│       │   │   ├── schema_diff_constraints.py # FK / constraints diffing
│       │   │   └── sqlite_rebuild.py          # Safe table rebuild helpers
│       │   ├── pipelines.py     # Pipeline DAOs and helpers
│       │   ├── runs.py          # Run/agent output DAOs, metrics persistence
│       │   └── services.py      # AgentService, RunService, PipelineService (transactional APIs)
│       └── ui/
│           ├── __init__.py      # UI subpackage
│           └── app.py           # Streamlit application (presentation only)
├── data/
│   ├── db/
│   │   └── multi_agent_dashboard_runs.db  # Auto-created SQLite database (not tracked)
│   └── migrations/
│       ├── 000_create_base_tables.sql          # Initial schema
│       ├── 001_add_agent_output_metadata.sql   # Schema evolution
│       ├── 002_add_runs_metadata.sql
│       ├── 003_normalize_agent_json.sql
│       └── ...                                 # Future migrations
└── tests/                      # Tests (optional / future expansion)
```

---

## 🧪 Development Notes

* The project uses a **`src/` layout** to avoid accidental imports from the working directory.
* Always install the project with `pip install -e .` during development.
* UI code is isolated under `multi_agent_dashboard/ui` and should not depend directly on DB internals.
* Core logic (engine, models, DB, services) is UI-agnostic and safe for reuse in CLI tools or tests.
* Database access is layered:
  * `infra` for low-level connections, schema, and migrations
  * DAO modules (`agents.py`, `pipelines.py`, `runs.py`) for structured persistence
  * `services.py` for transactional, higher-level operations used by the UI
* Schema changes must go through the migration system (see `data/migrations` and `db/infra` tools).
* Logging is configured centrally and consumed via module-level loggers; avoid ad-hoc logging setup in UI or DB code.

---

## 🤝 Contributing

Contributions are welcome.

Please:

* Keep UI changes confined to `multi_agent_dashboard/ui`
* Avoid introducing `sys.path` manipulation
* Use DAOs and `services.py` instead of direct SQLite access
* Include database migrations for schema changes
* Preserve engine/UI separation and keep the engine free of Streamlit dependencies
* Add or extend tests for new engine, DB, or migration behavior

---

## 📄 License

This project is licensed under the terms described in the `LICENSE` file.

---

## 📝 Project History

The project evolved from a single-file Streamlit dashboard into a modular, package-based architecture featuring:

* A decoupled multi-agent execution engine with structured results and hooks
* Versioned prompt management with atomic agent operations
* Persistent execution history with rich metadata and FK-aware migrations
* A clean `src/`-based layout for long-term maintainability
* A DAO + service-based database layer, decoupled from Streamlit
* First-class observability: logging panel, pipeline warnings, input/output contracts, and cost/latency metrics
* Advanced UX features: file attachments, JSON agent import/export, ad-hoc pipelines, and improved run selection