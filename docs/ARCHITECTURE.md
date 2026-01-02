# 🏗️ Architecture (expanded)

This document expands the short architecture overview in the README with responsibilities and common extension points.

## High-level layers

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

### A typical flow:

```text
UI (Streamlit)
  → Services (AgentService / RunService / PipelineService)
    → DAOs (agents / pipelines / runs)
      → DB (SQLite)
  → Engine (orchestrates agents & LLM calls)
    → LLMClient (OpenAI Responses API + tools)
```

### Extension points

- Add new tools: implement tool adapters and register them in the engine/tool registry.
- Add new persistence backends: replace DAO internals while keeping service contracts.
- Add integrations (e.g., telemetry): use a pluggable logging/metrics interface in `config.py`.

### Read the code

Important files to review:

- `src/multi_agent_dashboard/ui/app.py`
- `src/multi_agent_dashboard/engine.py`
- `src/multi_agent_dashboard/llm_client.py`
- `src/multi_agent_dashboard/db/infra/schema.py`

---

## 🗂️ Repository Structure

```text
repo_root/
├── .env                         # Environment variables (API keys, log level; not committed)
├── .env.template
├── .gitignore                   # Ignore sensitive/generated files
├── LICENSE                      # Project license
├── pyproject.toml               # Project metadata, dependencies, packaging config
├── README.md                    # User-facing documentation
├── docs/                        # Advanced and maintainer-focused documentation
├── scripts/                     # One-click starter scripts
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