# Multi-Agent Dashboard - Agent Guidelines

This document provides essential information for AI agents working in this codebase. It covers commands, patterns, conventions, and gotchas.

## Overview

The Multi-Agent Dashboard is a Streamlit-based Python application for building, managing, and running multi-agent LLM pipelines. It features:

- **UI-agnostic execution engine** (`engine.py`) for reusable agent orchestration
- **Persistent SQLite storage** with automatic migrations
- **Rich observability** (cost, latency, logs, history)
- **Tool calling** with per-agent controls
- **Provider-agnostic LLM integration** (OpenAI, DeepSeek, Ollama)
- **Structured output** with JSON schema validation

The codebase follows a clean separation between UI (`src/multi_agent_dashboard/ui/`) and engine (`src/multi_agent_dashboard/`). Database access is layered with low-level infra (`db/infra/`), DAOs (`db/*.py`), and high-level services (`db/services.py`).

## Essential Commands

### Installation & Setup

```bash
# Install in editable mode (required for development)
pip install -e .

# Create .env file with required API keys
echo 'OPENAI_API_KEY=your_key_here' > .env
# Optional: DEEPSEEK_API_KEY, etc.
```

**Convenience scripts** (optional):
- `scripts/quick_start.sh` – creates venv, installs, copies .env, starts app (Linux/macOS)
- `scripts/quick_start.ps1` – same for Windows (PowerShell)


### Running the Application

```bash
# Start the Streamlit dashboard
streamlit run src/multi_agent_dashboard/ui/app.py
```

### Testing

```bash
# Run pytest (tests are in tests/)
pytest

# Run specific test file
pytest tests/test_llm_client_instrumentation_output.py
```

### Database Migrations

**Schema changes workflow:**

1. Update canonical schema in `src/multi_agent_dashboard/db/infra/schema.py`
2. Preview migration SQL (dry-run):
   ```bash
   python -m multi_agent_dashboard.db.infra.generate_migration my_change --dry-run
   ```
3. Generate migration SQL file:
   ```bash
   python -m multi_agent_dashboard.db.infra.generate_migration my_change
   ```
4. Apply migrations: Start the app (calls `init_db`) or run a script that calls the init routine.

**Rebuilding SQLite database (for non-empty DBs with `_REQUIRES_REBUILD` migrations):**

```bash
# Preview rebuild plan
python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --dry-run --all-with-diffs data/db/multi_agent_runs.db

# Execute rebuild (creates backups automatically)
python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --all-with-diffs data/db/multi_agent_runs.db
```

**Important:** Run `sqlite_rebuild.py` as a direct script from the repository root (not as a module) to ensure the internal `generate_migration.py` subprocess can be located.

### Pruning Agent Snapshots

```bash
# Run snapshot pruning (keeps latest N snapshots per agent)
python -m multi_agent_dashboard.db.infra.prune_snapshots --keep 100

# Dry-run to see what would be deleted
python -m multi_agent_dashboard.db.infra.prune_snapshots --dry-run

# Prune snapshots for a specific agent on a custom DB file
python -m multi_agent_dashboard.db.infra.prune_snapshots my_agent my_custom.db --keep 50
```

**Note:** The first positional argument is agent name (optional), second is DB path (optional). Defaults to all agents and the configured DB path.

## Project Structure

```
src/multi_agent_dashboard/
├── __init__.py
├── config.py              # Global constants, environment variables, pricing
├── engine.py              # Core multi-agent orchestration engine
├── llm_client.py          # LLM provider integration (OpenAI, DeepSeek, Ollama)
├── models.py              # Data classes (AgentSpec, AgentRuntime, etc.)
├── structured_schemas.py  # JSON schema resolution for structured output
├── runtime_hooks.py       # Runtime hooks for agent execution
├── utils.py               # Utility functions (safe_format, etc.)
├── ui/                    # Streamlit UI components
│   ├── app.py             # Main Streamlit application
│   ├── bootstrap.py       # UI initialization
│   ├── agent_editor_mode.py
│   ├── history_mode.py
│   ├── run_mode.py
│   └── ...
└── db/                    # Database layer
    ├── infra/             # Low-level DB infrastructure
    │   ├── schema.py      # Canonical SQL schema
    │   ├── generate_migration.py
    │   ├── sqlite_rebuild.py
    │   ├── migrations.py
    │   └── ...
    ├── agents.py          # Agent DAO
    ├── pipelines.py       # Pipeline DAO
    ├── runs.py            # Run DAO
    └── services.py        # High-level transactional APIs

data/                      # Runtime data (created on first run)
├── db/                    # SQLite database files
├── migrations/            # Generated migration SQL files
└── logs/                  # Application logs

tests/                     # Unit tests (pytest)
docs/                      # Project documentation
```

## Code Patterns & Conventions

### Import Style

- Use absolute imports within the package: `from multi_agent_dashboard.models import AgentSpec`
- Keep Streamlit-specific code isolated in `ui/` modules
- Engine and services remain UI-agnostic for reuse in scripts/tests

### Configuration Access

- Global constants and paths are defined in `config.py`
- Environment variables are loaded via `dotenv` and accessible as `config.OPENAI_API_KEY`, etc.
- Use `config.configure_logging()` to set up logging with rotating file handler

### LLM Provider Integration

- Provider‑agnostic client in `llm_client.py` with factory pattern
- Supported providers: `openai`, `deepseek`, `ollama`
- Provider‑specific logic is encapsulated in `_build_structured_output_adapter` and `_compute_cost`
- Structured output uses `with_structured_output(method="function_calling")` where available
- Token accounting and pricing are preserved across all providers

**Key provider‑aware functions:**
- `LLMClient._build_structured_output_adapter()` – creates provider‑specific response format
- `engine._compute_cost()` – uses `OPENAI_PRICING`/`DEEPSEEK_PRICING` from config

### Database Access Pattern

```python
from multi_agent_dashboard.db.services import get_agent_by_id, save_agent

# Use high‑level services for business logic
agent = get_agent_by_id(agent_id)
# Use DAOs directly for simple queries
from multi_agent_dashboard.db.agents import AgentDAO
dao = AgentDAO()
agents = dao.list_all()
```

### Error Handling

- Use `logger.getLogger(__name__)` for module‑specific logging
- Middleware must not raise exceptions (catch and log)
- Validation errors are captured in `agent_schema_validation_failed` flags

### Type Hints

- Use Python type hints extensively (`from typing import Dict, List, Optional, Any`)
- Dataclasses are used for data containers (`AgentSpec`, `AgentRuntime`)

### Code Style & Formatting

- No automated linting or formatting tools are configured (no `.flake8`, `.black`, `pylintrc`, etc.)
- Follow existing patterns in the codebase (indentation: 4 spaces, snake_case for variables/functions, PascalCase for classes)
- Keep imports grouped: standard library, third-party, local modules
- Use descriptive variable names; avoid one-letter names except in trivial loops

## Testing

- Tests are located in `tests/` and use plain pytest functions (no fixtures).
- Mock LLM clients to avoid API calls.
- Focus on engine behavior, schema validation, and instrumentation.
- Unit tests are optional for new features (see `PLAN.md`); smoke/manual validation is acceptable early in the cycle.

**Example test pattern:**

```python
import pytest
from multi_agent_dashboard.engine import MultiAgentEngine

def test_something():
    engine = MultiAgentEngine(llm_client=DummyClient())
    # assertions
```

## Database & Migrations

### Schema Changes

- The canonical schema is defined in `db/infra/schema.py`.
- Migrations are generated automatically via `generate_migration.py`.
- Migration files are stored in `data/migrations/` with sequential numbering.
- Filenames ending with `_REQUIRES_REBUILD` indicate destructive changes that require `sqlite_rebuild.py`.

### Fresh DB Heuristic

A database is considered “fresh” if no user‑created tables exist or existing user tables are empty. In that case, some rebuilds may be auto‑applied. For non‑empty DBs always use explicit `sqlite_rebuild.py` with backups.

### Safety Checklist

1. Back up your DB: `cp data/db/multi_agent_runs.db data/db/multi_agent_runs.db.bak`
2. Run `generate_migration --dry-run` and inspect SQL.
3. If rebuild required, run `sqlite_rebuild.py --dry-run` and review the plan.
4. Test migration application on a copy of your DB before applying to production/user data.

## Environment & Configuration

### Required Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `DEEPSEEK_API_KEY` | Optional | DeepSeek API key |
| `DB_FILE` | Optional | Override default database filename |
| `LOG_LEVEL` | Optional | Logging level (INFO, DEBUG, etc.) |

### Pricing Tables

Pricing per 1M tokens is defined in `config.py` (`OPENAI_PRICING`, `DEEPSEEK_PRICING`). The `_compute_cost` function uses these tables based on `provider_id`.

### Agent Caps

- `AGENT_INPUT_CAP = 40_000` – maximum input token count per agent
- `AGENT_OUTPUT_CAP = 50_000` – maximum output token count per agent

### UI Colors & Symbols

Color themes and emoji symbols are centralized in `config.UI_COLORS`. Avoid hardcoding colors/symbols in UI components.

## Gotchas & Known Issues

1. **SQLite rebuild invocation** – Run `sqlite_rebuild.py` as a direct script from the repository root, not as a module (`python -m ...`), to ensure the internal `generate_migration.py` subprocess can be located.

2. **DeepSeek‑reasoner structured output** – On 400 tool_choice error, the client automatically retries with `json_mode` to enforce JSON formatting without tool invocation.

3. **LangChain optional dependency** – The codebase handles missing LangChain gracefully (`_LANGCHAIN_AVAILABLE` flag). Tests mock LangChain components to avoid import errors.

4. **Provider‑feature detection** – When an AgentSpec doesn’t provide `provider_features`, the engine attempts to derive them from the `detected_provider_profile` in the raw metrics.

5. **LLM client caching** – The `LLMClient` caches chat model instances based on a fingerprint of provider features. Changing provider_features may not create a new model instance if the fingerprint remains identical.

6. **Strict schema validation & early exit** – When `strict_schema_validation=True` and a schema is missing or empty, the run sets `agent_schema_validation_failed` and may early‑exit (triggered by `strict_schema_exit` flag). The UI displays badges (`[strict schema]`, `[schema failed]`) in agent configuration titles and shows error messages for runs that exited early.

7. **Migration naming** – Migration files are numbered sequentially (e.g., `016_add_strict_schema_validation_flags.sql`). Do not rename or reorder them.

8. **Snapshot auto‑creation** – By default, saving an agent does **not** automatically create a snapshot (`AGENT_SNAPSHOTS_AUTO = False`). Change this flag in `config.py` to enable.

9. **Attachment file types** – The UI file uploader can be restricted to specific extensions via `ATTACHMENT_FILE_TYPES_RESTRICTED` and `ATTACHMENT_FILE_TYPES` in `config.py`.

10. **Provider features normalization** – Provider features (`provider_features`) are normalized to keys `tool_calling`, `structured_output`, `reasoning`, `image_inputs`, `max_input_tokens`. The engine also accepts variations (`tool_calls`, `toolcalling`, `toolCalling`, etc.) but stores them normalized.

## Further Reading

- `README.md` – Quick start and overview
- `docs/DEVELOPMENT.md` – Developer notes and schema‑change workflow
- `docs/MIGRATIONS.md` – Detailed migration and rebuild guidance
- `docs/CONFIG.md` – Configuration reference
- `PLAN.md` – Current development plan
