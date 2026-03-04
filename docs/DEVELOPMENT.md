# 🧪 Development Notes

This document expands developer-focused guidance present in the [README](../README.md).

## 📁 Project Structure

- **Modular package layout** (see [docs/ARCHITECTURE.md](ARCHITECTURE.md) for full tree):
  - `engine/` – Multi-agent orchestration engine (8 focused modules)
  - `runtime/` – Agent execution logic (6 modules including `safe_format` in `utils.py`)
  - `shared/` – Utilities shared between engine and runtime
  - `llm_client/` – Provider-agnostic LLM client with modular `core/` subpackage
  - `provider_data/` – Dynamic provider capabilities & pricing data
  - `tool_integration/` – Tool registry and provider-specific adapter
  - `ui/` – Streamlit UI components (kept UI‑specific)
  - `db/` – Database layer with low‑level `infra/`, DAOs, and high‑level `services.py`
  - `observability/` – Langfuse integration for distributed tracing

- **Configuration**: YAML files in `config/` directory, loaded via `multi_agent_dashboard.config` package.

- **Data directories**: Created at first run under `data/` (DB, migrations, provider models, logs).

- **Always develop with `pip install -e .`** for module‑style imports and runs.

## 🎯 Development Patterns

### Import Style
- Use absolute imports within the package: `from multi_agent_dashboard.models import AgentSpec`
- Import runtime classes from the runtime package: `from multi_agent_dashboard.runtime import AgentRuntime`
- Keep Streamlit‑specific code isolated in `ui/` modules
- Engine and services remain UI‑agnostic for reuse in scripts/tests

### Configuration Access
- Global constants and paths are defined in YAML files under `config/` and loaded via the `config` package.
- Environment variables are loaded via `dotenv` and accessible as `config.OPENAI_API_KEY`, etc.
- Use `config.configure_logging()` to set up logging with rotating file handler.

### LLM Provider Integration
- Provider‑agnostic client in `llm_client/` subpackage with factory pattern; uses LangChain's unified `init_chat_model` interface with provider‑specific adapters for OpenAI, DeepSeek, and Ollama.
- Modular subpackage with separate modules for instrumentation, tool binding, structured output, response normalization, and provider adapters.
- Core `LLMClient` internals decomposed into focused modules under `llm_client/core/`.
- Supported providers: `openai`, `deepseek`, `ollama`.
- Provider‑specific logic encapsulated in `_build_structured_output_adapter`, `_build_input_with_files`, and `_compute_cost`.
- Structured output uses provider‑specific methods: OpenAI JSON Schema, DeepSeek function‑calling/json‑mode, Ollama raw schema.
- Tool calling integration uses provider‑specific tool adapter with dynamic capability data (loaded from `provider_models.json` and local Ollama overrides).
- File handling uses provider‑specific message building for multimodal inputs (images, PDFs, text).
- Token accounting and pricing are preserved across all providers.

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

### Prompt Safety & Caps
- Use `multi_agent_dashboard.runtime.utils.safe_format` for prompt formatting.
- Respect centralized caps defined in `config/agents.yaml` (`AGENT_INPUT_CHAR_CAP`, `AGENT_OUTPUT_CHAR_CAP`, `AGENT_OUTPUT_TOKEN_CAP`).
- Environment variables `AGENT_INPUT_CHAR_CAP`, `AGENT_OUTPUT_CHAR_CAP`, `AGENT_OUTPUT_TOKEN_CAP` override YAML defaults.
- Token limit precedence: `STRICT_OUTPUT_TOKEN_CAP_OVERRIDE` (if `true`) enforces global cap; otherwise smallest non‑zero value among global cap and per‑agent `max_output`.

### UI Theming & Symbols
- Shared color and emoji schemes live in `config/ui.yaml` (`UI_COLORS` constant).
- Avoid hardcoding colors/symbols in UI components.

### Error Handling
- Use `logger.getLogger(__name__)` for module‑specific logging.
- Middleware must not raise exceptions (catch and log).
- Validation errors are captured in `agent_schema_validation_failed` flags.

### Type Hints & Code Style
- Use Python type hints extensively (`from typing import Dict, List, Optional, Any`).
- Dataclasses are used for data containers (`AgentSpec`, `PipelineSpec`).
- No automated linting or formatting tools are configured (no `.flake8`, `.black`, `pylintrc`, etc.).
- Follow existing patterns in the codebase (indentation: 4 spaces, snake_case for variables/functions, PascalCase for classes).
- Keep imports grouped: standard library, third-party, local modules.
- Use descriptive variable names; avoid one-letter names except in trivial loops.

## 🧪 Tests & CI

- Tests are located in `tests/` and use plain pytest functions (no fixtures).
- Mock LLM clients to avoid API calls.
- Focus on engine behavior, schema validation, and instrumentation.
- Unit tests are optional for new features; smoke/manual validation is acceptable early in the cycle.

**Example test pattern:**

```python
import pytest
from multi_agent_dashboard.engine import MultiAgentEngine

def test_something():
    engine = MultiAgentEngine(llm_client=DummyClient())
    # assertions
```

## 🔁 Schema-Change Flow (developer checklist)

### Standard Workflow
1. Update the canonical schema: Edit `src/multi_agent_dashboard/db/infra/schema.py`.
2. Preview changes (dry-run): `python -m multi_agent_dashboard.db.infra.generate_migration my_change --dry-run`
3. Generate migration SQL: `python -m multi_agent_dashboard.db.infra.generate_migration my_change`
4. Apply migrations: Start the app (or run a script which calls `init_db`) to apply migrations automatically.

### Rebuilding SQLite Database (for non‑empty DBs and migrations that include rebuild metadata)
- Preview rebuild plan: `python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --dry-run --all-with-diffs data/db/multi_agent_runs.db`
- Execute rebuild (creates backups automatically): `python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --all-with-diffs data/db/multi_agent_runs.db`

**Important:** Run `sqlite_rebuild.py` as a direct script from the repository root (not as a module) to ensure the internal `generate_migration.py` subprocess can be located.

### Fresh DB Heuristic
A database is considered “fresh” if no user‑created tables exist or existing user tables are empty. In that case, some rebuilds may be auto‑applied. For non‑empty DBs always use explicit `sqlite_rebuild.py` with backups.

### Safety Checklist
1. Back up your DB: `cp data/db/multi_agent_runs.db data/db/multi_agent_runs.db.bak`
2. Run `generate_migration --dry-run` and inspect SQL.
3. If rebuild required, run `sqlite_rebuild.py --dry-run` and review the plan.
4. Test migration application on a copy of your DB before applying to production/user data.

> For detailed migration caveats and examples, see [docs/MIGRATIONS.md](MIGRATIONS.md).

## 🔧 Status & Known Gaps

- Unit tests: Basic pytest suite exists.
- CHANGELOG: not currently maintained — a `CHANGELOG.md` would be a helpful addition for releases.
- CI: add checks for linting and tests once a test suite grows.

---

If you are working on a larger change touching schema, migrations, engine behavior, or persistence, please document the intended migration plan in your PR and include sample DB copies where appropriate for reviewer testing.

## 📚 Further Reading

- [AGENTS.md](../AGENTS.md) – Agent guidelines, code patterns, gotchas.
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) – System architecture overview.
- [docs/CONFIG.md](CONFIG.md) – Configuration reference and YAML format.
- [docs/MIGRATIONS.md](MIGRATIONS.md) – Detailed database migration guide.
- [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) – Troubleshooting common issues.
- [docs/OBSERVABILITY.md](OBSERVABILITY.md) - Langfuse setup and usage.