# Multi-Agent Dashboard - Agent Guidelines

This document provides essential information for AI agents working in this codebase. It covers commands, patterns, conventions, and gotchas.

## Overview

The Multi-Agent Dashboard is a Streamlit-based Python application for building, managing, and running multi-agent LLM pipelines. It features:

- **UI-agnostic execution engine** (`engine/` and `runtime/` packages) for reusable agent orchestration
- **Persistent SQLite storage** with automatic migrations
- **Rich observability** (cost, latency, logs, history)
- **Tool calling** with per-agent controls
- **Provider-agnostic LLM integration** (OpenAI, DeepSeek, Ollama) via provider-specific LangChain implementations with dynamic capability data
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

For detailed database migration workflow, see **APPENDIX A: Database Migrations**.

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
├── config/                           # YAML-based configuration package
│   ├── __init__.py                  # Public API (same constants as before)
│   ├── core.py                      # Core configuration loading
│   └── loader.py                    # YAML validation with Pydantic
├── engine/                             # Modular multi-agent orchestration engine
│   ├── __init__.py
│   ├── agent_executor.py
│   ├── engine_orchestrator.py
│   ├── metrics_aggregator.py
│   ├── progress_reporter.py
│   ├── schema_validator.py
│   ├── snapshot_builder.py
│   ├── state_manager.py
│   ├── types.py
│   └── utils.py
├── runtime/                            # AgentRuntime class and execution logic
│   ├── __init__.py
│   ├── agent_runtime.py                # Main AgentRuntime class
│   ├── file_processor.py               # File type detection & content decoding
│   ├── tool_converter.py               # Tool configuration merging & provider conversion
│   ├── metrics_extractor.py            # Token extraction & provider profile detection
│   ├── structured_output_detector.py   # 4‑path detection & state writeback
│   └── utils.py                        # Utility functions (safe_format, etc.)
├── shared/                             # Shared utilities between engine and runtime
│   ├── __init__.py
│   ├── instrumentation.py              # Helper functions for metrics/instrumentation extraction
│   ├── provider_capabilities.py        # Legacy module (deprecated) - capabilities now loaded from dynamic data
│   ├── runtime_hooks.py                # Runtime hooks for agent execution
│   └── structured_schemas.py           # JSON schema resolution for structured output
├── llm_client/                         # Modular LLM provider integration subpackage
│   ├── __init__.py
│   ├── chat_model_factory.py
│   ├── core.py                         # Main LLMClient class
│   ├── instrumentation.py
│   ├── multimodal/                     # Multimodal file handling
│   │   ├── __init__.py
│   │   └── multimodal_handler.py
│   ├── provider_adapters.py
│   ├── response_normalizer.py
│   ├── structured_output.py
│   ├── tool_binder.py
│   └── wrappers.py
├── models.py                           # Data classes (AgentSpec, PipelineSpec) – pure dataclasses
├── tool_integration/                   # Tool registry and provider-specific tool adapter
│   ├── __init__.py
│   ├── provider_tool_adapter.py
│   ├── registry.py
│   ├── web_fetch_tool.py
│   └── search/                         # Web search tools
│       ├── __init__.py
│       ├── duckduckgo_base.py
│       └── duckduckgo_tool.py
├── ui/                                 # Streamlit UI components
│   ├── app.py                          # Main Streamlit application
│   ├── bootstrap.py                    # UI initialization
│   ├── agent_editor_mode.py
│   ├── history_mode.py
│   ├── run_mode.py
│   └── ...
└── db/                                 # Database layer
    ├── infra/                          # Low-level DB infrastructure
    │   ├── schema.py                   # Canonical SQL schema
    │   ├── generate_migration.py
    │   ├── sqlite_rebuild.py
    │   ├── migrations.py
    │   └── ...
    ├── db.py                           # Low-level DB connection and re‑exports
    ├── agents.py                       # Agent DAO
    ├── pipelines.py                    # Pipeline DAO
    ├── runs.py                         # Run DAO
    └── services.py                     # High-level transactional APIs

data/                                   # Runtime data (created on first run)
├── db/                                 # SQLite database files
├── migrations/                         # Generated migration SQL files
├── provider_models/                    # Dynamic provider capabilities & pricing data
└── logs/                               # Application logs

tests/                                  # Unit tests (pytest)
docs/                                   # Project documentation
config/                                 # Centralized YAML‑based configuration
```

## Code Patterns & Conventions

### Import Style

- Use absolute imports within the package: `from multi_agent_dashboard.models import AgentSpec`
- Import runtime classes from the runtime package: `from multi_agent_dashboard.runtime import AgentRuntime`
- Keep Streamlit-specific code isolated in `ui/` modules
- Engine and services remain UI-agnostic for reuse in scripts/tests

### Configuration Access

- Global constants and paths are defined in YAML files under `config/` and loaded via the `config` package.
- Environment variables are loaded via `dotenv` and accessible as `config.OPENAI_API_KEY`, etc.
- Use `config.configure_logging()` to set up logging with rotating file handler

### LLM Provider Integration

- Provider‑agnostic client in `llm_client/` subpackage with factory pattern; uses LangChain's unified `init_chat_model` interface with provider‑specific adapters for OpenAI, DeepSeek, and Ollama
- Modular subpackage with separate modules for instrumentation, tool binding, structured output, response normalization, and provider adapters
- Supported providers: `openai`, `deepseek`, `ollama`
- Provider‑specific logic is encapsulated in `_build_structured_output_adapter`, `_build_input_with_files`, and `_compute_cost`
- Structured output uses provider‑specific methods: OpenAI JSON Schema, DeepSeek function‑calling/json‑mode, Ollama raw schema
- Tool calling integration uses provider‑specific tool adapter with dynamic capability data (loaded from `provider_models.json` and local Ollama overrides)
- File handling uses provider‑specific message building for multimodal inputs (images, PDFs, text)
- Token accounting and pricing are preserved across all providers

**Key provider‑aware functions:**

- `LLMClient._build_structured_output_adapter()` – creates provider‑specific response format
- `LLMClient._build_input_with_files()` – constructs provider‑specific message parts for file uploads
- `engine._compute_cost()` – uses dynamic pricing data from provider models

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
- Dataclasses are used for data containers (`AgentSpec`, `PipelineSpec`)

### Code Style & Formatting

- No automated linting or formatting tools are configured (no `.flake8`, `.black`, `pylintrc`, etc.)
- Follow existing patterns in the codebase (indentation: 4 spaces, snake_case for variables/functions, PascalCase for classes)
- Keep imports grouped: standard library, third-party, local modules
- Use descriptive variable names; avoid one-letter names except in trivial loops

## Testing

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

## Database & Migrations

Refer to **APPENDIX A: Database Migrations** for schema changes, fresh DB heuristic, and safety checklist.

## Environment & Configuration

### Required Environment Variables (`.env`)

| Variable           | Required | Description                                          |
| ------------------ | -------- | ---------------------------------------------------- |
| `OPENAI_API_KEY`   | Yes      | OpenAI API key                                       |
| `DEEPSEEK_API_KEY` | Optional | DeepSeek API key                                     |
| `OLLAMA_HOST`      | Optional | Ollama server URL (for LangChain Ollama integration) |
| `DB_FILE`          | Optional | Override default database filename                   |
| `LOG_LEVEL`        | Optional | Logging level (INFO, DEBUG, etc.)                    |

### Agent Caps

- `AGENT_INPUT_CAP = 40_000` – maximum input character count per agent
- `AGENT_OUTPUT_CAP = 50_000` – maximum output character count per agent

### UI Colors & Symbols

Color themes and emoji symbols are centralized in `config.UI_COLORS`. Avoid hardcoding colors/symbols in UI components.

## Configuration Files

Configuration is now YAML‑based, located in the `config/` directory at the project root. The system uses five YAML files:

- `paths.yaml` – directory and file names
- `agents.yaml` – agent limits and snapshot settings
- `providers.yaml` – provider‑data file names and URLs
- `ui.yaml` – UI colors and attachment file types
- `logging.yaml` – default log level

Each file is validated with Pydantic at import time; missing or malformed keys raise immediate errors.

### Environment Variables & Overrides

Secrets (API keys) and user‑specific preferences remain in `.env` (git‑ignored). The following environment variables override YAML defaults:

- `DB_FILE` – override the database file name
- `LOG_LEVEL` – override the default log level (still falls back to YAML if not set)

All other constants are defined in YAML and cannot be overridden by environment variables.

### Adding a New Configuration Value

1. Add the key‑value pair to the appropriate YAML file.
2. Update the corresponding Pydantic model in `src/multi_agent_dashboard/config/loader.py`.
3. Add the constant to `src/multi_agent_dashboard/config/core.py` (read from `_yaml_config`).
4. Export the constant in `src/multi_agent_dashboard/config/__init__.py`.

The import pattern for consumers stays unchanged:

```python
from multi_agent_dashboard.config import AGENT_INPUT_CAP, UI_COLORS, ...
```

## Dynamic Pricing & Capabilities

- Provider model capabilities and pricing per 1M tokens are loaded from external data (`provider_models.json`) with optional local overrides for Ollama models.
- The `_compute_cost` function uses these dynamic tables based on `provider_id` and model name.
- Local Ollama models default to `0.0` pricing.

### Customizing Local Ollama Models Data (Capabilities & pricing)

1. Copy the template file to create your local configuration:
   
   ```bash
   cp data/provider_models/template_ollama_models.json data/provider_models/local_ollama_models.json
   ```

2. Edit `local_ollama_models.json` to add, remove, or modify Ollama model entries.

3. Restart the application – local models are automatically loaded and take precedence over external definitions for the same `ollama|model` composite keys.

**Note**: The local file is git‑ignored; your customizations persist across updates.

### Manual Data Updates

To refresh the external provider data (OpenAI, DeepSeek, etc.):

- Delete `data/provider_models/provider_models_all.json` to trigger a fresh download from the upstream source.
- Delete `data/provider_models/provider_models.json` to re‑extract the canonical data from the downloaded file.

Local Ollama models remain unaffected by these updates.

### File‑State Management

The system maintains two primary JSON files in `data/provider_models/`:

- `provider_models_all.json` – raw downloaded data from the upstream source (`https://models.dev/api.json`)
- `provider_models.json` – filtered copy containing only the supported providers (`openai`, `deepseek`) with all fields preserved
- **Composite cache keys**: Internal lookups use `"provider|model"` as unique identifier to disambiguate duplicate model IDs across providers.

**Initialization flow** (first run):

1. Both files missing → download `api.json`, save as `provider_models_all.json`
2. Extract filtered copy → `provider_models.json`
3. Load `provider_models.json` into memory cache

**Update flow** (manual):

- Delete `provider_models_all.json` → re‑download on next run
- Delete `provider_models.json` → re‑extract from `provider_models_all.json`
- Modify `provider_models.json` manually → changes persist (file never overwritten)

**Local Ollama files**:

- `template_ollama_models.json` – static template (git‑tracked)
- `local_ollama_models.json` – user‑customizable (git‑ignored)
  Local entries take precedence over external definitions for the same `ollama|model` composite keys.

**Error recovery**:

- Network failure → log ERROR, keep existing files
- Malformed JSON → log ERROR, fallback to existing `provider_models.json` if present
- Missing fields → log WARNING, treat missing capability as `False`, missing pricing as `0.0`

### Advisory Usage

Capability data is used **advisory only** – the primary source of truth for agent features is the actual agent configuration. The dynamic data provides up‑to‑date defaults and warnings but never overrides user configuration.

## Provider-Specific LangChain Architecture

**Architecture**: The codebase uses LangChain's unified `init_chat_model` interface with provider‑specific adapters and **advisory** capability data loaded from external sources and local overrides. This simplifies the codebase, eliminates static mapping maintenance, and provides full control over supported providers (OpenAI, DeepSeek, Ollama).

**High‑Level Directives**:

1. **Advisory Capability Data**:
   
   - Provider capabilities (tool calling, structured output, vision, etc.) are loaded from `provider_models.json` (external data) with optional local Ollama overrides in `local_ollama_models.json`.
   - Use `get_capabilities(provider_id, model)` and `supports_feature(provider_id, feature, model)` for advisory detection (warnings, UI defaults).
   - Agent configuration remains the primary source of truth; dynamic data does not override user choices.

2. **Provider‑Specific Implementations**:
   
   - Use LangChain's unified `init_chat_model` interface, which internally selects the appropriate provider integration based on `provider_id`.
   - Implement provider‑specific adapters for tool calling (`provider_tool_adapter.py`), structured output (`_build_structured_output_adapter`), and file handling (`_build_input_with_files`).
   - Maintain existing working implementations for OpenAI, DeepSeek and Ollama.

3. **Unified Interface**:
   
   - Maintain provider‑agnostic client interface in `llm_client/` subpackage with factory pattern.
   - Internal implementations are provider‑specific but expose a consistent API to the engine and UI.
   - All code paths use LangChain's unified `init_chat_model` interface, which internally selects the appropriate provider integration while exposing a consistent API to the engine and UI.

4. **Advisory Capability Detection**:
   
   - Feature detection uses dynamic capability data for advisory purposes (warnings, UI defaults); actual feature binding respects agent configuration.
   - Model‑specific capabilities (e.g., Ollama `llava` vs `llama3`) are loaded from external data or local overrides for advisory guidance.
   - Any dynamic capability detection must be based on provider APIs, not heuristics or hardcoded model names.
   - Never implement prompt‑based tool‑calling emulation; tool invocation via prompts remains the user's responsibility via manual agent prompt configuration.

5. **Adding New Providers**:
   
   - To add a new provider, ensure its capabilities are included in the external provider data (`provider_models.json`) and implement provider‑specific adapters for tools, structured output, and file handling.
   - New providers integrate via LangChain's unified `init_chat_model` interface, which uses the provider's native library internally.

6. **Modular Design Approach**:
   
   - Prefer a highly modularized approach over nested branches. Split existing code affected by the refactor into separate helper functions, modules, and packages.
   - Avoid deep nesting of provider‑specific branches; create focused modules (e.g., `tool_integration/provider_tool_adapter.py`, `multimodal_handler.py`).
   - Keep modules focused on single responsibility and preserve existing interfaces while refactoring internals.

**Key Benefit**: Adding a new provider requires ensuring its capabilities are included in the external data and implementing provider‑specific adapters, providing full control over provider integration without maintaining static maps.

## Further Reading

- `README.md` – Quick start and overview
- `docs/DEVELOPMENT.md` – Developer notes and schema‑change workflow
- `docs/MIGRATIONS.md` – Detailed migration and rebuild guidance
- `docs/CONFIG.md` – Configuration reference
- `PLAN.md` – Current development plan

## APPENDIX A: Database Migrations

### Schema Changes Workflow

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

### Rebuilding SQLite Database

**Rebuilding SQLite database (for non-empty DBs with `_REQUIRES_REBUILD` migrations):**

```bash
# Preview rebuild plan
python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --dry-run --all-with-diffs data/db/multi_agent_runs.db

# Execute rebuild (creates backups automatically)
python src/multi_agent_dashboard/db/infra/sqlite_rebuild.py --all-with-diffs data/db/multi_agent_runs.db
```

**Important:** Run `sqlite_rebuild.py` as a direct script from the repository root (not as a module) to ensure the internal `generate_migration.py` subprocess can be located.

### Schema Changes Details

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

## APPENDIX B: Gotchas & Known Issues

1. **SQLite rebuild invocation** – Run `sqlite_rebuild.py` as a direct script from the repository root, not as a module (`python -m ...`), to ensure the internal `generate_migration.py` subprocess can be located.

2. **DeepSeek‑reasoner structured output** – On 400 tool_choice error, the client automatically retries with `json_mode` to enforce JSON formatting without tool invocation.

3. **LangChain optional dependency** – The codebase handles missing LangChain gracefully (`_LANGCHAIN_AVAILABLE` flag). Tests mock LangChain components to avoid import errors.

4. **Provider‑feature detection & normalization** – The `provider_features` field is a legacy artifact stored for backward compatibility only. When an AgentSpec doesn’t provide `provider_features`, the engine may derive them from `detected_provider_profile` in raw metrics. The field is normalized to keys `tool_calling`, `structured_output`, `reasoning`, `image_inputs`, `max_input_tokens` (accepting variations like `tool_calls`, `toolcalling`, etc.). This functionality is already deactivated in the UI and will be removed in a future release. Agent configuration (tools, structured output, etc.) remains the source of truth.

5. **Provider capability mapping** – Provider capabilities (tool calling, structured output, vision, etc.) are loaded dynamically from `provider_models.json` and local Ollama overrides for each provider/model combination as an advisory reference; agent configuration remains the primary source of truth.

6. **Multimodal file handling** – File uploads are processed by provider‑specific message building in `LLMClient._build_input_with_files()`. Images are base64‑encoded for vision‑capable providers (OpenAI). Text files are decoded as UTF‑8 text parts. PDFs are extracted as text if `pypdf` is installed. If provider does not support vision, files are concatenated as plain text with headers. **Vision/tool support detection uses dynamic capability data as advisory; actual feature binding respects agent configuration.**

7. **Tool calling adapter** – Tool integration uses provider‑specific tool adapter (`provider_tool_adapter.py`) that respects exact tool configuration from `AgentSpec.tools`; dynamic capability data is used only for warnings and UI defaults (no automatic conversions).

8. **Structured output methods** – Structured output uses provider‑specific methods: OpenAI JSON Schema, DeepSeek function‑calling/json‑mode, Ollama raw schema. The appropriate method is selected via provider detection; capability mapping provides advisory hints.

9. **LLM client caching** – The `LLMClient` caches chat model instances based on a fingerprint of provider and model configuration. Changing `provider_features` (a legacy field) does not affect caching.

10. **Strict schema validation & early exit** – When `strict_schema_validation=True` and a schema is missing or empty, the run sets `agent_schema_validation_failed` and may early‑exit (triggered by `strict_schema_exit` flag). The UI displays badges (`[strict schema]`, `[schema failed]`) in agent configuration titles and shows error messages for runs that exited early.

11. **Migration naming** – Migration files are numbered sequentially (e.g., `016_add_strict_schema_validation_flags.sql`). Do not rename or reorder them.

12. **Snapshot auto‑creation** – By default, saving an agent does **not** automatically create a snapshot (`AGENT_SNAPSHOTS_AUTO = False`). Change this flag in `agents.yaml` to enable.

13. **Attachment file types** – The UI file uploader can be restricted to specific extensions via `ATTACHMENT_FILE_TYPES_RESTRICTED` and `ATTACHMENT_FILE_TYPES` in `ui.yaml`.
