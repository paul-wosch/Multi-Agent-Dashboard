# Configuration Migration Feasibility Report (V3)

## Executive Summary

**Confidence Level: 99/100** – Migration of hardcoded constants from `config.py` to dedicated YAML configuration files is **highly feasible** with minimal risk and clear separation of concerns.

### Key Findings

1. **24 hardcoded constants** identified in `config.py` across 5 logical domains
2. **Zero technical blockers** – all constants are imported directly via module‑level variables
3. **Clean separation** already exists between environment‑sensitive settings (`.env`) and developer‑configurable constants
4. **Backward compatibility** can be maintained by moving `config.py` into a package and exposing constants via `__init__.py` (consistent with existing packaging patterns)

## Constraints (File Locations, Packaging Pattern & Flow)

1. **YAML files location**: `config/` directory at project root (not inside `src/`)
2. **Config package location**: `src/multi_agent_dashboard/config/` Python package containing all config‑related modules
3. **Public API**: Constants exposed via `__init__.py` (same as `engine/`, `runtime/`, `llm_client/` packages)
4. **Flow**: `config/*.yaml` → loaded by config package → modules import from `multi_agent_dashboard.config`
5. **Environment variables**: Remain in `.env` and loaded by config package

## Current State Analysis

### `config.py` Constants Inventory

| Domain | Constants | Usage Count | Notes |
|--------|-----------|-------------|-------|
| **Paths & Directories** | `DATA_DIR`, `DB_DIR`, `MIGRATIONS_DIR`, `LOGS_DIR`, `DOTENV_FILE`, `LOG_FILE`, `PROVIDER_DATA_DIR` | 7 | Internal path resolution; could be configurable |
| **Database Defaults** | `DB_FILE` (default) | 1 | Overridable via `DB_FILE` env var |
| **Agent Configuration** | `AGENT_INPUT_CAP`, `AGENT_OUTPUT_CAP`, `AGENT_SNAPSHOTS_AUTO`, `AGENT_SNAPSHOT_PRUNE_AUTO`, `AGENT_SNAPSHOT_PRUNE_KEEP` | 5 | Agent behavior controls |
| **Provider Data** | `PROVIDER_MODELS_ALL_FILE`, `PROVIDER_MODELS_FILE`, `TEMPLATE_OLLAMA_MODELS_FILE`, `LOCAL_OLLAMA_MODELS_FILE`, `MODELS_DEV_URL` | 5 | Dynamic pricing/capabilities data sources |
| **UI Configuration** | `ATTACHMENT_FILE_TYPES_RESTRICTED`, `ATTACHMENT_FILE_TYPES`, `UI_COLORS` | 3 | UI appearance and file‑upload restrictions |
| **Logging Defaults** | `LOG_LEVEL` (default) | 1 | Default log level (env‑overridable) |

### Import Dependencies

- **20+ files** import config constants directly
- **Most referenced**: `DB_FILE_PATH` (database), `UI_COLORS` (UI), `AGENT_SNAPSHOT_PRUNE_KEEP` (snapshots)
- **No dynamic imports** – all references are static module‑level imports
- **Environment variables** (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `LOG_LEVEL`) already separated in `.env`

## Proposed YAML Configuration Structure

### Directory Layout
```
config/                           # Project root (new)
├── paths.yaml                    # Directory names & file defaults
├── agents.yaml                   # Agent caps & snapshot behavior
├── providers.yaml                # Provider‑data file names & URLs
├── ui.yaml                       # UI colors, file‑upload settings
└── logging.yaml                  # Default log level (optional)

src/multi_agent_dashboard/config/ # Python config package (new)
├── __init__.py                   # Exports all public constants (UI_COLORS, DB_FILE_PATH, etc.)
├── core.py                       # Path resolution, .env loading, YAML integration
├── loader.py                     # YAML loading & validation (used by core.py)
├── models.py                     # Pydantic models (optional)
└── constants.py                  # Default fallback values (optional)
```

### Package‑Public‑API Pattern (consistent with existing packages)

**`src/multi_agent_dashboard/config/__init__.py`**:
```python
# Public API – exposes exactly the same names as current config.py
from .core import (
    UI_COLORS,
    DB_FILE_PATH,
    AGENT_INPUT_CAP,
    AGENT_OUTPUT_CAP,
    AGENT_SNAPSHOTS_AUTO,
    AGENT_SNAPSHOT_PRUNE_AUTO,
    AGENT_SNAPSHOT_PRUNE_KEEP,
    ATTACHMENT_FILE_TYPES_RESTRICTED,
    ATTACHMENT_FILE_TYPES,
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    LOG_LEVEL,
    configure_logging,
    # … all other constants
)

__all__ = [
    "UI_COLORS",
    "DB_FILE_PATH",
    # … all exported names
]
```

**`src/multi_agent_dashboard/config/core.py`**:
- Contains the logic currently in `config.py`
- Loads YAML via `loader.py`
- Loads `.env` file
- Computes derived paths (`DATA_PATH`, `DB_FILE_PATH`, etc.)
- Exports all constants as module attributes

### File‑Content Examples (YAML files identical to V2)

**`config/paths.yaml`**
```yaml
# Directory names (relative to PROJECT_ROOT)
data_dir: "data"
db_dir: "db"
migrations_dir: "migrations"
logs_dir: "logs"

# Default file names
dotenv_file: ".env"
log_file: "application.log"
db_file: "multi_agent_runs.db"          # can be overridden by DB_FILE env
provider_data_dir: "data/provider_models"
```

**`config/agents.yaml`**
```yaml
# Token caps per agent
agent_input_cap: 40000
agent_output_cap: 50000

# Snapshot behavior
agent_snapshots_auto: false
agent_snapshot_prune_auto: false
agent_snapshot_prune_keep: 100
```

**`config/providers.yaml`**
```yaml
# Provider‑data file names
provider_models_all_file: "provider_models_all.json"
provider_models_file: "provider_models.json"
template_ollama_models_file: "template_ollama_models.json"
local_ollama_models_file: "local_ollama_models.json"

# External data source
models_dev_url: "https://models.dev/api.json"
```

**`config/ui.yaml`**
```yaml
# File‑upload restrictions
attachment_file_types_restricted: false
attachment_file_types:
  - "txt"
  - "pdf"
  - "csv"
  - "md"
  - "json"
  - "log"
  - "py"
  - "sql"
  - "patch"
  - "LICENSE"
  - "toml"
  - "gitignore"

# Color palette & symbols
ui_colors:
  red:
    value: "#dc3545"
    symbol: "🔴"
  orange:
    value: "#fd7e14"
    symbol: "🟠"
  # … (remaining colors unchanged)
```

**`config/logging.yaml` (optional)**
```yaml
# Default log level (can still be overridden by LOG_LEVEL env)
default_log_level: "INFO"
```

## Migration Approach

### Phase 1: Package & Loader Setup
1. Create `config/` directory at project root with YAML files (current defaults)
2. Create `src/multi_agent_dashboard/config/` package with `__init__.py`
3. Implement `loader.py` with YAML loading via `yaml.safe_load()` and Pydantic validation
4. Implement `core.py` that:
   - Loads `.env` (copy logic from current `config.py`)
   - Loads YAML files via `loader.py`
   - Computes derived paths (`DATA_PATH`, `DB_FILE_PATH`, etc.)
   - Exports all constants as module attributes

### Phase 2: Update Public API
1. Populate `__init__.py` with imports from `core.py` (all public constants)
2. **Remove** the standalone `src/multi_agent_dashboard/config.py` file
3. Verify all imports still work (`from multi_agent_dashboard.config import UI_COLORS`)

### Phase 3: Gradual Migration
1. Move **one domain at a time** (e.g., UI constants first) from hardcoded values in `core.py` to YAML
2. Test each migration with manual agent runs
3. Update `core.py` to read from YAML for that domain
4. Verify all imports still work unchanged

### Phase 4: Final Cleanup
1. Remove all hardcoded constants from `core.py` (they now come from YAML)
2. Add fallback defaults in loader for missing YAML keys
3. Document new configuration structure in `AGENTS.md`

## Backward‑Compatibility Guarantees

- **No changes to import statements** – modules continue using `from multi_agent_dashboard.config import UI_COLORS`
- **Environment variables remain in `.env`** – no migration needed
- **Path‑resolution logic stays in config package** – only source of values changes
- **Default values preserved** – YAML files ship with current defaults
- **Public API identical** – same constant names, same module‑level access

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| YAML parsing errors | Low | Medium | Use `yaml.safe_load()`; validate with Pydantic |
| Missing YAML keys | Low | Low | Loader provides same defaults as today |
| Performance impact | Negligible | Low | Load once at module import; cache |
| Path‑resolution order | Low | Medium | Load YAML before computing derived paths |
| Root‑config path resolution | Low | Medium | Use `PROJECT_ROOT / "config"` (already exists) |
| Import breakage | Low | High | Keep `__init__.py` exports identical; test all imports |

## Recommendations

### Immediate Actions

1. **Start with UI configuration** – lowest risk, immediate visual verification
2. **Add Pydantic validation** – ensures type safety and early error detection
3. **Update `AGENTS.md`** – document new configuration files and location

### Implementation Details

- **Use `pyyaml`** – add to `requirements.txt` or `pyproject.toml`
- **Pydantic optional** – but recommended for validation and type hints
- **Package structure** – follow existing patterns (`engine/`, `runtime/`, `llm_client/`)
- **Path resolution** – `core.py` will have `PROJECT_ROOT`; use it to locate `config/`

### Long‑Term Considerations

- **Keep `.env` for secrets** – API keys, database passwords
- **Environment‑specific configs** – future possibility (e.g., `config/dev/`, `config/prod/`)
- **Centralized validation** – all config validation in one place

## Implementation Checklist

- [ ] Install `pyyaml` dependency (add to `requirements.txt` or `pyproject.toml`)
- [ ] Create `config/` directory at project root with YAML files (current defaults)
- [ ] Create `src/multi_agent_dashboard/config/` package with `__init__.py`
- [ ] Implement `loader.py` with YAML loading and validation
- [ ] Implement `core.py` with `.env` loading, YAML integration, path resolution
- [ ] Populate `__init__.py` with exports from `core.py`
- [ ] Remove standalone `config.py` (or keep as backward‑compatibility shim if needed)
- [ ] Verify all imports still work (run existing tests)
- [ ] Manually test agent creation, file upload, and UI colors
- [ ] Update `AGENTS.md` with new configuration documentation

## Conclusion

Migration of hardcoded `config.py` constants to YAML files is **strongly recommended**. It provides:

1. **Clean separation** – developer‑configurable settings vs. environment‑sensitive secrets
2. **Improved maintainability** – logical grouping, comments, version‑control friendliness
3. **Future‑proofing** – easy addition of environment‑specific configurations
4. **Zero disruption** – existing code continues to import from `multi_agent_dashboard.config` unchanged
5. **Consistent packaging** – follows the same public‑API pattern as `engine/`, `runtime/`, `llm_client/` packages
6. **Adherence to best practices** – config files in root, Python modules in `src/`