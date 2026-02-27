# Migration to YAML Configuration: Implementation Strategy

**Date**: 2025‑02‑25 (created); 2025-02-26 (completed)
**Based on**: `docs/implementation‑strategies/260225‑1736_config_migration_feasibility_report_V3.md`  
**Confidence**: 99/100 (no technical blockers, incremental migration possible)

## Executive Summary

This document provides a **progressive, incremental, step‑by‑step strategy** to migrate hard‑coded constants from `src/multi_agent_dashboard/config.py` to dedicated YAML configuration files. The migration keeps the public API unchanged (`from multi_agent_dashboard.config import UI_COLORS`), follows the existing packaging pattern (`engine/`, `runtime/`, `llm_client/`), and separates developer‑configurable settings (YAML) from environment‑sensitive secrets (`.env`).

### Goals
- Move **24 hard‑coded constants** across 5 domains to YAML files in `config/` (project root)
- Create a config package (`src/multi_agent_dashboard/config/`) with public API identical to current `config.py`
- Zero disruption to existing imports
- Maintain clear separation: `.env` for secrets and user overrides, YAML for developer‑configurable values
- Enable future environment‑specific configurations

### Non‑Goals
- Changing the way other modules import configuration (import statements stay identical)
- Migrating environment variables (they remain in `.env`)
- Adding new configuration features beyond the current scope

## Prerequisites

### 1. PyYAML
PyYAML (`PyYAML==6.0.3`) is already installed as a dependency of other packages.

### 2. Pydantic validation (already installed)
Pydantic (`pydantic==2.12.5`) is already installed as a dependency of other packages. The configuration loader **must** use Pydantic models for validation; there is no optional fallback.

### 3. Ensure a clean working state
- No uncommitted changes in `config.py`
- Run existing tests to have a baseline
- Back up your database (optional safety step)

## ✅ Phase 1: Foundation Setup (YAML Files)

**Objective**: Create the YAML files in `config/` with current default values. No code changes yet.

### ✅ Step 1.1 – Create `config/` directory at project root
```bash
mkdir -p config
```

### ✅ Step 1.2 – Write YAML files

Create each file with the exact content below. These values mirror the current hard‑coded constants in `config.py`.

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
  yellow:
    value: "#FFFF00"
    symbol: "🟡"
  green:
    value: "#198754"
    symbol: "🟢"
  blue:
    value: "#0000FF"
    symbol: "🔵"
  purple:
    value: "#842029"
    symbol: "🟣"
  grey:
    value: "#6c757d"
    symbol: "⚪"
  brown:
    value: "#A52A2A"
    symbol: "🟤"
  black:
    value: "#000000"
    symbol: "⚫️"
  default:
    value: "#6c757d"
    symbol: "⚪"
```

**`config/logging.yaml`**
```yaml
# Default log level (can still be overridden by LOG_LEVEL env)
default_log_level: "INFO"
```

### Verification (Phase 1)
- ✅ `config/` directory exists with 5 YAML files
- ✅ YAML syntax is valid (no tabs, consistent 2‑space indentation)
- ✅ Values match current `config.py` constants (cross‑check)

## ✅ Phase 2: Config Package Creation

**Objective**: Create the Python config package that will replace `config.py`. This package loads `.env` and YAML files, computes derived paths, and exposes the same constants.

### ✅ Step 2.1 – Create package directory
```bash
mkdir -p src/multi_agent_dashboard/config
```

### ✅ Step 2.2 – Implement `loader.py`

**Mandatory Pydantic validation**: The loader must use Pydantic models for type safety and validation. Pydantic is already installed as a dependency (`pydantic==2.12.5`). Missing YAML keys will raise validation errors; there are no hard‑coded fallback defaults.

**`src/multi_agent_dashboard/config/loader.py`**:
```python
"""Load and validate YAML configuration files."""
import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, ValidationError


class PathsConfig(BaseModel):
    data_dir: str
    db_dir: str
    migrations_dir: str
    logs_dir: str
    dotenv_file: str
    log_file: str
    db_file: str
    provider_data_dir: str


class AgentsConfig(BaseModel):
    agent_input_cap: int
    agent_output_cap: int
    agent_snapshots_auto: bool
    agent_snapshot_prune_auto: bool
    agent_snapshot_prune_keep: int


class ProvidersConfig(BaseModel):
    provider_models_all_file: str
    provider_models_file: str
    template_ollama_models_file: str
    local_ollama_models_file: str
    models_dev_url: str


class UIConfig(BaseModel):
    attachment_file_types_restricted: bool
    attachment_file_types: list[str]
    ui_colors: Dict[str, Dict[str, str]]


class LoggingConfig(BaseModel):
    default_log_level: str


def load_yaml_config(config_root: Path) -> Dict[str, Any]:
    """Load all YAML files from config_root and return validated dict."""
    config = {}
    
    # Helper to load and validate a single file
    def load_and_validate(filename: str, model_cls: type[BaseModel]) -> Dict[str, Any]:
        path = config_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return model_cls(**data).dict()
    
    try:
        config['paths'] = load_and_validate('paths.yaml', PathsConfig)
        config['agents'] = load_and_validate('agents.yaml', AgentsConfig)
        config['providers'] = load_and_validate('providers.yaml', ProvidersConfig)
        config['ui'] = load_and_validate('ui.yaml', UIConfig)
        config['logging'] = load_and_validate('logging.yaml', LoggingConfig)
    except ValidationError as e:
        raise ValueError(f"YAML validation error: {e}") from e
    
    return config
```

### ✅ Step 2.3 – Implement `core.py`

This module contains the logic currently in `config.py` but reads values from YAML.

**`src/multi_agent_dashboard/config/core.py`**:
```python
"""Core configuration module – loads .env, YAML, computes derived paths."""
from pathlib import Path
from dotenv import dotenv_values
import logging
from typing import Optional

# Import the loader (adjust if Pydantic not used)
from .loader import load_yaml_config


# --- Path resolution (unchanged from config.py) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load YAML configuration
CONFIG_ROOT = PROJECT_ROOT / "config"
_yaml_config = load_yaml_config(CONFIG_ROOT)

# Extract domains for easier access
_paths = _yaml_config['paths']
_agents = _yaml_config['agents']
_providers = _yaml_config['providers']
_ui = _yaml_config['ui']
_logging = _yaml_config.get('logging', {})

# --- Directory & file names from YAML ---
DATA_DIR = Path(_paths['data_dir'])
DB_DIR = Path(_paths['db_dir'])
MIGRATIONS_DIR = Path(_paths['migrations_dir'])
LOGS_DIR = Path(_paths['logs_dir'])
DOTENV_FILE = Path(_paths['dotenv_file'])
LOG_FILE = Path(_paths['log_file'])
DB_FILE = Path(_paths['db_file'])  # can be overridden by env
PROVIDER_DATA_DIR = Path(_paths['provider_data_dir'])

# --- Derived paths (computed as before) ---
DATA_PATH = (PROJECT_ROOT / DATA_DIR).resolve()
DB_PATH = (DATA_PATH / DB_DIR).resolve()
MIGRATIONS_PATH = (DATA_PATH / MIGRATIONS_DIR).resolve()
LOGS_PATH = (DATA_PATH / LOGS_DIR).resolve()
DOTENV_FILE_PATH = (PROJECT_ROOT / DOTENV_FILE).resolve()
PROVIDER_DATA_PATH = (DATA_PATH.parent / PROVIDER_DATA_DIR).resolve()

# Ensure folders exist
DATA_PATH.mkdir(exist_ok=True)
DB_PATH.mkdir(exist_ok=True)
MIGRATIONS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
PROVIDER_DATA_PATH.mkdir(exist_ok=True)

# --- Load .env (unchanged) ---
_env = dotenv_values(DOTENV_FILE_PATH)
_db_file_env = _env.get("DB_FILE")
if _db_file_env is not None and str(_db_file_env).strip():
    DB_FILE = Path(str(_db_file_env).strip())

DB_FILE_PATH = (DATA_PATH / DB_DIR / DB_FILE).resolve()

# --- Environment variables ---
OPENAI_API_KEY = _env.get("OPENAI_API_KEY", None)
DEEPSEEK_API_KEY = _env.get("DEEPSEEK_API_KEY", None)
LOG_LEVEL = _env.get("LOG_LEVEL", _logging['default_log_level']).upper()

LOG_FILE_PATH = (LOGS_PATH / LOG_FILE).resolve()

# --- Agent configuration from YAML ---
AGENT_INPUT_CAP = _agents['agent_input_cap']
AGENT_OUTPUT_CAP = _agents['agent_output_cap']
AGENT_SNAPSHOTS_AUTO = _agents['agent_snapshots_auto']
AGENT_SNAPSHOT_PRUNE_AUTO = _agents['agent_snapshot_prune_auto']
AGENT_SNAPSHOT_PRUNE_KEEP = _agents['agent_snapshot_prune_keep']

# --- Provider‑data configuration from YAML ---
PROVIDER_MODELS_ALL_FILE = _providers['provider_models_all_file']
PROVIDER_MODELS_FILE = _providers['provider_models_file']
TEMPLATE_OLLAMA_MODELS_FILE = _providers['template_ollama_models_file']
LOCAL_OLLAMA_MODELS_FILE = _providers['local_ollama_models_file']
MODELS_DEV_URL = _providers['models_dev_url']

# --- UI configuration from YAML ---
ATTACHMENT_FILE_TYPES_RESTRICTED = _ui['attachment_file_types_restricted']
ATTACHMENT_FILE_TYPES = _ui['attachment_file_types']
UI_COLORS = _ui['ui_colors']


# --- Functions (unchanged) ---
def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        return

    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main():
    """Print global constants (optional)."""
    files_and_paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "DB_DIR": DB_DIR,
        "MIGRATIONS_DIR": MIGRATIONS_DIR,
        "DATA_PATH": DATA_PATH,
        "DB_FILE": DB_FILE,
        "DB_PATH": DB_PATH,
        "DB_FILE_PATH": DB_FILE_PATH,
        "MIGRATIONS_PATH": MIGRATIONS_PATH,
        "LOGS_PATH": LOGS_PATH,
        "LOG_FILE_PATH": LOG_FILE_PATH,
    }
    print("Current file and path resolutions:")
    print("----------------------------------")
    for label, file_path in files_and_paths.items():
        print(f"{label}: {file_path}")
    print("\nSecret environment variables:")
    print("-----------------------------")
    print(f"OPENAI_API_KEY: {'***REDACTED***' if OPENAI_API_KEY else None}")
    if DEEPSEEK_API_KEY:
        print(f"DEEPSEEK_API_KEY: {'***REDACTED***'}")


if __name__ == "__main__":
    main()
```

### ✅ Step 2.4 – Create `__init__.py` (Public API)

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
    # Path constants
    DATA_DIR,
    DB_DIR,
    MIGRATIONS_DIR,
    LOGS_DIR,
    DOTENV_FILE,
    LOG_FILE,
    DB_FILE,
    PROVIDER_DATA_DIR,
    DATA_PATH,
    DB_PATH,
    MIGRATIONS_PATH,
    LOGS_PATH,
    DOTENV_FILE_PATH,
    PROVIDER_DATA_PATH,
    LOG_FILE_PATH,
    # Provider‑data constants
    PROVIDER_MODELS_ALL_FILE,
    PROVIDER_MODELS_FILE,
    TEMPLATE_OLLAMA_MODELS_FILE,
    LOCAL_OLLAMA_MODELS_FILE,
    MODELS_DEV_URL,
)

__all__ = [
    "UI_COLORS",
    "DB_FILE_PATH",
    "AGENT_INPUT_CAP",
    "AGENT_OUTPUT_CAP",
    "AGENT_SNAPSHOTS_AUTO",
    "AGENT_SNAPSHOT_PRUNE_AUTO",
    "AGENT_SNAPSHOT_PRUNE_KEEP",
    "ATTACHMENT_FILE_TYPES_RESTRICTED",
    "ATTACHMENT_FILE_TYPES",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "LOG_LEVEL",
    "configure_logging",
    "DATA_DIR",
    "DB_DIR",
    "MIGRATIONS_DIR",
    "LOGS_DIR",
    "DOTENV_FILE",
    "LOG_FILE",
    "DB_FILE",
    "PROVIDER_DATA_DIR",
    "DATA_PATH",
    "DB_PATH",
    "MIGRATIONS_PATH",
    "LOGS_PATH",
    "DOTENV_FILE_PATH",
    "PROVIDER_DATA_PATH",
    "LOG_FILE_PATH",
    "PROVIDER_MODELS_ALL_FILE",
    "PROVIDER_MODELS_FILE",
    "TEMPLATE_OLLAMA_MODELS_FILE",
    "LOCAL_OLLAMA_MODELS_FILE",
    "MODELS_DEV_URL",
]
```

### ✅ Step 2.5 – Remove config.py and verify imports

Since the new config package provides the same public API via `src/multi_agent_dashboard/config/__init__.py`, the old `config.py` file can be deleted. However, we must verify that no modules import configuration using misaligned import patterns (relative `from config import` or absolute `import config`). The correct import path is `from multi_agent_dashboard.config import ...`.

**Delete `config.py`**:
```bash
rm src/multi_agent_dashboard/config.py
```

**Verify import alignment** (run from project root):
```bash
# Check for any remaining direct imports of 'config' (should be none)
grep -r "from config import" src/ tests/ 2>/dev/null || echo "No misaligned imports found"
grep -r "import config" src/ tests/ 2>/dev/null || echo "No misaligned imports found"
```

If any matches appear, update those imports to use the full package path: `from multi_agent_dashboard.config import ...`.

### Verification (Phase 2)
- ✅ `src/multi_agent_dashboard/config/` package exists with `__init__.py`, `core.py`, `loader.py`
- ✅ YAML loading works (run a quick script to test)
- ✅ `config.py` deleted (no shim)
- ✅ Import alignment verified (no `from config import` or `import config` patterns)
- ✅ All constants are accessible via `from multi_agent_dashboard.config import ...`

## ✅ Phase 3: Gradual Migration by Domain

**Objective**: Migrate constants domain by domain, verifying after each step. This minimizes risk and allows early detection of issues.

### ✅ Step 3.1 – UI Configuration (Lowest Risk)
1. In `core.py`, ensure `ATTACHMENT_FILE_TYPES_RESTRICTED`, `ATTACHMENT_FILE_TYPES`, `UI_COLORS` are read from `_ui` dict (already done in Step 2.3)
2. **Manual verification**:
   - ✅ Start the Streamlit app (`streamlit run src/multi_agent_dashboard/ui/app.py`)
   - ✅ Verify UI colors appear correctly
   - ✅ Upload a file in run mode – check allowed extensions
   - ✅ If anything looks wrong, check YAML syntax and reload

### ✅ Step 3.2 – Agent Configuration
1. Verify `AGENT_INPUT_CAP`, `AGENT_OUTPUT_CAP`, snapshot constants are read from `_agents`
2. **Manual verification**:
   - ✅ Create a new agent with a prompt that would exceed caps (optional)
   - ✅ Check agent‑editor UI for snapshot‑related defaults
   - ✅ Run a simple agent pipeline to confirm caps are respected

### ✅ Step 3.3 – Provider‑Data Configuration
1. Verify provider‑data constants are read from `_providers`
2. **Manual verification**:
   - ✅ Trigger provider‑data download/update (if configured)
   - ✅ Ensure files are saved to correct location (`PROVIDER_DATA_PATH`)
   - ✅ Check that dynamic pricing/capabilities still work

### ✅ Step 3.4 – Paths & Directories
1. Verify all path constants (`DATA_DIR`, `DB_DIR`, etc.) are read from `_paths`
2. **Manual verification**:
   - ✅ Start app – ensure database is created in correct location
   - ✅ Check that migrations can be applied
   - ✅ Verify logs are written to `LOGS_PATH`

### ✅ Step 3.5 – Logging Default
1. Verify `LOG_LEVEL` fallback uses `_logging['default_log_level']` if env not set
2. **Manual verification**:
   - ✅ Set `LOG_LEVEL` in `.env` to `DEBUG` – confirm it overrides YAML
   - ✅ Remove `LOG_LEVEL` from `.env` – confirm default from YAML is used

## ✅ Phase 4: Finalization & Cleanup

**Objective**: Remove any remaining hard‑coded fallbacks, update documentation, and ensure long‑term maintainability.

### ✅ Step 4.1 – Remove Hard‑Coded Defaults from `core.py`
- Ensure no constant in `core.py` has a hard‑coded value; all must come from YAML or `.env`
- No fallback defaults - missing YAML keys will raise validation errors

### ✅ Step 4.2 – Update `AGENTS.md`
Add a new section **Configuration Files** describing the YAML structure, location, and how to modify settings. Include:
- Location of `config/` directory
- Purpose of each YAML file
- How environment variables override YAML defaults
- Example of adding a new configuration value

## Verification Steps (After Each Phase)

### Automated Checks
```bash
# Run existing tests (they should still pass)
pytest

# Quick syntax check of Python files
python -m py_compile src/multi_agent_dashboard/config/*.py
```

### Manual Checks
1. **Start the Streamlit app** – verify no import errors
2. **Create and run a simple agent** – confirm pipeline executes
3. **Check UI colors and file‑upload restrictions** – visual confirmation
4. **Verify database operations** – create agent, save, load
5. **Check logs** – ensure logging works and files are written to correct location

### Configuration‑Specific Tests
```python
# Quick interactive test (run in Python console)
from multi_agent_dashboard.config import UI_COLORS, DB_FILE_PATH, AGENT_INPUT_CAP
print("UI_COLORS keys:", list(UI_COLORS.keys()))
print("DB_FILE_PATH:", DB_FILE_PATH)
print("AGENT_INPUT_CAP:", AGENT_INPUT_CAP)
```

## Rollback Plan

If any step fails, roll back to previous working state:

### Immediate Rollback (During Migration)
1. **Restore `config.py`** from backup (if you made one)
2. **Comment out YAML‑loading code** in `core.py` and revert to hard‑coded values
3. **Remove `config/` directory** if not needed

### Full Rollback (Abandon Migration)
1. Delete `src/multi_agent_dashboard/config/` package
2. Restore original `config.py`
3. Remove `config/` directory
4. Uninstall `pyyaml` (optional)

## Appendix: YAML File Templates (Complete)

See Phase 1.2 for full templates.

## Decisions Made

1. **Pydantic validation** – Mandatory (already installed as `pydantic==2.12.5`). No optional fallback.
2. **Shim `config.py`** – Delete `config.py` (no shim). Verify import alignment across codebase.
3. **Fallback defaults** – No hard‑coded defaults for missing YAML keys. Validation will fail on missing keys.
4. **Environment‑specific configs** – No separate `config/dev/` or `config/prod/` for now.
5. **Dependency management** – Project uses `pyproject.toml` (no `requirements.txt`).
6. **Default values policy** – No silent fallbacks; misconfiguration must be explicit.

## Next Steps

1. **Execute Phase 1** (create YAML files)
2. **Proceed with Phase 2–4** incrementally, verifying after each domain

**Stop**: This document is updated with user decisions. Ready for implementation.