"""Core configuration module – loads .env, YAML, computes derived paths."""
from pathlib import Path
from dotenv import dotenv_values
import logging
from typing import Optional

# Import the loader (adjust if Pydantic not used)
from .loader import load_yaml_config


# --- Path resolution (adjusted for config/ package location) ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load YAML configuration
CONFIG_ROOT = PROJECT_ROOT / "config"
_yaml_config = load_yaml_config(CONFIG_ROOT)

# Extract domains for easier access
_paths = _yaml_config['paths']
_agents = _yaml_config['agents']
_providers = _yaml_config['providers']
_ui = _yaml_config['ui']
_logging = _yaml_config['logging']

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