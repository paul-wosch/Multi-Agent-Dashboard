"""Provide global constants for the project."""
from pathlib import Path
from dotenv import dotenv_values
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_FILE = Path("multi_agent_runs.db")

DATA_DIR = Path("data")
DB_DIR = Path("db")
MIGRATIONS_DIR = Path("migrations")
LOGS_DIR = Path("logs")

DATA_PATH = (PROJECT_ROOT / DATA_DIR).resolve()
DB_PATH = (DATA_PATH / DB_DIR).resolve()
MIGRATIONS_PATH = (DATA_PATH / MIGRATIONS_DIR).resolve()
LOGS_PATH = (DATA_PATH / LOGS_DIR).resolve()

# Ensure folders are created if not existing
DATA_PATH.mkdir(exist_ok=True)
DB_PATH.mkdir(exist_ok=True)
MIGRATIONS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

DOTENV_FILE = Path(".env")
DOTENV_FILE_PATH = (PROJECT_ROOT / DOTENV_FILE).resolve()

_env = dotenv_values(DOTENV_FILE_PATH)

# If DB_FILE is provided in the .env and not empty, use it.
_db_file_env = _env.get("DB_FILE")
if _db_file_env is not None and str(_db_file_env).strip():
    DB_FILE = Path(str(_db_file_env).strip())

DB_FILE_PATH = (DATA_PATH / DB_DIR / DB_FILE).resolve()

OPENAI_API_KEY = _env.get("OPENAI_API_KEY", None)
DEEPSEEK_API_KEY = _env.get("DEEPSEEK_API_KEY", None)

LOG_LEVEL = _env.get("LOG_LEVEL", "INFO").upper()



LOG_FILE = Path("application.log")
LOG_FILE_PATH = (LOGS_PATH / LOG_FILE).resolve()

AGENT_INPUT_CAP = 40_000
AGENT_OUTPUT_CAP = 50_000

# Control whether saving an agent automatically creates a snapshot.
# Default is False to remain backwards-compatible.
AGENT_SNAPSHOTS_AUTO = False

# Control whether automatic pruning runs at app start.
# Set to True to run pruning automatically when the app bootstraps.
# Default: False (do not run automatic prune).
AGENT_SNAPSHOT_PRUNE_AUTO = False

# Default number of snapshots to keep per agent when pruning.
# You can override this value when calling the prune helper from the UI.
AGENT_SNAPSHOT_PRUNE_KEEP = 100

OPENAI_PRICING = {
    # https://platform.openai.com/docs/pricing
    # latest update: 251230
    # price per 1M tokens
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
    },
    "gpt-4.1-nano": {
        "input": 0.10,
        "output": 0.40,
    },
    "gpt-5-mini": {
        "input": 0.25,
        "output": 2.00,
    },
    "gpt-5-nano": {
        "input": 0.05,
        "output": 0.40,
    },
    "gpt-5-search-api": {
        "input": 1.25,
        "output": 10.00,
    },
    "gpt-5.1": {
        "input": 1.25,
        "output": 10.00,
    },
    "gpt-5.1-codex-mini": {
        "input": 0.25,
        "output": 2.00,
    },
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.00,
    },
    # Add more models as needed
}

# DeepSeek pricing (USD per 1M tokens)
# Sources: https://api-docs.deepseek.com/quick_start/pricing (accessed 2026-01)
DEEPSEEK_PRICING = {
    "deepseek-chat": {
        "input": 0.28,
        "output": 0.42,
    },
    "deepseek-reasoner": {
        "input": 0.28,
        "output": 0.42,
    },
}

# Dynamic pricing & capabilities configuration
PROVIDER_DATA_DIR = "data/provider_models"
PROVIDER_MODELS_ALL_FILE = "provider_models_all.json"
PROVIDER_MODELS_FILE = "provider_models.json"
TEMPLATE_OLLAMA_MODELS_FILE = "template_ollama_models.json"
LOCAL_OLLAMA_MODELS_FILE = "local_ollama_models.json"
MODELS_DEV_URL = "https://models.dev/api.json"

# Toggle whether the UI file uploader restricts selectable file extensions.
# If True, the uploader will only allow extensions listed in ATTACHMENT_FILE_TYPES.
# If False, the uploader will accept any file extension (Streamlit default: None).
ATTACHMENT_FILE_TYPES_RESTRICTED = False

# Allowed attachment file extensions for the file uploader (used in the UI)
# Keep in sync with any documentation / allowed agent input types.
ATTACHMENT_FILE_TYPES = [
    "txt",
    "pdf",
    "csv",
    "md",
    "json",
    "log",
    "py",
    "sql",
    "patch",
    "LICENSE",
    "toml",
    "gitignore",
]

UI_COLORS = {
    "red": {
        "value": "#dc3545",
        "symbol": "🔴",
    },
    "orange": {
        "value": "#fd7e14",
        "symbol": "🟠",
    },
    "yellow": {
        "value": "#FFFF00",
        "symbol": "🟡",
    },
    "green": {
        "value": "#198754",
        "symbol": "🟢",
    },
    "blue": {
        "value": "#0000FF",
        "symbol": "🔵",
    },
    "purple": {
        "value": "#842029",
        "symbol": "🟣",
    },
    "grey": {
        "value": "#6c757d",
        "symbol": "⚪",
    },
    "brown": {
        "value": "#A52A2A",
        "symbol": "🟤",
    },
    "black": {
        "value": "#000000",
        "symbol": "⚫️",
    },
    "default": {
        "value": "#6c757d",
        "symbol": "⚪",
    },
}


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        return

    # NEW: add rotating file handler in addition to basicConfig (stdout)
    from logging.handlers import RotatingFileHandler

    # console/basic config
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # file handler
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
    """Print global constants."""
    files_and_paths = {"PROJECT_ROOT": PROJECT_ROOT,
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
