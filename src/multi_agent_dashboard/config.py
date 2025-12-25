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

DB_FILE_PATH = (DATA_PATH / DB_DIR / DB_FILE).resolve()

DOTENV_FILE = Path(".env")
DOTENV_FILE_PATH = (PROJECT_ROOT / DOTENV_FILE).resolve()

OPENAI_API_KEY = dotenv_values(DOTENV_FILE_PATH).get("OPENAI_API_KEY", None)

LOG_LEVEL = dotenv_values(DOTENV_FILE_PATH).get("LOG_LEVEL", "INFO").upper()

LOG_FILE = Path("application.log")
LOG_FILE_PATH = (LOGS_PATH / LOG_FILE).resolve()

AGENT_INPUT_CAP = 40_000
AGENT_OUTPUT_CAP = 50_000

OPENAI_PRICING = {
    # https://platform.openai.com/docs/pricing
    # latest update: 251224
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
    "gpt-5-nano": {
        "input": 0.05,
        "output": 0.40,
    },
    "gpt-5.1": {
        "input": 1.25,
        "output": 10.00,
    },
    # Add more models as needed
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
    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")


if __name__ == "__main__":
    main()
