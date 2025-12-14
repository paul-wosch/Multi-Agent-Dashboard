"""Provide global constants for the project."""
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent

DB_FILE = Path("multi_agent_runs.db")

DATA_DIR = Path("data")
DB_DIR = Path("db")
MIGRATIONS_DIR = Path("migrations")

DATA_PATH = (PROJECT_ROOT / DATA_DIR).resolve()
DB_PATH = (DATA_PATH / DB_DIR).resolve()
MIGRATIONS_PATH = (DATA_PATH / MIGRATIONS_DIR).resolve()

# Ensure folders are created if not existing
DATA_PATH.mkdir(exist_ok=True)
DB_PATH.mkdir(exist_ok=True)
MIGRATIONS_PATH.mkdir(exist_ok=True)

DB_FILE_PATH = (DATA_PATH / DB_DIR / DB_FILE).resolve()

DOTENV_FILE = Path(".env")
DOTENV_FILE_PATH = (PROJECT_ROOT / DOTENV_FILE).resolve()

OPENAI_API_KEY = dotenv_values(DOTENV_FILE_PATH).get("OPENAI_API_KEY", None)


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
