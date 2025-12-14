"""generate_migration.py

Recommended Workflow:
1. Update schema.py
2. Run generator
3. Review SQL
4. Commit migration file
5. App applies it via apply_migrations()

✅ Resulting CLI Behavior
python generate_migration.py add_agents	            --> uses default DB
python generate_migration.py add_agents --dry-run   --> preview only
python generate_migration.py custom.db add_agents	--> ❌ wrong order
python generate_migration.py add_agents custom.db	--> ✅ override DB

🧪 Example Output
$ python generate_migration.py add_indexes --dry-run

--- DRY RUN: schema diff preview ---

CREATE TABLE indexes (
    ...
);

--- end dry run ---
"""
import os
import sqlite3
from schema import SCHEMA
from schema_diff import get_db_schema

import sys
from pathlib import Path
# Get the parent directory using pathlib
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
# Import the module
from config import DB_FILE_PATH, MIGRATIONS_PATH


MIGRATIONS_DIR = MIGRATIONS_PATH
SYSTEM_TABLES = {"migrations"}

import re


def next_migration_prefix(migrations_dir: str, width: int = 3) -> str:
    """
    Returns next zero-padded migration number based on existing files.
    Example: '003'
    """
    nums = []

    for fname in os.listdir(migrations_dir):
        match = re.match(r"(\d+)_", fname)
        if match:
            nums.append(int(match.group(1)))

    next_num = max(nums, default=-1) + 1
    return str(next_num).zfill(width)


def generate_migration(db_path: str, name: str, dry_run: bool = False):
    conn = sqlite3.connect(db_path)
    db_schema = get_db_schema(conn)

    statements = []

    for table, columns in SCHEMA.items():

        # --- GUARD: system tables (migrations) ---
        if table in SYSTEM_TABLES:
            if table not in db_schema:
                cols = ",\n    ".join(
                    f"{name} {type_}" for name, type_ in columns.items()
                )
                statements.append(
                    f"CREATE TABLE {table} (\n    {cols}\n);"
                )
            continue
        # ----------------------------------------

        if table not in db_schema:
            cols = ",\n    ".join(
                f"{name} {type_}" for name, type_ in columns.items()
            )
            statements.append(
                f"CREATE TABLE {table} (\n    {cols}\n);"
            )
        else:
            existing = db_schema[table]
            for col, col_type in columns.items():
                if col not in existing:
                    statements.append(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type};"
                    )

    if not statements:
        print("No schema changes detected.")
        return

    # ---------- DRY RUN ----------
    if dry_run:
        print("\n--- DRY RUN: schema diff preview ---\n")
        for stmt in statements:
            print(stmt)
        print("\n--- end dry run ---\n")
        return
    # -----------------------------

    prefix = next_migration_prefix(MIGRATIONS_DIR)
    fname = f"{prefix}_{name}.sql"
    path = os.path.join(MIGRATIONS_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write("-- auto-generated migration\n\n")
        f.write("\n".join(statements))

    print(f"Generated migration: {path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-generate SQLite migration files from schema.py"
    )

    parser.add_argument(
        "name",
        help="Migration name (used in filename)"
    )

    parser.add_argument(
        "db_path",
        nargs="?",
        default=DB_FILE_PATH,
        help=f"Path to SQLite database file (default: {DB_FILE_PATH})"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema diffs without writing a migration file"
    )

    args = parser.parse_args()

    generate_migration(
        db_path=args.db_path,
        name=args.name,
        dry_run=args.dry_run,
    )
