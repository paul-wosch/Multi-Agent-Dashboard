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

= Sanity checklist (run these) =

# No changes
python generate_migration.py add_x --dry-run

# Constraint audit
python generate_migration.py add_constraints --dry-run --enable-constraints
"""
import os
import sqlite3
from schema import SCHEMA
from schema_diff import get_db_schema
from schema_diff_constraints import (
    get_db_foreign_keys,
    diff_foreign_keys,
)

from multi_agent_dashboard.config import DB_FILE_PATH, MIGRATIONS_PATH

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


def normalize_table_def(table_def):
    """
    Supports both:
    - flat dict (legacy)
    - { columns: {...}, constraints: {...} }
    """
    if "columns" in table_def:
        return table_def["columns"], table_def.get("constraints", {})
    return table_def, {}


def generate_migration(
    db_path: str,
    name: str,
    dry_run: bool = False,
    enable_constraints: bool = False,
):
    conn = sqlite3.connect(db_path)
    db_schema = get_db_schema(conn)

    statements = []

    # Collect schema constraints once
    schema_constraints = {}
    for table, table_def in SCHEMA.items():
        if isinstance(table_def, dict):
            schema_constraints[table] = table_def.get("constraints", {})

    # ------------------------------
    # PHASE 1: Column diffs
    # ------------------------------
    for table, table_def in SCHEMA.items():
        columns, constraints = normalize_table_def(table_def)

        # --- GUARD: system tables ---
        if table in SYSTEM_TABLES:
            if table not in db_schema:
                cols = ",\n    ".join(
                    f"{name} {type_}" for name, type_ in columns.items()
                )
                statements.append(
                    f"CREATE TABLE {table} (\n    {cols}\n);"
                )
            continue
        # ----------------------------

        if table not in db_schema:
            defs = [f"{name} {type_}" for name, type_ in columns.items()]

            # Constraints allowed ONLY on CREATE TABLE
            for fk in constraints.get("foreign_keys", []):
                clause = (
                    f"FOREIGN KEY({fk['column']}) "
                    f"REFERENCES {fk['references']}"
                )
                if fk.get("on_delete"):
                    clause += f" ON DELETE {fk['on_delete']}"
                defs.append(clause)

            statements.append(
                f"CREATE TABLE {table} (\n    {',\n    '.join(defs)}\n);"
            )
        else:
            existing = db_schema[table]["columns"]
            for col, col_type in columns.items():
                if col not in existing:
                    statements.append(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type};"
                    )

    # ------------------------------
    # PHASE 2: Constraint diffs (OPT-IN)
    # ------------------------------
    if enable_constraints:
        db_fks = get_db_foreign_keys(conn)
        missing = diff_foreign_keys(schema_constraints, db_fks)

        for table, fks in missing.items():
            statements.append(
                f"-- NOTE: {table} requires rebuild to add foreign keys"
            )
            for fk in fks:
                statements.append(
                    f"--   FOREIGN KEY({fk['column']}) REFERENCES {fk['references']}"
                )

    # ------------------------------
    # PHASE 3: Output
    # ------------------------------
    if not statements:
        print("No schema changes detected.")
        return

    if dry_run:
        print("\n--- DRY RUN: schema diff preview ---\n")
        for stmt in statements:
            print(stmt)
        print("\n--- end dry run ---\n")
        return

    prefix = next_migration_prefix(MIGRATIONS_DIR)
    path = os.path.join(MIGRATIONS_DIR, f"{prefix}_{name}.sql")

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

    parser.add_argument(
        "--enable-constraints",
        action="store_true",
        help="Enable constraint diffing and rebuild migrations"
    )

    args = parser.parse_args()

    generate_migration(
        db_path=args.db_path,
        name=args.name,
        dry_run=args.dry_run,
        enable_constraints=args.enable_constraints,
    )
