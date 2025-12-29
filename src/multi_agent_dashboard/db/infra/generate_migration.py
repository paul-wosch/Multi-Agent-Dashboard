"""generate_migration.py

Recommended Workflow:
1. Update schema.py
2. Run generator
3. Review SQL
4. Commit migration file
5. App applies it via apply_migrations()

Optionally: Rebuild table with sqlite_rebuild.py when FK constraints change was detected

✅ Resulting CLI Behavior
python generate_migration.py add_agents	                            --> uses default DB
python generate_migration.py add_agents --dry-run                   --> preview only
python generate_migration.py add_constraints --disable-constraints  --> disable FK-constraints check
python generate_migration.py custom.db add_agents                   --> ❌ wrong order
python generate_migration.py add_agents custom.db                   --> ✅ override DB
"""
import os
import sqlite3
import re
from typing import Dict, Any

from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.db.infra.schema_diff import get_db_schema
from multi_agent_dashboard.db.infra.schema_diff_constraints import (
    get_db_foreign_keys,
    diff_foreign_keys,
)

from multi_agent_dashboard.config import DB_FILE_PATH, MIGRATIONS_PATH

MIGRATIONS_DIR = MIGRATIONS_PATH
SYSTEM_TABLES = {"migrations"}


def next_migration_prefix(migrations_dir: str, width: int = 3) -> str:
    """Return next zero-padded migration number based on existing files.

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


def _format_fk_clause(fk: Dict[str, Any]) -> str:
    clause = f"FOREIGN KEY({fk['column']}) REFERENCES {fk['references']}"
    if fk.get("on_delete"):
        clause += f" ON DELETE {fk['on_delete']}"
    return clause


def generate_migration(
    db_path: str,
    name: str,
    dry_run: bool = False,
    enable_constraints: bool = True,
):
    conn = sqlite3.connect(db_path)
    try:
        db_schema = get_db_schema(conn)
        statements: list[str] = []

        # Collect schema constraints once
        schema_constraints: Dict[str, Dict[str, Any]] = {}
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

            is_new_table = table not in db_schema

            if is_new_table:
                defs = [f"{name} {type_}" for name, type_ in columns.items()]

                # Constraints allowed ONLY on CREATE TABLE
                for fk in constraints.get("foreign_keys", []):
                    defs.append(_format_fk_clause(fk))

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

        # Will collect tables that require rebuild for FK diffs
        tables_needing_rebuild: list[str] = []

        # ------------------------------
        # PHASE 2: Constraint diffs (default ON, can be disabled)
        # ------------------------------
        if enable_constraints:
            db_fks = get_db_foreign_keys(conn)
            fk_diffs = diff_foreign_keys(schema_constraints, db_fks)

            for table, diff in fk_diffs.items():
                # If the table didn't exist when Phase 1 ran, it will be created
                # with full constraints; no rebuild note is needed.
                is_new_table = table not in db_schema
                if is_new_table:
                    continue

                adds = diff.get("add", [])
                removes = diff.get("remove", [])
                changes = diff.get("change", [])

                if not (adds or removes or changes):
                    continue

                tables_needing_rebuild.append(table)

                # High-level note
                stmt_note = f"-- NOTE: {table} requires rebuild to "
                actions = []
                if adds:
                    actions.append("add foreign keys")
                if removes:
                    actions.append("remove foreign keys")
                if changes:
                    actions.append("change foreign keys")
                stmt_note += ", ".join(actions)
                statements.append(stmt_note)

                # Detailed before/after preview
                if removes:
                    statements.append(f"--   Existing foreign keys to REMOVE in DB:")
                    for fk in removes:
                        statements.append(f"--     { _format_fk_clause(fk) }")

                if adds:
                    statements.append(f"--   New foreign keys to ADD from schema:")
                    for fk in adds:
                        statements.append(f"--     { _format_fk_clause(fk) }")

                if changes:
                    statements.append(f"--   Foreign keys to CHANGE (DB -> schema):")
                    for pair in changes:
                        from_fk = pair["from"]
                        to_fk = pair["to"]
                        statements.append(
                            f"--     { _format_fk_clause(from_fk) }"
                            f"  ==>  { _format_fk_clause(to_fk) }"
                        )

                statements.append(
                    f"--   Use sqlite_rebuild.py to rebuild this table "
                    f"with constraints from schema.py"
                )
                # Ready-to-copy helper command for this table
                statements.append(
                    f"--   e.g.: python sqlite_rebuild.py --all-with-diffs {db_path}"
                )

        # ------------------------------
        # PHASE 3: Output
        # ------------------------------
        if not statements:
            print("No schema changes detected.")
            return

        # DRY RUN: always show rebuild NOTE / helper in output if present
        if dry_run:
            print("\n--- DRY RUN: schema diff preview ---\n")
            for stmt in statements:
                print(stmt)

            # Helper block: ready-to-copy sqlite_rebuild.py commands
            if enable_constraints and tables_needing_rebuild:
                print("\n--- Tables requiring rebuild (FK diffs detected) ---")
                print("You can rebuild all of them automatically with:")
                print(f"  python sqlite_rebuild.py --all-with-diffs {db_path}\n")
                print("Or rebuild individually:")
                for table in tables_needing_rebuild:
                    print(f"  python sqlite_rebuild.py {db_path} {table}")
                print("--- end rebuild helper ---\n")

            print("\n--- end dry run ---\n")
            return

        # Determine filename suffix when manual table rebuild is required
        manual_suffix = ""
        if enable_constraints and tables_needing_rebuild:
            # Suffix is intentionally short but explicit
            manual_suffix = "_REQUIRES_REBUILD"

        prefix = next_migration_prefix(MIGRATIONS_DIR)
        path = os.path.join(
            MIGRATIONS_DIR,
            f"{prefix}_{name}{manual_suffix}.sql",
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("-- auto-generated migration\n\n")
            f.write("\n".join(statements))

        print(f"Generated migration: {path}")

        # ------------------------------
        # NEW: Explicit CLI warning for rebuild
        # ------------------------------
        if enable_constraints and tables_needing_rebuild:
            print()
            print("WARNING: This migration involves foreign key changes that")
            print("require a full table rebuild; the migration filename is")
            print("suffixed with '_REQUIRES_REBUILD'.")
            print()
            print("Run this command for a detailed preview of the required")
            print("rebuild operations and foreign key diffs:")
            print(f"  python generate_migration.py {name} {db_path} --dry-run")
            print()
            print("Then rebuild the affected tables with sqlite_rebuild.py, e.g.:")
            print(f"  python sqlite_rebuild.py --all-with-diffs {db_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-generate SQLite migration files from schema.py"
    )

    # Keep name first (per docstring examples), db_path optional
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
        "--disable-constraints",
        action="store_true",
        help="Disable constraint diffing and rebuild notes"
    )

    args = parser.parse_args()

    generate_migration(
        db_path=args.db_path,
        name=args.name,
        dry_run=args.dry_run,
        enable_constraints=not args.disable_constraints,
    )
