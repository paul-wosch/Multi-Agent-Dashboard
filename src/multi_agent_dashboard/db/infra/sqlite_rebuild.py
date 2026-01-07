"""sqlite_rebuild.py

CLI helper for rebuilding SQLite tables with constraints from schema.py.

Usage examples:

    # Rebuild a single table using constraints from schema.py
    python sqlite_rebuild.py my.db agent_outputs

    # Dry run: just show what would happen
    python sqlite_rebuild.py my.db agent_outputs --dry-run

    # Rebuild all tables that generate_migration --dry-run says need a rebuild
    python sqlite_rebuild.py --all-with-diffs my.db

This tool:
- Creates a timestamped backup before modifying the DB:
  "my_backup_251224-1715.sqlite"
- Rebuilds the table (create temp, copy data, drop, rename)
- Re-enables foreign key enforcement and checks for violations
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Set

from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.config import DB_FILE_PATH


def rebuild_table_with_constraints(conn, table, columns, constraints):
    """
    Rebuilds a table with constraints:
    - Creates temp table
    - Copies data
    - Drops old table
    - Renames temp table
    """

    tmp = f"{table}__rebuild"

    conn.execute("PRAGMA foreign_keys = OFF")

    col_defs = [f"{c} {t}" for c, t in columns.items()]

    for fk in constraints.get("foreign_keys", []):
        clause = (
            f"FOREIGN KEY({fk['column']}) "
            f"REFERENCES {fk['references']}"
        )
        if fk.get("on_delete"):
            clause += f" ON DELETE {fk['on_delete']}"
        col_defs.append(clause)

    ddl = f"""
        CREATE TABLE {tmp} (
            {", ".join(col_defs)}
        );
        """

    conn.executescript(ddl)

    cols = ", ".join(columns.keys())
    conn.execute(
        f"INSERT INTO {tmp} ({cols}) SELECT {cols} FROM {table}"
    )

    conn.executescript(f"""
            DROP TABLE {table};
            ALTER TABLE {tmp} RENAME TO {table};
        """)

    conn.execute("PRAGMA foreign_keys = ON")

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"Foreign key violations after rebuild: {violations}")


def _timestamp_suffix() -> str:
    # YYMMDD-HHMM
    return datetime.now().strftime("%y%m%d-%H%M")


def _make_backup_path(db_path: str) -> str:
    base, ext = os.path.splitext(db_path)
    if not ext:
        ext = ".sqlite"
    return f"{base}_backup_{_timestamp_suffix()}{ext}"


def _get_table_def_from_schema(table: str) -> tuple[Dict[str, str], Dict[str, Any]]:
    if table not in SCHEMA:
        raise KeyError(f"Table '{table}' not found in schema.py SCHEMA")

    table_def = SCHEMA[table]
    if "columns" in table_def:
        return table_def["columns"], table_def.get("constraints", {})
    # legacy style
    return table_def, {}


def _parse_tables_needing_rebuild_from_generate_migration(db_path: str) -> List[str]:
    """
    Runs `python generate_migration.py <temp_name> --dry-run` and parses its output
    for tables that have FK rebuild notes.

    We do not depend on any special machine-readable format in the output:
    - Look for lines starting with '-- NOTE: <table> requires rebuild to ...'
    """
    cmd = [sys.executable, "generate_migration.py", "tmp_rebuild_scan", db_path, "--dry-run"]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Failed to run generate_migration.py for --all-with-diffs:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}"
        )

    tables: Set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        # matches lines like:
        # -- NOTE: some_table requires rebuild to add foreign keys
        if line.startswith("-- NOTE:") and "requires rebuild to" in line:
            # "-- NOTE: <table> requires rebuild to ..."
            rest = line[len("-- NOTE:"):].strip()
            # rest is "<table> requires rebuild to ..."
            parts = rest.split(" ", 1)
            if parts:
                table = parts[0].strip()
                if table:
                    tables.add(table)

    return sorted(tables)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild SQLite tables using constraints from schema.py"
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "table",
        nargs="?",
        help="Table name to rebuild"
    )
    group.add_argument(
        "--all-with-diffs",
        action="store_true",
        help=(
            "Automatically rebuild all tables that "
            "`generate_migration.py --dry-run` marks as requiring a rebuild "
            "due to foreign key constraint diffs."
        ),
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
        help="Show what would be done, but do not modify the database"
    )

    args = parser.parse_args()

    db_path = args.db_path

    if not os.path.exists(db_path):
        raise SystemExit(f"Database file not found: {db_path}")

    if args.all_with_diffs:
        # Determine tables needing rebuild by calling generate_migration
        tables = _parse_tables_needing_rebuild_from_generate_migration(db_path)
        if not tables:
            print(
                "No tables require rebuild according to "
                "generate_migration.py --dry-run."
            )
            return

        print(
            "Tables requiring rebuild (detected via generate_migration.py --dry-run):"
        )
        for t in tables:
            print(f"  - {t}")

        if args.dry_run:
            print("\n[DRY RUN] Would rebuild the above tables in:", db_path)
            backup_path = _make_backup_path(db_path)
            print(f"[DRY RUN] Would create backup at: {backup_path}")
            return

        # Create a single backup for all-table rebuild
        backup_path = _make_backup_path(db_path)
        shutil.copy2(db_path, backup_path)
        print(f"Created backup: {backup_path}")

        conn = sqlite3.connect(db_path)
        try:
            for table in tables:
                print(f"Rebuilding table '{table}'...")
                try:
                    # Confirm table exists in DB
                    row = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    if not row:
                        print(
                            f"  Skipping '{table}': table not found in database."
                        )
                        continue

                    columns, constraints = _get_table_def_from_schema(table)
                    rebuild_table_with_constraints(conn, table, columns, constraints)
                    conn.commit()
                    print(f"  Successfully rebuilt '{table}'.")
                except Exception as e:
                    conn.rollback()
                    print(f"  ERROR rebuilding '{table}': {e}")
                    raise
        finally:
            conn.close()

        print("All requested tables processed.")
        print("Backup remains at:", backup_path)
        return

    # Single-table mode
    table = args.table
    if not table:
        raise SystemExit(
            "You must specify either a table name or --all-with-diffs.\n"
            "Examples:\n"
            f"  python {os.path.basename(__file__)} {db_path} my_table\n"
            f"  python {os.path.basename(__file__)} --all-with-diffs {db_path}"
        )

    # Confirm table exists in DB
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not row:
            raise SystemExit(f"Table '{table}' does not exist in database {db_path}")
    finally:
        conn.close()

    # Fetch schema definition
    try:
        columns, constraints = _get_table_def_from_schema(table)
    except KeyError as e:
        raise SystemExit(str(e))

    if args.dry_run:
        print(f"[DRY RUN] Would rebuild table '{table}' in {db_path}")
        print(f"[DRY RUN] Columns: {columns}")
        print(f"[DRY RUN] Constraints: {constraints}")
        backup_path = _make_backup_path(db_path)
        print(f"[DRY RUN] Would create backup at: {backup_path}")
        return

    # Create backup
    backup_path = _make_backup_path(db_path)
    shutil.copy2(db_path, backup_path)
    print(f"Created backup: {backup_path}")

    # Rebuild in-place
    conn = sqlite3.connect(db_path)
    try:
        rebuild_table_with_constraints(conn, table, columns, constraints)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Successfully rebuilt table '{table}' in {db_path}")
    print("Backup remains at:", backup_path)


if __name__ == "__main__":
    main()
