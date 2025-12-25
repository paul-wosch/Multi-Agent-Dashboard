import os
from datetime import datetime, UTC

from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.db.infra.sqlite_rebuild import rebuild_table_with_constraints


def _is_fresh_database(conn) -> bool:
    """
    Heuristic: DB is 'fresh' if it only contains system tables and no user rows.
    We avoid doing automatic destructive rebuilds on non-trivial DBs.
    """
    # Ignore SQLite internal tables and our migrations table
    system_tables = {"sqlite_sequence", "sqlite_schema", "sqlite_master", "migrations"}

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    user_tables = [name for (name,) in rows if name not in system_tables]

    if not user_tables:
        # No user tables at all → definitely fresh
        return True

    # Check if any user table has rows; if so, treat as non-fresh
    for table in user_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            # Be conservative: if anything goes wrong, don't treat as fresh
            return False
        if count > 0:
            return False

    # All user tables exist but are empty → safe to treat as fresh
    return True


def _tables_from_requires_rebuild_migration(migration_sql: str) -> list[str]:
    """
    Parse the SQL of a *_REQUIRES_REBUILD migration and extract table names
    from lines like:
        -- NOTE: agent_outputs requires rebuild to ...
    """
    tables: list[str] = []
    for line in migration_sql.splitlines():
        line = line.strip()
        if line.startswith("-- NOTE:") and "requires rebuild to" in line:
            # "-- NOTE: <table> requires rebuild to ..."
            rest = line[len("-- NOTE:"):].strip()
            parts = rest.split(" ", 1)
            if parts:
                table = parts[0].strip()
                if table:
                    tables.append(table)
    return tables


def apply_migrations(conn, migrations_dir: str):
    """
    Apply SQL migrations exactly once, in filename order.

    Special handling:
    - Migrations whose filename ends with '_REQUIRES_REBUILD.sql' are treated
      as 'marker' migrations for foreign-key changes that normally require
      a manual table rebuild via sqlite_rebuild.py.
    - On a *fresh* database, after applying such a migration, we automatically
      rebuild the affected tables to match schema.py so that a first-time
      clone gets the correct final schema without extra steps.
    - On a non-fresh database, we only apply the SQL (usually comments/no-op)
      and emit a warning; the user must still run sqlite_rebuild.py manually.
    """

    # Ensure migrations table exists before reading history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)

    applied = {
        row[0]
        for row in conn.execute("SELECT id FROM migrations").fetchall()
    }

    # Evaluate freshness once at the beginning
    fresh_db = _is_fresh_database(conn)

    for fname in sorted(os.listdir(migrations_dir)):
        if not fname.endswith(".sql"):
            continue

        migration_id = fname
        if migration_id in applied:
            continue

        path = os.path.join(migrations_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read()

        # Apply the migration SQL first
        conn.executescript(sql)

        # Special handling for *_REQUIRES_REBUILD migrations
        if fname.endswith("_REQUIRES_REBUILD.sql"):
            tables = _tables_from_requires_rebuild_migration(sql)

            if fresh_db and tables:
                # Safe to auto-rebuild on a fresh/empty DB:
                # use schema.py as the single source of truth.
                for table in tables:
                    table_def = SCHEMA.get(table)
                    if not table_def:
                        # If schema.py has no definition, skip silently
                        continue

                    if "columns" in table_def:
                        columns = table_def["columns"]
                        constraints = table_def.get("constraints", {})
                    else:
                        # legacy style: flat dict of columns
                        columns = table_def
                        constraints = {}

                    rebuild_table_with_constraints(conn, table, columns, constraints)
            elif tables:
                # Non-fresh DB: we warn but do not perform destructive rebuild.
                # Logging is preferred, but we keep it print-only to avoid
                # pulling logging config here.
                print(
                    f"[migrations] Migration '{fname}' marked as REQUIRES_REBUILD "
                    f"for tables {tables}, but database is not empty. "
                    f"Please run sqlite_rebuild.py to rebuild these tables "
                    f"manually if you want the new foreign key behavior."
                )

        conn.execute(
            "INSERT INTO migrations (id, applied_at) VALUES (?, ?)",
            (migration_id, datetime.now(UTC).isoformat()),
        )

        conn.commit()
