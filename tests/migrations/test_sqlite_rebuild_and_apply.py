# tests/migrations/test_sqlite_rebuild_and_apply.py
import os
import sqlite3

import pytest

from multi_agent_dashboard.config import MIGRATIONS_PATH
from multi_agent_dashboard.db.infra.migration_meta import parse_migration_meta
from multi_agent_dashboard.db.infra.sqlite_rebuild import rebuild_tables_batch
from multi_agent_dashboard.db.infra.migrations import apply_migrations


def test_sqlite_rebuild_from_migration(tmp_path):
    """
    Create a small DB with runs + agent_outputs (agent_outputs initially has
    a FK without ON DELETE CASCADE). Use the migration 005_* MIGRATION-META
    rebuild_defs to rebuild agent_outputs with ON DELETE CASCADE and assert
    that PRAGMA foreign_key_list shows ON DELETE CASCADE afterward.
    """
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        # Ensure FK enforcement toggles are effective when used by rebuild helpers
        conn.execute("PRAGMA foreign_keys = ON")

        # Create minimal tables in the DB (agent_outputs without ON DELETE CASCADE)
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)"
        )
        conn.execute(
            "CREATE TABLE agent_outputs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "run_id INTEGER,"
            "agent_name TEXT,"
            "output TEXT,"
            "FOREIGN KEY(run_id) REFERENCES runs(id)"
            ")"
        )

        # Insert matching rows to ensure no FK violations
        conn.execute("INSERT INTO runs (id, timestamp) VALUES (NULL, 'now')")
        conn.execute(
            "INSERT INTO agent_outputs (run_id, agent_name, output) VALUES (1, 'a', 'o')"
        )
        conn.commit()

        # Parse migration meta of the included migration file
        mig_file = os.path.join(
            MIGRATIONS_PATH, "005_fix_agent_outputs_constraints_REQUIRES_REBUILD.sql"
        )
        with open(mig_file, "r", encoding="utf-8") as f:
            text = f.read()
        meta = parse_migration_meta(text, fname=mig_file)
        assert meta and "rebuild" in meta

        rebuild_defs = meta["rebuild"]["rebuild_defs"]
        assert "agent_outputs" in rebuild_defs
        rebuild_map = {"agent_outputs": rebuild_defs["agent_outputs"]}

        # Perform the batch rebuild (single-table via the batch helper)
        rebuild_tables_batch(conn, rebuild_map)

        # After rebuild, PRAGMA foreign_key_list should include on_delete='CASCADE'
        fk_rows = conn.execute("PRAGMA foreign_key_list('agent_outputs')").fetchall()
        # pragma rows: id, seq, table, from, to, on_update, on_delete, match
        found = any(
            (r[3] == "run_id")
            and (r[6] is not None and str(r[6]).upper() == "CASCADE")
            for r in fk_rows
        )
        assert found, f"Expected ON DELETE CASCADE in foreign_key_list after rebuild, got: {fk_rows}"
    finally:
        conn.close()


def test_apply_migrations_refuses_destructive_on_nonfresh_db(tmp_path):
    """
    Create a non-fresh DB (has table(s) with rows) and run apply_migrations on it.
    The included migration 005_* requires a rebuild; apply_migrations should
    refuse to auto-apply destructive rebuilds and raise a RuntimeError with an
    actionable message.
    """
    db_file = tmp_path / "test2.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # Create a simple table and insert a row to ensure DB is non-fresh
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)")
        conn.execute("INSERT INTO runs (id, timestamp) VALUES (NULL, 'now')")
        conn.commit()

        with pytest.raises(RuntimeError) as ei:
            apply_migrations(conn, MIGRATIONS_PATH)

        msg = str(ei.value)
        assert "destructive table rebuilds" in msg or "requires destructive table rebuilds" in msg or "includes destructive table rebuilds" in msg
    finally:
        conn.close()
