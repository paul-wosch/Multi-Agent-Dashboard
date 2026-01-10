# tests/migrations/test_rebuild_failure_and_restore.py
import os
import sqlite3

import pytest

from multi_agent_dashboard.config import MIGRATIONS_PATH
from multi_agent_dashboard.db.infra.migrations import apply_migrations
import multi_agent_dashboard.db.infra.migrations as migrations_mod


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return bool(row)


def test_rebuild_failure_triggers_restore(tmp_path, monkeypatch):
    """
    Simulate a rebuild failure during apply_migrations (on the fresh DB auto-rebuild path).
    The test patches the rebuild_tables_batch used by apply_migrations to create a
    transient artifact (a table) and then raise. After apply_migrations fails and
    the code attempts restore, the transient artifact should not exist (backup restore succeeded).
    """
    db_file = tmp_path / "fail_restore.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        # Patch the rebuild function referenced by the migrations module so apply_migrations
        # will call our fake and it will raise after making a visible change.
        def fake_rebuild(conn_arg, rebuild_map):
            # Simulate partial work that is committed by the rebuild step (a transient artifact)
            conn_arg.execute("CREATE TABLE temp_during_rebuild (id INTEGER PRIMARY KEY)")
            conn_arg.commit()
            # Then fail
            raise RuntimeError("simulated rebuild failure")

        monkeypatch.setattr(migrations_mod, "rebuild_tables_batch", fake_rebuild)

        with pytest.raises(RuntimeError):
            apply_migrations(conn, MIGRATIONS_PATH)

        # After apply_migrations failure, the DB should have been restored from the backup;
        # the transient table created inside fake_rebuild must NOT exist.
        conn2 = sqlite3.connect(str(db_file))
        try:
            row = conn2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='temp_during_rebuild'").fetchone()
            assert row is None, "Expected transient table to be removed by backup restore"
        finally:
            conn2.close()
    finally:
        conn.close()
