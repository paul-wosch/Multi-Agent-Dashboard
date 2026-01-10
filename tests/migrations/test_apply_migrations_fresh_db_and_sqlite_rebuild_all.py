# tests/migrations/test_apply_migrations_fresh_db_and_sqlite_rebuild_all.py
import os
import sqlite3

import pytest

from multi_agent_dashboard.config import MIGRATIONS_PATH
from multi_agent_dashboard.db.infra.sqlite_rebuild import _scan_migrations_for_rebuilds, rebuild_tables_batch
from multi_agent_dashboard.db.infra.migrations import apply_migrations


def _fk_has_on_delete(conn: sqlite3.Connection, table: str, column: str, expected_on_delete: str) -> bool:
    """
    Helper: return True if PRAGMA foreign_key_list(table) includes a FK on `column`
    with on_delete matching expected_on_delete (case-insensitive).
    """
    rows = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
    for r in rows:
        # pragma row format: id, seq, table, from, to, on_update, on_delete, match
        fk_from = r[3]
        fk_on_delete = r[6]
        if fk_from == column and fk_on_delete and str(fk_on_delete).upper() == expected_on_delete.upper():
            return True
    return False


def test_apply_migrations_fresh_db_auto_rebuild(tmp_path):
    """
    On a fresh DB (no user rows), apply_migrations should auto-apply migrations and
    perform embedded rebuilds using MIGRATION-META rebuild_defs when present.
    This test verifies that the included migration 005_* rebuilds agent_outputs to
    include ON DELETE CASCADE as specified in its MIGRATION-META.
    """
    db_file = tmp_path / "fresh_apply.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        # Ensure foreign-keys can be toggled during rebuilds
        conn.execute("PRAGMA foreign_keys = ON")

        # Run the migrator across the repository migrations directory.
        apply_migrations(conn, MIGRATIONS_PATH)

        # Verify the agent_outputs foreign key now includes ON DELETE CASCADE
        assert _fk_has_on_delete(conn, "agent_outputs", "run_id", "CASCADE"), \
            "Expected agent_outputs.run_id to have ON DELETE CASCADE after applying migrations."

        # Verify the migrations history table contains an entry referencing the 005 migration id
        rows = conn.execute("SELECT id FROM migrations").fetchall()
        ids = [r[0] for r in rows]
        assert any("005_fix_agent_outputs_constraints" in i for i in ids), \
            f"Expected migrations table to include a 005 migration id; found {ids}"

    finally:
        conn.close()


def test_sqlite_rebuild_all_with_diffs_scans_and_rebuilds(tmp_path):
    """
    Verify sqlite_rebuild's scan for MIGRATION-META headers returns rebuildable tables,
    and that the batch rebuild path (rebuild_tables_batch) successfully applies the
    rebuild_defs (PRAGMA foreign_key_check passes and ON DELETE CASCADE appears).
    """
    # Prepare a minimal DB state: runs + agent_outputs (agent_outputs without ON DELETE CASCADE)
    db_file = tmp_path / "rebuild_all_with_diffs.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)")
        conn.execute(
            "CREATE TABLE agent_outputs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "run_id INTEGER,"
            "agent_name TEXT,"
            "output TEXT,"
            "FOREIGN KEY(run_id) REFERENCES runs(id)"
            ")"
        )
        # insert minimal data so DB is non-empty (we're testing rebuild path directly)
        conn.execute("INSERT INTO runs (id, timestamp) VALUES (NULL, 'now')")
        conn.execute("INSERT INTO agent_outputs (run_id, agent_name, output) VALUES (1, 'a', 'o')")
        conn.commit()

        # Scan migrations directory for rebuild metadata
        tables, rebuild_defs, file_map = _scan_migrations_for_rebuilds(MIGRATIONS_PATH)

        # Expect agent_outputs to be listed among tables requiring rebuild (per our repo migrations)
        assert "agent_outputs" in tables, f"Expected agent_outputs in scanned tables, got: {tables}"
        assert "agent_outputs" in rebuild_defs, "Expected rebuild_defs to contain agent_outputs entry"

        # Perform the rebuild using the batch helper for the scanned table(s)
        rebuild_map = {t: rebuild_defs[t] for t in tables if t in rebuild_defs and rebuild_defs[t].get("columns")}
        assert rebuild_map, "No valid rebuild map constructed from scanned rebuild_defs"

        rebuild_tables_batch(conn, rebuild_map)

        # After rebuild, the FK should have ON DELETE CASCADE per migration-provided rebuild_defs
        assert _fk_has_on_delete(conn, "agent_outputs", "run_id", "CASCADE"), \
            "Expected agent_outputs.run_id to have ON DELETE CASCADE after batch rebuild"

    finally:
        conn.close()
