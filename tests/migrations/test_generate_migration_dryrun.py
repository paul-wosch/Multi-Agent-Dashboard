# tests/migrations/test_generate_migration_dryrun.py
import json
import re
import sqlite3
import tempfile
from pathlib import Path

import pytest

from multi_agent_dashboard.db.infra.generate_migration import generate_migration


def _extract_migration_meta_from_output(output: str):
    """
    Find the printed line that starts with '-- MIGRATION-META:' and parse the JSON
    payload that follows. Return the parsed dict or raise ValueError if not found.
    """
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("-- MIGRATION-META:"):
            payload = line[len("-- MIGRATION-META:"):].strip()
            return json.loads(payload)
    raise ValueError("MIGRATION-META header not found in output")


def test_generate_migration_dryrun_outputs_meta(tmp_path, capsys):
    # Use a temporary DB so generator sees an empty DB and can emit CREATE TABLE statements
    db_file = tmp_path / "test.db"
    # ensure file exists (generator will connect and inspect schema)
    conn = sqlite3.connect(str(db_file))
    conn.close()

    # Run generator in dry-run mode; this should print MIGRATION-META to stdout
    generate_migration(
        db_path=str(db_file),
        name="unit_test_dryrun",
        dry_run=True,
        enable_constraints=True,
        allow_drop_table=False,
        allow_drop_column=False,
        assume_rename_table=None,
        assume_rename_column=None,
    )

    captured = capsys.readouterr()
    out = captured.out

    # Extract header JSON
    meta = _extract_migration_meta_from_output(out)
    assert isinstance(meta, dict)
    assert "id" in meta
    assert "created_at" in meta
    assert "safe_sql" in meta
    assert isinstance(meta["safe_sql"], list)
    # For an empty DB and this SCHEMA, generator should suggest at least one CREATE TABLE
    assert any("CREATE TABLE" in s.upper() or s.strip().upper().startswith("CREATE TABLE") for s in meta["safe_sql"])
