# tests/migrations/test_migration_meta.py
import json
import os
import tempfile
import pytest

from multi_agent_dashboard.db.infra.migration_meta import (
    parse_migration_meta,
    write_migration_meta,
    validate_migration_meta,
    MigrationMetaValidationError,
    MigrationMetaError,
)

SIMPLE_META = {
    "id": "001_test_migration",
    "created_at": "2025-12-01T12:00:00Z",
    "safe_sql": ["SELECT 1"]
}


def test_parse_migration_meta_happy_path():
    sql = f"-- MIGRATION-META: {json.dumps(SIMPLE_META)}\n\n-- rest of migration\nSELECT 1;\n"
    meta = parse_migration_meta(sql)
    assert isinstance(meta, dict)
    assert meta["id"] == SIMPLE_META["id"]
    assert "safe_sql" in meta


def test_parse_migration_meta_absent_header_returns_none():
    sql = "-- some other comment\nCREATE TABLE foo(id INTEGER);\n"
    meta = parse_migration_meta(sql)
    assert meta is None


def test_parse_migration_meta_invalid_json_raises_with_filename():
    sql = "-- MIGRATION-META: { bad json }\nSELECT 1;\n"
    with pytest.raises(MigrationMetaValidationError) as ei:
        parse_migration_meta(sql, fname="badfile.sql")
    assert "badfile.sql" in str(ei.value) or "badfile.sql" in repr(ei.value)


def test_write_migration_meta_inserts_header_as_first_non_empty_line(tmp_path):
    fpath = tmp_path / "mig.sql"
    # create file with initial blank lines and some comments
    fpath.write_text("\n\n-- existing comment\nSELECT 1;\n", encoding="utf-8")

    write_migration_meta(str(fpath), SIMPLE_META)

    text = fpath.read_text(encoding="utf-8")
    # first non-empty line must be the MIGRATION-META header
    lines = [L for L in text.splitlines() if L.strip()]
    assert lines[0].startswith("-- MIGRATION-META:")
    # the JSON must load
    header_json = lines[0].split(":", 1)[1].strip()
    parsed = json.loads(header_json)
    assert parsed["id"] == SIMPLE_META["id"]


def test_write_migration_meta_refuses_files_with_existing_meta(tmp_path):
    fpath = tmp_path / "mig2.sql"
    # include a MIGRATION-META somewhere in the file (not necessarily first line)
    fpath.write_text("-- some comment\n-- MIGRATION-META: {\"id\":\"x\"}\nSELECT 1;\n", encoding="utf-8")

    meta = {
        "id": "will_fail",
        "created_at": "2025-12-01T12:00:00Z",
        "safe_sql": ["SELECT 1"]
    }

    with pytest.raises(MigrationMetaError):
        write_migration_meta(str(fpath), meta)
