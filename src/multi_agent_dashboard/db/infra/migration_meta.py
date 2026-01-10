# multi_agent_dashboard/db/infra/migration_meta.py
"""
MIGRATION-META utilities.

Spec (short):
- The first non-empty line of a migration SQL file MUST be a single-line comment:
  -- MIGRATION-META: { ...json... }

This module:
- Parses that header (parse_migration_meta)
- Validates the structure (validate_migration_meta)
- Writes the header into an existing migration file (write_migration_meta)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

MIGRATION_META_PREFIX = "-- MIGRATION-META:"


class MigrationMetaError(Exception):
    """Base exception for migration meta parsing/validation/writing errors."""


class MigrationMetaValidationError(MigrationMetaError):
    """Raised when MIGRATION-META JSON is syntactically invalid or missing required fields."""


def parse_migration_meta(sql_text: str, fname: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse the MIGRATION-META header from the given SQL text.

    Behavior:
    - Only the first non-empty line is considered.
    - If the first non-empty line starts with '-- MIGRATION-META:', the remainder
      is parsed as JSON and returned as dict.
    - If the first non-empty line is present but not the header, returns None.
    - If the header is present but JSON fails to parse, raises MigrationMetaValidationError.
    - If fname is provided, include filename context in validation error messages.
    """
    if not isinstance(sql_text, str):
        return None

    context = f" in {fname}" if fname else ""

    for line in sql_text.splitlines():
        if not line or not line.strip():
            # skip empty lines until the first non-empty line
            continue

        first = line.strip()
        if first.startswith(MIGRATION_META_PREFIX):
            json_part = first[len(MIGRATION_META_PREFIX):].strip()
            if not json_part:
                raise MigrationMetaValidationError(
                    f"MIGRATION-META header present but no JSON payload found{context}."
                )
            try:
                meta = json.loads(json_part)
            except json.JSONDecodeError as e:
                raise MigrationMetaValidationError(
                    f"Failed to parse MIGRATION-META JSON{context}: {e.msg} (at pos {e.pos})"
                )
            return meta
        # first non-empty line exists but is not the MIGRATION-META header -> no meta
        return None

    return None


def _parse_iso8601_lenient(s: str) -> datetime:
    """
    Parse an ISO8601-ish timestamp leniently:
    - Accept both '2025-01-07T12:00:00+00:00' and '2025-01-07T12:00:00Z' forms.
    - Raise MigrationMetaValidationError on failure.
    """
    if not isinstance(s, str):
        raise MigrationMetaValidationError("Timestamp must be a string in ISO8601 format.")
    # Accept 'Z' by converting it to +00:00 for datetime.fromisoformat
    ts = s
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except Exception as e:
        raise MigrationMetaValidationError(f"Invalid ISO8601 timestamp: {e}")


def validate_migration_meta(meta: Dict[str, Any]) -> None:
    """
    Validate basic well-formedness of a migration meta dict.

    Required top-level keys:
      - id: string
      - created_at: ISO8601 string (will be parsed by datetime.fromisoformat)

    If rebuild is present:
      - rebuild.requires_rebuild must be a list of table names
      - rebuild.rebuild_defs must be a dict mapping those table names to dicts
        that contain at least "columns": { ... }
      - rebuild_defs may optionally include "indexes" and "triggers" lists (SQL strings)
    Additionally: at least one of 'safe_sql' (non-empty list) or 'rebuild' with
    a non-empty 'requires_rebuild' list must be present to be considered a meaningful migration header.
    """
    if not isinstance(meta, dict):
        raise MigrationMetaValidationError("MIGRATION-META must be a JSON object.")

    mid = meta.get("id")
    if not mid or not isinstance(mid, str):
        raise MigrationMetaValidationError("MIGRATION-META 'id' is required and must be a string.")

    created_at = meta.get("created_at")
    if not created_at or not isinstance(created_at, str):
        raise MigrationMetaValidationError("MIGRATION-META 'created_at' is required and must be an ISO8601 string.")
    # Parse timestamp leniently (accept 'Z' suffix)
    _parse_iso8601_lenient(created_at)

    # safe_sql must be a list of strings if present
    safe_sql = meta.get("safe_sql")
    if safe_sql is not None:
        if not isinstance(safe_sql, list) or not all(isinstance(s, str) for s in safe_sql):
            raise MigrationMetaValidationError("MIGRATION-META 'safe_sql' must be a list of SQL string statements.")

    rebuild = meta.get("rebuild")
    if rebuild is not None:
        if not isinstance(rebuild, dict):
            raise MigrationMetaValidationError("MIGRATION-META 'rebuild' must be an object when present.")
        requires = rebuild.get("requires_rebuild", [])
        if not isinstance(requires, list) or not all(isinstance(t, str) for t in requires):
            raise MigrationMetaValidationError("rebuild.requires_rebuild must be a list of table names.")
        rebuild_defs = rebuild.get("rebuild_defs", {})
        if not isinstance(rebuild_defs, dict):
            raise MigrationMetaValidationError("rebuild.rebuild_defs must be an object mapping table->definition.")
        for t in requires:
            if t not in rebuild_defs:
                raise MigrationMetaValidationError(
                    f"rebuild.rebuild_defs missing entry for table '{t}' required by rebuild.requires_rebuild."
                )
            td = rebuild_defs[t]
            if not isinstance(td, dict):
                raise MigrationMetaValidationError(f"rebuild_defs['{t}'] must be an object.")
            cols = td.get("columns")
            if not isinstance(cols, dict):
                raise MigrationMetaValidationError(f"rebuild_defs['{t}'].columns must be an object mapping name->type.")
            # Optional: indexes/triggers must be lists of SQL strings if present
            idxs = td.get("indexes")
            if idxs is not None:
                if not isinstance(idxs, list) or not all(isinstance(s, str) for s in idxs):
                    raise MigrationMetaValidationError(f"rebuild_defs['{t}'].indexes must be a list of SQL strings.")
            trgs = td.get("triggers")
            if trgs is not None:
                if not isinstance(trgs, list) or not all(isinstance(s, str) for s in trgs):
                    raise MigrationMetaValidationError(f"rebuild_defs['{t}'].triggers must be a list of SQL strings.")

    # Require at least one of safe_sql (non-empty list) or rebuild.requires_rebuild (non-empty)
    safe_sql_present = bool(safe_sql)
    rebuild_requires_present = False
    if rebuild is not None and isinstance(rebuild, dict):
        rr = rebuild.get("requires_rebuild", [])
        rebuild_requires_present = bool(rr)

    if not (safe_sql_present or rebuild_requires_present):
        raise MigrationMetaValidationError(
            "MIGRATION-META must include either a non-empty 'safe_sql' list or a 'rebuild' object with a non-empty 'requires_rebuild' list describing required rebuilds."
        )


def write_migration_meta(fname: str, meta: Dict[str, Any]) -> None:
    """
    Insert the MIGRATION-META single-line header as the first non-empty line
    in the file located at fname.

    Raises MigrationMetaError if the file already has MIGRATION-META or on IO errors.
    """

    # Validate meta before attempting to write
    try:
        validate_migration_meta(meta)
    except MigrationMetaValidationError as e:
        raise MigrationMetaError(f"Provided MIGRATION-META is invalid: {e}")

    # Ensure meta is serializable
    try:
        header_json = json.dumps(meta, separators=(",", ":"), sort_keys=True)
    except Exception as e:
        raise MigrationMetaError(f"Could not serialize MIGRATION-META to JSON: {e}")

    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()

    existing_text = "".join(lines)
    # Detect any existing MIGRATION-META header anywhere in the file.
    for idx, line in enumerate(lines):
        if line and line.strip().startswith(MIGRATION_META_PREFIX):
            # If any MIGRATION-META line exists, refuse to insert a second one.
            raise MigrationMetaError(f"File {fname} already contains a MIGRATION-META header (line {idx + 1}).")

    # Find first non-empty line index
    first_non_empty = 0
    while first_non_empty < len(lines) and not lines[first_non_empty].strip():
        first_non_empty += 1

    # Put a blank line after the header for readability (matching generator output)
    header_line = f"{MIGRATION_META_PREFIX} {header_json}\n"

    # If file is empty or only whitespace, insert at start followed by a blank line
    if first_non_empty >= len(lines):
        new_lines = [header_line, "\n"]
    else:
        new_lines = lines[:first_non_empty] + [header_line, "\n"] + lines[first_non_empty:]

    with open(fname, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
