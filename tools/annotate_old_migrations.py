#!/usr/bin/env python3
"""
tools/annotate_old_migrations.py

One-time helper to annotate legacy migration files with MIGRATION-META headers.

Usage:
  # Annotate all SQL files in migrations directory (default)
  python tools/annotate_old_migrations.py --migrations-dir data/migrations

  # Dry-run for directory
  python tools/annotate_old_migrations.py --migrations-dir data/migrations --dry-run

  # Annotate one or more specific migration files (repeatable)
  python tools/annotate_old_migrations.py --migration-file data/migrations/005_fix_agent_outputs_constraints.sql

This improved version provides a concise per-file summary showing which inferred tables
have fully-populated rebuild_defs (columns present) and which require manual completion.
"""
import os
import re
import json
import copy
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from multi_agent_dashboard.config import MIGRATIONS_PATH
from multi_agent_dashboard.db.infra.migration_meta import (
    parse_migration_meta,
    write_migration_meta,
    MigrationMetaValidationError,
    MigrationMetaError,
)
from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.db.infra.cli_utils import print_user_message


def _infer_tables_from_legacy_comments(sql_text: str) -> List[str]:
    """
    Best-effort extraction of table names from legacy comment blocks used by
    the old generate_migration output.
    """
    tables = set()

    lines = sql_text.splitlines()

    # Pattern 1: lines like "-- NOTE: agent_outputs requires rebuild [..]"
    note_pattern = re.compile(r"^\s*--\s*NOTE:\s*([A-Za-z0-9_]+)\b.*\brequires\b.*\brebuild\b", re.I)

    # Pattern 2: variations like "-- agent_outputs REQUIRES_REBUILD" or "-- requires_rebuild: agent_outputs"
    inline_requires_pattern = re.compile(
        r"^\s*--.*\b([A-Za-z0-9_]+)\b.*\bREQUIRES[_\- ]?REBUILD\b", re.I
    )
    requires_key_value_pattern = re.compile(
        r"^\s*--.*\brequires[_\- ]?rebuild[:\s]*([A-Za-z0-9_]+)\b", re.I
    )

    for line in lines:
        if not line or not line.strip():
            continue

        m = note_pattern.search(line)
        if m:
            tables.add(m.group(1))
            continue

        m2 = inline_requires_pattern.search(line)
        if m2:
            name = m2.group(1)
            if name:
                tables.add(name)
                continue

        m3 = requires_key_value_pattern.search(line)
        if m3:
            tables.add(m3.group(1))
            continue

    return sorted(tables)


def _create_provisional_meta(fname: str, inferred_tables: List[str]) -> Dict[str, Any]:
    # Use filename without extension as canonical id to be consistent with generated migrations.
    base_id = os.path.splitext(os.path.basename(fname))[0]
    meta = {
        "id": base_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rebuild": {
            "requires_rebuild": inferred_tables,
            "rebuild_defs": {},
            "batch": True,
        },
        "author_notes": "Auto-annotated from legacy comments. PLEASE REVIEW and adjust rebuild_defs before applying.",
        "legacy_inferred": True,
    }

    for t in inferred_tables:
        # Try to populate rebuild_defs from schema.py if possible (safer default)
        if t in SCHEMA:
            td = SCHEMA[t]
            if "columns" in td:
                cols = td["columns"]
                cons = td.get("constraints", {})
            else:
                cols = td
                cons = {}
            meta["rebuild"]["rebuild_defs"][t] = {"columns": copy.deepcopy(cols), "constraints": copy.deepcopy(cons)}
        else:
            meta["rebuild"]["rebuild_defs"][t] = {"columns": {}, "constraints": {}}

    return meta


def _resolve_target_files(migrations_dir: str, migration_file_args: Optional[List[str]]) -> List[str]:
    """
    Return canonical list of migration file paths to process.
    """
    resolved: List[str] = []

    if migration_file_args:
        for p in migration_file_args:
            if os.path.isabs(p) and os.path.exists(p):
                resolved.append(os.path.abspath(p))
                continue
            if os.path.exists(p):
                resolved.append(os.path.abspath(p))
                continue
            alt = os.path.join(migrations_dir, p)
            if os.path.exists(alt):
                resolved.append(os.path.abspath(alt))
                continue
            # Not found: record as missing (we'll handle missing upstream)
            resolved.append(os.path.abspath(alt))
        return resolved

    # No specific files: return all SQL files in directory
    if not os.path.isdir(migrations_dir):
        raise SystemExit(f"Migrations directory not found: {migrations_dir}")

    files = sorted(os.listdir(migrations_dir))
    for fname in files:
        if not fname.endswith(".sql"):
            continue
        resolved.append(os.path.join(migrations_dir, fname))
    return resolved


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Annotate legacy migration files with MIGRATION-META headers.")
    parser.add_argument("--migrations-dir", default=MIGRATIONS_PATH, help=f"Directory of migration SQL files (default: {MIGRATIONS_PATH})")
    parser.add_argument("--migration-file", "-m", action="append", help="Path to a migration SQL file to annotate (repeatable). If a bare filename is provided it will be resolved relative to --migrations-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed annotations without modifying files")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--verbose", action="store_true", help="Print extra developer detail")
    args = parser.parse_args()
    md = args.migrations_dir

    target_files = _resolve_target_files(md, args.migration_file)

    annotated = []
    annotated_details = []
    skipped = []
    failed = []

    for path in target_files:
        fname = os.path.basename(path)
        # Basic sanity: ensure file exists
        if not os.path.exists(path):
            failed.append((fname, f"file_not_found: {path}"))
            continue

        if not fname.endswith(".sql"):
            skipped.append((fname, "not_sql"))
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # First, quick scan: if the file contains a MIGRATION-META anywhere, skip
        # because either it's already correctly annotated (maybe in the wrong place)
        # or it's an edge case needing manual inspection.
        if "-- MIGRATION-META:" in text:
            # Use parse_migration_meta to determine if a valid header exists at the
            # required first-non-empty-line position. If parse returns a dict, the
            # file already has a valid header in the canonical place and we skip.
            try:
                existing = parse_migration_meta(text, fname=path)
            except MigrationMetaValidationError as e:
                # The file contains MIGRATION-META text but it's invalid JSON or malformed.
                # Treat as a failed case to be reviewed.
                failed.append((fname, f"invalid_existing_meta: {e}"))
                continue

            if existing is not None:
                skipped.append((fname, "already_has_meta"))
                continue
            else:
                # There's a MIGRATION-META text somewhere, but not at first non-empty line.
                skipped.append((fname, "contains_meta_not_first_line"))
                continue

        # If we reach here, file does not contain MIGRATION-META text anywhere.
        inferred = _infer_tables_from_legacy_comments(text)

        # Also look for filename-based hint
        if not inferred and "_REQUIRES_REBUILD" in fname:
            # Attempt to extract probable table from leading comment lines (best-effort)
            m = re.search(r"--\s*NOTE:\s*([A-Za-z0-9_]+)\s+requires rebuild", text, re.I)
            if m:
                inferred = [m.group(1)]
            else:
                # fallback: can't infer table name automatically
                inferred = []

        if not inferred:
            skipped.append((fname, "no_inference"))
            continue

        meta = _create_provisional_meta(fname, inferred)
        # Build completeness summary per-file
        complete_tables = [t for t in inferred if meta["rebuild"]["rebuild_defs"].get(t, {}).get("columns")]
        incomplete_tables = [t for t in inferred if not meta["rebuild"]["rebuild_defs"].get(t, {}).get("columns")]

        if args.dry_run:
            print(f"[DRY RUN] Would annotate {fname} with MIGRATION-META:")
            print(json.dumps(meta, indent=2))
            print(f"[DRY RUN] completeness: complete={complete_tables} incomplete={incomplete_tables}")
            annotated.append((fname, meta))
            annotated_details.append((fname, meta, {"complete": complete_tables, "incomplete": incomplete_tables}))
            continue

        # Attempt to write the header. If write_migration_meta complains that a header
        # already exists somewhere (should not happen due to earlier check), treat as skipped.
        try:
            write_migration_meta(path, meta)
            annotated.append((fname, meta))
            annotated_details.append((fname, meta, {"complete": complete_tables, "incomplete": incomplete_tables}))
        except MigrationMetaError as e:
            # If write_migration_meta refuses because a MIGRATION-META exists, skip.
            # Otherwise, record failure so the operator can inspect.
            msg = str(e)
            if "already contains a MIGRATION-META header" in msg:
                skipped.append((fname, "existing_meta_detected_on_write"))
            else:
                failed.append((fname, msg))

    # Summary
    if not args.quiet:
        print("\n=== annotate_old_migrations summary ===")
        print(f"Annotated files: {len(annotated)}")
        for a in annotated[:50]:
            print("  -", a[0])
        if annotated_details:
            print("\nPer-file completeness details (review files marked 'incomplete'):")
            for fname, _, details in annotated_details[:50]:
                comp = details.get("complete", [])
                incon = details.get("incomplete", [])
                print(f"  - {fname}: complete={comp} incomplete={incon}")
        print(f"\nSkipped files (no inference or already present): {len(skipped)}")
        for s in skipped[:50]:
            print("  -", s[0], s[1])
        if failed:
            print(f"\nFailed to annotate {len(failed)} files:")
            for f in failed[:50]:
                print("  -", f[0], "error:", f[1])
            raise SystemExit("Some files failed to annotate; review the output above.")

    if annotated and not args.quiet:
        print("Done. Please manually review inserted MIGRATION-META headers before applying migrations.")
        print("Search for 'legacy_inferred' in migration files to find auto-annotated entries.")


if __name__ == "__main__":
    main()
