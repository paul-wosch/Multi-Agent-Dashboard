"""multi_agent_dashboard/db/infra/generate_migration.py

Generate migration files with a single-line MIGRATION-META header.

Modifications:
 - Use sqlite_features.is_add_column_safe to avoid emitting unsafe ALTER ADD COLUMN
   statements on runtimes that may reject them.
 - Include sqlite_capabilities in generator_options in header for reproducibility.
 - Add --quiet / --verbose CLI flags (default behavior preserved).
 - Harden emitted SQL by quoting identifiers via sql_utils.quote_ident and
   quoting column lists using quote_column_list where needed.
"""
import json
import os
import sqlite3
import re
import copy
from typing import Dict, Any, List, Tuple, Optional

from datetime import datetime, timezone

from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.db.infra.schema_diff import (
    get_db_schema,
    detect_table_renames,
    detect_column_renames,
    get_db_views,
)
from multi_agent_dashboard.db.infra.schema_diff_constraints import (
    get_db_foreign_keys,
    diff_foreign_keys,
)

from multi_agent_dashboard.config import DB_FILE_PATH, MIGRATIONS_PATH

# Validate headers before writing
from multi_agent_dashboard.db.infra.migration_meta import validate_migration_meta

# New helpers
from multi_agent_dashboard.db.infra.sqlite_features import (
    version_tuple_from_string,
    get_capabilities,
    is_add_column_safe,
)
from multi_agent_dashboard.db.infra.cli_utils import print_user_message
from multi_agent_dashboard.db.infra.sql_utils import quote_ident, quote_column_list

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
    """
    Build a safe FOREIGN KEY clause. Quote local column names; leave the REFERENCES
    text as provided (authors commonly supply 'other_table(col)').
    """
    col_spec = fk.get("column", "")
    if not isinstance(col_spec, str):
        col_spec = str(col_spec)
    return f"FOREIGN KEY({quote_column_list(col_spec)}) REFERENCES {fk['references']}" + (
        (f" ON DELETE {fk['on_delete']}" if fk.get("on_delete") else "")
    )


def _normalize_type(t: str) -> str:
    if not isinstance(t, str):
        return str(t)
    return " ".join(t.strip().lower().split())


def _parse_assume_rename_table(list_in: List[str] | None) -> Dict[str, str]:
    """Parse --assume-rename-table old:new into a mapping {old: new}."""
    mapping: Dict[str, str] = {}
    if not list_in:
        return mapping
    for item in list_in:
        if ":" not in item:
            continue
        old, new = item.split(":", 1)
        old = old.strip()
        new = new.strip()
        if old and new:
            mapping[old] = new
    return mapping


def _parse_assume_rename_column(list_in: List[str] | None) -> Dict[str, Dict[str, str]]:
    """
    Parse --assume-rename-column table.old:new into mapping:
      { table: { old_col: new_col, ... }, ... }
    """
    mapping: Dict[str, Dict[str, str]] = {}
    if not list_in:
        return mapping
    for item in list_in:
        if ":" not in item or "." not in item:
            continue
        left, new = item.split(":", 1)
        if "." not in left:
            continue
        table, old = left.split(".", 1)
        table = table.strip()
        old = old.strip()
        new = new.strip()
        if table and old and new:
            mapping.setdefault(table, {})[old] = new
    return mapping


def _sqlite_version_tuple(conn: sqlite3.Connection) -> Tuple[int, int, int]:
    ver = conn.execute("select sqlite_version()").fetchone()[0]
    parts = ver.split(".")
    # pad to 3
    parts = (parts + ["0", "0"])[:3]
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _version_ge(vtuple: Tuple[int, int, int], maj: int, min_: int, patch: int) -> bool:
    return vtuple >= (maj, min_, patch)


def _collect_existing_index_and_trigger_sql(db_schema: Dict[str, Any], table: str) -> Tuple[List[str], List[str]]:
    """
    Helper to extract SQL of indexes and triggers that exist for `table` in db_schema.
    Returns (index_sqls, trigger_sqls) lists (strings). Filters out None entries.
    """
    index_sqls: List[str] = []
    trigger_sqls: List[str] = []
    if table in db_schema:
        idxs = db_schema[table].get("indexes", {}) or {}
        for idx_name, idx_info in idxs.items():
            sql = idx_info.get("sql")
            if sql and isinstance(sql, str) and sql.strip():
                index_sqls.append(sql)
        for tr in db_schema[table].get("triggers", []) or []:
            sql = tr.get("sql")
            if sql and isinstance(sql, str) and sql.strip():
                trigger_sqls.append(sql)
    return index_sqls, trigger_sqls


def generate_migration(
    db_path: str,
    name: str,
    dry_run: bool = False,
    enable_constraints: bool = True,
    allow_drop_table: bool = False,
    allow_drop_column: bool = False,
    assume_rename_table: List[str] | None = None,
    assume_rename_column: List[str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
):
    conn = sqlite3.connect(db_path)
    try:
        db_schema = get_db_schema(conn)
        db_views = get_db_views(conn)
        safe_statements: List[str] = []
        # Collect schema constraints once
        schema_constraints: Dict[str, Dict[str, Any]] = {}
        for table, table_def in SCHEMA.items():
            if isinstance(table_def, dict):
                schema_constraints[table] = table_def.get("constraints", {})

        sqlite_version = tuple(
            int(p) for p in conn.execute("select sqlite_version()").fetchone()[0].split(".")[:3]
        )
        assume_table_map = _parse_assume_rename_table(assume_rename_table)
        assume_col_map = _parse_assume_rename_column(assume_rename_column)

        # ------------------------------
        # PHASE 1: Column / table diffs -> safe statements where possible
        # ------------------------------
        # We'll collect per-table column diffs to include in the migration diff summary
        per_table_column_diffs: Dict[str, Dict[str, Any]] = {}
        tables_needing_rebuild: List[str] = []
        rebuild_defs: Dict[str, Dict[str, Any]] = {}

        # Also collect index & trigger diffs summary
        per_table_index_diffs: Dict[str, Dict[str, Any]] = {}
        per_table_trigger_diffs: Dict[str, Dict[str, Any]] = {}

        for table, table_def in SCHEMA.items():
            columns, constraints = normalize_table_def(table_def)

            # --- GUARD: system tables ---
            if table in SYSTEM_TABLES:
                if table not in db_schema:
                    cols = ",\n    ".join(f"{quote_ident(name)} {type_}" for name, type_ in columns.items())
                    safe_statements.append(f"CREATE TABLE {quote_ident(table)} (\n    {cols}\n);")
                continue
            # ----------------------------

            is_new_table = table not in db_schema

            if is_new_table:
                defs = [f"{quote_ident(name)} {type_}" for name, type_ in columns.items()]

                # Constraints allowed ONLY on CREATE TABLE
                for fk in constraints.get("foreign_keys", []):
                    defs.append(_format_fk_clause(fk))

                safe_statements.append(f"CREATE TABLE {quote_ident(table)} (\n    {',\n    '.join(defs)}\n);")
                # record no column diffs: newly created table -> all columns are 'added'
                per_table_column_diffs[table] = {
                    "added": sorted(list(columns.keys())),
                    "removed": [],
                    "type_changed": [],
                    "renamed": [],
                }

                # If SCHEMA declares indexes or triggers for new table, emit them as safe_sql (CREATE INDEX / CREATE TRIGGER)
                declared_indexes = []
                declared_triggers = []
                if isinstance(table_def, dict):
                    declared_indexes = table_def.get("indexes", []) or []
                    declared_triggers = table_def.get("triggers", []) or []

                for idx in declared_indexes:
                    if isinstance(idx, str):
                        safe_statements.append(idx if idx.strip().endswith(";") else idx + ";")
                    elif isinstance(idx, dict) and "sql" in idx:
                        safe_statements.append(idx["sql"] if idx["sql"].strip().endswith(";") else idx["sql"] + ";")

                for tr in declared_triggers:
                    if isinstance(tr, str):
                        safe_statements.append(tr if tr.strip().endswith(";") else tr + ";")
                    elif isinstance(tr, dict) and "sql" in tr:
                        safe_statements.append(tr["sql"] if tr["sql"].strip().endswith(";") else tr["sql"] + ";")

                continue
            else:
                existing = db_schema[table]["columns"]
                # existing: mapping col->type (from PRAGMA table_info)
                added_cols = []
                removed_cols = []
                type_changed = []
                renamed_cols = []

                existing_col_types = {c: _normalize_type(t) for c, t in existing.items()}
                new_col_types = {c: _normalize_type(t) for c, t in columns.items()}

                existing_cols_set = set(existing.keys())
                new_cols_set = set(columns.keys())

                added_cols = sorted(list(new_cols_set - existing_cols_set))
                removed_cols = sorted(list(existing_cols_set - new_cols_set))

                # Detect columns present in both with type changes
                for c in sorted(list(existing_cols_set & new_cols_set)):
                    old_t = existing_col_types.get(c)
                    new_t = new_col_types.get(c)
                    if old_t != new_t:
                        type_changed.append({"column": c, "from": existing.get(c), "to": columns.get(c)})

                # Apply candidate column renames from CLI mapping for this table:
                if table in assume_col_map:
                    # mapping old->new
                    for old_col, new_col in assume_col_map[table].items():
                        if old_col in removed_cols and new_col in added_cols:
                            # Treat as rename candidate
                            renamed_cols.append({"from": old_col, "to": new_col})
                            # Remove from added/removed so they are not treated as destructive changes
                            removed_cols = [c for c in removed_cols if c != old_col]
                            added_cols = [c for c in added_cols if c != new_col]

                per_table_column_diffs[table] = {
                    "added": added_cols,
                    "removed": removed_cols,
                    "type_changed": type_changed,
                    "renamed": renamed_cols,
                }

                # Handle the "only added columns" case but conservatively: if any added column
                # would violate ALTER TABLE ADD COLUMN restrictions on this runtime, prefer rebuild.
                if added_cols and not removed_cols and not type_changed and not renamed_cols:
                    # Partition added columns into safe vs unsafe for ALTER
                    unsafe_found = False
                    for col in added_cols:
                        col_type = columns[col]
                        if not is_add_column_safe(col_type, sqlite_version):
                            unsafe_found = True
                            break

                    if not unsafe_found:
                        # All added columns are safe -> emit ALTER ADD COLUMN statements
                        for col in added_cols:
                            col_type = columns[col]
                            safe_statements.append(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(col)} {col_type};")
                    else:
                        # One or more added columns are unsafe to emit as ALTER -> schedule rebuild
                        tables_needing_rebuild.append(table)
                        # populate rebuild_defs (use final SCHEMA definition)
                        table_def = SCHEMA.get(table)
                        if table_def is not None:
                            cols, cons = normalize_table_def(table_def)
                            declared_indexes = []
                            declared_triggers = []
                            if isinstance(table_def, dict):
                                declared_indexes = table_def.get("indexes", []) or []
                                declared_triggers = table_def.get("triggers", []) or []

                            existing_index_sqls, existing_trigger_sqls = _collect_existing_index_and_trigger_sql(db_schema, table)

                            declared_index_sqls: List[str] = []
                            for idx in declared_indexes:
                                if isinstance(idx, str):
                                    declared_index_sqls.append(idx)
                                elif isinstance(idx, dict) and "sql" in idx and isinstance(idx["sql"], str):
                                    declared_index_sqls.append(idx["sql"])

                            declared_trigger_sqls: List[str] = []
                            for tr in declared_triggers:
                                if isinstance(tr, str):
                                    declared_trigger_sqls.append(tr)
                                elif isinstance(tr, dict) and "sql" in tr and isinstance(tr["sql"], str):
                                    declared_trigger_sqls.append(tr["sql"])

                            final_index_sqls = declared_index_sqls[:]
                            for s in existing_index_sqls:
                                if s not in final_index_sqls:
                                    final_index_sqls.append(s)

                            final_trigger_sqls = declared_trigger_sqls[:]
                            for s in existing_trigger_sqls:
                                if s not in final_trigger_sqls:
                                    final_trigger_sqls.append(s)

                            rebuild_defs[table] = {
                                "columns": copy.deepcopy(cols),
                                "constraints": copy.deepcopy(cons),
                                "indexes": copy.deepcopy(final_index_sqls),
                                "triggers": copy.deepcopy(final_trigger_sqls),
                            }
                else:
                    # Complex/destructive path (type changes, removed cols, renames not handled)
                    can_emit_renames = _version_ge(sqlite_version, 3, 25, 0)
                    emitted_rename_columns = []
                    if renamed_cols:
                        for r in renamed_cols:
                            old = r["from"]
                            new = r["to"]
                            if can_emit_renames and table in assume_col_map and assume_col_map[table].get(old) == new:
                                safe_statements.append(f"ALTER TABLE {quote_ident(table)} RENAME COLUMN {quote_ident(old)} TO {quote_ident(new)};")
                                emitted_rename_columns.append((old, new))

                        per_table_column_diffs[table]["renamed"] = [
                            r for r in renamed_cols if (r["from"], r["to"]) not in emitted_rename_columns
                        ]

                    remaining_removed = per_table_column_diffs[table]["removed"]
                    remaining_type_changed = per_table_column_diffs[table]["type_changed"]
                    remaining_renamed = per_table_column_diffs[table]["renamed"]

                    if (
                        remaining_removed
                        and not remaining_type_changed
                        and not remaining_renamed
                        and allow_drop_column
                        and _version_ge(sqlite_version, 3, 35, 0)
                    ):
                        for rc in remaining_removed:
                            safe_statements.append(f"ALTER TABLE {quote_ident(table)} DROP COLUMN {quote_ident(rc)};")
                        per_table_column_diffs[table]["removed"] = []
                        remaining_removed = []

                    if remaining_removed or remaining_type_changed or remaining_renamed:
                        tables_needing_rebuild.append(table)
                        table_def = SCHEMA.get(table)
                        if table_def is not None:
                            cols, cons = normalize_table_def(table_def)

                            declared_indexes = []
                            declared_triggers = []
                            if isinstance(table_def, dict):
                                declared_indexes = table_def.get("indexes", []) or []
                                declared_triggers = table_def.get("triggers", []) or []

                            existing_index_sqls, existing_trigger_sqls = _collect_existing_index_and_trigger_sql(db_schema, table)

                            declared_index_sqls: List[str] = []
                            for idx in declared_indexes:
                                if isinstance(idx, str):
                                    declared_index_sqls.append(idx)
                                elif isinstance(idx, dict) and "sql" in idx and isinstance(idx["sql"], str):
                                    declared_index_sqls.append(idx["sql"])

                            declared_trigger_sqls: List[str] = []
                            for tr in declared_triggers:
                                if isinstance(tr, str):
                                    declared_trigger_sqls.append(tr)
                                elif isinstance(tr, dict) and "sql" in tr and isinstance(tr["sql"], str):
                                    declared_trigger_sqls.append(tr["sql"])

                            final_index_sqls = declared_index_sqls[:]
                            for s in existing_index_sqls:
                                if s not in final_index_sqls:
                                    final_index_sqls.append(s)

                            final_trigger_sqls = declared_trigger_sqls[:]
                            for s in existing_trigger_sqls:
                                if s not in final_trigger_sqls:
                                    final_trigger_sqls.append(s)

                            rebuild_defs[table] = {
                                "columns": copy.deepcopy(cols),
                                "constraints": copy.deepcopy(cons),
                                "indexes": copy.deepcopy(final_index_sqls),
                                "triggers": copy.deepcopy(final_trigger_sqls),
                            }

        # ------------------------------
        # PHASE 2: Constraint diffs (default ON, can be disabled)
        # ------------------------------
        fk_diffs = {}
        if enable_constraints:
            db_fks = get_db_foreign_keys(conn)
            fk_diffs = diff_foreign_keys(schema_constraints, db_fks)

            for table, diff in fk_diffs.items():
                is_new_table = table not in db_schema
                if is_new_table:
                    continue

                adds = diff.get("add", [])
                removes = diff.get("remove", [])
                changes = diff.get("change", [])

                if not (adds or removes or changes):
                    continue

                if table not in tables_needing_rebuild:
                    tables_needing_rebuild.append(table)
                table_def = SCHEMA.get(table)
                if table_def is not None and table not in rebuild_defs:
                    cols, cons = normalize_table_def(table_def)

                    existing_index_sqls, existing_trigger_sqls = _collect_existing_index_and_trigger_sql(db_schema, table)

                    declared_indexes = []
                    declared_triggers = []
                    if isinstance(table_def, dict):
                        declared_indexes = table_def.get("indexes", []) or []
                        declared_triggers = table_def.get("triggers", []) or []

                    declared_index_sqls: List[str] = []
                    for idx in declared_indexes:
                        if isinstance(idx, str):
                            declared_index_sqls.append(idx)
                        elif isinstance(idx, dict) and "sql" in idx and isinstance(idx["sql"], str):
                            declared_index_sqls.append(idx["sql"])

                    declared_trigger_sqls: List[str] = []
                    for tr in declared_triggers:
                        if isinstance(tr, str):
                            declared_trigger_sqls.append(tr)
                        elif isinstance(tr, dict) and "sql" in tr and isinstance(tr["sql"], str):
                            declared_trigger_sqls.append(tr["sql"])

                    final_index_sqls = declared_index_sqls[:]
                    for s in existing_index_sqls:
                        if s not in final_index_sqls:
                            final_index_sqls.append(s)

                    final_trigger_sqls = declared_trigger_sqls[:]
                    for s in existing_trigger_sqls:
                        if s not in final_trigger_sqls:
                            final_trigger_sqls.append(s)

                    rebuild_defs[table] = {
                        "columns": copy.deepcopy(cols),
                        "constraints": copy.deepcopy(cons),
                        "indexes": copy.deepcopy(final_index_sqls),
                        "triggers": copy.deepcopy(final_trigger_sqls),
                    }

        # ------------------------------
        # PHASE 3: Build MIGRATION-META and write file (or dry-run)
        # ------------------------------
        db_tables = set(db_schema.keys())
        schema_tables = set(SCHEMA.keys())

        # Respect user-provided table rename assumptions: old:new
        tables_renamed: List[Dict[str, Any]] = []
        if assume_table_map:
            for old, new in assume_table_map.items():
                if old in db_tables and new in schema_tables:
                    tables_renamed.append({"from": old, "to": new})
                    if old in db_tables:
                        db_tables.remove(old)
                    if new in schema_tables:
                        schema_tables.remove(new)
                    safe_statements.append(f"ALTER TABLE {quote_ident(old)} RENAME TO {quote_ident(new)};")

        removed_candidates = db_tables - schema_tables
        added_candidates = schema_tables - db_tables
        try:
            detected_table_renames = detect_table_renames(
                conn,
                db_schema,
                SCHEMA,
                candidate_old_tables=removed_candidates,
                candidate_new_tables=added_candidates,
                threshold=0.6,
            )
            for cand in detected_table_renames:
                tables_renamed.append({"from": cand["from"], "to": cand["to"], "confidence": cand.get("confidence", 0.0)})
                if cand["from"] in db_tables:
                    db_tables.discard(cand["from"])
                if cand["to"] in schema_tables:
                    schema_tables.discard(cand["to"])
        except Exception:
            pass

        for table, col_diff in list(per_table_column_diffs.items()):
            if table in db_schema and table in SCHEMA:
                try:
                    desired_cols, _ = normalize_table_def(SCHEMA[table])
                    col_renames = detect_column_renames(conn, table, db_schema[table]["columns"], desired_cols, threshold=0.6)
                    existing_pairs = {(r.get("from"), r.get("to")) for r in col_diff.get("renamed", [])}
                    for c in col_renames:
                        pair = (c["from"], c["to"])
                        if pair not in existing_pairs:
                            col_diff.setdefault("renamed", []).append(c)
                except Exception:
                    pass

        final_removed_tables = sorted(list(db_tables - schema_tables))
        if allow_drop_table and final_removed_tables:
            for rt in final_removed_tables:
                if rt in SYSTEM_TABLES or rt.startswith("sqlite_"):
                    continue
                safe_statements.append(f"DROP TABLE {quote_ident(rt)};")

        db_view_names = set(db_views.keys())
        schema_view_names = set()
        if isinstance(SCHEMA, dict):
            if "VIEWS" in SCHEMA and isinstance(SCHEMA["VIEWS"], dict):
                schema_view_names |= set(SCHEMA["VIEWS"].keys())
            if "views" in SCHEMA and isinstance(SCHEMA["views"], dict):
                schema_view_names |= set(SCHEMA["views"].keys())

        diff_summary = {
            "tables": {
                "added": sorted(list(schema_tables - db_tables)),
                "removed": sorted(list(db_tables - schema_tables)),
                "renamed": tables_renamed,
            },
            "columns": per_table_column_diffs,
            "foreign_keys": fk_diffs,
            "indexes": per_table_index_diffs,
            "triggers": per_table_trigger_diffs,
            "views": {
                "db_only": sorted(list(db_view_names - schema_view_names)),
                "schema_only": sorted(list(schema_view_names - db_view_names)),
                "both": sorted(list(db_view_names & schema_view_names)),
            },
        }

        if (
            not safe_statements
            and not tables_needing_rebuild
            and not diff_summary["tables"]["added"]
            and not diff_summary["tables"]["removed"]
            and not diff_summary["tables"]["renamed"]
            and not diff_summary.get("foreign_keys")
        ):
            print("No schema changes detected.")
            return

        prefix = next_migration_prefix(MIGRATIONS_DIR)
        meta_id = f"{prefix}_{name}"
        meta: Dict[str, Any] = {
            "id": meta_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "diff": diff_summary,
            "safe_sql": safe_statements,
            "generator_options": {
                "allow_drop_table": allow_drop_table,
                "allow_drop_column": allow_drop_column,
                "assume_rename_table": assume_rename_table or [],
                "assume_rename_column": assume_rename_column or [],
                "sqlite_version": conn.execute("select sqlite_version()").fetchone()[0],
                "sqlite_capabilities": get_capabilities(sqlite_version),
            },
        }

        if tables_needing_rebuild:
            meta["rebuild"] = {
                "requires_rebuild": tables_needing_rebuild,
                "rebuild_defs": rebuild_defs,
                "batch": True,
            }

        if dry_run:
            # Preserve original dry-run outputs (tests depend on MIGRATION-META preview line)
            print("\n--- DRY RUN: MIGRATION-META preview ---\n")
            print("-- MIGRATION-META:", json.dumps(meta, separators=(',', ':'), sort_keys=True))
            print("\n--- safe_sql statements ---\n")
            for s in safe_statements:
                print(s)
            if tables_needing_rebuild:
                print("\n--- REBUILD REQUIRED for tables: ---")
                for t in tables_needing_rebuild:
                    print(f"  - {t}")
                print("\n(rebuild_defs present in MIGRATION-META preview)")
            print("\n--- Detailed diff (tables, columns & foreign_keys) ---\n")
            print(json.dumps(diff_summary, indent=2))
            print("\n--- end dry run ---\n")
            return

        path = os.path.join(MIGRATIONS_DIR, f"{prefix}_{name}.sql")

        header_json: Dict[str, Any] = {
            "id": meta["id"],
            "created_at": meta["created_at"],
            "diff": meta.get("diff"),
            "safe_sql": meta.get("safe_sql", []),
            "generator_options": meta.get("generator_options", {}),
        }

        if "rebuild" in meta:
            # Deepcopy rebuild metadata to avoid accidental mutation later
            h_rebuild = copy.deepcopy(meta["rebuild"])
            header_json["rebuild"] = h_rebuild

        try:
            validate_migration_meta(header_json)
        except Exception as e:
            raise RuntimeError(f"Constructed MIGRATION-META header is invalid: {e}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"-- MIGRATION-META: {json.dumps(header_json, separators=(',', ':'), sort_keys=True)}\n\n")
            f.write("-- auto-generated migration\n\n")
            if safe_statements:
                for stmt in safe_statements:
                    f.write(stmt)
                    if not stmt.strip().endswith(";"):
                        f.write(";")
                    f.write("\n\n")

            if tables_needing_rebuild:
                f.write("-- NOTE: This migration includes rebuild metadata in the MIGRATION-META header.\n")
                f.write("-- The rebuild will be performed by sqlite_rebuild.py using the embedded definitions.\n\n")

        print_user_message(f"Generated migration: {path}", verbose=verbose, quiet=quiet)

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-generate SQLite migration files with MIGRATION-META headers"
    )

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
        help="Print schema diffs and MIGRATION-META preview without writing a migration file"
    )

    parser.add_argument(
        "--disable-constraints",
        action="store_true",
        help="Disable constraint diffing and rebuild notes"
    )

    parser.add_argument(
        "--allow-drop-table",
        action="store_true",
        help="Allow generator to emit DROP TABLE (must be explicitly enabled)"
    )

    parser.add_argument(
        "--allow-drop-column",
        action="store_true",
        help="Allow generator to emit DROP COLUMN (must be explicitly enabled)"
    )

    parser.add_argument(
        "--assume-rename-table",
        action="append",
        help="Assume a table rename in the form old:new (repeatable)"
    )

    parser.add_argument(
        "--assume-rename-column",
        action="append",
        help="Assume a column rename in the form table.old:new (repeatable)"
    )

    # CLI UX flags
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra developer detail"
    )

    args = parser.parse_args()

    generate_migration(
        db_path=args.db_path,
        name=args.name,
        dry_run=args.dry_run,
        enable_constraints=not args.disable_constraints,
        allow_drop_table=args.allow_drop_table,
        allow_drop_column=args.allow_drop_column,
        assume_rename_table=args.assume_rename_table,
        assume_rename_column=args.assume_rename_column,
        verbose=args.verbose,
        quiet=args.quiet,
    )
