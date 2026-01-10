"""multi_agent_dashboard/db/infra/sqlite_rebuild.py

Rebuild SQLite tables using rebuild definitions embedded in MIGRATION-META headers.

Changes:
 - Improved view detection (token-based) to avoid substring false positives.
 - Introduced 'force' flag to allow proceeding when view-detection is ambiguous.
 - Quote identifiers consistently (via sql_utils.quote_ident).
 - Use copy.deepcopy when importing migration-provided rebuild_defs to avoid mutation surprises.
 - Print concise actionable messages using cli_utils.print_user_message.
"""
import os
import sqlite3
import re
import copy
from typing import Dict, Any, List, Tuple, Optional

from multi_agent_dashboard.db.infra.schema import SCHEMA
from multi_agent_dashboard.db.infra.migration_meta import parse_migration_meta, MigrationMetaValidationError
from multi_agent_dashboard.config import DB_FILE_PATH, MIGRATIONS_PATH

from multi_agent_dashboard.db.infra.backup_utils import (
    make_backup_path,
    create_backup_from_file,
    restore_backup_file_copy,
)

from multi_agent_dashboard.db.infra.cli_utils import print_user_message
from multi_agent_dashboard.db.infra.sql_utils import quote_ident, quote_column_list

# -----------------------
# Minimal SQL tokenizer helpers (robust enough for identifier detection)
# -----------------------
_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RE_LINE_COMMENT = re.compile(r"--[^\n]*")
_RE_SINGLE_QUOTED = re.compile(r"'(?:''|[^'])*'")  # matches SQL single-quoted string literals
# Identifier/token regex: double-quoted identifiers, backtick-quoted, or bare identifiers
_RE_IDENTIFIER = re.compile(r'"([^"]+)"|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*)', re.UNICODE)


def _strip_comments_and_strings(sql: str) -> str:
    """Remove block comments, line comments, and single-quoted string literals."""
    if not sql:
        return ""
    s = _RE_BLOCK_COMMENT.sub(" ", sql)
    s = _RE_LINE_COMMENT.sub(" ", s)
    s = _RE_SINGLE_QUOTED.sub("''", s)
    return s


def _identifiers_from_sql(sql: str) -> List[str]:
    """
    Extract identifier-like tokens from SQL (quoted or unquoted).
    Returns a list of identifier strings (without surrounding quotes).
    """
    cleaned = _strip_comments_and_strings(sql)
    ids: List[str] = []
    for m in _RE_IDENTIFIER.finditer(cleaned):
        if m.group(1):
            ids.append(m.group(1))
        elif m.group(2):
            ids.append(m.group(2))
        elif m.group(3):
            ids.append(m.group(3))
    return ids


def _view_refers_to_any_table(view_sql: str, tables: List[str]) -> bool:
    """
    Return True if any identifier token in view_sql matches any name in `tables`.
    Uses tokenization so that substrings or comments do not produce false positives.
    """
    tokens = _identifiers_from_sql(view_sql)
    if not tokens:
        return False
    token_set = {t.lower() for t in tokens}
    for t in tables:
        if t.lower() in token_set:
            return True
    return False


def _ensure_no_active_transaction(conn: sqlite3.Connection) -> None:
    """
    Ensure there is no active transaction on the connection.
    PRAGMA foreign_keys toggling is a no-op inside an active transaction.
    """
    try:
        in_tx = getattr(conn, "in_transaction", False)
    except Exception:
        in_tx = False
    if in_tx:
        conn.commit()


def rebuild_table_with_constraints(conn, table, columns, constraints):
    """
    Rebuild a single table by delegating to the batch rebuild path.
    """
    rebuild_tables_batch(conn, {table: {"columns": columns, "constraints": constraints}})


def _get_table_def_from_schema(table: str) -> tuple[Dict[str, str], Dict[str, Any]]:
    if table not in SCHEMA:
        raise KeyError(f"Table '{table}' not found in schema.py SCHEMA")

    table_def = SCHEMA[table]
    if "columns" in table_def:
        return table_def["columns"], table_def.get("constraints", {})
    # legacy style
    return table_def, {}


def _scan_migrations_for_rebuilds(migrations_dir: str) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """
    (unchanged) scan migrations directory and extract rebuild metadata...
    """
    tables_ordered: List[str] = []
    rebuild_defs: Dict[str, Dict[str, Any]] = {}
    file_map: Dict[str, List[str]] = {}
    invalid_meta_files: List[Tuple[str, str]] = []

    for fname in sorted(os.listdir(migrations_dir)):
        if not fname.endswith(".sql"):
            continue
        path = os.path.join(migrations_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            meta = parse_migration_meta(text, fname=path)
        except MigrationMetaValidationError as e:
            invalid_meta_files.append((fname, str(e)))
            continue
        except Exception:
            meta = None

        if not meta:
            continue

        rebuild = (meta.get("rebuild") or {})
        requires = rebuild.get("requires_rebuild") or []
        defs = rebuild.get("rebuild_defs") or {}

        if not requires:
            continue

        file_map[fname] = list(requires)
        for t in requires:
            if t not in rebuild_defs:
                if t in defs:
                    td = defs[t]
                    # Use deepcopy to avoid accidental mutations of migration metadata
                    td_copy = copy.deepcopy(td)
                    td_copy["_inference_source"] = "migration"
                    rebuild_defs[t] = td_copy
                else:
                    try:
                        cols, cons = _get_table_def_from_schema(t)
                        rebuild_defs[t] = {"columns": copy.deepcopy(cols), "constraints": copy.deepcopy(cons), "_inferred_from_schema": True}
                    except KeyError:
                        rebuild_defs[t] = {"columns": {}, "constraints": {}, "_inferred_from_schema": False}

            if t not in tables_ordered:
                tables_ordered.append(t)

    if invalid_meta_files:
        print("Warning: some migration files contain invalid MIGRATION-META headers and were skipped:")
        for fn, err in invalid_meta_files:
            print(f"  - {fn}: {err}")

    return tables_ordered, rebuild_defs, file_map


def rebuild_tables_batch(conn: sqlite3.Connection, rebuild_map: Dict[str, Dict[str, Any]], force: bool = False):
    """
    Rebuild multiple tables in a single connection with foreign_keys disabled once.

    If view detection is ambiguous (substring match but no token match), and force is False,
    this will abort with a clear message rather than silently drop/recreate views.
    """
    if not rebuild_map:
        return

    _ensure_no_active_transaction(conn)

    try:
        conn.execute("PRAGMA foreign_keys = OFF")
    except Exception as e:
        raise RuntimeError(f"Failed to set PRAGMA foreign_keys = OFF: {e}")

    try:
        all_tables = set(rebuild_map.keys())
        view_rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='view' AND sql IS NOT NULL").fetchall()

        views_to_recreate: List[Tuple[str, str]] = []
        ambiguous_views: List[Tuple[str, str]] = []

        for row in view_rows:
            name, sql = row[0], row[1]
            if not sql:
                continue
            # Token-based detection (preferred)
            if _view_refers_to_any_table(sql, list(all_tables)):
                views_to_recreate.append((name, sql))
                continue
            # Fallback substring heuristic: if substring appears but tokenization didn't match,
            # mark as ambiguous so operator can review.
            sql_lower = sql.lower()
            if any(t.lower() in sql_lower for t in all_tables):
                ambiguous_views.append((name, sql))

        if ambiguous_views and not force:
            msg_lines = [
                "Ambiguous view dependencies detected. Some views may reference tables scheduled for rebuild but tokenization could not confirm identifiers.",
                "To avoid accidental DROP/CREATE of views, run sqlite_rebuild with --force after manual review, or inspect the following views:",
            ]
            for vn, vsql in ambiguous_views:
                msg_lines.append(f"  - {vn}")
            msg_lines.append("Re-run with --force to proceed despite ambiguity.")
            raise RuntimeError("\n".join(msg_lines))

        # Begin a single explicit transaction for all table rewrites
        conn.execute("BEGIN")

        # Drop the confidently-identified views inside the transaction so we can safely rewrite tables.
        for view_name, _ in views_to_recreate:
            quoted = quote_ident(view_name)
            conn.execute(f"DROP VIEW IF EXISTS {quoted}")

        for table, td in rebuild_map.items():
            tmp = f"{table}__rebuild"
            columns = td.get("columns", {})
            constraints = td.get("constraints", {})

            if not isinstance(columns, dict) or not columns:
                raise RuntimeError(f"Cannot rebuild table '{table}': rebuild_defs lack a non-empty 'columns' mapping.")

            td_indexes = td.get("indexes")
            td_triggers = td.get("triggers")

            if isinstance(td_indexes, list) and td_indexes:
                index_sqls = [s for s in td_indexes if s and isinstance(s, str)]
            else:
                index_rows = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                    (table,),
                ).fetchall()
                index_sqls = [r[0] for r in index_rows if r and r[0]]

            if isinstance(td_triggers, list) and td_triggers:
                trigger_sqls = [s for s in td_triggers if s and isinstance(s, str)]
            else:
                trigger_rows = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL",
                    (table,),
                ).fetchall()
                trigger_sqls = [r[0] for r in trigger_rows if r and r[0]]

            # Quote column names in the CREATE TABLE temporary definition
            col_defs = [f"{quote_ident(c)} {typ}" for c, typ in columns.items()]

            for fk in constraints.get("foreign_keys", []):
                clause = f"FOREIGN KEY({quote_column_list(fk['column'])}) REFERENCES {fk['references']}"
                if fk.get("on_delete"):
                    clause += f" ON DELETE {fk['on_delete']}"
                col_defs.append(clause)

            conn.execute(f"DROP TABLE IF EXISTS {quote_ident(tmp)}")
            conn.execute(f"CREATE TABLE {quote_ident(tmp)} ({', '.join(col_defs)})")

            old_cols_rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            old_cols = [r[1] for r in old_cols_rows]
            new_cols = list(columns.keys())
            common_cols = [c for c in new_cols if c in old_cols]

            if common_cols:
                cols_quoted = ", ".join(quote_ident(c) for c in common_cols)
                conn.execute(f"INSERT INTO {quote_ident(tmp)} ({cols_quoted}) SELECT {cols_quoted} FROM {quote_ident(table)}")

            conn.execute(f"DROP TABLE {quote_ident(table)}")
            conn.execute(f"ALTER TABLE {quote_ident(tmp)} RENAME TO {quote_ident(table)}")

            for sql in index_sqls:
                if sql and sql.strip():
                    conn.execute(sql)

            for sql in trigger_sqls:
                if sql and sql.strip():
                    conn.execute(sql)

        # Commit all changes as one atomic transaction
        conn.commit()

        # After commit, recreate views we dropped earlier (outside the transaction)
        for view_name, view_sql in views_to_recreate:
            if view_sql and view_sql.strip():
                conn.execute(view_sql)

        # Re-enable foreign keys and validate (outside transaction)
        _ensure_no_active_transaction(conn)
        conn.execute("PRAGMA foreign_keys = ON")

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign key violations after rebuild: {violations}")

    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            _ensure_no_active_transaction(conn)
            conn.execute("PRAGMA foreign_keys = ON")
        except Exception:
            pass
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild SQLite tables using MIGRATION-META rebuild_defs"
    )

    parser.add_argument(
        "table",
        nargs="?",
        help="Table name to rebuild"
    )

    parser.add_argument(
        "--all-with-diffs",
        action="store_true",
        help=(
            "Automatically rebuild all tables that have rebuild metadata in migrations"
        ),
    )

    parser.add_argument(
        "--from-migration",
        help="Rebuild tables listed in the given migration file's MIGRATION-META header"
    )

    parser.add_argument(
        "db_path",
        nargs="?",
        default=DB_FILE_PATH,
        help=f"Path to SQLite database file (default: {DB_FILE_PATH})"
    )
    parser.add_argument(
        "--migrations-dir",
        default=MIGRATIONS_PATH,
        help=f"Path to migrations directory (default: {MIGRATIONS_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done, but do not modify the database"
    )

    # New CLI UX flags
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if view dependency detection is ambiguous (use with care)"
    )
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

    modes_used = 0
    if args.table:
        modes_used += 1
    if args.all_with_diffs:
        modes_used += 1
    if args.from_migration:
        modes_used += 1

    if modes_used > 1:
        raise SystemExit("Specify exactly one of: a positional <table>, --all-with-diffs, or --from-migration.")

    db_path = args.db_path
    migrations_dir = args.migrations_dir

    if not os.path.exists(db_path):
        raise SystemExit(f"Database file not found: {db_path}")

    if args.all_with_diffs:
        tables, rebuild_defs, file_map = _scan_migrations_for_rebuilds(migrations_dir)
        if not tables:
            print_user_message("No tables require rebuild according to MIGRATION-META headers in migrations directory.", verbose=args.verbose, quiet=args.quiet)
            return

        missing_defs = [t for t in tables if not rebuild_defs.get(t) or not isinstance(rebuild_defs.get(t).get("columns", None), dict) or not rebuild_defs.get(t).get("columns")]
        if missing_defs:
            print_user_message("Error: some rebuild_defs are missing or incomplete. Please annotate migrations and review.", verbose=args.verbose, quiet=args.quiet)
            raise SystemExit("Aborting due to incomplete rebuild_defs. Do not proceed until metadata is reviewed.")

        if not args.quiet:
            print("Tables requiring rebuild (detected via MIGRATION-META):")
            for t in tables:
                origins = [f for f, ts in file_map.items() if t in ts]
                inferred_note = ""
                if rebuild_defs.get(t, {}).get("_inferred_from_schema"):
                    inferred_note = " (rebuild_def inferred from current schema.py; please review)"
                print(f"  - {t}{inferred_note} (from: {origins})")

        if args.dry_run:
            if not args.quiet:
                print("\n[DRY RUN] Would rebuild the above tables in:", db_path)
                backup_path = make_backup_path(db_path)
                print(f"[DRY RUN] Would create backup at: {backup_path}")
            return

        try:
            backup_path = create_backup_from_file(db_path)
            print_user_message(f"Created backup: {backup_path}", verbose=args.verbose, quiet=args.quiet)
        except Exception as e:
            raise SystemExit(f"Failed to create DB backup: {e}")

        conn = sqlite3.connect(db_path)
        try:
            try:
                if not args.quiet:
                    print("Beginning batch rebuild...")
                rebuild_tables_batch(conn, {t: rebuild_defs[t] for t in tables}, force=args.force)
                if not args.quiet:
                    print("Successfully completed batch rebuild.")
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                print_user_message(f"ERROR during rebuild: {e}. Restoring from backup...", verbose=args.verbose, quiet=args.quiet)
                restore_backup_file_copy(backup_path, db_path)
                raise SystemExit(f"Rebuild failed and DB restored from backup: {backup_path}\nError: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        if not args.quiet:
            print("All requested tables processed.")
            print("Backup remains at:", backup_path)
        return

    if args.from_migration:
        migration_file = args.from_migration
        if not os.path.exists(migration_file):
            raise SystemExit(f"Migration file not found: {migration_file}")

        with open(migration_file, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            meta = parse_migration_meta(text, fname=migration_file)
        except MigrationMetaValidationError as e:
            raise SystemExit(f"Invalid MIGRATION-META in {migration_file}: {e}")

        if not meta:
            raise SystemExit(f"No MIGRATION-META found in {migration_file}")

        rebuild = (meta.get("rebuild") or {})
        requires = rebuild.get("requires_rebuild") or []
        rebuild_defs = rebuild.get("rebuild_defs") or {}

        if not requires:
            print_user_message(f"No tables require rebuild in MIGRATION-META of {migration_file}", verbose=args.verbose, quiet=args.quiet)
            return

        missing_defs = [t for t in requires if t not in rebuild_defs]
        if missing_defs:
            raise SystemExit(
                f"The MIGRATION-META in {migration_file} lists requires_rebuild={requires} but is missing rebuild_defs for: {missing_defs}.\n"
                f"Please annotate the migration with full rebuild_defs, then re-run."
            )

        rebuild_map: Dict[str, Dict[str, Any]] = {}
        for t in requires:
            td = rebuild_defs.get(t)
            if not isinstance(td, dict):
                raise SystemExit(f"Invalid rebuild_def for '{t}' in {migration_file}: expected object with 'columns' key.")
            cols = td.get("columns")
            if not isinstance(cols, dict) or not cols:
                raise SystemExit(f"rebuild_defs['{t}'].columns missing or empty in {migration_file}.")
            cons = td.get("constraints", {})
            idxs = td.get("indexes", [])
            trgs = td.get("triggers", [])
            rebuild_map[t] = {"columns": cols, "constraints": cons, "indexes": idxs, "triggers": trgs}

        print_user_message(f"Rebuild requested by migration {migration_file}: {requires}", verbose=args.verbose, quiet=args.quiet)

        if args.dry_run:
            if not args.quiet:
                print("[DRY RUN] Would rebuild tables:", requires)
                backup_path = make_backup_path(db_path)
                print(f"[DRY RUN] Would create backup at: {backup_path}")
            return

        try:
            backup_path = create_backup_from_file(db_path)
            print_user_message(f"Created backup: {backup_path}", verbose=args.verbose, quiet=args.quiet)
        except Exception as e:
            raise SystemExit(f"Failed to create DB backup: {e}")

        conn = sqlite3.connect(db_path)
        try:
            try:
                rebuild_tables_batch(conn, rebuild_map, force=args.force)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                print_user_message(f"ERROR during rebuild: {e}. Restoring from backup...", verbose=args.verbose, quiet=args.quiet)
                restore_backup_file_copy(backup_path, db_path)
                raise SystemExit(f"Rebuild failed and DB restored from backup: {backup_path}\nError: {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        print_user_message(f"Successfully rebuilt tables from {migration_file}", verbose=args.verbose, quiet=args.quiet)
        if not args.quiet:
            print("Backup remains at:", backup_path)
        return

    table = args.table
    if not table:
        raise SystemExit(
            "You must specify exactly one of: a positional <table>, --from-migration, or --all-with-diffs.\n"
            "Examples:\n"
            f"  python {os.path.basename(__file__)} my_table {db_path}\n"
            f"  python {os.path.basename(__file__)} --from-migration data/migrations/005_fix_agent_outputs_constraints.sql {db_path}\n"
            f"  python {os.path.basename(__file__)} --all-with-diffs {db_path}\n"
        )

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

    try:
        columns, constraints = _get_table_def_from_schema(table)
    except KeyError as e:
        raise SystemExit(str(e))

    if args.dry_run:
        if not args.quiet:
            print(f"[DRY RUN] Would rebuild table '{table}' in {db_path}")
            print(f"[DRY RUN] Columns: {columns}")
            print(f"[DRY RUN] Constraints: {constraints}")
            backup_path = make_backup_path(db_path)
            print(f"[DRY RUN] Would create backup at: {backup_path}")
        return

    try:
        backup_path = create_backup_from_file(db_path)
        print_user_message(f"Created backup: {backup_path}", verbose=args.verbose, quiet=args.quiet)
    except Exception as e:
        raise SystemExit(f"Failed to create DB backup: {e}")

    conn = sqlite3.connect(db_path)
    try:
        rebuild_tables_batch(conn, {table: {"columns": columns, "constraints": constraints}}, force=args.force)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    print_user_message(f"Successfully rebuilt table '{table}' in {db_path}", verbose=args.verbose, quiet=args.quiet)
    if not args.quiet:
        print("Backup remains at:", backup_path)
