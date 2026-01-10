# multi_agent_dashboard/db/infra/migrations.py
import os
import re
import sqlite3
import copy
from datetime import datetime, timezone

from multi_agent_dashboard.db.infra.sqlite_rebuild import (
    rebuild_table_with_constraints,
    rebuild_tables_batch,
)
from multi_agent_dashboard.db.infra.migration_meta import (
    parse_migration_meta,
    validate_migration_meta,
    MigrationMetaValidationError,
)
from multi_agent_dashboard.db.infra.backup_utils import (
    create_backup_from_conn,
    restore_backup_to_conn,
    restore_backup_file_copy,
)
from multi_agent_dashboard.db.infra.cli_utils import print_user_message
from multi_agent_dashboard.db.infra.sql_utils import quote_ident

# Use centralized sqlite feature helpers for runtime gating
from multi_agent_dashboard.db.infra.sqlite_features import (
    supports_rename_column,
    supports_drop_column,
)


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
    # rows may be list of tuples; normalize
    user_tables = [name for (name,) in rows if name not in system_tables]

    if not user_tables:
        # No user tables at all → definitely fresh
        return True

    # Check if any user table has rows; if so, treat as non-fresh
    for table in user_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0]
        except Exception:
            # Be conservative: if anything goes wrong, don't treat as fresh
            return False
        if count > 0:
            return False

    # All user tables exist but are empty → safe to treat as fresh
    return True


def _sqlite_version_tuple_from_conn(conn) -> tuple[int, int, int]:
    """
    Return a 3-tuple version (major, minor, patch) for the sqlite engine backing
    this connection. On any parse failure, return (0,0,0).
    """
    try:
        ver = conn.execute("select sqlite_version()").fetchone()[0]
    except Exception:
        return (0, 0, 0)
    parts = ver.split(".")
    parts = (parts + ["0", "0"])[:3]
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _version_ge(vtuple: tuple[int, int, int], maj: int, min_: int, patch: int) -> bool:
    return vtuple >= (maj, min_, patch)


def apply_migrations(conn, migrations_dir: str, dry_run: bool = False, verbose: bool = False, quiet: bool = False):
    """
    Apply SQL migrations exactly once, in filename order.

    New behavior:
    - Migrations that include a MIGRATION-META header are applied according to that metadata.
      safe_sql statements are executed automatically.
    - If MIGRATION-META.rebuild.requires_rebuild is present:
      - On a fresh DB, the migration will auto-run rebuilds using the embedded rebuild_defs.
      - On a non-fresh DB, the migration will NOT auto-rebuild; the operator must run sqlite_rebuild.py
        (apply_migrations will print an actionable instruction and abort).
    - Legacy comment-based markers are NOT parsed at runtime. If a migration filename contains
      the historical '_REQUIRES_REBUILD' suffix and the file lacks MIGRATION-META, the migrator
      will abort and instruct the operator to run the annotation helper to add a MIGRATION-META header.
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

        # Prepare a couple of canonical identifiers for this migration:
        # - fname (e.g., "005_fix_x.sql")
        # - basename (no extension, e.g., "005_fix_x")
        file_basename = os.path.splitext(fname)[0]

        # If this migration has already been recorded in migrations table under either
        # the filename or the header id (common patterns), skip it.
        if fname in applied or file_basename in applied:
            continue

        path = os.path.join(migrations_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read()

        # Try to parse MIGRATION-META from the migration file
        try:
            meta = parse_migration_meta(sql, fname=path)
        except MigrationMetaValidationError as e:
            # If the header exists but is invalid JSON or malformed, abort.
            raise RuntimeError(f"Invalid MIGRATION-META in {fname}: {e}")

        # If no meta found:
        if meta is None:
            # If the migration contains a MIGRATION-META marker anywhere in the file
            # but it's not the canonical first-non-empty-line header, treat this as a
            # malformed header and abort. The header must be the first non-empty line
            # and runtime must not silently accept misplaced headers.
            if "-- MIGRATION-META:" in sql:
                raise RuntimeError(
                    f"Migration file '{fname}' contains a MIGRATION-META marker but it is not the first non-empty line of the file.\n"
                    "The canonical MIGRATION-META header must be the first non-empty line.\n"
                    "Please move/fix the header (or run the annotation helper) and re-run.\n"
                    f"Example: python tools/annotate_old_migrations.py --migration-file {os.path.join(migrations_dir, fname)}"
                )

            # Only rely on filename marker to detect legacy "requires rebuild" migrations.
            legacy_marker_in_name = "_REQUIRES_REBUILD" in fname

            if legacy_marker_in_name:
                # Refuse to silently accept legacy migrations that likely
                # require table rewrites. Instruct the operator to run the annotation helper.
                raise RuntimeError(
                    "This migration requires table rebuilds but lacks MIGRATION-META. "
                    f"Run tools/annotate_old_migrations.py {os.path.join(migrations_dir, fname)} to annotate, review, and re-run migration."
                )

            # For plain SQL-only migrations without MIGRATION-META, apply directly (but warn).
            if dry_run:
                print_user_message(f"[DRY RUN] Would apply legacy migration '{fname}' without MIGRATION-META header.", verbose=verbose, quiet=quiet)
                continue

            print(f"[migrations] Warning: applying legacy migration '{fname}' without MIGRATION-META header.")
            conn.executescript(sql)
            # Use the full filename (including extension) as canonical id for legacy plain SQL migrations
            id_to_insert = fname
            conn.execute(
                "INSERT INTO migrations (id, applied_at) VALUES (?, ?)",
                (id_to_insert, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            # Update in-memory applied set so rest of this run recognizes it
            applied.add(id_to_insert)
            applied.add(fname)
            applied.add(file_basename)
            continue

        # Validate meta structure
        try:
            validate_migration_meta(meta)
        except MigrationMetaValidationError as e:
            raise RuntimeError(f"MIGRATION-META validation failed for {fname}: {e}")

        # If meta.id already recorded in migrations, skip applying.
        meta_id = meta.get("id")
        if meta_id and meta_id in applied:
            # Already applied under canonical id
            continue

        # If dry-run, print plan & continue without modifying DB
        if dry_run:
            safe_sql_list = meta.get("safe_sql") or []
            rebuild_meta = meta.get("rebuild") or {}
            requires = rebuild_meta.get("requires_rebuild") or []
            summary_lines = [f"[DRY RUN] Migration: {meta.get('id') or file_basename}"]
            if safe_sql_list:
                summary_lines.append(f"  safe_sql statements: {len(safe_sql_list)}")
            if requires:
                summary_lines.append(f"  requires_rebuild: {requires}")
            print_user_message("\n".join(summary_lines), verbose=verbose, quiet=quiet)
            continue

        # Begin applying migration atomically (best-effort within sqlite3 limitations).
        try:
            # Run any rebuild.preconditions (if provided)
            rebuild_meta = meta.get("rebuild", {}) or {}
            preconds = rebuild_meta.get("preconditions", []) or []
            for pre in preconds:
                # Precondition must return at least one row to be considered satisfied.
                cur = conn.execute(pre)
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"Precondition failed for migration '{fname}': {pre}")

            # Determine rebuild requirements and prepare a backup BEFORE making changes
            requires = rebuild_meta.get("requires_rebuild", []) or []
            rebuild_defs = rebuild_meta.get("rebuild_defs", {}) or {}

            backup_path = None
            db_file = None

            # If the migration requires destructive rebuilds but the DB is not fresh -> abort early.
            if requires and not fresh_db:
                # Non-fresh DB and table rebuilds are required: do not auto-apply destructive rebuilds.
                db_list_row = conn.execute('PRAGMA database_list').fetchone()
                db_file = db_list_row[2] if db_list_row and len(db_list_row) > 2 else "<db_path>"
                raise RuntimeError(
                    f"Migration '{fname}' includes destructive table rebuilds for {requires}, but the database is not empty.\n"
                    f"Please run the sqlite_rebuild tool manually using the migration header for guidance, e.g.:\n"
                    f"  python -m multi_agent_dashboard.db.infra.sqlite_rebuild --from-migration {os.path.join(migrations_dir, fname)} {db_file}\n"
                    f"Or inspect/annotate migrations first using:\n"
                    f"  python tools/annotate_old_migrations.py --migration-file {os.path.join(migrations_dir, fname)}\n"
                )

            # If this migration will perform destructive rebuilds and DB is fresh,
            # create a pre-migration backup so we can restore the DB to its original
            # state if anything fails later.
            if requires and fresh_db:
                try:
                    db_list_row = conn.execute('PRAGMA database_list').fetchone()
                    db_file = db_list_row[2] if db_list_row and len(db_list_row) > 2 else None
                except Exception:
                    db_file = None

                try:
                    # Centralized helper: create a backup from the existing open connection.
                    # Important: create the backup BEFORE executing any safe_sql so the backup
                    # represents the pre-migration state and can be used to fully restore.
                    backup_path = create_backup_from_conn(conn, db_file)
                    print(f"Created DB backup before migration: {backup_path}")
                except Exception as e:
                    # Fail fast: do not proceed if we cannot create a reliable backup.
                    raise RuntimeError(f"Failed to create DB backup before migration '{fname}': {e}")

            # Execute safe_sql statements; prefer meta.safe_sql if present.
            safe_sql_list = meta.get("safe_sql") or []
            if safe_sql_list:
                # Runtime-check: ensure the local sqlite engine supports any ALTER variants present
                sqlite_ver = _sqlite_version_tuple_from_conn(conn)
                for stmt in safe_sql_list:
                    up = stmt.upper()
                    # Use sqlite_features helpers instead of hard-coded version checks
                    if "RENAME COLUMN" in up and not supports_rename_column(sqlite_ver):
                        raise RuntimeError(
                            f"Migration '{fname}' contains an ALTER TABLE RENAME COLUMN statement which requires SQLite >= 3.25.0.\n"
                            f"Runtime SQLite version: {sqlite_ver}. You can either regenerate this migration for the target environment or perform the table rewrite using the sqlite_rebuild tool:\n"
                            f"  python -m multi_agent_dashboard.db.infra.sqlite_rebuild --from-migration {os.path.join(migrations_dir, fname)} <db_path>\n"
                        )
                    if "DROP COLUMN" in up and not supports_drop_column(sqlite_ver):
                        raise RuntimeError(
                            f"Migration '{fname}' contains an ALTER TABLE DROP COLUMN statement which requires SQLite >= 3.35.0.\n"
                            f"Runtime SQLite version: {sqlite_ver}. Consider using sqlite_rebuild to perform the safe rewrite or re-generate the migration for the target environment.\n"
                            f"  python -m multi_agent_dashboard.db.infra.sqlite_rebuild --from-migration {os.path.join(migrations_dir, fname)} <db_path>\n"
                        )

                # If checks passed, execute safe statements
                for stmt in safe_sql_list:
                    # Each safe_sql entry may include semicolons; use executescript for safety.
                    conn.executescript(stmt)
            else:
                # IMPORTANT: Do NOT execute the file body SQL automatically if a MIGRATION-META header is present.
                # MIGRATION-META header is required to be the authoritative source of runtime decisions.
                # Executing unclassified SQL in the file body could cause accidental destructive changes.
                print(f"[migrations] Notice: migration '{fname}' contains a MIGRATION-META header but no 'safe_sql' statements. The SQL in the file body will NOT be executed automatically. "
                      "If you intended statements to be applied automatically, add them to 'safe_sql' in the MIGRATION-META header.")

            # Handle rebuilds (authoritative list in meta)
            if requires:
                # Rebuild defs are authoritative for this migration; prefer them over schema.py
                rebuild_defs = rebuild_meta.get("rebuild_defs", {}) or {}

                if fresh_db and requires:
                    # Safe to auto-rebuild on a fresh DB: use provided rebuild_defs.
                    # Commit any prior safe_sql statements so PRAGMA foreign_keys toggles are outside a transaction.
                    conn.commit()

                    try:
                        if len(requires) > 1:
                            # Build a map for batch rebuild
                            rebuild_map = {}
                            for table in requires:
                                table_def = rebuild_defs.get(table)
                                if not table_def:
                                    # If migration didn't include rebuild_defs for a required table, abort.
                                    raise RuntimeError(
                                        f"Migration '{fname}' requests a rebuild of '{table}' but no rebuild_def was provided in MIGRATION-META."
                                    )

                                if "columns" in table_def:
                                    columns = copy.deepcopy(table_def["columns"])
                                    constraints = copy.deepcopy(table_def.get("constraints", {}))
                                    indexes = copy.deepcopy(table_def.get("indexes", []))
                                    triggers = copy.deepcopy(table_def.get("triggers", []))
                                else:
                                    # legacy-like structure: accept flat mapping
                                    columns = copy.deepcopy(table_def)
                                    constraints = {}
                                    indexes = []
                                    triggers = {}

                                rebuild_map[table] = {"columns": columns, "constraints": constraints, "indexes": indexes, "triggers": triggers}

                            # Use batch rebuild to toggle foreign_keys once and validate afterwards
                            rebuild_tables_batch(conn, rebuild_map)
                        else:
                            # Single-table rebuilds: call per-table helper
                            for table in requires:
                                table_def = rebuild_defs.get(table)
                                if not table_def:
                                    # If migration didn't include rebuild_defs for a required table, abort.
                                    raise RuntimeError(
                                        f"Migration '{fname}' requests a rebuild of '{table}' but no rebuild_def was provided in MIGRATION-META."
                                    )

                                if "columns" in table_def:
                                    columns = copy.deepcopy(table_def["columns"])
                                    constraints = copy.deepcopy(table_def.get("constraints", {}))
                                    indexes = copy.deepcopy(table_def.get("indexes", []))
                                    triggers = copy.deepcopy(table_def.get("triggers", []))
                                else:
                                    # legacy-like structure: accept flat mapping
                                    columns = copy.deepcopy(table_def)
                                    constraints = {}
                                    indexes = []
                                    triggers = {}

                                rebuild_table_with_constraints(conn, table, columns, constraints)
                    except Exception as e:
                        # Ensure any active transaction is rolled back before attempting restore.
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                        # Attempt best-effort restore from backup (if we created one).
                        if backup_path and os.path.exists(backup_path):
                            try:
                                # Preferred: restore into the open connection using sqlite backup API.
                                restore_backup_to_conn(backup_path, conn)
                                print(f"Restored DB from backup (sqlite backup): {backup_path}")
                            except Exception as restore_exc:
                                print(f"Failed to restore via sqlite backup API: {restore_exc}. Attempting file-level restore...")
                                try:
                                    try:
                                        conn.close()
                                    except Exception:
                                        pass
                                    if db_file:
                                        restore_backup_file_copy(backup_path, db_file)
                                        print(f"Restored DB from backup (file copy): {backup_path}")
                                except Exception as file_restore_exc:
                                    print(f"Failed to restore DB from backup by file copy: {file_restore_exc}")
                        # Surface the original rebuild error
                        raise RuntimeError(f"Rebuild failed for migration '{fname}': {e}")
                elif requires:
                    # Non-fresh DB and table rebuilds are required: do not auto-apply destructive rebuilds.
                    db_list_row = conn.execute('PRAGMA database_list').fetchone()
                    db_file = db_list_row[2] if db_list_row and len(db_list_row) > 2 else "<db_path>"
                    raise RuntimeError(
                        f"Migration '{fname}' includes destructive table rebuilds for {requires}, but the database is not empty.\n"
                        f"Please run the sqlite_rebuild tool manually using the migration header for guidance, e.g.:\n"
                        f"  python -m multi_agent_dashboard.db.infra.sqlite_rebuild --from-migration {os.path.join(migrations_dir, fname)} {db_file}\n"
                        f"Or inspect/annotate migrations first using:\n"
                        f"  python tools/annotate_old_migrations.py --migration-file {os.path.join(migrations_dir, fname)}\n"
                    )

            # Run post-checks if present
            post_checks = rebuild_meta.get("post_checks", []) or []
            for check in post_checks:
                cur = conn.execute(check)
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"Post-check failed for migration '{fname}': {check}")

            # If we get here, migration applied successfully; persist history
            # Use meta.id when present to record canonical id, else use full filename
            id_to_insert = meta.get("id") if meta else fname
            conn.execute(
                "INSERT INTO migrations (id, applied_at) VALUES (?, ?)",
                (id_to_insert, datetime.now(timezone.utc).isoformat()),
            )

            conn.commit()
            # Update in-memory applied set so rest of this run recognizes the newly-applied migration
            applied.add(id_to_insert)
            applied.add(fname)
            applied.add(file_basename)

        except Exception:
            conn.rollback()
            raise
