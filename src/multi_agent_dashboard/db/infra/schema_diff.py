# multi_agent_dashboard/db/infra/schema_diff.py
import sqlite3
import copy
from typing import Dict, List, Any, Optional

# Import foreign-key helpers from the dedicated module to avoid duplication
from multi_agent_dashboard.db.infra.schema_diff_constraints import (
    ForeignKeyDef,
    get_db_foreign_keys,
    diff_foreign_keys,
)
from multi_agent_dashboard.db.infra.sql_utils import quote_ident

# -------------------------
# DB schema introspection
# -------------------------
def get_db_schema(conn: sqlite3.Connection):
    """
    Returns a dictionary describing the database schema with additional metadata:

    {
        table_name: {
            "columns": { colname: type, ... },
            "pk": [col1, ...],
            "indexes": { index_name: { "unique": bool, "columns": [...], "sql": "CREATE INDEX ..." } },
            "triggers": [ { "name": ..., "sql": ... }, ... ],
            "create_sql": "CREATE TABLE ...",
        }
    }

    Note: the returned per-column "type" string is composed from PRAGMA table_info
    values and the table's CREATE SQL to include PRIMARY KEY and AUTOINCREMENT
    and any DEFAULT value reported by PRAGMA table_info. This produces a
    representation that matches the canonical SCHEMA declarations used by the
    generator, avoiding spurious diffs for PRIMARY KEY / DEFAULT / AUTOINCREMENT.
    """
    schema: Dict[str, Dict[str, Any]] = {}
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (table, create_sql) in tables:
        if table == "sqlite_sequence":
            continue

        cols = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()

        # Build a richer "type" mapping per column that incorporates:
        # - the declared type (PRAGMA table_info.type)
        # - whether the column is part of the primary key (PRAGMA table_info.pk)
        # - AUTOINCREMENT (detected from the CREATE TABLE SQL)
        # - DEFAULT value (PRAGMA table_info.dflt_value)
        columns_map: Dict[str, str] = {}
        create_sql_lower = (create_sql or "").lower()

        for c in cols:
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            col_name = c[1]
            col_type = c[2] or ""
            pk_flag = int(c[5] or 0)
            default_val = c[4]

            # Normalize/compose: include PRIMARY KEY if the column is a PK
            if pk_flag and "primary key" not in col_type.lower():
                if col_type:
                    col_type = f"{col_type} PRIMARY KEY"
                else:
                    col_type = "PRIMARY KEY"

            # If the table's CREATE SQL contains AUTOINCREMENT and this column is the PK,
            # reflect that in the returned type string (so it matches "INTEGER PRIMARY KEY AUTOINCREMENT").
            if pk_flag and "autoincrement" in create_sql_lower and "autoincrement" not in col_type.lower():
                col_type = f"{col_type} AUTOINCREMENT" if col_type else "AUTOINCREMENT"

            # Append DEFAULT if PRAGMA reports one and the type string does not already include DEFAULT.
            if default_val is not None:
                if "default" not in col_type.lower():
                    # default_val is the raw default expression as returned by PRAGMA,
                    # include it verbatim so comparisons against SCHEMA (which contain DEFAULT ...)
                    # will match after normalization.
                    col_type = f"{col_type} DEFAULT {default_val}" if col_type else f"DEFAULT {default_val}"

            columns_map[col_name] = col_type

        pk_cols = [c[1] for c in cols if c[5] and c[5] > 0]

        # Indexes
        indexes = {}
        idx_rows = conn.execute(f"PRAGMA index_list({quote_ident(table)})").fetchall()
        for idx in idx_rows:
            # idx: seq, name, unique, origin, partial
            idx_name = idx[1]
            unique = bool(idx[2])
            idx_info = conn.execute(f"PRAGMA index_info({quote_ident(idx_name)})").fetchall()
            idx_columns = [r[2] for r in idx_info]
            idx_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                (idx_name,),
            ).fetchone()
            idx_sql = idx_sql_row[0] if idx_sql_row else None
            indexes[idx_name] = {
                "unique": unique,
                "columns": idx_columns,
                "sql": idx_sql,
            }

        # Triggers
        triggers = []
        trigger_rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?",
            (table,),
        ).fetchall()
        for tr in trigger_rows:
            triggers.append({"name": tr[0], "sql": tr[1]})

        schema[table] = {
            "columns": columns_map,
            "pk": pk_cols,
            "indexes": indexes,
            "triggers": triggers,
            "create_sql": create_sql,
        }

    return schema


def get_db_views(conn: sqlite3.Connection) -> Dict[str, str]:
    """
    Return a mapping of view_name -> create_sql for all views in the database.

    This helper is intended to support sqlite_rebuild and generation tools that
    need to inspect and potentially recreate views when tables are rewritten.
    The returned mapping only includes views that have an explicit SQL definition
    in sqlite_master (i.e., sql IS NOT NULL).
    """
    views: Dict[str, str] = {}
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='view' AND sql IS NOT NULL").fetchall()
    for name, sql in rows:
        if not name or not sql:
            continue
        views[name] = sql
    return views


# -------------------------
# Rename-detection heuristics
# -------------------------
def _normalize_schema_table_def(table_def: Any) -> Dict[str, str]:
    """
    Normalize a table definition from SCHEMA (which may be either a flat mapping
    or a dict with a 'columns' key) into a simple columns mapping name->type.
    """
    if isinstance(table_def, dict) and "columns" in table_def:
        return table_def["columns"]
    if isinstance(table_def, dict):
        # Legacy/flat mapping
        return table_def
    raise TypeError("Unsupported table_def format in desired schema")


def _normalize_type(t: Optional[str]) -> str:
    if t is None:
        return ""
    if not isinstance(t, str):
        return str(t)
    return " ".join(t.strip().lower().split())


def detect_table_renames(
    conn: sqlite3.Connection,
    db_schema: Dict[str, Dict[str, Any]],
    desired_schema: Dict[str, Any],
    candidate_old_tables: Optional[set] = None,
    candidate_new_tables: Optional[set] = None,
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Heuristic detection of table renames between a DB schema and the desired schema.

    - Compute Jaccard similarity over the set of column names between each
      candidate pair (old -> new). If similarity >= threshold, propose the
      pair as a rename candidate with a 'confidence' equal to the Jaccard score.

    Returns:
        List of dicts: { "from": old_table, "to": new_table, "confidence": 0.72, "common_columns": [...] }
    """
    db_tables = set(db_schema.keys())
    desired_tables = set(desired_schema.keys())

    if candidate_old_tables is None:
        candidate_old_tables = db_tables - desired_tables
    else:
        candidate_old_tables = set(candidate_old_tables)

    if candidate_new_tables is None:
        candidate_new_tables = desired_tables - db_tables
    else:
        candidate_new_tables = set(candidate_new_tables)

    candidates: List[Dict[str, Any]] = []

    for old in sorted(candidate_old_tables):
        if old not in db_schema:
            continue
        old_cols = set(db_schema[old]["columns"].keys())
        if not old_cols:
            continue
        for new in sorted(candidate_new_tables):
            try:
                new_cols_map = _normalize_schema_table_def(desired_schema[new])
            except Exception:
                continue
            new_cols = set(new_cols_map.keys())
            if not new_cols:
                continue
            inter = old_cols & new_cols
            union = old_cols | new_cols
            jaccard = len(inter) / len(union) if union else 0.0
            if jaccard >= threshold:
                candidates.append({
                    "from": old,
                    "to": new,
                    "confidence": round(jaccard, 3),
                    "common_columns": sorted(list(inter)),
                })

    # Sort by confidence descending
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


def detect_column_renames(
    conn: sqlite3.Connection,
    table: str,
    existing_columns: Dict[str, str],
    desired_columns: Dict[str, str],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Heuristic detection of column rename candidates within a single table.

    Strategy (best-effort, lightweight):
    - For each removed column (in existing_columns but not in desired_columns)
      and each added column (in desired_columns but not in existing_columns),
      compute a heuristic confidence that 'removed' was renamed to 'added' using:
        - type match (strong signal)
        - simple name similarity (charset overlap)

    Returns:
        List of { "from": old_col, "to": new_col, "confidence": float }
    """
    existing_set = set(existing_columns.keys())
    desired_set = set(desired_columns.keys())

    removed = sorted(list(existing_set - desired_set))
    added = sorted(list(desired_set - existing_set))

    results: List[Dict[str, Any]] = []

    if not removed or not added:
        return results

    # Precompute normalized types
    existing_types = {c: _normalize_type(existing_columns.get(c)) for c in existing_columns}
    desired_types = {c: _normalize_type(desired_columns.get(c)) for c in desired_columns}

    for old in removed:
        for new in added:
            old_type = existing_types.get(old, "")
            new_type = desired_types.get(new, "")
            type_match = 1.0 if old_type and new_type and old_type == new_type else 0.0

            # Name similarity: charset overlap ratio (simple & cheap)
            old_chars = set(old.lower())
            new_chars = set(new.lower())
            denom = max(len(old_chars | new_chars), 1)
            char_overlap = len(old_chars & new_chars) / denom

            # Boost when names contain/are substrings
            substring_boost = 0.0
            if old.lower() == new.lower():
                substring_boost = 1.0
            elif old.lower() in new.lower() or new.lower() in old.lower():
                substring_boost = 0.9

            name_score = max(char_overlap, substring_boost)

            # Combine: weight type heavily (0.7) and name lightly (0.3)
            confidence = 0.7 * type_match + 0.3 * name_score

            # If types are missing (both blank), rely more on name similarity
            if not old_type and not new_type:
                confidence = name_score

            if confidence >= threshold:
                results.append({
                    "from": old,
                    "to": new,
                    "confidence": round(float(confidence), 3),
                    "type_from": old_type,
                    "type_to": new_type,
                })

    # Sort by confidence desc
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
