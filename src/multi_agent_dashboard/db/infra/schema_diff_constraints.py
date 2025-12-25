# schema_diff_constraints.py

import sqlite3
from typing import Dict, List, Any, TypedDict


class ForeignKeyDef(TypedDict, total=False):
    column: str
    references: str
    on_delete: str | None


def get_db_foreign_keys(conn: sqlite3.Connection) -> Dict[str, List[ForeignKeyDef]]:
    """
    Returns:
    {
        table: [
            {
                column: str,
                references: "other_table(col)",
                on_delete: str | None,
            }
        ]
    }
    """
    fks: Dict[str, List[ForeignKeyDef]] = {}

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (table,) in tables:
        rows = conn.execute(
            f"PRAGMA foreign_key_list({table})"
        ).fetchall()

        if not rows:
            continue

        table_fks: List[ForeignKeyDef] = []
        for r in rows:
            # SQLite pragma columns:
            # id, seq, table, from, to, on_update, on_delete, match
            on_delete = r[6]
            if on_delete == "NO ACTION":
                on_delete = None

            table_fks.append({
                "column": r[3],
                "references": f"{r[2]}({r[4]})",
                "on_delete": on_delete,
            })

        fks[table] = table_fks

    return fks


def _fk_key(fk: ForeignKeyDef) -> tuple[str, str]:
    """
    Logical identity of a FK: table column + referenced target.
    on_delete is *not* part of identity, because changing it
    should be detected as a "change", not an add/remove of a
    different FK.
    """
    return (fk["column"], fk["references"])


def diff_foreign_keys(
    schema_constraints: Dict[str, Dict[str, Any]],
    db_constraints: Dict[str, List[ForeignKeyDef]],
) -> Dict[str, Dict[str, List[ForeignKeyDef]]]:
    """
    Compute foreign key differences between desired schema and DB.

    Returns per-table:
    {
        table_name: {
            "add": [fk_defs...],       # new FKs to add
            "remove": [fk_defs...],    # FKs present in DB but not in schema
            "change": [                # FKs whose on_delete changed
                {
                    "from": fk_in_db,
                    "to": fk_in_schema,
                },
                ...
            ],
            "existing_in_db": bool,    # True if table already exists in DB
        }
    }

    Notes:
    - Adding/removing/changing FKs on existing SQLite tables requires a rebuild.
    - For *new* tables (not present in db_constraints), these FKs can be added
      directly in the CREATE TABLE statement; no rebuild note is necessary.
    """
    result: Dict[str, Dict[str, Any]] = {}

    # All tables that either have schema constraints or DB constraints
    all_tables = set(schema_constraints.keys()) | set(db_constraints.keys())

    for table in sorted(all_tables):
        wanted_list: List[ForeignKeyDef] = schema_constraints.get(table, {}).get("foreign_keys", []) or []
        existing_list: List[ForeignKeyDef] = db_constraints.get(table, []) or []

        existing_in_db = bool(existing_list or (table in db_constraints))

        # Build index maps by logical key
        wanted_by_key = {_fk_key(fk): fk for fk in wanted_list}
        existing_by_key = {_fk_key(fk): fk for fk in existing_list}

        adds: List[ForeignKeyDef] = []
        removes: List[ForeignKeyDef] = []
        changes: List[Dict[str, ForeignKeyDef]] = []

        # Detect adds & changes
        for key, wanted_fk in wanted_by_key.items():
            if key not in existing_by_key:
                # New FK
                adds.append(wanted_fk)
            else:
                db_fk = existing_by_key[key]
                # Compare on_delete; column+references already equal
                if db_fk.get("on_delete") != wanted_fk.get("on_delete"):
                    changes.append({"from": db_fk, "to": wanted_fk})

        # Detect removals
        for key, db_fk in existing_by_key.items():
            if key not in wanted_by_key:
                removes.append(db_fk)

        if adds or removes or changes:
            result[table] = {
                "add": adds,
                "remove": removes,
                "change": changes,
                "existing_in_db": existing_in_db,
            }

    return result
