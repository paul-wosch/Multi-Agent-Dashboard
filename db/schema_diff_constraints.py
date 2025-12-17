# schema_diff_constraints.py

import sqlite3

def get_db_foreign_keys(conn: sqlite3.Connection):
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
    fks = {}

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (table,) in tables:
        rows = conn.execute(
            f"PRAGMA foreign_key_list({table})"
        ).fetchall()

        if not rows:
            continue

        table_fks = []
        for r in rows:
            # SQLite pragma columns:
            # id, seq, table, from, to, on_update, on_delete, match
            table_fks.append({
                "column": r[3],
                "references": f"{r[2]}({r[4]})",
                "on_delete": r[6] if r[6] != "NO ACTION" else None,
            })

        fks[table] = table_fks

    return fks


def diff_foreign_keys(schema_constraints, db_constraints):
    """
    Returns missing FKs that exist in schema but not DB.
    Does NOT detect removals automatically (rebuild-only).
    """
    missing = {}

    for table, constraints in schema_constraints.items():
        wanted = constraints.get("foreign_keys", [])
        existing = db_constraints.get(table, [])

        for fk in wanted:
            if fk not in existing:
                missing.setdefault(table, []).append(fk)

    return missing
