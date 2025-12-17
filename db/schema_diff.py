import sqlite3

def get_db_schema(conn: sqlite3.Connection):
    schema = {}
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for (table,) in tables:
        if table == "sqlite_sequence":
            continue

        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        schema[table] = {
            "columns": {c[1]: c[2] for c in cols}
        }

    return schema
