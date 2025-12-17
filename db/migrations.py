import os
from datetime import datetime


def apply_migrations(conn, migrations_dir: str):
    """
    Apply SQL migrations exactly once, in filename order.
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

    for fname in sorted(os.listdir(migrations_dir)):
        if not fname.endswith(".sql"):
            continue

        migration_id = fname
        if migration_id in applied:
            continue

        path = os.path.join(migrations_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            sql = f.read()

        conn.executescript(sql)
        conn.execute(
            "INSERT INTO migrations (id, applied_at) VALUES (?, ?)",
            (migration_id, datetime.utcnow().isoformat()),
        )

        conn.commit()
