# db/infra/core.py
import json
import logging
import sqlite3

from contextlib import contextmanager

from multi_agent_dashboard.config import MIGRATIONS_PATH
from multi_agent_dashboard.db.infra.migrations import apply_migrations

logger = logging.getLogger(__name__)

def init_db(db_path: str):
    """
    Initialize the database:
    - open connection
    - apply migrations
    """
    logger.info("Initializing database at %s", db_path)
    try:
        with get_conn(db_path) as conn:
            apply_migrations(conn, MIGRATIONS_PATH)
    except Exception:
        logger.exception("Database initialization failed")
        raise


def safe_json_loads(value: str | None, default):
    """
    Safely load JSON from DB fields.
    Returns default if value is None, empty, or invalid.
    """
    if not value or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in DB field")
        return default


# -----------------------
# Connection helper
# -----------------------

@contextmanager
def get_conn(path):
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
