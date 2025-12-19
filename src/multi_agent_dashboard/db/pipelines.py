# db/pipelines.py
import json
import logging

from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple

from multi_agent_dashboard.db.infra.core import get_conn, safe_json_loads

logger = logging.getLogger(__name__)


# -----------------------
# READ operations
# -----------------------

def load_pipelines_from_db(db_path: str) -> list[dict]:
    logger.debug("Loading pipelines from DB")
    try:
        with get_conn(db_path) as conn:
            rows = conn.execute(
                """
                SELECT pipeline_name, steps_json, metadata_json, timestamp
                FROM pipelines
                ORDER BY pipeline_name
                """
            ).fetchall()
    except Exception:
        logger.exception("Failed to load pipelines from DB")
        raise

    result = []
    for row in rows:
        result.append({
            "pipeline_name": row["pipeline_name"],
            "steps": safe_json_loads(row["steps_json"], []),
            "metadata": safe_json_loads(row["metadata_json"], {}),
            "timestamp": row["timestamp"],
        })
    return result


# -----------------------
# WRITE operations
# -----------------------

def save_pipeline_to_db(
        db_path: str,
        pipeline_name: str,
        steps: List[str],
        metadata: Optional[dict] = None
    ):
    logger.info("Saving pipeline %s to DB", pipeline_name)
    try:
        with get_conn(db_path) as conn:
            c = conn.cursor()

            ts = datetime.now(UTC).isoformat()
            steps_json = json.dumps(steps)
            metadata_json = json.dumps(metadata or {})

            c.execute("""
                INSERT OR REPLACE INTO pipelines (pipeline_name, steps_json, metadata_json, timestamp)
                VALUES (?, ?, ?, ?)
            """, (pipeline_name, steps_json, metadata_json, ts))
    except Exception:
        logger.exception("Failed to save pipeline %s to DB", pipeline_name)
        raise


def delete_pipeline_from_db(db_path: str, pipeline_name: str):
    logger.info("Deleting pipeline %s from DB", pipeline_name)
    try:
        with get_conn(db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM pipelines WHERE pipeline_name = ?", (pipeline_name,))
    except Exception:
        logger.exception("Failed to delete pipeline %s from DB", pipeline_name)
        raise
