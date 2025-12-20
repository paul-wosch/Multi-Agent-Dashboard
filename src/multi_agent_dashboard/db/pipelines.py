"""db/pipelines.py

Construction modes:

- PipelineDAO(db_path="...")
- PipelineDAO(conn=sqlite3.Connection)

Support for atomic multi-step operations:

with pipeline_dao(db_path) as dao:
    dao.save("analysis", ["planner", "executor"])
    dao.save("final", ["planner", "executor", "finalizer"])
"""
import json
import logging
import warnings
from datetime import datetime, UTC
from typing import List, Optional
from contextlib import contextmanager

from multi_agent_dashboard.db.infra.core import get_conn, safe_json_loads

logger = logging.getLogger(__name__)


class PipelineDAO:
    def __init__(self, db_path: Optional[str] = None, conn=None):
        if conn is None and db_path is None:
            raise ValueError("PipelineDAO requires either db_path or conn")

        self._db_path = db_path
        self._conn = conn

    # -----------------------
    # internal helpers
    # -----------------------

    @contextmanager
    def _connection(self):
        """
        Yield a connection.
        If DAO was constructed with a connection, reuse it.
        Otherwise, open a new one.
        """
        if self._conn is not None:
            yield self._conn
        else:
            with get_conn(self._db_path) as conn:
                yield conn

    # -----------------------
    # READ operations
    # -----------------------

    def list(self) -> list[dict]:
        logger.debug("Loading pipelines from DB")
        try:
            with self._connection() as conn:
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
            result.append(
                {
                    "pipeline_name": row["pipeline_name"],
                    "steps": safe_json_loads(row["steps_json"], []),
                    "metadata": safe_json_loads(row["metadata_json"], {}),
                    "timestamp": row["timestamp"],
                }
            )
        return result

    # -----------------------
    # WRITE operations
    # -----------------------

    def save(
        self,
        pipeline_name: str,
        steps: List[str],
        metadata: Optional[dict] = None,
    ) -> None:
        logger.info("Saving pipeline %s to DB", pipeline_name)
        try:
            with self._connection() as conn:
                ts = datetime.now(UTC).isoformat()
                steps_json = json.dumps(steps)
                metadata_json = json.dumps(metadata or {})

                conn.execute(
                    """
                    INSERT OR REPLACE INTO pipelines
                        (pipeline_name, steps_json, metadata_json, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (pipeline_name, steps_json, metadata_json, ts),
                )
        except Exception:
            logger.exception("Failed to save pipeline %s to DB", pipeline_name)
            raise

    def delete(self, pipeline_name: str) -> None:
        logger.info("Deleting pipeline %s from DB", pipeline_name)
        try:
            with self._connection() as conn:
                conn.execute(
                    "DELETE FROM pipelines WHERE pipeline_name = ?",
                    (pipeline_name,),
                )
        except Exception:
            logger.exception("Failed to delete pipeline %s from DB", pipeline_name)
            raise


# -----------------------
# Transaction-scoped DAO
# -----------------------

@contextmanager
def pipeline_dao(db_path: str):
    """
    Yield a PipelineDAO bound to a single transaction.
    """
    with get_conn(db_path) as conn:
        yield PipelineDAO(conn=conn)


# -----------------------
# Compatibility wrappers
# -----------------------

def load_pipelines_from_db(db_path: str) -> list[dict]:
    warnings.warn(
        "load_pipelines_from_db is deprecated; use PipelineDAO.list",
        DeprecationWarning,
    )
    return PipelineDAO(db_path=db_path).list()


def save_pipeline_to_db(
    db_path: str,
    pipeline_name: str,
    steps: List[str],
    metadata: Optional[dict] = None,
    ):
    warnings.warn(
        "save_pipeline_to_db is deprecated; use PipelineDAO.save",
        DeprecationWarning,
    )
    return PipelineDAO(db_path=db_path).save(pipeline_name, steps, metadata)


def delete_pipeline_from_db(db_path: str, pipeline_name: str):
    warnings.warn(
        "delete_pipeline_from_db is deprecated; use PipelineDAO.delete",
        DeprecationWarning,
    )
    return PipelineDAO(db_path=db_path).delete(pipeline_name)
