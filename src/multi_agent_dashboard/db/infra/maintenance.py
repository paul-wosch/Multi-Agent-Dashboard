# db/infra/maintenance.py
"""
Database maintenance helpers.
"""
import logging
from typing import Optional

from multi_agent_dashboard.config import DB_FILE_PATH
from multi_agent_dashboard.db.infra.core import get_conn

logger = logging.getLogger(__name__)


def prune_agent_snapshots(agent_name: Optional[str] = None, keep: int = 100) -> int:
    """
    Prune agent_snapshots table keeping at most `keep` most-recent snapshots
    per agent (ordered by version DESC). If `agent_name` is None, prune all
    agents (per-agent basis). Returns the number of rows deleted.

    Implementation notes:
    - This function opens a single DB transaction and performs per-agent deletes.
    - To avoid relying on database-specific LIMIT parameter binding quirks,
      we fetch ordered ids and slice in Python to determine ids to KEEP,
      then delete rows not in that set.
    """
    if keep is None or keep < 0:
        raise ValueError("keep must be a non-negative integer")

    total_deleted = 0

    try:
        with get_conn(DB_FILE_PATH) as conn:
            before = conn.total_changes

            if agent_name:
                # Get ordered snapshot ids for this agent, most recent first
                rows = conn.execute(
                    "SELECT id FROM agent_snapshots WHERE agent_name = ? ORDER BY version DESC",
                    (agent_name,),
                ).fetchall()
                ids_keep = [r["id"] for r in rows[:keep]]

                if ids_keep:
                    placeholders = ",".join("?" for _ in ids_keep)
                    params = [agent_name] + ids_keep
                    conn.execute(
                        f"DELETE FROM agent_snapshots WHERE agent_name = ? AND id NOT IN ({placeholders})",
                        tuple(params),
                    )
                else:
                    # No kept ids -> delete all snapshots for this agent
                    conn.execute(
                        "DELETE FROM agent_snapshots WHERE agent_name = ?",
                        (agent_name,),
                    )
            else:
                # Prune for all agents: iterate distinct agent_name values
                agent_rows = conn.execute(
                    "SELECT DISTINCT agent_name FROM agent_snapshots"
                ).fetchall()
                for ar in agent_rows:
                    name = ar["agent_name"]
                    rows = conn.execute(
                        "SELECT id FROM agent_snapshots WHERE agent_name = ? ORDER BY version DESC",
                        (name,),
                    ).fetchall()
                    ids_keep = [r["id"] for r in rows[:keep]]
                    if ids_keep:
                        placeholders = ",".join("?" for _ in ids_keep)
                        params = [name] + ids_keep
                        conn.execute(
                            f"DELETE FROM agent_snapshots WHERE agent_name = ? AND id NOT IN ({placeholders})",
                            tuple(params),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM agent_snapshots WHERE agent_name = ?",
                            (name,),
                        )

            after = conn.total_changes
            total_deleted = after - before

            logger.info(
                "Pruned %d agent snapshots (keep=%d) for agent=%s",
                total_deleted,
                keep,
                agent_name or "<all>",
            )
    except Exception:
        logger.exception("Failed to prune agent snapshots")
        raise

    return total_deleted
