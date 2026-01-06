#!/usr/bin/env python3
# db/infra/prune_snapshots.py
"""
prune_snapshots.py

CLI helper to prune the `agent_snapshots` table.

Behavior:
- By default runs on the DB path configured in multi_agent_dashboard.config.DB_FILE_PATH.
- Accepts an optional agent_name to prune snapshots for a single agent (omit to prune all agents).
- Accepts an optional db_path positional to operate on a different SQLite file.
- --keep N controls how many recent snapshots to keep per-agent (default from config).
- --dry-run prints the exact snapshot rows (id, version, created_at) that WOULD be deleted,
  without modifying the database.

Examples:
  # Dry run for all agents (default DB)
  python prune_snapshots.py --dry-run

  # Dry run for a single agent on a custom DB
  python prune_snapshots.py my_agent my_custom.db --keep 50 --dry-run

  # Actually delete older snapshots for all agents, keeping 100
  python prune_snapshots.py --keep 100
"""
import os
import sys
import argparse
from typing import Optional, Dict, List, Any

from multi_agent_dashboard.config import DB_FILE_PATH, AGENT_SNAPSHOT_PRUNE_KEEP
from multi_agent_dashboard.db.infra.core import get_conn
import multi_agent_dashboard.db.infra.maintenance as maintenance


def _preview_deletions(conn, agent_name: Optional[str], keep: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Determine which snapshot rows WOULD be deleted if pruning were executed.

    Returns a mapping: { agent_name: [ {id, version, created_at}, ... ], ... }
    Only agents with >= keep+1 rows will appear in the result.
    """
    # Ensure the table exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_snapshots'"
    ).fetchone()
    if not row:
        raise RuntimeError("Table 'agent_snapshots' not found in database")

    result: Dict[str, List[Dict[str, Any]]] = {}

    if agent_name:
        rows = conn.execute(
            "SELECT id, version, created_at FROM agent_snapshots WHERE agent_name = ? ORDER BY version DESC",
            (agent_name,),
        ).fetchall()
        to_delete = [
            {"id": r["id"], "version": r["version"], "created_at": r["created_at"]}
            for r in rows[keep:]
        ]
        if to_delete:
            result[agent_name] = to_delete
        return result

    # All agents
    agent_rows = conn.execute("SELECT DISTINCT agent_name FROM agent_snapshots").fetchall()
    for ar in agent_rows:
        name = ar["agent_name"]
        rows = conn.execute(
            "SELECT id, version, created_at FROM agent_snapshots WHERE agent_name = ? ORDER BY version DESC",
            (name,),
        ).fetchall()
        to_delete = [
            {"id": r["id"], "version": r["version"], "created_at": r["created_at"]}
            for r in rows[keep:]
        ]
        if to_delete:
            result[name] = to_delete

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prune agent snapshots, keeping the latest N per agent."
    )

    parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="Agent name to prune snapshots for (omit to prune all agents)",
    )

    parser.add_argument(
        "db_path",
        nargs="?",
        default=DB_FILE_PATH,
        help=f"Path to SQLite database file (default: {DB_FILE_PATH})",
    )

    parser.add_argument(
        "--keep",
        type=int,
        default=AGENT_SNAPSHOT_PRUNE_KEEP,
        help=f"Keep latest N snapshots per agent (default: {AGENT_SNAPSHOT_PRUNE_KEEP})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which snapshots would be deleted without modifying the database",
    )

    args = parser.parse_args()

    db_path = args.db_path

    if not os.path.exists(db_path):
        raise SystemExit(f"Database file not found: {db_path}")

    # Ensure the maintenance helper runs against the requested DB file.
    # The maintenance module imports DB_FILE_PATH at module scope, so override
    # its variable before calling the function.
    maintenance.DB_FILE_PATH = db_path

    if args.dry_run:
        try:
            with get_conn(db_path) as conn:
                preview = _preview_deletions(conn, args.agent_name, args.keep)
        except Exception as e:
            raise SystemExit(f"Failed to preview deletions: {e}")

        if not preview:
            print(f"No snapshots would be deleted (keep={args.keep}).")
            return

        total = 0
        print("\n--- DRY RUN: Snapshots that would be deleted ---\n")
        for name, rows in preview.items():
            print(f"Agent: {name} — {len(rows)} snapshot(s) would be deleted")
            total += len(rows)
            # Avoid overwhelming the terminal; show up to 100 ids per agent
            for r in rows[:100]:
                created_at = r.get("created_at")
                print(f"  - id={r['id']}  version={r['version']}  created_at={created_at}")
            if len(rows) > 100:
                print(f"    ... and {len(rows) - 100} more ids omitted")
            print()
        print(f"Total snapshots that would be deleted: {total}")
        print("\nRun without --dry-run to actually delete the above snapshots.\n")
        return

    # Actual deletion: call the existing maintenance helper
    try:
        deleted = maintenance.prune_agent_snapshots(agent_name=args.agent_name, keep=args.keep)
    except Exception as e:
        raise SystemExit(f"Pruning failed: {e}")

    print(f"Pruned {deleted} snapshot(s) (keep={args.keep}) for agent={args.agent_name or '<all>'}")


if __name__ == "__main__":
    main()
