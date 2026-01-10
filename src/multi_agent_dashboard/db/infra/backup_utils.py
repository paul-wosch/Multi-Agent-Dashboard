# multi_agent_dashboard/db/infra/backup_utils.py
"""
Centralized DB backup & restore helpers used by migrations and sqlite_rebuild.

Functions:
- make_backup_path(db_path) -> str
- create_backup_from_conn(conn, db_file=None) -> str
- create_backup_from_file(db_path) -> str
- restore_backup_to_conn(backup_path, conn) -> None
- restore_backup_file_copy(backup_path, db_file) -> None

Behavior:
- Prefer the sqlite3 online backup API (sqlite3.Connection.backup) for consistent snapshots.
- Fall back to file-level copy when necessary.
- All paths returned are filesystem paths to the created backup file.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from typing import Optional


def make_backup_path(db_path: str) -> str:
    base, ext = os.path.splitext(db_path)
    if not ext:
        ext = ".sqlite"
    ts = datetime.now().strftime("%y%m%d-%H%M")
    return f"{base}_backup_{ts}{ext}"


def create_backup_from_conn(conn: sqlite3.Connection, db_file: Optional[str] = None) -> str:
    """
    Create a backup of an *open* sqlite3.Connection.

    Args:
        conn: open sqlite3.Connection (source)
        db_file: optional canonical DB file path (used to build backup filename). If not
                 provided or doesn't exist, a temporary file will be created.

    Returns:
        Path to the backup file.

    Raises:
        Exception on failure to create a reliable backup.
    """
    if db_file and os.path.exists(db_file):
        backup_path = make_backup_path(db_file)
    else:
        tmpf = tempfile.NamedTemporaryFile(prefix="db_backup_", suffix=".sqlite", delete=False)
        tmpf.close()
        backup_path = tmpf.name

    try:
        dest_conn = sqlite3.connect(backup_path)
        try:
            # Copy from conn -> dest_conn
            conn.backup(dest_conn)
            dest_conn.commit()
        finally:
            try:
                dest_conn.close()
            except Exception:
                pass
        return backup_path
    except Exception as e:
        # Cleanup any partial backup file (best-effort)
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except Exception:
            pass
        raise RuntimeError(f"Failed to create DB backup from connection: {e}")


def create_backup_from_file(db_path: str) -> str:
    """
    Create a backup from an on-disk SQLite file path using the online backup API
    (preferred) and fallback to file copy if necessary.

    Returns:
        Path to backup file.

    Raises:
        RuntimeError on failure.
    """
    if not os.path.exists(db_path):
        raise RuntimeError(f"DB file not found for backup: {db_path}")

    backup_path = make_backup_path(db_path)

    # Preferred: open source DB and create a new destination DB and use .backup
    try:
        src_conn = sqlite3.connect(db_path)
        dest_conn = sqlite3.connect(backup_path)
        try:
            src_conn.backup(dest_conn)
            dest_conn.commit()
            return backup_path
        finally:
            try:
                dest_conn.close()
            except Exception:
                pass
            try:
                src_conn.close()
            except Exception:
                pass
    except Exception:
        # Fallback: copy the file
        try:
            shutil.copy2(db_path, backup_path)
            return backup_path
        except Exception as e:
            # Cleanup partial backup if created
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            except Exception:
                pass
            raise RuntimeError(f"Failed to create DB backup via backup API and file copy: {e}")


def restore_backup_to_conn(backup_path: str, conn: sqlite3.Connection) -> None:
    """
    Restore the backup file into an open sqlite3.Connection using the online backup API.

    This will copy data from the backup file into the provided connection.

    Raises:
        Exception if the restore fails.
    """
    if not os.path.exists(backup_path):
        raise RuntimeError(f"Backup file not found: {backup_path}")

    try:
        src_conn = sqlite3.connect(backup_path)
        try:
            src_conn.backup(conn)
            conn.commit()
        finally:
            try:
                src_conn.close()
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to restore backup into connection: {e}")


def restore_backup_file_copy(backup_path: str, db_file: str) -> None:
    """
    Restore the backup file to the given db_file path using a file copy.

    Note: caller is responsible for ensuring no other process is using the DB file.
    """
    if not os.path.exists(backup_path):
        raise RuntimeError(f"Backup file not found: {backup_path}")
    if not db_file:
        raise RuntimeError("Destination DB file path not provided for file-level restore.")
    try:
        shutil.copy2(backup_path, db_file)
    except Exception as e:
        raise RuntimeError(f"Failed to restore DB file from backup via copy: {e}")
