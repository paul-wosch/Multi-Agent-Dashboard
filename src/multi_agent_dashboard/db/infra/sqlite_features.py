# multi_agent_dashboard/db/infra/sqlite_features.py
"""
Small SQLite capability matrix & helpers used to decide whether an ALTER statement
is safe to emit for a target runtime, or whether a rebuild should be preferred.

The rules are conservative and based on official SQLite documentation (ALTER TABLE
limitations, version history). The goal is to centralize runtime feature checks.

Key features:
 - rename_column: supported since 3.25.0
 - drop_column: supported since 3.35.0
 - added_constraint_testing_on_add_column: testing of added constraints is present since 3.37.0

Also exposes is_add_column_safe() to decide if an ADD COLUMN with the provided
column-def is safe to emit as ALTER in the given runtime.
"""
from __future__ import annotations

from typing import Tuple, Dict


def version_tuple_from_string(s: str) -> Tuple[int, int, int]:
    parts = (s or "").split(".")
    parts = (parts + ["0", "0"])[:3]
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def supports_rename_column(ver: Tuple[int, int, int]) -> bool:
    return ver >= (3, 25, 0)


def supports_drop_column(ver: Tuple[int, int, int]) -> bool:
    return ver >= (3, 35, 0)


def supports_added_constraint_testing(ver: Tuple[int, int, int]) -> bool:
    # Testing of added CHECK/NOT NULL on generated columns became an enhancement in 3.37.0
    return ver >= (3, 37, 0)


def get_capabilities(ver: Tuple[int, int, int]) -> Dict[str, bool]:
    return {
        "rename_column": supports_rename_column(ver),
        "drop_column": supports_drop_column(ver),
        "added_constraint_testing": supports_added_constraint_testing(ver),
    }


def _norm_col_def(col_def: str) -> str:
    if not isinstance(col_def, str):
        return ""
    return " ".join(col_def.strip().lower().split())


def is_add_column_safe(col_def: str, ver: Tuple[int, int, int]) -> bool:
    """
    Decide whether emitting 'ALTER TABLE ADD COLUMN <col> <col_def>' is safe
    (i.e., allowed by SQLite and unlikely to trigger preexisting-row constraint checks
    or unsupported restrictions).

    Conservative rules (based on SQLite docs):
    - Cannot add PRIMARY KEY or UNIQUE with ALTER -> mark unsafe.
    - Cannot add DEFAULT CURRENT_* or expression defaults -> mark unsafe.
    - GENERATED ... STORED is not allowed; VIRTUAL allowed -> treat GENERATED as unsafe.
    - If 'NOT NULL' is present, a DEFAULT (non-NULL) must be present (else unsafe).
    - If REFERENCES (foreign key) appears, SQLite requires the added column to have DEFAULT NULL
      when foreign keys are enabled; treat addition of REFERENCES without explicit DEFAULT as unsafe.
    - CHECK constraints and certain generated-column NOT NULL cases may trigger testing of existing rows
      starting in 3.37.0; we conservatively mark CHECK and GENERATED cases as unsafe unless the runtime supports
      the enhanced testing (we still prefer rebuilds for safety).
    """
    s = _norm_col_def(col_def)

    # Disallowed / problematic tokens
    if "primary key" in s or "unique" in s:
        return False
    if "generated" in s:
        # GENERATED ALWAYS ... STORED not allowed in ADD COLUMN
        return False
    if "current_timestamp" in s or "current_time" in s or "current_date" in s:
        return False
    if "references" in s:
        # Must ensure default NULL; conservative: require rebuild unless DEFAULT NULL explicitly present
        if "default null" not in s:
            return False
    if "not null" in s and "default" not in s:
        # SQLite requires a default other than NULL for NOT NULL on ADD COLUMN
        return False
    if "check(" in s:
        # CHECK constraints on added columns may trigger row checks (3.37.0 change).
        # Conservative: prefer rebuild.
        return False

    # If none of the flagged patterns present, treat as safe
    return True
