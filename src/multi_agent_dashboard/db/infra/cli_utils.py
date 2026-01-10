# multi_agent_dashboard/db/infra/cli_utils.py
"""
CLI message helpers for consistent, compact messages.

Provides:
 - print_user_message(summary, action=None, details=None, verbose=False, quiet=False)

Pattern:
 - One-line summary always printed (unless --quiet).
 - Optional one-line actionable command (prefixed) shown next.
 - Optional details block printed only when verbose=True.

This is intentionally lightweight and uses print() so it works in both CLI and
test harnesses (captured by pytest capsys).
"""
from __future__ import annotations

import textwrap
import sys
from typing import Optional


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def print_user_message(
    summary: str,
    action: Optional[str] = None,
    details: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """
    Print a consistent CLI message.

    - summary: short human-facing one-line summary.
    - action: short actionable command or next step (printed on its own).
    - details: multi-line developer detail printed only with verbose=True.
    - verbose: if True, print details (if any).
    - quiet: if True, suppress output entirely.
    """
    if quiet:
        return

    # Ensure summary is one line
    first_line = summary.strip().splitlines()[0] if summary else ""
    print(first_line)

    if action:
        # Keep action block compact and visually distinct
        print()
        print("Actionable:")
        for ln in action.strip().splitlines():
            print("  " + ln.rstrip())

    if verbose and details:
        print()
        print("Details:")
        print(_indent(details.strip(), prefix="  "))


def format_action_command(cmd: str) -> str:
    """
    Helper to produce a nicely indented action command block for messages.
    """
    return textwrap.dedent(cmd).strip()
