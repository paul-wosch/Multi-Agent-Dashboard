# multi_agent_dashboard/db/infra/sql_utils.py
"""
SQL identifier helpers: safe quoting and basic validation.

This small utility centralizes:
 - validate_identifier(name)    -> raises on suspicious/invalid names
 - quote_ident(name)            -> returns a double-quoted, SQL-escaped identifier
 - quote_ident_list(iterable)   -> comma-joined quoted identifiers
 - quote_column_list(spec)      -> accepts either "a,b" or ["a","b"] and returns quoted list

Notes:
- Uses SQL double-quote identifier quoting and escapes internal double-quotes by doubling them,
  per the SQLite tokenizer / identifier rules (SQL standard). See the SQLite tokenizer docs.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union


def _ensure_str(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("Identifier must be a string")
    return name


def validate_identifier(name: str) -> None:
    """
    Basic validation for identifier strings. This is intentionally conservative:
    - Must be a non-empty string.
    - Must not contain NUL/newline/carriage-return characters (obvious injection risks).
    - Other characters are allowed (SQLite supports many forms of identifiers when quoted),
      but control characters are rejected.
    Raises ValueError/TypeError on invalid input.
    """
    name = _ensure_str(name)
    if not name:
        raise ValueError("Identifier must not be empty")
    if "\x00" in name or "\n" in name or "\r" in name:
        raise ValueError("Identifier contains disallowed control characters")


def quote_ident(name: str) -> str:
    """
    Quote an SQL identifier using double quotes and escape embedded double-quotes.

    Examples:
      quote_ident('table') -> '"table"'
      quote_ident('we"ir d') -> '"we""ir d"'

    This function DOES NOT attempt to normalize or validate schema-qualified names
    (e.g., 'main.table'); callers may perform higher-level parsing if needed.
    """
    validate_identifier(name)
    return '"' + name.replace('"', '""') + '"'


def quote_ident_list(names: Iterable[str]) -> str:
    """
    Quote and join an iterable of identifier names into a comma-separated list.
    """
    return ", ".join(quote_ident(n) for n in names)


def quote_column_list(spec: Union[str, Sequence[str]]) -> str:
    """
    Given either:
      - a comma-separated string "a,b,c"
      - a sequence/list of column names ["a", "b"]
    Return a comma-separated list of quoted identifiers: '"a","b","c"'
    """
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = list(spec)
    if not parts:
        raise ValueError("Empty column list provided")
    return ", ".join(quote_ident(p) for p in parts)
