"""sqlite_rebuild.py

If the generator says: “This table requires a rebuild”

Respond by:
rebuild_table_with_constraints(
    conn,
    "agent_outputs",
    SCHEMA["agent_outputs"]["columns"],
    SCHEMA["agent_outputs"]["constraints"],
)
"""

def rebuild_table_with_constraints(conn, table, columns, constraints):
    """
    Rebuilds a table with constraints:
    - Creates temp table
    - Copies data
    - Drops old table
    - Renames temp table
    """

    tmp = f"{table}__rebuild"

    conn.execute("PRAGMA foreign_keys = OFF")

    col_defs = [f"{c} {t}" for c, t in columns.items()]

    for fk in constraints.get("foreign_keys", []):
        clause = (
            f"FOREIGN KEY({fk['column']}) "
            f"REFERENCES {fk['references']}"
        )
        if fk.get("on_delete"):
            clause += f" ON DELETE {fk['on_delete']}"
        col_defs.append(clause)

    ddl = f"""
        CREATE TABLE {tmp} (
            {", ".join(col_defs)}
        );
        """

    conn.executescript(ddl)

    cols = ", ".join(columns.keys())
    conn.execute(
        f"INSERT INTO {tmp} ({cols}) SELECT {cols} FROM {table}"
    )

    conn.executescript(f"""
            DROP TABLE {table};
            ALTER TABLE {tmp} RENAME TO {table};
        """)

    conn.execute("PRAGMA foreign_keys = ON")

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"Foreign key violations after rebuild: {violations}")
