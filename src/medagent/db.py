"""Read-only DuckDB access: connection, schema introspection, guarded queries."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from medagent.config import ROW_CAP
from medagent.sqlguard import validate_sql

MAX_CELL_CHARS = 500


def get_conn(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def list_tables(conn: duckdb.DuckDBPyConnection) -> str:
    rows = conn.execute(
        "select table_name, estimated_size from duckdb_tables() order by table_name"
    ).fetchall()
    if not rows:
        return "The database contains no tables."
    lines = [f"  {name}  (~{size:,} rows)" for name, size in rows]
    return "Tables:\n" + "\n".join(lines)


def describe_table(conn: duckdb.DuckDBPyConnection, table: str) -> str:
    names = {
        r[0] for r in conn.execute("select table_name from duckdb_tables()").fetchall()
    }
    if table not in names:
        return f"Unknown table '{table}'. Available tables: {', '.join(sorted(names))}"
    cols = conn.execute(f'describe "{table}"').fetchall()
    col_lines = [f"  {c[0]}: {c[1]}" for c in cols]
    sample = conn.execute(f'select * from "{table}" limit 3').fetch_df()
    return (
        f"Table {table} — columns:\n"
        + "\n".join(col_lines)
        + "\n\nSample rows (CSV):\n"
        + sample.to_csv(index=False)
    )


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool


def run_query(
    conn: duckdb.DuckDBPyConnection, sql: str, cap: int = ROW_CAP
) -> QueryResult:
    """Validate and execute a SELECT, returning at most `cap` rows.

    fetchmany(cap + 1) instead of LIMIT-wrapping: DuckDB executes lazily, and
    wrapping would interact with the query's own ORDER BY / LIMIT semantics.
    """
    cleaned = validate_sql(sql)
    cur = conn.execute(cleaned)
    rows = cur.fetchmany(cap + 1)
    columns = [d[0] for d in cur.description or []]
    truncated = len(rows) > cap
    df = pd.DataFrame(rows[:cap], columns=columns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(_clip_cell)
    return QueryResult(df=df, truncated=truncated)


def _clip_cell(v: object) -> object:
    if isinstance(v, str) and len(v) > MAX_CELL_CHARS:
        return v[:MAX_CELL_CHARS] + "…"
    return v
