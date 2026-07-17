"""Tool definitions for the Gemini agent, plus execution of the auto tools.

`run_sql` is the only human-gated tool; it is executed in agent.resolve_sql
after the user approves. `get_schema` and `plot` execute automatically.
"""

from dataclasses import dataclass
from typing import Any

import altair as alt
import duckdb
import pandas as pd

from medagent import db
from medagent.config import ROW_CAP

FUNCTION_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "get_schema",
        "description": (
            "Inspect the MIMIC-IV DuckDB database. With no arguments, lists all "
            "tables with approximate row counts. With a table name, returns that "
            "table's columns and types plus 3 sample rows. Always inspect the "
            "relevant tables before writing SQL against them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Optional table name to describe, e.g. 'admissions'.",
                }
            },
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Execute a read-only SQL SELECT against the MIMIC-IV DuckDB database. "
            "The user must approve every query before it runs and may decline. "
            f"Results are capped at {ROW_CAP} rows — prefer aggregation (GROUP BY, "
            "count, avg, median) over pulling raw rows. DuckDB dialect."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SELECT or WITH...SELECT statement.",
                },
                "purpose": {
                    "type": "string",
                    "description": (
                        "One sentence: what this query answers. Shown to the user "
                        "in the approval prompt."
                    ),
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "plot",
        "description": (
            "Render an Altair chart from the result of the MOST RECENT successful "
            "run_sql call. Column names must exist in that result. Call run_sql first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mark": {"type": "string", "enum": ["bar", "line", "point", "area"]},
                "x": {
                    "type": "string",
                    "description": "Column for the x encoding (Altair shorthand allowed, e.g. 'admittime:T').",
                },
                "y": {
                    "type": "string",
                    "description": "Column for the y encoding, or 'count()' for a count aggregate.",
                },
                "color": {
                    "type": "string",
                    "description": "Optional column for color encoding.",
                },
                "title": {"type": "string"},
            },
            "required": ["mark", "x", "y"],
        },
    },
]


@dataclass
class ToolEvent:
    """Display record for the UI — one per executed tool call."""

    name: str
    input: dict[str, Any]
    summary: str
    detail: str | None = None
    df: pd.DataFrame | None = None
    chart_spec: dict[str, Any] | None = None


def execute_auto_tool(
    name: str,
    tool_input: dict[str, Any],
    conn: duckdb.DuckDBPyConnection | None,
    last_df: pd.DataFrame | None,
) -> tuple[str, bool, ToolEvent]:
    """Execute a non-gated tool. Returns (result_content, is_error, event)."""
    if name == "get_schema":
        if conn is None:
            return (
                "No database is connected. Run the ingest first.",
                True,
                ToolEvent(name, tool_input, "no database"),
            )
        table = tool_input.get("table")
        content = db.describe_table(conn, table) if table else db.list_tables(conn)
        summary = f"described {table}" if table else "listed tables"
        return content, False, ToolEvent(name, tool_input, summary, detail=content)

    if name == "plot":
        if last_df is None or last_df.empty:
            return (
                "No query result is available to plot. Call run_sql first.",
                True,
                ToolEvent(name, tool_input, "no data to plot"),
            )
        missing = _missing_columns(tool_input, last_df)
        if missing:
            return (
                f"Column(s) {missing} not in the last result. "
                f"Available columns: {list(last_df.columns)}",
                True,
                ToolEvent(name, tool_input, "bad column in chart spec"),
            )
        spec = {
            k: tool_input[k]
            for k in ("mark", "x", "y", "color", "title")
            if k in tool_input
        }
        return (
            "Chart rendered for the user.",
            False,
            ToolEvent(
                name, tool_input, f"{spec['mark']} chart", df=last_df, chart_spec=spec
            ),
        )

    return f"Unknown tool: {name}", True, ToolEvent(name, tool_input, "unknown tool")


def _missing_columns(tool_input: dict[str, Any], df: pd.DataFrame) -> list[str]:
    missing = []
    for key in ("x", "y", "color"):
        value = tool_input.get(key)
        if not value:
            continue
        base = value.split(":")[0]
        if "(" in base:  # aggregate shorthand like count() — let Altair handle it
            continue
        if base not in df.columns:
            missing.append(value)
    return missing


def build_chart(spec: dict[str, Any], df: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(df)
    mark_method = getattr(chart, f"mark_{spec['mark']}")
    encodings: dict[str, Any] = {"x": spec["x"], "y": spec["y"]}
    if spec.get("color"):
        encodings["color"] = spec["color"]
    chart = mark_method().encode(**encodings)
    if spec.get("title"):
        chart = chart.properties(title=spec["title"])
    return chart
