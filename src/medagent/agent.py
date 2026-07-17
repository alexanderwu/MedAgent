"""Hand-rolled Claude tool-calling loop, designed to survive Streamlit reruns.

The loop is a resumable state object, not a blocking function: when Claude
requests run_sql, run_until_blocked() *returns* with status "awaiting_approval"
and the pending SQL parked in session state. resolve_sql() feeds the verdict
back in and re-enters the loop once every gated call in the turn is resolved.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import anthropic
import duckdb
import pandas as pd

from medagent import db, tools
from medagent.config import EFFORT, MAX_TOOL_ITERATIONS, MODEL
from medagent.sqlguard import SQLValidationError

SYSTEM_PROMPT = """\
You are MedAgent, a clinical-data analyst querying the MIMIC-IV dataset \
({dataset}) stored in a local DuckDB database.

Workflow:
1. Use get_schema (list tables, then describe specific ones) before writing SQL \
against tables you have not inspected in this conversation.
2. Write a single DuckDB SELECT (or WITH...SELECT) via run_sql. Every query \
requires the user's explicit approval before execution; they may decline — if \
so, adjust your approach rather than resubmitting the same SQL.
3. Results are capped at 200 rows. Prefer aggregates (COUNT, AVG, median(col), \
percentile_cont, GROUP BY) over raw row dumps.
4. When a result has a natural visual form (distributions, trends over time, \
group comparisons), offer or produce a chart with the plot tool using the exact \
column names from the last result.

DuckDB dialect notes: median(col) works directly; date arithmetic via \
date_diff/age; timestamps are TIMESTAMP columns; string matching is \
case-sensitive (use ilike for case-insensitive).

MIMIC-IV notes: hosp- and icu-module tables join on subject_id / hadm_id / \
stay_id. Hospital length of stay = dischtime - admittime on admissions; ICU \
length of stay is the los column (days) on icustays. The data is deidentified \
and dates are shifted into the future — never present them as real calendar \
dates.

Answer concisely: lead with the number or finding, then state the SQL logic in \
one line.
"""

Status = Literal["running", "awaiting_approval", "done", "error"]


@dataclass
class PendingSQL:
    tool_use_id: str
    sql: str
    purpose: str


@dataclass
class AgentSession:
    """One user question = one session = one fresh API transcript."""

    question: str
    client: anthropic.Anthropic
    conn: duckdb.DuckDBPyConnection | None
    dataset_label: str

    status: Status = "running"
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending: list[PendingSQL] = field(default_factory=list)
    resolved_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_content: list[Any] = field(default_factory=list)
    events: list[tools.ToolEvent] = field(default_factory=list)
    final_text: str | None = None
    last_df: pd.DataFrame | None = None
    error: str | None = None
    iterations: int = 0

    def __post_init__(self) -> None:
        self.messages = [{"role": "user", "content": self.question}]
        self.system = SYSTEM_PROMPT.format(dataset=self.dataset_label)

    # -- driving the loop ---------------------------------------------------

    def run_until_blocked(self) -> None:
        while self.status == "running":
            self._step()

    def resolve_sql(self, tool_use_id: str, approved: bool) -> None:
        """Feed the user's verdict on one pending run_sql call back in."""
        assert self.status == "awaiting_approval"
        item = next(p for p in self.pending if p.tool_use_id == tool_use_id)
        self.pending.remove(item)

        if not approved:
            self.resolved_results[tool_use_id] = _tool_result(
                tool_use_id,
                "The user declined to run this query. Ask what they'd like "
                "instead or propose a different query.",
                is_error=True,
            )
            self.events.append(
                tools.ToolEvent("run_sql", {"sql": item.sql}, "declined by user")
            )
        else:
            self._execute_sql(item)

        if not self.pending:
            self._flush_tool_results()
            self.status = "running"
            self.run_until_blocked()

    # -- internals ----------------------------------------------------------

    def _step(self) -> None:
        if self.iterations >= MAX_TOOL_ITERATIONS:
            self.status = "error"
            self.error = f"Stopped after {MAX_TOOL_ITERATIONS} tool iterations."
            return
        self.iterations += 1

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=self.system,
                tools=cast(Any, tools.TOOLS if self.conn is not None else []),
                output_config=cast(Any, {"effort": EFFORT}),
                messages=cast(Any, self.messages),
            )
        except anthropic.RateLimitError as e:
            self._fail(f"Rate limited by the Anthropic API: {e.message}")
            return
        except anthropic.APIStatusError as e:
            self._fail(f"Anthropic API error ({e.status_code}): {e.message}")
            return
        except anthropic.APIConnectionError:
            self._fail("Could not reach the Anthropic API — check your network.")
            return

        # Append the assistant turn verbatim (thinking blocks included).
        self.messages.append({"role": "assistant", "content": response.content})
        self.current_content = list(response.content)

        if response.stop_reason == "refusal":
            self._fail("Claude declined to answer this request.")
            return
        if response.stop_reason == "pause_turn":
            return  # loop continues; server resumes the paused turn
        if response.stop_reason != "tool_use":
            text = _text_of(response.content)
            if response.stop_reason == "max_tokens":
                text += "\n\n*(response truncated at the token limit)*"
            self.final_text = text
            self.status = "done"
            return

        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_input: dict[str, Any] = dict(block.input)  # already parsed by the SDK
            if block.name == "run_sql":
                self.pending.append(
                    PendingSQL(
                        block.id,
                        tool_input.get("sql", ""),
                        tool_input.get("purpose", ""),
                    )
                )
            else:
                content, is_error, event = tools.execute_auto_tool(
                    block.name, tool_input, self.conn, self.last_df
                )
                self.events.append(event)
                self.resolved_results[block.id] = _tool_result(
                    block.id, content, is_error
                )

        if self.pending:
            self.status = "awaiting_approval"
            return
        self._flush_tool_results()

    def _execute_sql(self, item: PendingSQL) -> None:
        assert self.conn is not None
        try:
            result = db.run_query(self.conn, item.sql)
        except (SQLValidationError, duckdb.Error) as e:
            self.resolved_results[item.tool_use_id] = _tool_result(
                item.tool_use_id, f"Query failed: {e}", is_error=True
            )
            self.events.append(
                tools.ToolEvent("run_sql", {"sql": item.sql}, f"error: {e}")
            )
            return

        self.last_df = result.df
        note = (
            f"\n(first {len(result.df)} rows shown — result exceeded the cap; "
            "refine with aggregation)"
            if result.truncated
            else f"\n({len(result.df)} rows)"
        )
        self.resolved_results[item.tool_use_id] = _tool_result(
            item.tool_use_id, result.df.to_csv(index=False) + note
        )
        self.events.append(
            tools.ToolEvent(
                "run_sql",
                {"sql": item.sql, "purpose": item.purpose},
                f"{len(result.df)} rows" + (" (truncated)" if result.truncated else ""),
                df=result.df,
            )
        )

    def _flush_tool_results(self) -> None:
        """Send all tool_results for the turn as ONE user message, in block order."""
        ordered = [
            self.resolved_results.pop(block.id)
            for block in self.current_content
            if block.type == "tool_use"
        ]
        if ordered:
            self.messages.append({"role": "user", "content": ordered})
        self.current_content = []

    def _fail(self, message: str) -> None:
        self.status = "error"
        self.error = message


def _tool_result(
    tool_use_id: str, content: str, is_error: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        result["is_error"] = True
    return result


def _text_of(content: list[Any]) -> str:
    return "".join(b.text for b in content if b.type == "text")
