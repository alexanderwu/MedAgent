"""Hand-rolled tool-calling loop, designed to survive Streamlit reruns.

The loop is a resumable state object, not a blocking function: when the model
requests run_sql, run_until_blocked() *returns* with status "awaiting_approval"
and the pending SQL parked in session state. resolve_sql() feeds the verdict
back in and re-enters the loop once every gated call in the turn is resolved.

Everything provider-specific (Gemini vs. local Ollama) lives behind the
Provider protocol in providers.py; the transcript holds provider-native
messages the loop never inspects. Neither API supplies usable call ids, so
each call gets a synthetic id and tool results are sent back in the original
call order.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb
import pandas as pd

from medagent import db, tools
from medagent.config import MAX_TOOL_ITERATIONS
from medagent.providers import Provider, ProviderError
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
    provider: Provider
    conn: duckdb.DuckDBPyConnection | None
    dataset_label: str

    status: Status = "running"
    contents: list[Any] = field(default_factory=list)
    pending: list[PendingSQL] = field(default_factory=list)
    # synthetic id -> (function name, {"result": ...} | {"error": ...})
    resolved_results: dict[str, tuple[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    current_calls: list[str] = field(default_factory=list)  # synthetic ids, call order
    events: list[tools.ToolEvent] = field(default_factory=list)
    final_text: str | None = None
    last_df: pd.DataFrame | None = None
    error: str | None = None
    iterations: int = 0

    def __post_init__(self) -> None:
        self.contents = [self.provider.user_text(self.question)]
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
            self.resolved_results[tool_use_id] = (
                "run_sql",
                {
                    "error": "The user declined to run this query. Ask what they'd "
                    "like instead or propose a different query."
                },
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
            self._fail(f"Stopped after {MAX_TOOL_ITERATIONS} tool iterations.")
            return
        self.iterations += 1

        try:
            step = self.provider.generate(
                self.system, self.contents, use_tools=self.conn is not None
            )
        except ProviderError as e:
            self._fail(str(e))
            return

        self.contents.append(step.message)

        if not step.function_calls:
            text = step.text
            if step.truncated:
                text += "\n\n*(response truncated at the token limit)*"
            if not text:
                self._fail("The model stopped without a reply.")
                return
            self.final_text = text
            self.status = "done"
            return

        self.current_calls = []
        for index, call in enumerate(step.function_calls):
            call_id = f"call_{self.iterations}_{index}"
            self.current_calls.append(call_id)
            if call.name == "run_sql":
                self.pending.append(
                    PendingSQL(
                        call_id,
                        call.args.get("sql", ""),
                        call.args.get("purpose", ""),
                    )
                )
            else:
                result_text, is_error, event = tools.execute_auto_tool(
                    call.name, call.args, self.conn, self.last_df
                )
                self.events.append(event)
                payload = (
                    {"error": result_text} if is_error else {"result": result_text}
                )
                self.resolved_results[call_id] = (call.name, payload)

        if self.pending:
            self.status = "awaiting_approval"
            return
        self._flush_tool_results()

    def _execute_sql(self, item: PendingSQL) -> None:
        assert self.conn is not None
        try:
            result = db.run_query(self.conn, item.sql)
        except (SQLValidationError, duckdb.Error) as e:
            self.resolved_results[item.tool_use_id] = (
                "run_sql",
                {"error": f"Query failed: {e}"},
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
        self.resolved_results[item.tool_use_id] = (
            "run_sql",
            {"result": result.df.to_csv(index=False) + note},
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
        """Send all tool results for the turn back at once, in call order."""
        ordered = [self.resolved_results.pop(call_id) for call_id in self.current_calls]
        if ordered:
            self.contents.extend(self.provider.tool_results(ordered))
        self.current_calls = []

    def _fail(self, message: str) -> None:
        self.status = "error"
        self.error = message
