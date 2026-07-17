"""MedAgent Streamlit chat UI.

Run with: uv run streamlit run src/medagent/app.py

All mutable agent state lives in st.session_state; the AgentSession never
blocks on user input — it returns in "awaiting_approval" and the Approve /
Reject buttons re-enter it. Chat history is UI-only (no conversation memory).
"""

from pathlib import Path

import streamlit as st
from google import genai

from medagent import db, tools
from medagent.agent import AgentSession
from medagent.config import DB_DEMO, DB_FULL, MODEL

st.set_page_config(page_title="MedAgent", layout="wide")

DATASETS = {
    "demo": (DB_DEMO, "demo 2.2 (100 patients)"),
    "full": (DB_FULL, "full 3.1"),
}


@st.cache_resource
def _conn(db_path: str):
    return db.get_conn(Path(db_path))


@st.cache_resource
def _client() -> genai.Client:
    return genai.Client()


def _init_state() -> None:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("dataset", "demo")


def _render_events(events: list[tools.ToolEvent]) -> None:
    for event in events:
        with st.expander(f"🔧 {event.name} — {event.summary}"):
            if "sql" in event.input:
                st.code(event.input["sql"], language="sql")
            elif event.input:
                st.json(event.input)
            if event.detail and event.name == "get_schema":
                st.text(event.detail)
            if event.df is not None and event.chart_spec is None:
                st.dataframe(event.df)
            if event.chart_spec is not None and event.df is not None:
                st.altair_chart(
                    tools.build_chart(event.chart_spec, event.df),
                    use_container_width=True,
                )


def main() -> None:
    _init_state()

    with st.sidebar:
        st.title("MedAgent")
        available = [k for k, (p, _) in DATASETS.items() if p.exists()]
        if not available:
            st.error("No DuckDB database found. Run `just ingest-demo` first.")
            st.stop()
        dataset = st.radio("Dataset", available, key="dataset")
        missing = [k for k in DATASETS if k not in available]
        if missing:
            st.caption(f"({', '.join(missing)} not ingested yet)")
        st.caption(f"Model: {MODEL}")
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.session_state.agent = None
            st.rerun()

    db_path, dataset_label = DATASETS[dataset]

    # A dataset switch mid-question would leave the agent pointing at the old
    # connection — drop it.
    agent: AgentSession | None = st.session_state.agent
    if agent is not None and agent.dataset_label != dataset_label:
        st.session_state.agent = None
        agent = None

    for entry in st.session_state.chat:
        with st.chat_message(entry["role"]):
            _render_events(entry.get("events", []))
            if entry.get("text"):
                st.markdown(entry["text"])

    if agent is not None:
        if agent.status == "awaiting_approval":
            with st.chat_message("assistant"):
                _render_events(agent.events)
                item = agent.pending[0]
                st.markdown("**MedAgent wants to run this query:**")
                if item.purpose:
                    st.caption(item.purpose)
                st.code(item.sql, language="sql")
                st.caption(
                    "Approving runs this query locally and sends the result rows "
                    "to the Google Gemini API."
                )
                col_a, col_b, _ = st.columns([1, 1, 4])
                approve = col_a.button("✅ Approve", key=f"approve_{item.tool_use_id}")
                reject = col_b.button("❌ Reject", key=f"reject_{item.tool_use_id}")
                if approve or reject:
                    with st.spinner("Thinking…"):
                        agent.resolve_sql(item.tool_use_id, approved=approve)
                    st.rerun()
        elif agent.status in ("done", "error"):
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "text": agent.final_text
                    if agent.status == "done"
                    else f"⚠️ {agent.error}",
                    "events": agent.events,
                }
            )
            st.session_state.agent = None
            st.rerun()

    question = st.chat_input(
        "Ask about the MIMIC-IV data…",
        disabled=agent is not None,
    )
    if question:
        st.session_state.chat.append({"role": "user", "text": question})
        session = AgentSession(
            question=question,
            client=_client(),
            conn=_conn(str(db_path)),
            dataset_label=dataset_label,
        )
        st.session_state.agent = session
        with st.spinner("Thinking…"):
            session.run_until_blocked()
        st.rerun()


main()
