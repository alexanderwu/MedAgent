"""CLI harness for the agent loop: python -m medagent.cli "your question".

Uses the demo database by default (--full for the full one). If no database
has been ingested yet, runs with no tools — useful as a bare-loop smoke test.
"""

import argparse
import sys

from google import genai

from medagent import db
from medagent.agent import AgentSession
from medagent.config import DB_DEMO, DB_FULL, MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask MedAgent a question.")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument(
        "--full", action="store_true", help="Use the full MIMIC-IV database"
    )
    args = parser.parse_args()

    db_path = DB_FULL if args.full else DB_DEMO
    dataset_label = "full 3.1" if args.full else "demo 2.2 (100 patients)"
    if db_path.exists():
        conn = db.get_conn(db_path)
    else:
        conn = None
        print(
            f"[no database at {db_path} — running without tools; run `just ingest-demo` first]"
        )

    print(f"[model: {MODEL} · db: {db_path.name if conn else 'none'}]")
    session = AgentSession(
        question=args.question,
        client=genai.Client(),
        conn=conn,
        dataset_label=dataset_label,
    )
    session.run_until_blocked()

    while session.status == "awaiting_approval":
        for item in list(session.pending):
            print("\n--- MedAgent proposes this query ---")
            if item.purpose:
                print(f"Purpose: {item.purpose}")
            print(item.sql)
            answer = input("Run this query? Results go to the Gemini API. [y/N] ")
            session.resolve_sql(item.tool_use_id, answer.strip().lower() == "y")

    print()
    for event in session.events:
        print(f"[tool] {event.name}: {event.summary}")
    print()
    if session.status == "done":
        print(session.final_text)
        return 0
    print(f"ERROR: {session.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
