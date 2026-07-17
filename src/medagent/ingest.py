"""Build a DuckDB database from the MIMIC-IV csv.gz files.

Usage: python -m medagent.ingest --demo | --full

Note: DuckDB allows one writer OR many readers — close the Streamlit app
before re-ingesting.
"""

import argparse
import time

import duckdb

from medagent.config import DB_DEMO, DB_FULL, P_DEMO, P_DUCKDB, P_MIMIC

# Per-table read_csv_auto overrides, e.g. for columns that mis-sniff as numeric.
# Populate only when a load fails or verification shows wrong types.
OVERRIDES: dict[str, str] = {}


def ingest(dataset: str) -> int:
    root = P_DEMO if dataset == "demo" else P_MIMIC
    db_path = DB_DEMO if dataset == "demo" else DB_FULL
    P_DUCKDB.mkdir(parents=True, exist_ok=True)

    files = sorted((root / "hosp").glob("*.csv.gz")) + sorted(
        (root / "icu").glob("*.csv.gz")
    )
    if not files:
        print(f"No csv.gz files found under {root}")
        return 1

    conn = duckdb.connect(str(db_path))  # the only writable connection in the project
    conn.execute("SET preserve_insertion_order = false")  # stream large CSVs

    failures: list[tuple[str, str]] = []
    for path in files:
        table = path.name.removesuffix(".csv.gz")
        src = path.as_posix()
        options = OVERRIDES.get(table, "")
        t0 = time.perf_counter()
        try:
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS '
                f"SELECT * FROM read_csv_auto('{src}'{options})"
            )
            count_row = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            n = count_row[0] if count_row else 0
            print(f"  {table:<24} {n:>12,} rows   {time.perf_counter() - t0:6.1f}s")
        except duckdb.Error as e:
            failures.append((table, str(e)))
            print(f"  {table:<24} FAILED: {e}")

    conn.close()
    print(f"\nDatabase written to {db_path}")
    if failures:
        print(f"{len(failures)} table(s) failed:")
        for table, err in failures:
            print(f"  {table}: {err}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MIMIC-IV csv.gz into DuckDB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Demo subset (~19 MB)")
    group.add_argument(
        "--full", action="store_true", help="Full MIMIC-IV 3.1 (long run)"
    )
    args = parser.parse_args()
    return ingest("demo" if args.demo else "full")


if __name__ == "__main__":
    raise SystemExit(main())
