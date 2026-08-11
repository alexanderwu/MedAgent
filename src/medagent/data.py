from pathlib import Path
from functools import cache

import pandas as pd

from medagent.config import P_MIMIC, P_DEMO


# @cache
# def load_data(table="hosp/admissions", demo=False) -> pd.DataFrame:
#     """Load MIMIC-IV table

#     Sources:
#     - https://physionet.org/content/mimiciv/3.1/
#     - https://physionet.org/content/mimic-iv-demo/2.2/

#     Returns:
#         pd.DataFrame: table from MIMIC-IV
#     """
#     P_src = P_MIMIC if not demo else P_DEMO
#     matches = list(P_src.rglob(f"{table}.csv.gz"))

#     if len(matches) != 1:
#      raise FileNotFoundError(
#         f"Expected exactly one file for '{table}', found {len(matches)} under {P_src}"
#     )

#     return pd.read_csv(matches[0])



from functools import cache
from pathlib import Path
from typing import Iterator

import pandas as pd

from medagent.config import P_DEMO, P_MIMIC


def find_table_path(
    table: str,
    demo: bool = False,
) -> Path:
    """Return the unique CSV.gz path for a MIMIC table."""

    source_root = P_DEMO if demo else P_MIMIC

    matches = list(source_root.rglob(f"{table}.csv.gz"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for '{table}', "
            f"found {len(matches)} under {source_root}"
        )

    return matches[0]


@cache
def load_data(
    table: str = "hosp/admissions",
    demo: bool = False,
) -> pd.DataFrame:
    """
    Load a complete MIMIC table.

    Use only for reasonably sized tables. Do not use this for
    chartevents, labevents, microbiologyevents, or prescriptions.
    """

    return pd.read_csv(find_table_path(table, demo=demo))


def read_table_chunks(
    table: str,
    *,
    usecols: list[str],
    parse_dates: list[str] | None = None,
    chunksize: int = 1_000_000,
    demo: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Stream a large MIMIC table in pandas DataFrame chunks.

    Example:
        for chunk in read_table_chunks(
            "hosp/microbiologyevents",
            usecols=["hadm_id", "charttime"],
        ):
            ...
    """

    return pd.read_csv(
        find_table_path(table, demo=demo),
        usecols=usecols,
        parse_dates=parse_dates,
        chunksize=chunksize,
    )
