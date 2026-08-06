from pathlib import Path
from functools import cache

import pandas as pd

from medagent.config import P_MIMIC, P_DEMO


@cache
def load_data(table="hosp/admissions", demo=False) -> pd.DataFrame:
    """Load MIMIC-IV table

    Sources:
    - https://physionet.org/content/mimiciv/3.1/
    - https://physionet.org/content/mimic-iv-demo/2.2/

    Returns:
        pd.DataFrame: table from MIMIC-IV
    """
    P_src = P_MIMIC if not demo else P_DEMO
    matches = list(P_src.rglob(f"{table}.csv.gz"))

    if len(matches) != 1:
     raise FileNotFoundError(
        f"Expected exactly one file for '{table}', found {len(matches)} under {P_src}"
    )

    return pd.read_csv(matches[0])
