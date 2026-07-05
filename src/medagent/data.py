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
    P_data_list = [p for p in Path(P_src).rglob(f"{table}.csv.gz")]
    assert len(P_data_list) == 1, "Unique table not found"

    P_data = P_data_list[0]
    data = pd.read_csv(P_data)
    return data
