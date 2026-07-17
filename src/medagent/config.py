import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

P_ROOT = Path(__file__).resolve().parents[2]
P_RAW = P_ROOT / "data/raw"
P_MIMIC = P_RAW / "physionet.org/files/mimiciv/3.1"
P_DEMO = P_RAW / "physionet.org/files/mimic-iv-demo/2.2"

P_DUCKDB = P_ROOT / "data/duckdb"
DB_DEMO = P_DUCKDB / "mimic_demo.duckdb"
DB_FULL = P_DUCKDB / "mimic.duckdb"

MODEL = os.getenv("MEDAGENT_MODEL", "gemini-2.5-flash")
PROVIDER = os.getenv("MEDAGENT_PROVIDER", "gemini")
LOCAL_MODEL = os.getenv("MEDAGENT_LOCAL_MODEL", "gpt-oss:20b")
LOCAL_NUM_CTX = int(os.getenv("MEDAGENT_LOCAL_NUM_CTX", "16384"))
ROW_CAP = 200
MAX_TOOL_ITERATIONS = 15


def hello():
    print("Hello from MedAgent!")


def debug():
    print(sys.executable)
    assert P_DEMO.exists()
    assert P_MIMIC.exists()
