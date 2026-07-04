import sys
from pathlib import Path

P_ROOT = Path(__name__).resolve().parents[1]
P_RAW = P_ROOT / "data/raw"
P_MIMIC = P_RAW / "physionet.org/files/mimiciv/3.1"
P_DEMO = P_RAW / "physionet.org/files/mimic-iv-demo/2.2"


def hello():
    print("Hello from MedAgent!")


def debug():
    print(sys.executable)
    assert P_DEMO.exists()
    assert P_MIMIC.exists()
