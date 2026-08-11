import sys
from pathlib import Path

P_ROOT = Path(__file__).resolve().parents[2]
P_RAW = P_ROOT / "data/raw"
P_MIMIC = P_RAW / "physionet.org/files/mimiciv/3.1"
P_DEMO = P_RAW / "physionet.org/files/mimic-iv-demo/2.2"


P_CHECKPOINTS = P_ROOT / "checkpoints"
P_HOSP = P_MIMIC / "hosp"
P_ICU = P_MIMIC / "icu"

# READMISSION_WINDOW_DAYS = 30
EXTENDED_STAY_THRESHOLD_DAYS = 7
FIRST_24_HOURS = 24
RANDOM_STATE = 42
TEST_SIZE = 0.20

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220210: "resp_rate",
    220277: "spo2",
    220179: "sbp_noninvasive",
    220180: "dbp_noninvasive",
    223761: "temp_f",
    223900: "gcs_verbal",
}

LAB_ITEMIDS = {
    50912: "creatinine",
    51006: "bun",
    51301: "wbc",
    51222: "hemoglobin",
    51265: "platelet",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    50931: "glucose",
}


PREPROCESS_DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "last_careunit",
    "hosp_los_days",
    "discharge_location",
    "hospital_expire_flag",
    "language",
]

HARD_LEAKAGE_COLUMNS = [
    "los",
]

SOFT_LEAKAGE_COLUMNS = [
    "num_procedures",
    "num_transfers",
    "num_distinct_meds",
    "num_micro_tests",
    "num_diagnoses",
    "drg_severity",
    "drg_mortality",
]


def hello():
    print("Hello from MedAgent!")


def debug():
    print(sys.executable)
    assert P_DEMO.exists()
    assert P_MIMIC.exists()





CATEGORICAL_COLUMNS = [
    "first_careunit",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
    "admitting_service",
]

COUNT_FEATURE_COLUMNS = [
    "num_diagnoses",
    "num_procedures",
    "num_transfers",
    "num_distinct_meds",
    "num_micro_tests",
]


# ==========================================================
# Readmission-model configuration
# ==========================================================

READMISSION_WINDOW_DAYS = 30
READMISSION_RECALL_TARGET = 0.70

READMISSION_FEATURE_COLUMNS = [
    # Demographics and admission-time information
    "gender",
    "anchor_age",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "first_careunit",

    # Prior history only
    "prior_admission_count",
    "days_since_previous_discharge",
    "previous_hospital_los_days",
    "mean_prior_hospital_los_days",
    "prior_icu_admission_count",
    "prior_icu_stay_count",

    # First ICU 24-hour microbiology
    "micro_event_count_24h",
    "organism_detected_event_count_24h",
]

READMISSION_MODEL_SETTINGS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}