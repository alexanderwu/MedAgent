import pandas as pd
from medagent.config import (
    FIRST_24_HOURS,
    LAB_ITEMIDS,
    P_HOSP,
    P_ICU,
    VITAL_ITEMIDS,
)

from medagent.data import load_data

"""

steps in order:

feature_engineering.py
raw MIMIC tables → clinical feature table (df_experiment)

preprocessing.py
df_experiment → cleaned, consistently formatted model-ready features

modeling.py
clean features → train/test split → XGBoost model → metrics



"""

# First one is for the model for LOS prediction for less than or more than 7 days

def build_base_cohort(icustays: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ICU stay with the core ICU-stay columns."""
    icustays = icustays.copy()

    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    base_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ]

    return icustays[base_columns].copy()



def add_demographics(
    base_cohort: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Add patient demographic features to the ICU-stay cohort."""
    patient_columns = [
        "subject_id",
        "gender",
        "anchor_age",
    ]

    return base_cohort.merge(
        patients[patient_columns],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )


def add_admission_features(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Add admission details and derived admission-time features."""
    admissions = admissions.copy()

    for column in ["admittime", "dischtime", "edregtime", "edouttime"]:
        admissions[column] = pd.to_datetime(admissions[column])

    admissions["hosp_los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86400

    admissions["came_through_ed"] = admissions["edregtime"].notna().astype(int)

    admissions["ed_los_hours"] = (
        admissions["edouttime"] - admissions["edregtime"]
    ).dt.total_seconds() / 3600

    admission_columns = [
        "hadm_id",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "marital_status",
        "language",
        "hosp_los_days",
        "came_through_ed",
        "ed_los_hours",
        "hospital_expire_flag",
    ]

    return cohort.merge(
        admissions[admission_columns],
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )


def add_diagnosis_count(
    cohort: pd.DataFrame,
    diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """Add the number of recorded diagnoses for each hospital admission."""
    diagnosis_counts = (
        diagnoses.groupby("hadm_id")
        .size()
        .rename("num_diagnoses")
        .reset_index()
    )

    return cohort.merge(
        diagnosis_counts,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )


def add_admission_row_count(
    cohort: pd.DataFrame,
    table: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """Add a per-admission row count from a MIMIC table."""
    counts = (
        table.groupby("hadm_id")
        .size()
        .rename(feature_name)
        .reset_index()
    )

    return cohort.merge(
        counts,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )

""""

Example usage:

df = add_admission_row_count(
    df,
    load_data("hosp/procedures_icd"),
    "num_procedures",
)

df = add_admission_row_count(
    df,
    load_data("hosp/transfers"),
    "num_transfers",
)

df = add_admission_row_count(
    df,
    load_data("hosp/microbiologyevents"),
    "num_micro_tests",
)

"""


def add_distinct_medication_count(
    cohort: pd.DataFrame,
    prescriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Add the number of distinct medications recorded per admission."""
    medication_counts = (
        prescriptions.groupby("hadm_id")["drug"]
        .nunique()
        .rename("num_distinct_meds")
        .reset_index()
    )

    return cohort.merge(
        medication_counts,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )


def add_first_service(
    cohort: pd.DataFrame,
    services: pd.DataFrame,
) -> pd.DataFrame:
    """Add the first recorded hospital service for each admission."""
    services = services.copy()
    services["transfertime"] = pd.to_datetime(services["transfertime"])

    first_services = (
        services.sort_values(["hadm_id", "transfertime"])
        .groupby("hadm_id")["curr_service"]
        .first()
        .rename("admitting_service")
        .reset_index()
    )

    return cohort.merge(
        first_services,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )


def add_drg_scores(
    cohort: pd.DataFrame,
    drgcodes: pd.DataFrame,
) -> pd.DataFrame:
    """Add APR-DRG severity and mortality scores for each admission."""
    apr_drg = drgcodes.loc[
        drgcodes["drg_type"] == "APR",
        ["hadm_id", "drg_severity", "drg_mortality"],
    ]

    drg_scores = (
        apr_drg.groupby("hadm_id")[["drg_severity", "drg_mortality"]]
        .mean()
        .reset_index()
    )

    return cohort.merge(
        drg_scores,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )


def build_first_24h_vitals(
    icustays: pd.DataFrame,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Aggregate selected charted vitals from the first ICU 24 hours."""
    intime_lookup = (
        icustays.assign(intime=pd.to_datetime(icustays["intime"]))
        .set_index("stay_id")["intime"]
    )

    vital_chunks = []

    for chunk in pd.read_csv(
        P_ICU / "chartevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(VITAL_ITEMIDS)]

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["intime"] = chunk["stay_id"].map(intime_lookup)

        in_first_24h = (
            (chunk["charttime"] >= chunk["intime"])
            & (chunk["charttime"] <= chunk["intime"] + pd.Timedelta(hours=FIRST_24_HOURS))
        )

        vital_chunks.append(
            chunk.loc[in_first_24h, ["stay_id", "itemid", "valuenum"]]
        )

    vitals_long = pd.concat(vital_chunks, ignore_index=True)

    return (
        vitals_long.groupby(["stay_id", "itemid"])["valuenum"]
        .mean()
        .unstack("itemid")
        .rename(columns=VITAL_ITEMIDS)
        .add_prefix("mean_")
        .reset_index()
    )


def add_first_24h_vitals(
    cohort: pd.DataFrame,
    vitals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge first-24-hour mean vital features onto the ICU-stay cohort."""
    return cohort.merge(
        vitals,
        on="stay_id",
        how="left",
        validate="one_to_one",
    )


def build_first_24h_labs(
    admissions: pd.DataFrame,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Aggregate selected labs from the first 24 hospital-admission hours."""
    admittime_lookup = (
        admissions.assign(admittime=pd.to_datetime(admissions["admittime"]))
        .set_index("hadm_id")["admittime"]
    )

    lab_chunks = []

    for chunk in pd.read_csv(
        P_HOSP / "labevents.csv.gz",
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=chunksize,
    ):
        chunk = chunk[
            chunk["itemid"].isin(LAB_ITEMIDS)
            & chunk["hadm_id"].notna()
        ]

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["admittime"] = chunk["hadm_id"].map(admittime_lookup)

        in_first_24h = (
            (chunk["charttime"] >= chunk["admittime"])
            & (chunk["charttime"] <= chunk["admittime"] + pd.Timedelta(hours=FIRST_24_HOURS))
        )

        lab_chunks.append(
            chunk.loc[in_first_24h, ["hadm_id", "itemid", "valuenum"]]
        )

    labs_long = pd.concat(lab_chunks, ignore_index=True)

    return (
        labs_long.groupby(["hadm_id", "itemid"])["valuenum"]
        .mean()
        .unstack("itemid")
        .rename(columns=LAB_ITEMIDS)
        .add_prefix("mean_")
        .reset_index()
    )


def add_first_24h_labs(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge first-24-hour lab features onto the ICU-stay cohort."""
    return cohort.merge(
        labs,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )

def build_experiment_table() -> pd.DataFrame:
    """Build the full ICU-stay feature table from raw MIMIC-IV tables."""
    icustays = load_data("icu/icustays")
    patients = load_data("hosp/patients")
    admissions = load_data("hosp/admissions")

    df = build_base_cohort(icustays)
    df = add_demographics(df, patients)
    df = add_admission_features(df, admissions)

    df = add_diagnosis_count(
        df,
        load_data("hosp/diagnoses_icd"),
    )

    df = add_admission_row_count(
        df,
        load_data("hosp/procedures_icd"),
        "num_procedures",
    )

    df = add_admission_row_count(
        df,
        load_data("hosp/transfers"),
        "num_transfers",
    )

    df = add_distinct_medication_count(
        df,
        load_data("hosp/prescriptions"),
    )

    df = add_first_service(
        df,
        load_data("hosp/services"),
    )

    df = add_drg_scores(
        df,
        load_data("hosp/drgcodes"),
    )

    df = add_admission_row_count(
        df,
        load_data("hosp/microbiologyevents"),
        "num_micro_tests",
    )

    vitals = build_first_24h_vitals(icustays)
    df = add_first_24h_vitals(df, vitals)

    labs = build_first_24h_labs(admissions)
    df = add_first_24h_labs(df, labs)

    return df