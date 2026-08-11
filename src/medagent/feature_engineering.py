import pandas as pd
from medagent.config import (
    FIRST_24_HOURS,
    LAB_ITEMIDS,
    P_HOSP,
    P_ICU,
    VITAL_ITEMIDS,
    READMISSION_WINDOW_DAYS
)

from medagent.data import load_data

"""

steps in order:

feature_engineering.py
raw MIMIC tables
→ df_experiment

preprocessing.py
df_experiment
→ clean, consistently formatted features

modeling.py
clean features
→ train/test split
→ XGBoost model
→ metrics

persist.py
model + feature list + preprocessing metadata + metrics
→ one saved model bundle (.pkl)


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

    # df = add_distinct_medication_count(
    #     df,
    #     load_data("hosp/prescriptions"),
    # )

    df = add_first_service(
        df,
        load_data("hosp/services"),
    )

    df = add_drg_scores(
        df,
        load_data("hosp/drgcodes"),
    )

    # df = add_admission_row_count(
    #     df,
    #     load_data("hosp/microbiologyevents"),
    #     "num_micro_tests",
    # )

    vitals = build_first_24h_vitals(icustays)
    df = add_first_24h_vitals(df, vitals)

    labs = build_first_24h_labs(admissions)
    df = add_first_24h_labs(df, labs)

    return df



# ==========================================================
# Readmission-focus feature engineering
# ==========================================================

from medagent.config import READMISSION_WINDOW_DAYS


def build_readmission_admissions(
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Create the eligible 30-day readmission target per admission."""

    admissions = admissions.copy()

    for column in ["admittime", "dischtime", "deathtime"]:
        admissions[column] = pd.to_datetime(admissions[column])

    admissions = admissions.sort_values(
        ["subject_id", "admittime"]
    ).reset_index(drop=True)

    # Next hospital admission for the same patient.
    admissions["next_admittime"] = (
        admissions.groupby("subject_id")["admittime"].shift(-1)
    )

    admissions["days_to_next_admission"] = (
        admissions["next_admittime"] - admissions["dischtime"]
    ).dt.total_seconds() / 86400

    # A patient who dies in the hospital or within 30 days after discharge
    # cannot experience a 30-day readmission.
    admissions["days_to_death"] = (
        admissions["deathtime"] - admissions["dischtime"]
    ).dt.total_seconds() / 86400

    died_before_window_ends = (
        admissions["hospital_expire_flag"].eq(1)
        | admissions["days_to_death"].between(
            0,
            READMISSION_WINDOW_DAYS,
            inclusive="both",
        )
    )

    admissions["readmit_30d"] = pd.NA

    eligible = ~died_before_window_ends

    admissions.loc[eligible, "readmit_30d"] = (
        admissions.loc[eligible, "days_to_next_admission"]
        .between(0, READMISSION_WINDOW_DAYS, inclusive="both")
        .astype(int)
    )

    admissions["readmit_30d"] = admissions["readmit_30d"].astype(
        "Int64"
    )

    return admissions


def build_readmission_base_cohort(
    icustays: pd.DataFrame,
    readmission_admissions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep the first ICU stay for each hospital admission and attach
    the eligible 30-day readmission target.
    """

    icustays = icustays.copy()
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    # drop_duplicates preserves the actual first ICU-stay row.
    first_icu_stays = (
        icustays.sort_values(["hadm_id", "intime"])
        .drop_duplicates(subset="hadm_id", keep="first")
        .copy()
    )

    target_columns = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "readmit_30d",
    ]

    cohort = first_icu_stays.merge(
        readmission_admissions[target_columns],
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )

    # Remove admissions ineligible for the readmission outcome.
    return cohort.dropna(subset=["readmit_30d"]).copy()


def add_readmission_demographics(
    cohort: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Add demographic features known by the ICU admission."""

    patient_columns = [
        "subject_id",
        "gender",
        "anchor_age",
    ]

    return cohort.merge(
        patients[patient_columns],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )


def add_readmission_admission_features(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Add admission-time features safe for the readmission model."""

    admission_columns = [
        "subject_id",
        "hadm_id",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]

    return cohort.merge(
        admissions[admission_columns],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )


def add_prior_admission_history(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Add history features using only admissions before the current one."""

    admissions = admissions.copy()
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    admissions = admissions.sort_values(
        ["subject_id", "admittime"]
    ).reset_index(drop=True)

    admissions["current_hospital_los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86400

    # Number of ICU stays in each admission.
    icu_counts = (
        icustays.groupby(["subject_id", "hadm_id"])
        .size()
        .rename("icu_stay_count_this_admission")
        .reset_index()
    )

    admissions = admissions.merge(
        icu_counts,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    admissions["icu_stay_count_this_admission"] = (
        admissions["icu_stay_count_this_admission"]
        .fillna(0)
        .astype(int)
    )

    by_patient = admissions.groupby("subject_id")

    admissions["prior_admission_count"] = by_patient.cumcount()

    admissions["previous_dischtime"] = (
        by_patient["dischtime"].shift(1)
    )

    admissions["days_since_previous_discharge"] = (
        admissions["admittime"] - admissions["previous_dischtime"]
    ).dt.total_seconds() / 86400

    admissions["previous_hospital_los_days"] = (
        by_patient["current_hospital_los_days"].shift(1)
    )

    admissions["mean_prior_hospital_los_days"] = (
        by_patient["current_hospital_los_days"]
        .transform(lambda values: values.shift(1).expanding().mean())
    )

    admissions["prior_icu_admission_count"] = (
        by_patient["icu_stay_count_this_admission"]
        .transform(
            lambda values: values.gt(0).shift(
                1,
                fill_value=False,
            ).cumsum()
        )
    )

    admissions["prior_icu_stay_count"] = (
        by_patient["icu_stay_count_this_admission"]
        .transform(lambda values: values.shift(1, fill_value=0).cumsum())
    )

    history_columns = [
        "subject_id",
        "hadm_id",
        "prior_admission_count",
        "days_since_previous_discharge",
        "previous_hospital_los_days",
        "mean_prior_hospital_los_days",
        "prior_icu_admission_count",
        "prior_icu_stay_count",
    ]

    return cohort.merge(
        admissions[history_columns],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )


def add_prior_diagnosis_features(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """Add diagnosis history from admissions before the current admission."""

    admissions = admissions[
        ["subject_id", "hadm_id", "admittime"]
    ].copy()

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])

    diagnoses = diagnoses[
        ["subject_id", "hadm_id", "icd_code"]
    ].copy()

    diagnoses["icd_code"] = (
        diagnoses["icd_code"]
        .astype("string")
        .str.upper()
        .str.replace(".", "", regex=False)
        .fillna("")
    )

    diagnoses["diabetes_flag"] = (
        diagnoses["icd_code"].str.startswith(
            ("250", "E08", "E09", "E10", "E11", "E13")
        )
    ).astype(int)

    diagnoses["heart_failure_flag"] = (
        diagnoses["icd_code"].str.startswith(("428", "I50"))
    ).astype(int)

    diagnoses["ckd_flag"] = (
        diagnoses["icd_code"].str.startswith(("585", "N18"))
    ).astype(int)

    diagnoses["copd_flag"] = (
        diagnoses["icd_code"].str.startswith(
            (
                "490", "491", "492", "493", "494", "495", "496",
                "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
            )
        )
    ).astype(int)

    diagnoses["liver_disease_flag"] = (
        diagnoses["icd_code"].str.startswith(
            (
                "570", "571", "572", "573",
                "K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77",
            )
        )
    ).astype(int)

    icd9_prefix = pd.to_numeric(
        diagnoses["icd_code"].str[:3],
        errors="coerce",
    )

    diagnoses["cancer_flag"] = (
        diagnoses["icd_code"].str.startswith("C")
        | icd9_prefix.between(140, 209).fillna(False)
    ).astype(int)

    admission_diagnoses = (
        diagnoses.groupby(["subject_id", "hadm_id"])
        .agg(
            diagnosis_code_count=("icd_code", "size"),
            diabetes_flag=("diabetes_flag", "max"),
            heart_failure_flag=("heart_failure_flag", "max"),
            ckd_flag=("ckd_flag", "max"),
            copd_flag=("copd_flag", "max"),
            liver_disease_flag=("liver_disease_flag", "max"),
            cancer_flag=("cancer_flag", "max"),
        )
        .reset_index()
    )

    history = admissions.merge(
        admission_diagnoses,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    ).fillna(0)

    history = history.sort_values(
        ["subject_id", "admittime"]
    ).reset_index(drop=True)

    history["prior_diagnosis_code_count"] = (
        history.groupby("subject_id")["diagnosis_code_count"]
        .transform(lambda values: values.shift(1).cumsum())
        .fillna(0)
    )

    for condition in [
        "diabetes",
        "heart_failure",
        "ckd",
        "copd",
        "liver_disease",
        "cancer",
    ]:
        history[f"prior_{condition}_flag"] = (
            history.groupby("subject_id")[f"{condition}_flag"]
            .transform(lambda values: values.shift(1).cummax())
            .fillna(0)
            .astype(int)
        )

    feature_columns = [
        "prior_diagnosis_code_count",
        "prior_diabetes_flag",
        "prior_heart_failure_flag",
        "prior_ckd_flag",
        "prior_copd_flag",
        "prior_liver_disease_flag",
        "prior_cancer_flag",
    ]

    return cohort.merge(
        history[["subject_id", "hadm_id", *feature_columns]],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )


def build_first_24h_microbiology(
    cohort: pd.DataFrame,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Build first-24-hour microbiology features for each hospital admission.

    Returns a small table keyed by hadm_id. It does not merge it yet.
    """

    admission_windows = cohort[
        ["hadm_id", "intime"]
    ].drop_duplicates("hadm_id").copy()

    admission_windows["intime"] = pd.to_datetime(
        admission_windows["intime"]
    )

    admission_windows["first_24h_end"] = (
        admission_windows["intime"]
        + pd.Timedelta(hours=FIRST_24_HOURS)
    )

    chunk_summaries = []

    for chunk in pd.read_csv(
        P_HOSP / "microbiologyevents.csv.gz",
        usecols=[
            "hadm_id",
            "chartdate",
            "charttime",
            "org_name",
        ],
        parse_dates=["chartdate", "charttime"],
        chunksize=chunksize,
    ):
        # Rows without a hospital admission cannot be linked to the cohort.
        chunk = chunk[chunk["hadm_id"].notna()].copy()

        if chunk.empty:
            continue

        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

        # charttime is more precise; use chartdate when it is missing.
        chunk["event_time"] = chunk["charttime"].fillna(
            chunk["chartdate"]
        )

        # Keep only admissions that exist in our readmission cohort.
        chunk = chunk.merge(
            admission_windows,
            on="hadm_id",
            how="inner",
            validate="many_to_one",
        )

        # Keep only microbiology events from the first ICU 24 hours.
        chunk = chunk[
            (chunk["event_time"] >= chunk["intime"])
            & (chunk["event_time"] < chunk["first_24h_end"])
        ]

        if chunk.empty:
            continue

        chunk["organism_detected"] = (
            chunk["org_name"].notna()
        ).astype(int)

        chunk_summary = (
            chunk.groupby("hadm_id")
            .agg(
                micro_event_count_24h=("hadm_id", "size"),
                organism_detected_event_count_24h=(
                    "organism_detected",
                    "sum",
                ),
            )
            .reset_index()
        )

        chunk_summaries.append(chunk_summary)

    if not chunk_summaries:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "micro_event_count_24h",
                "organism_detected_event_count_24h",
            ]
        )

    return (
        pd.concat(chunk_summaries, ignore_index=True)
        .groupby("hadm_id", as_index=False)
        .sum()
    )


def build_readmission_experiment_table() -> pd.DataFrame:
    """
    Build the full, leakage-safe ICU readmission feature table.

    One row represents one hospital admission's first ICU stay.
    """

    print("Loading core MIMIC tables...")

    admissions = load_data("hosp/admissions")
    icustays = load_data("icu/icustays")
    patients = load_data("hosp/patients")
    diagnoses = load_data("hosp/diagnoses_icd")

    print("Creating the eligible 30-day readmission target...")
    readmission_admissions = build_readmission_admissions(
        admissions
    )

    print("Building the first-ICU-stay cohort...")
    cohort = build_readmission_base_cohort(
        icustays,
        readmission_admissions,
    )

    print("Adding demographics and admission-time features...")
    cohort = add_readmission_demographics(cohort, patients)

    cohort = add_readmission_admission_features(
        cohort,
        admissions,
    )

    print("Adding prior admission and ICU history...")
    cohort = add_prior_admission_history(
        cohort,
        admissions,
        icustays,
    )

    print("Adding prior diagnosis-history features...")
    cohort = add_prior_diagnosis_features(
        cohort,
        admissions,
        diagnoses,
    )

    print("Scanning microbiology events in first ICU 24 hours...")
    microbiology_features = build_first_24h_microbiology(
        cohort
    )

    cohort = cohort.merge(
        microbiology_features,
        on="hadm_id",
        how="left",
        validate="one_to_one",
    )

    microbiology_columns = [
        "micro_event_count_24h",
        "organism_detected_event_count_24h",
    ]

    # No microbiology event in the first 24 hours means zero events.
    cohort[microbiology_columns] = (
        cohort[microbiology_columns].fillna(0)
    )

    cohort["readmit_30d"] = cohort["readmit_30d"].astype(int)

    return cohort