"""Interactive MedAgent demo for ICU extended stay and readmission.

Run from the MedAgent project root:

    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# 1. Make the src/medagent package importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# These functions come from your own MedAgent .py files.
from medagent.persist import (
    load_model_bundle,
    load_readmission_bundle,
)
from medagent.preprocessing import transform_features


# ============================================================
# 2. Page configuration
# ============================================================

st.set_page_config(
    page_title="MedAgent",
    page_icon="🏥",
    layout="wide",
)


# ============================================================
# 3. Custom white-and-black styling
# ============================================================

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #ffffff;
        color: #111111;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }

    [data-testid="stMain"] {
        background-color: #ffffff;
    }

    /* Top toolbar/header */
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }

    /* All normal text */
    p, label, span, div {
        color: #111111;
    }

    /* Headings */
    h1, h2, h3 {
        color: #111111 !important;
    }

    /* Form container */
    [data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #d8d8d8;
        border-radius: 12px;
        padding: 22px;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #f5f5f5 !important;
        color: #111111 !important;
        border-color: #cfcfcf !important;
    }

    div[data-baseweb="select"] span {
        color: #111111 !important;
    }

    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }

    li[role="option"] {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    li[role="option"]:hover {
        background-color: #eeeeee !important;
    }

    /* Main buttons */
    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #111111 !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
        border-color: #333333 !important;
    }

    .stButton > button p,
    .stFormSubmitButton > button p,
    .stDownloadButton > button p {
        color: #ffffff !important;
    }

    /* Slider track */
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: #d3d3d3;
    }

    /* Slider filled section */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #111111 !important;
        border-color: #111111 !important;
    }

    /* Slider value */
    [data-testid="stSlider"] div {
        color: #111111 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #555555 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    [data-baseweb="tab-highlight"] {
        background-color: #111111 !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #111111 !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background-color: #f6f6f6;
        border: 1px solid #d8d8d8;
        border-radius: 10px;
        padding: 15px;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        color: #555555 !important;
    }

    /* Divider */
    hr {
        border-color: #dddddd;
    }

    /* Custom result card */
    .result-card {
        background-color: #ffffff;
        border: 2px solid #111111;
        border-radius: 10px;
        padding: 18px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .result-card-positive {
        border-left: 10px solid #111111;
    }

    .result-card-negative {
        border-left: 10px solid #777777;
    }

    .result-title {
        color: #111111;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .result-text {
        color: #333333;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. Load saved model bundles
# ============================================================

@st.cache_resource
def load_models() -> tuple[dict, dict]:
    """Load both models once and keep them in memory."""

    extended_stay_bundle = load_model_bundle()
    readmission_bundle = load_readmission_bundle()

    return extended_stay_bundle, readmission_bundle


# ============================================================
# 5. Result display
# ============================================================

def show_probability_result(
    probability: float,
    threshold: float,
    positive_text: str,
    negative_text: str,
) -> None:
    """Display the model probability and threshold decision."""

    predicted_positive = probability >= threshold

    if predicted_positive:
        label = positive_text
        card_class = "result-card-positive"
    else:
        label = negative_text
        card_class = "result-card-negative"

    st.divider()

    metric_column, threshold_column = st.columns(2)

    with metric_column:
        st.metric(
            label="Predicted probability",
            value=f"{probability:.1%}",
        )

    with threshold_column:
        st.metric(
            label="Classification threshold",
            value=f"{threshold:.0%}",
        )

    st.progress(float(probability))

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-title">Model result</div>
            <div class="result-text">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"A probability at or above {threshold:.2f} "
        "is classified as a positive result."
    )


# ============================================================
# 6. Readmission form
# ============================================================

def readmission_form(bundle: dict) -> None:
    """Collect readmission features and make one prediction."""

    feature_columns = bundle["selected_feature_columns"]

    st.subheader("30-day readmission risk")

    st.write(
        "Select the patient's admission information and prior history. "
        "Categorical information is on the left and numerical information "
        "is on the right."
    )

    with st.form("readmission_form"):
        categorical_column, numerical_column = st.columns(2, gap="large")

        # ----------------------------------------------------
        # Categorical inputs
        # ----------------------------------------------------

        with categorical_column:
            st.markdown("### Categorical information")

            gender = st.selectbox(
                "Gender",
                ["M", "F"],
            )

            admission_type = st.selectbox(
                "Admission type",
                [
                    "EMERGENCY",
                    "URGENT",
                    "ELECTIVE",
                    "DIRECT EMER.",
                    "EU OBSERVATION",
                    "EW EMER.",
                    "OBSERVATION ADMIT",
                    "SURGICAL SAME DAY ADMISSION",
                    "AMBULATORY OBSERVATION",
                    "DIRECT OBSERVATION",
                ],
            )

            admission_location = st.selectbox(
                "Admission location",
                [
                    "EMERGENCY ROOM",
                    "PHYSICIAN REFERRAL",
                    "TRANSFER FROM HOSPITAL",
                    "TRANSFER FROM SKILLED NURSING FACILITY",
                    "TRANSFER FROM OTHER HEALTHCARE FACILITY",
                    "CLINIC REFERRAL",
                    "PROCEDURE SITE",
                    "INTERNAL TRANSFER TO OR FROM PSYCH",
                    "WALK-IN OR SELF REFERRAL",
                    "INFORMATION NOT AVAILABLE",
                ],
            )

            insurance = st.selectbox(
                "Insurance",
                [
                    "Medicare",
                    "Medicaid",
                    "Private",
                    "Other",
                    "No charge",
                ],
            )

            language = st.selectbox(
                "Language",
                [
                    "English",
                    "Spanish",
                    "Portuguese",
                    "Chinese",
                    "Russian",
                    "Other",
                    "Unknown",
                ],
            )

            marital_status = st.selectbox(
                "Marital status",
                [
                    "MARRIED",
                    "SINGLE",
                    "DIVORCED",
                    "WIDOWED",
                    "SEPARATED",
                    "UNKNOWN",
                ],
            )

            race = st.selectbox(
                "Race",
                [
                    "WHITE",
                    "BLACK/AFRICAN AMERICAN",
                    "ASIAN",
                    "HISPANIC/LATINO",
                    "AMERICAN INDIAN/ALASKA NATIVE",
                    "OTHER",
                    "UNKNOWN",
                    "UNABLE TO OBTAIN",
                    "PATIENT DECLINED TO ANSWER",
                ],
            )

            first_careunit = st.selectbox(
                "First ICU care unit",
                [
                    "Medical Intensive Care Unit (MICU)",
                    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
                    "Surgical Intensive Care Unit (SICU)",
                    "Cardiac Vascular Intensive Care Unit (CVICU)",
                    "Coronary Care Unit (CCU)",
                    "Trauma SICU (TSICU)",
                    "Neuro Intermediate",
                    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
                    "Neuro Stepdown",
                ],
            )

        # ----------------------------------------------------
        # Numerical inputs
        # ----------------------------------------------------

        with numerical_column:
            st.markdown("### Numerical information")

            anchor_age = st.slider(
                "Age",
                min_value=18,
                max_value=120,
                value=65,
                step=1,
            )

            prior_admission_count = st.slider(
                "Prior hospital admissions",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            days_since_previous_discharge = st.slider(
                "Days since previous discharge",
                min_value=0.0,
                max_value=3650.0,
                value=0.0,
                step=1.0,
                help="Use 0 if there was no previous admission.",
            )

            previous_hospital_los_days = st.slider(
                "Previous hospital length of stay in days",
                min_value=0.0,
                max_value=200.0,
                value=0.0,
                step=0.5,
            )

            mean_prior_hospital_los_days = st.slider(
                "Mean prior hospital length of stay in days",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
            )

            prior_icu_admission_count = st.slider(
                "Prior admissions involving an ICU",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            prior_icu_stay_count = st.slider(
                "Prior ICU stays",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            micro_event_count_24h = st.slider(
                "Microbiology events in the first 24 hours",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )

            organism_detected_event_count_24h = st.slider(
                "Organism-detected events in the first 24 hours",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict readmission risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        patient_values = {
            "gender": gender,
            "anchor_age": anchor_age,
            "admission_type": admission_type,
            "admission_location": admission_location,
            "insurance": insurance,
            "language": language,
            "marital_status": marital_status,
            "race": race,
            "first_careunit": first_careunit,
            "prior_admission_count": prior_admission_count,
            "days_since_previous_discharge": days_since_previous_discharge,
            "previous_hospital_los_days": previous_hospital_los_days,
            "mean_prior_hospital_los_days": mean_prior_hospital_los_days,
            "prior_icu_admission_count": prior_icu_admission_count,
            "prior_icu_stay_count": prior_icu_stay_count,
            "micro_event_count_24h": micro_event_count_24h,
            "organism_detected_event_count_24h": (
                organism_detected_event_count_24h
            ),
        }

        patient = pd.DataFrame(
            [
                {
                    feature: patient_values.get(feature, 0)
                    for feature in feature_columns
                }
            ]
        )

        probability = float(
            bundle["pipeline"].predict_proba(patient)[0, 1]
        )

        show_probability_result(
            probability=probability,
            threshold=float(bundle["chosen_threshold"]),
            positive_text=(
                "Higher readmission risk — consider additional clinical review."
            ),
            negative_text="Lower predicted readmission risk.",
        )


# ============================================================
# 7. Extended-stay form helpers
# ============================================================

def get_los_categorical_options(
    bundle: dict,
    column: str,
) -> list[str]:
    """
    Infer categorical options from the encoded feature names saved in
    the extended-stay bundle.
    """

    prefix = f"{column}_"

    encoded_options = [
        feature[len(prefix):]
        for feature in bundle["best_cols"]
        if feature.startswith(prefix)
    ]

    encoded_options = sorted(set(encoded_options))

    # Because training used drop_first=True, the reference category is not
    # represented by its own encoded column. Selecting this option produces
    # zeros for every encoded category in this group.
    return ["Reference / other"] + encoded_options


def get_los_numeric_columns(bundle: dict) -> list[str]:
    """
    Identify original numeric LOS features that were retained by the model.
    """

    metadata = bundle["preprocessing_metadata"]
    categorical_columns = metadata["categorical_columns"]
    numeric_medians = metadata["numeric_medians"]

    numeric_columns: list[str] = []

    for column in numeric_medians:
        is_encoded_category = any(
            column.startswith(f"{category}_")
            for category in categorical_columns
        )

        if is_encoded_category:
            continue

        if column not in bundle["best_cols"]:
            continue

        numeric_columns.append(column)

    return sorted(numeric_columns)


def slider_settings(
    column: str,
    median: Any,
) -> tuple[float, float, float, float]:
    """
    Select reasonable demonstration slider boundaries based on the
    feature name and its training-set median.

    Returns:
        minimum, maximum, default, step
    """

    try:
        default = float(median)
    except (TypeError, ValueError):
        default = 0.0

    if math.isnan(default) or math.isinf(default):
        default = 0.0

    lower_name = column.lower()

    # Age
    if "age" in lower_name:
        return 18.0, 120.0, min(max(default, 18.0), 120.0), 1.0

    # Binary values
    if "binary" in lower_name or lower_name.startswith("is_"):
        return 0.0, 1.0, min(max(default, 0.0), 1.0), 1.0

    # Event and count features
    if (
        "count" in lower_name
        or lower_name.startswith("num_")
        or "number" in lower_name
    ):
        maximum = max(20.0, math.ceil(default * 4.0 + 10.0))
        maximum = min(maximum, 500.0)
        return 0.0, maximum, min(max(default, 0.0), maximum), 1.0

    # Percentages and proportions
    if "percent" in lower_name or "ratio" in lower_name:
        return 0.0, 1.0, min(max(default, 0.0), 1.0), 0.01

    # Glasgow Coma Scale features
    if "gcs" in lower_name:
        return 1.0, 15.0, min(max(default, 1.0), 15.0), 1.0

    # Oxygen saturation
    if "spo2" in lower_name or "oxygen" in lower_name:
        return 0.0, 100.0, min(max(default, 0.0), 100.0), 1.0

    # Temperature
    if "temp" in lower_name:
        return 85.0, 110.0, min(max(default, 85.0), 110.0), 0.1

    # Heart rate
    if "heart_rate" in lower_name:
        return 20.0, 250.0, min(max(default, 20.0), 250.0), 1.0

    # Respiratory rate
    if "resp_rate" in lower_name:
        return 1.0, 80.0, min(max(default, 1.0), 80.0), 1.0

    # Blood pressure
    if "sbp" in lower_name:
        return 40.0, 260.0, min(max(default, 40.0), 260.0), 1.0

    if "dbp" in lower_name:
        return 20.0, 180.0, min(max(default, 20.0), 180.0), 1.0

    # Time measurements
    if "hours" in lower_name:
        maximum = max(48.0, math.ceil(default * 4.0 + 24.0))
        maximum = min(maximum, 1000.0)
        return 0.0, maximum, min(max(default, 0.0), maximum), 0.5

    if "days" in lower_name:
        maximum = max(30.0, math.ceil(default * 4.0 + 10.0))
        maximum = min(maximum, 3650.0)
        return 0.0, maximum, min(max(default, 0.0), maximum), 0.5

    # General numeric fallback based on the training median
    spread = max(abs(default) * 3.0, 10.0)

    minimum = min(0.0, default - spread)
    maximum = max(10.0, default + spread)

    return (
        float(minimum),
        float(maximum),
        float(min(max(default, minimum), maximum)),
        0.1,
    )


def display_feature_name(column: str) -> str:
    """Convert a Python column name into a readable label."""

    return column.replace("_", " ").strip().title()


# ============================================================
# 8. Interactive extended-stay form
# ============================================================

def extended_stay_form(bundle: dict) -> None:
    """Build an interactive LOS form from saved feature metadata."""

    metadata = bundle["preprocessing_metadata"]
    categorical_columns = metadata["categorical_columns"]
    numeric_medians = metadata["numeric_medians"]

    model_categorical_columns = [
        column
        for column in categorical_columns
        if any(
            feature.startswith(f"{column}_")
            for feature in bundle["best_cols"]
        )
    ]

    model_numeric_columns = get_los_numeric_columns(bundle)

    st.subheader("ICU stay longer than 7 days")

    st.write(
        "Enter the patient's early ICU information. Categorical information "
        "is on the left and numerical information is on the right."
    )

    st.caption(
        f"This form creates the {len(bundle['best_cols'])} encoded features "
        "expected by the trained extended-stay model."
    )

    with st.form("extended_stay_form"):
        categorical_column, numerical_column = st.columns(2, gap="large")

        los_values: dict[str, Any] = {}

        # ----------------------------------------------------
        # Categorical LOS inputs
        # ----------------------------------------------------

        with categorical_column:
            st.markdown("### Categorical information")

            if not model_categorical_columns:
                st.info(
                    "This model does not contain retained categorical features."
                )

            for column in model_categorical_columns:
                options = get_los_categorical_options(
                    bundle=bundle,
                    column=column,
                )

                selected_value = st.selectbox(
                    display_feature_name(column),
                    options,
                    key=f"los_category_{column}",
                )

                # Reference/other is intentionally encoded as an unseen
                # category. Reindexing later converts its dummy variables to 0.
                if selected_value == "Reference / other":
                    los_values[column] = "__REFERENCE_CATEGORY__"
                else:
                    los_values[column] = selected_value

        # ----------------------------------------------------
        # Numerical LOS inputs
        # ----------------------------------------------------

        with numerical_column:
            st.markdown("### Numerical information")

            if not model_numeric_columns:
                st.info(
                    "No original numeric feature names were found in the bundle."
                )

            for column in model_numeric_columns:
                median = numeric_medians.get(column, 0.0)

                minimum, maximum, default, step = slider_settings(
                    column=column,
                    median=median,
                )

                los_values[column] = st.slider(
                    display_feature_name(column),
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=float(step),
                    key=f"los_numeric_{column}",
                    help=(
                        f"The default is the training-set median: "
                        f"{float(default):.2f}"
                    ),
                )

        submitted = st.form_submit_button(
            "Predict extended-stay risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        patient = pd.DataFrame([los_values])

        transformed = transform_features(
            patient,
            bundle["preprocessing_metadata"],
        )

        transformed = transformed.reindex(
            columns=bundle["best_cols"],
            fill_value=0,
        )

        probability = float(
            bundle["model"].predict_proba(transformed)[0, 1]
        )

        show_probability_result(
            probability=probability,
            threshold=0.50,
            positive_text=(
                "Higher probability of an ICU stay longer than seven days."
            ),
            negative_text=(
                "Lower probability of an ICU stay longer than seven days."
            ),
        )


# ============================================================
# 9. Main application
# ============================================================

st.title("MedAgent ICU Risk Demo")

st.caption(
    "Research demonstration only — model output is not medical advice "
    "or a replacement for clinician judgment."
)


try:
    extended_bundle, readmission_bundle = load_models()

except FileNotFoundError as error:
    st.error(str(error))

    st.write(
        "One or both trained model bundles are missing. Run the training "
        "commands below before starting the application."
    )

    st.code(
        "$env:PYTHONPATH = 'src'\n"
        "python -m medagent.main\n"
        "python -m medagent.readmission_main\n"
        "python -m streamlit run streamlit_app.py",
        language="powershell",
    )

    st.stop()


readmission_tab, extended_stay_tab, about_tab = st.tabs(
    [
        "Readmission",
        "Extended stay",
        "About",
    ]
)


with readmission_tab:
    readmission_form(readmission_bundle)


with extended_stay_tab:
    extended_stay_form(extended_bundle)


with about_tab:
    st.subheader("How the demo works")

    st.markdown(
        """
        1. The interface collects patient information.
        2. The values are placed into a one-row pandas DataFrame.
        3. The saved preprocessing rules convert the values into model features.
        4. XGBoost calculates a probability between 0 and 1.
        5. A decision threshold converts the probability into a classification.

        The application loads saved models from the `checkpoints` folder.
        It does not retrain the models or scan the raw MIMIC files when making
        predictions.
        """
    )

    st.markdown("### Model inputs")

    st.write(
        "**Readmission model:** admission information, demographics and "
        "patient history."
    )

    st.write(
        "**Extended-stay model:** early ICU information stored in the "
        "extended-stay model bundle."
    )

    st.markdown("### Important limitation")

    st.write(
        "This is a research and presentation demo. The models have not been "
        "validated or approved for direct clinical decision-making."
    )