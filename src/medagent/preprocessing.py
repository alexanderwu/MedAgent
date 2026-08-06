import pandas as pd


from medagent.config import (
    CATEGORICAL_COLUMNS,
    COUNT_FEATURE_COLUMNS,
    PREPROCESS_DROP_COLUMNS,
)

from medagent.config import (
    HARD_LEAKAGE_COLUMNS,
    RANDOM_STATE,
    SOFT_LEAKAGE_COLUMNS,
    TEST_SIZE,
)

def drop_preprocess_columns(df_experiment: pd.DataFrame) -> pd.DataFrame:
    """Remove IDs and features unavailable at prediction time."""
    return df_experiment.drop(
        columns=PREPROCESS_DROP_COLUMNS,
        errors="ignore",
    ).copy()


def apply_known_missing_value_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply missing-value rules determined by the meaning of each feature."""
    df = df.copy()

    # No ED visit means ED duration is meaningfully zero.
    df["ed_los_hours"] = df["ed_los_hours"].fillna(0)

    # Missing categories become an explicit category.
    for column in ["insurance", "marital_status", "admitting_service"]:
        df[column] = df[column].fillna("Unknown")

    return df

def prepare_unencoded_features(df_experiment: pd.DataFrame) -> pd.DataFrame:
    """Drop unsafe columns and apply fixed feature-cleaning rules."""
    df = drop_preprocess_columns(df_experiment)
    df = apply_known_missing_value_rules(df)

    df["gender_binary"] = (df["gender"] == "M").astype(int)
    df = df.drop(columns="gender")

    return df

"""
steps in order:

feature_engineering.py
raw MIMIC tables → clinical feature table (df_experiment)

preprocessing.py
df_experiment → cleaned, consistently formatted model-ready features

modeling.py
clean features → train/test split → XGBoost model → metrics

"""


def fit_preprocessing_metadata(X_train: pd.DataFrame) -> dict:
    """Learn encoding columns and numeric medians from training features only."""
    encoded_train = pd.get_dummies(
        X_train,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )

    for column in COUNT_FEATURE_COLUMNS:
        if column in encoded_train.columns:
            encoded_train[column] = encoded_train[column].fillna(0)

    numeric_columns = encoded_train.select_dtypes(include="number").columns

    return {
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoded_columns": encoded_train.columns.tolist(),
        "numeric_medians": encoded_train[numeric_columns].median().to_dict(),
    }

def transform_features(
    X: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Transform features using preprocessing metadata learned from training data."""
    df = X.copy()

    columns_to_encode = [
        column
        for column in metadata["categorical_columns"]
        if column in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=columns_to_encode,
        drop_first=True,
    )

    for column in COUNT_FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    for column, median in metadata["numeric_medians"].items():
        if column in df.columns:
            df[column] = df[column].fillna(median)

    return df.reindex(
        columns=metadata["encoded_columns"],
        fill_value=0,
    )