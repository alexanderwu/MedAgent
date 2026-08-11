import pandas as pd

from medagent.config import EXTENDED_STAY_THRESHOLD_DAYS
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from medagent.config import (
    HARD_LEAKAGE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    SOFT_LEAKAGE_COLUMNS,
)


"""
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



def add_extended_stay_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target: ICU LOS greater than the configured threshold."""
    df = df.copy()

    df["extended_stay"] = (
        df["los"] > EXTENDED_STAY_THRESHOLD_DAYS
    ).astype(int)

    return df


def split_extended_stay_data(
    df_extended: pd.DataFrame,
    patient_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split extended-stay data so no patient appears in both sets.
    same patient, multiple ICU stays
        ↓
    all of their stays go to train OR test
        ↓
    never both
    """
    X = df_extended.drop(
        columns=HARD_LEAKAGE_COLUMNS + ["extended_stay"],
    )
    y = df_extended["extended_stay"]

    patient_ids = patient_ids.loc[df_extended.index]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=patient_ids)
    )

    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
    )


def select_leakage_free_features(X: pd.DataFrame) -> pd.DataFrame:
    """Remove features not available early enough for a real LOS prediction."""
    return X.drop(
        columns=SOFT_LEAKAGE_COLUMNS,
        errors="ignore",
    ).copy()


def train_extended_stay_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """Train the leakage-free extended-stay XGBoost classifier."""
    scale_pos_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_extended_stay_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate an extended-stay classifier on untouched test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from medagent.config import (
    RANDOM_STATE,
    READMISSION_MODEL_SETTINGS,
    READMISSION_RECALL_TARGET,
)
from medagent.preprocessing import build_readmission_preprocessor


def split_readmission_train_validation_test(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Make patient-safe 60% / 20% / 20% splits.

    A patient can appear in exactly one of:
    train, validation, or final test.
    """

    groups = cohort["subject_id"]
    y = cohort["readmit_30d"]

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        outer_splitter.split(cohort, y, groups=groups)
    )

    train_validation_cohort = cohort.iloc[
        train_validation_idx
    ].copy()

    test_cohort = cohort.iloc[test_idx].copy()

    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, validation_idx = next(
        inner_splitter.split(
            train_validation_cohort,
            train_validation_cohort["readmit_30d"],
            groups=train_validation_cohort["subject_id"],
        )
    )

    train_cohort = train_validation_cohort.iloc[train_idx].copy()

    validation_cohort = train_validation_cohort.iloc[
        validation_idx
    ].copy()

    return train_cohort, validation_cohort, test_cohort


def build_readmission_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_settings: dict | None = None,
) -> Pipeline:
    """
    Build the complete pipeline:
    raw features -> preprocessing -> imbalance-weighted XGBoost.
    """

    if model_settings is None:
        model_settings = READMISSION_MODEL_SETTINGS

    preprocessor = build_readmission_preprocessor(X_train)

    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "Training data has no positive readmission examples."
        )

    scale_pos_weight = (y_train == 0).sum() / positive_count

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        **model_settings,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_readmission_pipeline(
    cohort: pd.DataFrame,
    feature_columns: list[str],
    model_settings: dict | None = None,
) -> Pipeline:
    """Fit the complete readmission pipeline on one cohort."""

    X_train = cohort[feature_columns].copy()
    y_train = cohort["readmit_30d"].copy()

    pipeline = build_readmission_pipeline(
        X_train,
        y_train,
        model_settings=model_settings,
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_readmission_model(
    pipeline: Pipeline,
    cohort: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> dict:
    """Evaluate a fitted readmission pipeline at one alert threshold."""

    X = cohort[feature_columns].copy()
    y_true = cohort["readmit_30d"].copy()

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
    }


def evaluate_readmission_threshold_grid(
    pipeline: Pipeline,
    validation_cohort: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Score candidate alert thresholds on validation data only.
    This is where we make the recall-vs-precision tradeoff.
    """

    X_validation = validation_cohort[feature_columns].copy()
    y_validation = validation_cohort["readmit_30d"].copy()

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    results = []

    for threshold in np.arange(0.10, 0.71, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)


def select_recall_first_threshold(
    threshold_results: pd.DataFrame,
    minimum_recall: float = READMISSION_RECALL_TARGET,
) -> float:
    """
    Pick the highest threshold that still meets the recall goal.

    Higher threshold = fewer false-positive alerts.
    We therefore choose the highest one that still catches at least
    the required percentage of true readmissions.
    """

    valid_thresholds = threshold_results[
        threshold_results["recall"] >= minimum_recall
    ]

    if valid_thresholds.empty:
        raise ValueError(
            f"No threshold reached recall >= {minimum_recall:.0%}."
        )

    best_row = valid_thresholds.sort_values(
        ["threshold", "precision"],
        ascending=[False, False],
    ).iloc[0]

    return float(best_row["threshold"])