#persist.py — save/load the complete model bundle

from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from medagent.config import (
    EXTENDED_STAY_THRESHOLD_DAYS,
    HARD_LEAKAGE_COLUMNS,
    P_CHECKPOINTS,
    RANDOM_STATE,
    SOFT_LEAKAGE_COLUMNS,
    TEST_SIZE,
)


def get_library_versions() -> dict[str, str]:
    """Return versions needed to reproduce or troubleshoot a saved bundle."""
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__,
    }


def build_model_bundle(
    model: Any,
    best_cols: list[str],
    preprocessing_metadata: dict,
    metrics: dict,
) -> dict:
    """
    Package everything required to make consistent future predictions.

    Does not include raw MIMIC patient data or train/test DataFrames.
    """
    return {
        "bundle_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "model": model,
        "best_cols": list(best_cols),
        "preprocessing_metadata": preprocessing_metadata,
        "metrics": metrics,

        "model_configuration": {
            "target_name": "extended_stay",
            "target_definition": (
                f"ICU length of stay > {EXTENDED_STAY_THRESHOLD_DAYS} days"
            ),
            "threshold_days": EXTENDED_STAY_THRESHOLD_DAYS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "hard_leakage_columns": HARD_LEAKAGE_COLUMNS,
            "soft_leakage_columns": SOFT_LEAKAGE_COLUMNS,
        },

        "library_versions": get_library_versions(),
    }


def save_model_bundle(
    bundle: dict,
    filename: str = "extended_stay_bundle.pkl",
) -> Path:
    """Save a complete model bundle inside the project's checkpoints folder."""
    P_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    bundle_path = P_CHECKPOINTS / filename

    with bundle_path.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)

    return bundle_path


def load_model_bundle(
    filename: str = "extended_stay_bundle.pkl",
) -> dict:
    """
    Load a previously saved model bundle.

    Only load pickle files created by your team or another trusted source.
    """
    bundle_path = P_CHECKPOINTS / filename

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No model bundle found at: {bundle_path}"
        )

    with bundle_path.open("rb") as file:
        bundle = pickle.load(file)

    required_keys = {
        "model",
        "best_cols",
        "preprocessing_metadata",
        "metrics",
        "model_configuration",
        "library_versions",
    }

    missing_keys = required_keys - bundle.keys()

    if missing_keys:
        raise ValueError(
            f"Invalid model bundle; missing keys: {sorted(missing_keys)}"
        )

    return bundle

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

def build_readmission_bundle(
    pipeline: Any,
    selected_feature_columns: list[str],
    chosen_threshold: float,
    final_metrics: dict,
    split_subject_ids: dict[str, list[int]],
    threshold_results: pd.DataFrame,
) -> dict:
    """Package everything needed to make future readmission predictions."""

    return {
        "bundle_version": 1,
        "model_type": "30_day_readmission_recall_first",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        # This one object contains preprocessing + XGBoost.
        "pipeline": pipeline,

        "selected_feature_columns": list(
            selected_feature_columns
        ),
        "chosen_threshold": float(chosen_threshold),

        "final_metrics": final_metrics,
        "split_subject_ids": split_subject_ids,
        "validation_threshold_results": threshold_results,

        "model_configuration": {
            "target_name": "readmit_30d",
            "target_definition": (
                "Eligible hospital readmission within 30 days "
                "after discharge"
            ),
            "threshold_selection_rule": (
                "Highest validation threshold meeting recall target"
            ),
        },

        "library_versions": get_library_versions(),
    }


def save_readmission_bundle(
    bundle: dict,
    filename: str = "readmission_recall_bundle.pkl",
) -> Path:
    """Save the complete readmission model bundle."""

    P_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    bundle_path = P_CHECKPOINTS / filename

    with bundle_path.open("wb") as file:
        pickle.dump(
            bundle,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return bundle_path


def load_readmission_bundle(
    filename: str = "readmission_recall_bundle.pkl",
) -> dict:
    """Load a trusted saved readmission model bundle."""

    bundle_path = P_CHECKPOINTS / filename

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No readmission bundle found at: {bundle_path}"
        )

    with bundle_path.open("rb") as file:
        bundle = pickle.load(file)

    required_keys = {
        "pipeline",
        "selected_feature_columns",
        "chosen_threshold",
        "final_metrics",
        "model_configuration",
        "library_versions",
    }

    missing_keys = required_keys - bundle.keys()

    if missing_keys:
        raise ValueError(
            "Invalid readmission bundle; missing keys: "
            f"{sorted(missing_keys)}"
        )

    return bundle