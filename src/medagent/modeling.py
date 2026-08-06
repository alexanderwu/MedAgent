import pandas as pd

from medagent.config import EXTENDED_STAY_THRESHOLD_DAYS
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from medagent.config import (
    HARD_LEAKAGE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)


"""
steps in order:

feature_engineering.py
raw MIMIC tables → clinical feature table (df_experiment)

preprocessing.py
df_experiment → cleaned, consistently formatted model-ready features

modeling.py
clean features → train/test split → XGBoost model → metrics

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