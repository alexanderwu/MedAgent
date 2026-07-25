"""
confidence_intervals.py

95% confidence intervals for the LOS (regression) model and the
readmission (classification) model, built on top of
load_medagent_artifacts.py.

Two methods:
  1. Percentile bootstrap -- works for ANY metric (MAE, RMSE, R^2,
     accuracy, recall, precision, F1, ROC-AUC), no distributional
     assumptions needed. This is the number to report.
  2. Analytic Wald interval -- closed-form, only valid for metrics
     that are themselves proportions (accuracy, recall, precision).
     Included as a fast algebraic sanity check against the bootstrap.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from load_medagent_artifacts import load_los_artifacts, load_readmit_artifacts


# ---------------------------------------------------------------------
# 1. Percentile bootstrap -- generic, works for every metric below
# ---------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    """
    Percentile bootstrap CI for any metric_fn(y_true, y_pred).

    Resamples the TEST SET (not the model) with replacement n_boot times,
    recomputes the metric on each resample, and reads off the alpha/2
    and 1 - alpha/2 percentiles of that distribution. The model is
    already fixed and trained -- what's uncertain is how representative
    this particular test set is, so that's what gets resampled.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # n indices, drawn WITH replacement
        boot_stats[b] = metric_fn(y_true[idx], y_pred[idx])

    point_estimate = metric_fn(y_true, y_pred)
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper


# ---------------------------------------------------------------------
# 2. Analytic Wald interval -- closed form, proportions only
#    p_hat +/- z * sqrt( p_hat*(1 - p_hat) / n )
# ---------------------------------------------------------------------
def wald_ci_proportion(successes, n, z=1.96):
    """95% Wald CI for a proportion p_hat = successes / n (valid when n*p_hat, n*(1-p_hat) >~ 10)."""
    p_hat = successes / n
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat, p_hat - z * se, p_hat + z * se


# ---------------------------------------------------------------------
# LOS (regression) confidence intervals
# ---------------------------------------------------------------------
def los_confidence_intervals():
    los = load_los_artifacts()
    model_v3 = los['model_v3']
    X_test, y_test = los['X_test'], los['y_test']

    # model_v3 was trained on ln(LOS); undo the transform to get days back
    y_pred = np.clip(np.expm1(model_v3.predict(X_test)), 0, None)
    y_true = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    metrics = {
        'MAE (days)':  lambda yt, yp: mean_absolute_error(yt, yp),
        'RMSE (days)': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'R2':          lambda yt, yp: r2_score(yt, yp),
    }

    print('=== LOS model (model_v3) -- 95% bootstrap CIs ===')
    for name, fn in metrics.items():
        point, lo, hi = bootstrap_ci(y_true, y_pred, fn)
        print(f'{name:12s}: {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]')


# ---------------------------------------------------------------------
# Readmission (classification) confidence intervals
# ---------------------------------------------------------------------
def readmit_confidence_intervals():
    readmit = load_readmit_artifacts()
    model_readmit = readmit['model_readmit']
    X_test, y_test = readmit['X_test'], readmit['y_test']

    y_pred = model_readmit.predict(X_test)
    y_proba = model_readmit.predict_proba(X_test)[:, 1]
    y_true = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    metrics = {
        'Accuracy':  lambda yt, yp: accuracy_score(yt, yp),
        'Precision': lambda yt, yp: precision_score(yt, yp),
        'Recall':    lambda yt, yp: recall_score(yt, yp),
        'F1':        lambda yt, yp: f1_score(yt, yp),
    }

    print('\n=== Readmission model -- 95% bootstrap CIs ===')
    for name, fn in metrics.items():
        point, lo, hi = bootstrap_ci(y_true, y_pred, fn)
        print(f'{name:10s}: {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]')

    def auc_fn(yt, yp_proba):
        return roc_auc_score(yt, yp_proba)
    point, lo, hi = bootstrap_ci(y_true, y_proba, auc_fn)
    print(f'{"ROC-AUC":10s}: {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]')

    # Analytic sanity check on recall -- recall IS a proportion:
    # (true positives) / (actual positives), so the Wald formula applies directly.
    n_actual_pos = int(y_true.sum())
    n_true_pos = int(((y_pred == 1) & (y_true == 1)).sum())
    p_hat, lo_a, hi_a = wald_ci_proportion(n_true_pos, n_actual_pos)
    print(f'\nAnalytic Wald check on recall: {p_hat:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]')


if __name__ == '__main__':
    los_confidence_intervals()
    readmit_confidence_intervals()
