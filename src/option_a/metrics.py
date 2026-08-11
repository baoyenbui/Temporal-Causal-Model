"""Frozen metrics for the Option A benchmark.

Primary metric is row-level mean absolute error on the primary target. Everything else is
secondary and reported alongside it, never in place of it.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_as_array(y_true) - _as_array(y_pred))))


def rmse(y_true, y_pred) -> float:
    diff = _as_array(y_true) - _as_array(y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def spearman(y_true, y_pred) -> float:
    """Rank correlation. Returns nan when either side is constant, which is the honest
    answer for a baseline that predicts one number for every row."""
    a, b = _as_array(y_true), _as_array(y_pred)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def sign_class(values) -> np.ndarray:
    """Three-class sign: -1 negative, 0 exactly zero, +1 positive.

    Exact zero is its own class. No tolerance band is applied, because a data-dependent
    dead zone would let the zero class absorb small errors and inflate agreement.
    """
    arr = _as_array(values)
    out = np.zeros_like(arr)
    out[arr > 0] = 1.0
    out[arr < 0] = -1.0
    return out


def sign_agreement(y_true, y_pred) -> float:
    return float(np.mean(sign_class(y_true) == sign_class(y_pred)))


def all_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Spearman": spearman(y_true, y_pred),
        "SignAgreement": sign_agreement(y_true, y_pred),
    }


def _resample_cluster_positions(
    clusters: np.ndarray, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Draw clusters with replacement and return the row positions they contribute."""
    unique = np.unique(clusters)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    positions = np.concatenate([np.flatnonzero(clusters == c) for c in drawn])
    return positions if len(positions) else None


def cluster_bootstrap_ci(
    y_true,
    y_pred,
    clusters,
    statistic=mae,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Percentile interval from resampling treatment constructs with replacement.

    Rows sharing a treatment construct are not independent, so the resampling unit is the
    construct. With only 15 clusters the interval is wide; report it as it comes out.
    """
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    clusters = np.asarray(clusters).ravel()
    point = float(statistic(y_true, y_pred))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        positions = _resample_cluster_positions(clusters, rng)
        if positions is None:
            continue
        draws.append(statistic(y_true[positions], y_pred[positions]))

    if not draws:
        return point, float("nan"), float("nan")
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return point, float(lower), float(upper)


def paired_difference_ci(
    y_true,
    pred_method,
    pred_baseline,
    clusters,
    statistic=mae,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Interval for statistic(method) - statistic(baseline) on the same resampled clusters.

    Negative means the method has lower error. An interval containing zero is a valid
    reportable result and must not trigger a change of method, sample, or target.
    """
    y_true = _as_array(y_true)
    pred_method, pred_baseline = _as_array(pred_method), _as_array(pred_baseline)
    clusters = np.asarray(clusters).ravel()

    point = float(statistic(y_true, pred_method)) - float(statistic(y_true, pred_baseline))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        positions = _resample_cluster_positions(clusters, rng)
        if positions is None:
            continue
        draws.append(
            statistic(y_true[positions], pred_method[positions])
            - statistic(y_true[positions], pred_baseline[positions])
        )

    if not draws:
        return point, float("nan"), float("nan")
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return point, float(lower), float(upper)


def format_results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})
