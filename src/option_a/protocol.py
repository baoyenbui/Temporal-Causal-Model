"""The frozen evaluation protocol: out-of-fold predictions, metrics, and a run manifest.

Every method sees the same 88 rows, the same folds, the same feature contract, and the same
failure policy. That is what makes the comparison fair; it is enforced here rather than
trusted to each method.
"""

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.option_a.benchmark import (
    PRIMARY_TARGET,
    N_BENCHMARK_ROWS,
    load_benchmark,
    verify_input_hashes,
)
from src.option_a.features import FeatureBuilder, NoFeatures
from src.option_a.folds import FOLD_KEY, Fold, leave_one_treatment_out
from src.option_a.metrics import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    all_metrics,
    cluster_bootstrap_ci,
    mae,
    paired_difference_ci,
)


@dataclass
class RunResult:
    target: str
    predictions: pd.DataFrame
    results_table: pd.DataFrame
    paired_differences: pd.DataFrame
    failures: List[Dict] = field(default_factory=list)
    manifest: Dict = field(default_factory=dict)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest().upper()


def run_protocol(
    methods: Sequence,
    bench: Optional[pd.DataFrame] = None,
    feature_builder: Optional[FeatureBuilder] = None,
    logs: Optional[pd.DataFrame] = None,
    target: str = PRIMARY_TARGET,
    data_dir: str = "data",
    reference_method: str = "TEMPORAL_PRIMARY",
) -> RunResult:
    """Run every method through leave-one-treatment-construct-out and score them together.

    A method that raises inside a fold does not abort the run. Its rows for that fold are
    recorded as failures and left unpredicted, so the failure is visible in the output
    rather than being silently absorbed into a shorter evaluation sample.
    """
    started = time.time()

    if bench is None:
        bench = load_benchmark(data_dir=data_dir)
    if target not in bench.columns:
        raise ValueError(f"Target '{target}' is not a column of the benchmark table")

    folds = leave_one_treatment_out(bench)
    builder = feature_builder if feature_builder is not None else NoFeatures()

    predictions = pd.DataFrame(index=bench.index)
    for method in methods:
        predictions[method.name] = np.nan

    failures: List[Dict] = []

    for fold in folds:
        train_rows = bench.iloc[fold.train_idx]
        test_rows = bench.iloc[fold.test_idx]

        needs_features = any(getattr(m, "requires_features", False) for m in methods)
        X_train = X_test = None
        if needs_features:
            try:
                builder.fit(train_rows, logs)
                X_train = builder.transform(train_rows, logs)
                X_test = builder.transform(test_rows, logs)
            except Exception as exc:
                failures.append(
                    {
                        "stage": "features",
                        "fold": fold.name,
                        "method": None,
                        "n_rows": fold.n_test,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

        for method in methods:
            use_features = getattr(method, "requires_features", False)
            try:
                method.fit(train_rows, X_train if use_features else None, target)
                predicted = np.asarray(
                    method.predict(test_rows, X_test if use_features else None), dtype=float
                )
                if len(predicted) != fold.n_test:
                    raise ValueError(
                        f"returned {len(predicted)} predictions for {fold.n_test} held-out rows"
                    )
                predictions.iloc[fold.test_idx, predictions.columns.get_loc(method.name)] = predicted
            except Exception as exc:
                failures.append(
                    {
                        "stage": "method",
                        "fold": fold.name,
                        "method": method.name,
                        "n_rows": fold.n_test,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    y_true = bench[target].to_numpy(dtype=float)
    clusters = bench[FOLD_KEY].to_numpy()

    rows = []
    for method in methods:
        column = predictions[method.name].to_numpy(dtype=float)
        covered = ~np.isnan(column)
        n_covered = int(covered.sum())

        if n_covered == 0:
            rows.append(
                {
                    "method": method.name,
                    "n_predicted": 0,
                    "n_missing": len(bench),
                    "MAE": np.nan,
                    "MAE_lo": np.nan,
                    "MAE_hi": np.nan,
                    "RMSE": np.nan,
                    "Spearman": np.nan,
                    "SignAgreement": np.nan,
                }
            )
            continue

        scores = all_metrics(y_true[covered], column[covered])
        point, lo, hi = cluster_bootstrap_ci(
            y_true[covered], column[covered], clusters[covered], statistic=mae
        )
        rows.append(
            {
                "method": method.name,
                "n_predicted": n_covered,
                "n_missing": len(bench) - n_covered,
                "MAE": point,
                "MAE_lo": lo,
                "MAE_hi": hi,
                "RMSE": scores["RMSE"],
                "Spearman": scores["Spearman"],
                "SignAgreement": scores["SignAgreement"],
            }
        )

    results_table = pd.DataFrame(rows).sort_values("MAE", na_position="last").reset_index(drop=True)

    paired = _paired_differences(
        bench, predictions, methods, y_true, clusters, target, reference_method
    )

    manifest = {
        "commit": _git_commit(),
        "target": target,
        "n_rows": int(len(bench)),
        "n_rows_expected": N_BENCHMARK_ROWS,
        "n_folds": len(folds),
        "fold_key": FOLD_KEY,
        "methods": [m.name for m in methods],
        "feature_builder": builder.name,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "input_hashes": {k: v["observed"] for k, v in verify_input_hashes(data_dir).items()},
        "input_hashes_all_match": all(v["match"] for v in verify_input_hashes(data_dir).values()),
        "n_failures": len(failures),
        "predictions_sha256": _frame_digest(predictions),
        "results_sha256": _frame_digest(results_table),
        "runtime_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    return RunResult(
        target=target,
        predictions=predictions,
        results_table=results_table,
        paired_differences=paired,
        failures=failures,
        manifest=manifest,
    )


def _paired_differences(
    bench, predictions, methods, y_true, clusters, target, reference_method
) -> pd.DataFrame:
    """MAE difference between the reference method and each other method, on shared rows.

    Reported whether or not the interval excludes zero.
    """
    names = [m.name for m in methods]
    if reference_method not in names:
        return pd.DataFrame(
            columns=["reference", "baseline", "n_shared", "MAE_diff", "diff_lo", "diff_hi"]
        )

    reference = predictions[reference_method].to_numpy(dtype=float)
    rows = []
    for name in names:
        if name == reference_method:
            continue
        other = predictions[name].to_numpy(dtype=float)
        shared = ~np.isnan(reference) & ~np.isnan(other)
        if shared.sum() == 0:
            continue
        point, lo, hi = paired_difference_ci(
            y_true[shared], reference[shared], other[shared], clusters[shared], statistic=mae
        )
        rows.append(
            {
                "reference": reference_method,
                "baseline": name,
                "n_shared": int(shared.sum()),
                "MAE_diff": point,
                "diff_lo": lo,
                "diff_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def placebo_targets(bench: pd.DataFrame, target: str = PRIMARY_TARGET, seed: int = 20260811):
    """Permute the target within year group, leaving everything else untouched.

    A method that scores comparably here is reading structure that has nothing to do with
    the experimental effect, which blocks evidence claims.
    """
    rng = np.random.default_rng(seed)
    permuted = bench.copy()
    values = permuted[target].to_numpy(dtype=float).copy()
    for year in permuted["Year"].unique():
        positions = np.flatnonzero((permuted["Year"] == year).to_numpy())
        values[positions] = rng.permutation(values[positions])
    permuted[target] = values
    return permuted


def write_manifest(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"manifest": result.manifest, "failures": result.failures},
            handle,
            indent=2,
            ensure_ascii=False,
        )
