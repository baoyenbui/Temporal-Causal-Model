"""The frozen evaluation protocol: out-of-fold predictions, metrics, and a run manifest.

Every method sees the same 88 rows, the same folds, the same feature contract, and the same
failure policy. That is what makes the comparison fair; it is enforced here rather than
trusted to each method.
"""

import copy
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
    assert_no_outcome_rows,
    filter_pre_outcome,
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


def _git_state() -> Dict[str, object]:
    """Return the exact revision plus whether uncommitted files affect the run.

    A commit SHA alone is insufficient when the worktree is dirty: two executions can
    report the same commit while running different source or configuration files.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit_value = commit.stdout.strip() if commit.returncode == 0 else None
        changed = [line.rstrip() for line in status.stdout.splitlines() if line.strip()]
        return {
            "commit": commit_value,
            "dirty": bool(changed) if status.returncode == 0 else None,
            "changed_paths": changed if status.returncode == 0 else [],
        }
    except Exception:
        return {"commit": None, "dirty": None, "changed_paths": []}


def _fresh_component(component, kind: str):
    """Clone an unfitted prototype so no fitted state crosses fold boundaries."""
    try:
        return copy.deepcopy(component)
    except Exception as exc:
        raise TypeError(
            f"Cannot create a fresh {kind} for each fold from "
            f"{type(component).__module__}.{type(component).__qualname__}: {exc}"
        ) from exc


def _json_safe_config(value, where: str):
    """Validate that declared component configuration can be written to a manifest."""
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{where} configuration is not JSON-serializable: {exc}") from exc
    return value


def _component_config(component) -> Dict[str, object]:
    """Record component identity and its declared, pre-fit configuration.

    Student components should expose ``get_config()``. Scikit-learn-style components may
    expose ``get_params()``. Components with neither still have their class identity
    recorded, but cannot silently inject a non-serializable fitted object into the manifest.
    """
    if callable(getattr(component, "get_config", None)):
        parameters = component.get_config()
        source = "get_config"
    elif callable(getattr(component, "get_params", None)):
        parameters = component.get_params(deep=False)
        source = "get_params"
    else:
        parameters = {}
        source = "class_only"

    if not isinstance(parameters, dict):
        raise TypeError(
            f"{type(component).__qualname__} configuration must be a dict, "
            f"found {type(parameters).__name__}"
        )
    return {
        "name": getattr(component, "name", type(component).__qualname__),
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
        "config_source": source,
        "parameters": _json_safe_config(parameters, type(component).__qualname__),
    }


def _validate_method_names(methods: Sequence) -> List[str]:
    names = [getattr(method, "name", None) for method in methods]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every method must declare a non-empty string 'name'")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Method names must be unique; duplicates={duplicates}")
    return names


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

    n_logs_in = 0 if logs is None else int(len(logs))
    filtered_logs = None if logs is None else filter_pre_outcome(logs)
    n_logs_used = 0 if filtered_logs is None else int(len(filtered_logs))
    if filtered_logs is not None:
        assert_no_outcome_rows(filtered_logs, where="run_protocol feature-builder logs")

    folds = leave_one_treatment_out(bench)
    builder_prototype = feature_builder if feature_builder is not None else NoFeatures()
    method_names = _validate_method_names(methods)
    method_configs = [_component_config(method) for method in methods]
    feature_builder_config = _component_config(builder_prototype)
    missing_feature_method_configs = [
        method_name
        for method_name, method, method_config in zip(
            method_names, methods, method_configs, strict=True
        )
        if getattr(method, "requires_features", False)
        and method_config["config_source"] == "class_only"
    ]
    if missing_feature_method_configs:
        raise ValueError(
            "Feature-driven methods must expose a JSON-serializable get_config() "
            "or get_params(deep=False): "
            + ", ".join(missing_feature_method_configs)
        )

    predictions = pd.DataFrame(index=bench.index)
    for name in method_names:
        predictions[name] = np.nan

    failures: List[Dict] = []

    for fold in folds:
        train_rows = bench.iloc[fold.train_idx]
        test_rows = bench.iloc[fold.test_idx]

        needs_features = any(getattr(m, "requires_features", False) for m in methods)
        X_train = X_test = None
        features_ready = not needs_features
        if needs_features:
            try:
                fold_builder = _fresh_component(builder_prototype, "feature builder")
                fold_builder.fit(train_rows, filtered_logs)
                X_train = fold_builder.transform(train_rows, filtered_logs)
                X_test = fold_builder.transform(test_rows, filtered_logs)
                features_ready = True
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

        for method_prototype in methods:
            use_features = getattr(method_prototype, "requires_features", False)
            if use_features and not features_ready:
                continue
            try:
                method = _fresh_component(method_prototype, "method")
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
                        "method": method_prototype.name,
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

    git_state = _git_state()
    hash_report = verify_input_hashes(data_dir)
    control_type = bench.attrs.get("control_type", "primary")
    control_seed = bench.attrs.get("control_seed")
    manifest = {
        "commit": git_state["commit"],
        "git_dirty": git_state["dirty"],
        "git_changed_paths": git_state["changed_paths"],
        "target": target,
        "control": {"type": control_type, "seed": control_seed},
        "n_rows": int(len(bench)),
        "n_rows_expected": N_BENCHMARK_ROWS,
        "n_logs_in": n_logs_in,
        "n_logs_used": n_logs_used,
        "n_folds": len(folds),
        "fold_key": FOLD_KEY,
        "methods": method_names,
        "method_configs": method_configs,
        "feature_builder": builder_prototype.name,
        "feature_builder_config": feature_builder_config,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "input_hashes": {k: v["observed"] for k, v in hash_report.items()},
        "input_hashes_all_match": all(v["match"] for v in hash_report.values()),
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
    permuted.attrs["control_type"] = "within_year_target_permutation"
    permuted.attrs["control_seed"] = int(seed)
    return permuted


def write_manifest(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"manifest": result.manifest, "failures": result.failures},
            handle,
            indent=2,
            ensure_ascii=False,
        )
