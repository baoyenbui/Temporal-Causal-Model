import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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
from src.option_a.features import FeatureBuilder
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


def _git_status() -> tuple:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None, None
    if out.returncode != 0:
        return None, None
    changed_paths = [line[3:].strip() for line in out.stdout.splitlines() if line.strip()]
    return len(changed_paths) > 0, changed_paths


def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest().upper()


def _prepare_logs(logs: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if logs is None:
        return None
    filtered = filter_pre_outcome(logs)
    assert_no_outcome_rows(filtered, where="protocol logs after filter_pre_outcome")
    return filtered


def _config_for(obj: Any, label: str, strict: bool) -> Dict[str, Any]:
    get_config = getattr(obj, "get_config", None)
    if not callable(get_config):
        if strict:
            raise ValueError(
                f"{label} requires features and must expose a JSON-serializable get_config()"
            )
        config: Dict[str, Any] = {}
    else:
        config = get_config()
        if not isinstance(config, dict):
            config = {"value": config}
        else:
            config = dict(config)

    config.setdefault(
        "class", f"{type(obj).__module__}.{type(obj).__qualname__}"
    )

    try:
        json.dumps(config)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} requires features and must expose a JSON-serializable get_config(): {exc}"
        )
    return config


def run_protocol(
    methods: Sequence,
    bench: Optional[pd.DataFrame] = None,
    feature_builder: Optional[FeatureBuilder] = None,
    logs: Optional[pd.DataFrame] = None,
    target: str = PRIMARY_TARGET,
    data_dir: str = "data",
    reference_method: str = "TEMPORAL_PRIMARY",
) -> RunResult:
    started = time.time()

    if bench is None:
        bench = load_benchmark(data_dir=data_dir)
    if target not in bench.columns:
        raise ValueError(f"Target '{target}' is not a column of the benchmark table")

    names = [m.name for m in methods]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"method names must be unique; duplicate name(s): {dupes}")

    method_configs: List[Dict[str, Any]] = []
    needs_features = False
    for method in methods:
        strict = bool(getattr(method, "requires_features", False))
        if strict:
            needs_features = True
        method_configs.append(_config_for(method, f"Method '{method.name}'", strict))

    if needs_features and feature_builder is None:
        raise ValueError(
            "feature_builder must be provided because at least one method requires features"
        )

    feature_builder_config: Dict[str, Any] = {}
    if feature_builder is not None:
        label = f"Feature builder '{getattr(feature_builder, 'name', type(feature_builder).__name__)}'"
        feature_builder_config = _config_for(feature_builder, label, False)

    n_logs_in = 0 if logs is None else int(len(logs))
    prepared_logs = _prepare_logs(logs)
    n_logs_used = 0 if prepared_logs is None else int(len(prepared_logs))

    control = bench.attrs.get("control", {"type": "primary", "seed": None})

    folds = leave_one_treatment_out(bench)

    predictions = pd.DataFrame(index=bench.index)
    for method in methods:
        predictions[method.name] = np.nan

    failures: List[Dict] = []

    for fold in folds:
        train_rows = bench.iloc[fold.train_idx]
        test_rows = bench.iloc[fold.test_idx]

        X_train = X_test = None
        features_ok = True
        if needs_features and feature_builder is not None:
            fold_builder = copy.deepcopy(feature_builder)
            try:
                fold_builder.fit(train_rows, prepared_logs)
                X_train = fold_builder.transform(train_rows, prepared_logs)
                X_test = fold_builder.transform(test_rows, prepared_logs)
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
                X_train = X_test = None
                features_ok = False

        for method in methods:
            use_features = bool(getattr(method, "requires_features", False))
            if use_features and not features_ok:
                continue
            fold_method = copy.deepcopy(method)
            try:
                fold_method.fit(train_rows, X_train if use_features else None, target)
                predicted = np.asarray(
                    fold_method.predict(test_rows, X_test if use_features else None),
                    dtype=float,
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

    commit = _git_commit()
    git_dirty, git_changed_paths = _git_status()

    manifest = {
        "commit": commit,
        "git_dirty": git_dirty,
        "git_changed_paths": git_changed_paths,
        "target": target,
        "control": control,
        "n_rows": int(len(bench)),
        "n_rows_expected": N_BENCHMARK_ROWS,
        "n_logs_in": n_logs_in,
        "n_logs_used": n_logs_used,
        "n_folds": len(folds),
        "fold_key": FOLD_KEY,
        "methods": [m.name for m in methods],
        "method_configs": method_configs,
        "feature_builder_config": feature_builder_config,
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
    rng = np.random.default_rng(seed)
    permuted = bench.copy()
    values = permuted[target].to_numpy(dtype=float).copy()
    for year in permuted["Year"].unique():
        positions = np.flatnonzero((permuted["Year"] == year).to_numpy())
        values[positions] = rng.permutation(values[positions])
    permuted[target] = values
    permuted.attrs["control"] = {"type": "within_year_target_permutation", "seed": seed}
    return permuted


def write_manifest(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"manifest": result.manifest, "failures": result.failures},
            handle,
            indent=2,
            ensure_ascii=False,
        )