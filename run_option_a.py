import argparse
import itertools
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.option_a.benchmark import PRIMARY_TARGET, SENSITIVITY_TARGET, load_benchmark
from src.option_a.folds import FOLD_KEY, describe_folds, leave_one_treatment_out
from src.option_a.methods import BASELINE_METHODS_WITHOUT_FEATURES
from src.option_a.metrics import paired_difference_ci, mae
from src.option_a.protocol import placebo_targets, run_protocol

from src.temporal import default_feature_builders, TemporalPrimary
from src.non_temporal import NonTemporalSameFeatures

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

EXPECTED_N_LOGS_IN = 641_490
EXPECTED_N_LOGS_USED = 563_117


def build_temporal_methods():
    return [TemporalPrimary(seed=42)]


def build_shuffled_methods():
    return [NonTemporalSameFeatures(seed=42)]


def build_baseline_methods():
    return [cls() for cls in BASELINE_METHODS_WITHOUT_FEATURES]


def combined_paired_ci(predictions: pd.DataFrame, y_true: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    rows = []
    method_names = list(predictions.columns)
    for a, b in itertools.combinations(method_names, 2):
        col_a = predictions[a].to_numpy(dtype=float)
        col_b = predictions[b].to_numpy(dtype=float)
        shared = ~np.isnan(col_a) & ~np.isnan(col_b)
        if shared.sum() == 0:
            continue
        point, lo, hi = paired_difference_ci(
            y_true[shared], col_a[shared], col_b[shared], clusters[shared], statistic=mae
        )
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "n_shared": int(shared.sum()),
                "MAE_diff_a_minus_b": point,
                "diff_lo": lo,
                "diff_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def build_combined_manifest(baseline_result, temporal_result, shuffled_result, predictions_all: pd.DataFrame) -> dict:
    import hashlib

    def digest(frame: pd.DataFrame) -> str:
        return hashlib.sha256(
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest().upper()

    manifests = {
        "baseline": baseline_result.manifest,
        "temporal": temporal_result.manifest,
        "shuffled": shuffled_result.manifest,
    }

    n_logs_in = temporal_result.manifest["n_logs_in"]
    n_logs_used = temporal_result.manifest["n_logs_used"]
    logs_consistent = (
        n_logs_in == shuffled_result.manifest["n_logs_in"]
        and n_logs_used == shuffled_result.manifest["n_logs_used"]
    )

    return {
        "commit": temporal_result.manifest["commit"],
        "git_dirty": temporal_result.manifest["git_dirty"],
        "target": temporal_result.manifest["target"],
        "control": temporal_result.manifest["control"],
        "n_rows": temporal_result.manifest["n_rows"],
        "n_folds": temporal_result.manifest["n_folds"],
        "fold_key": FOLD_KEY,
        "n_logs_in": n_logs_in,
        "n_logs_used": n_logs_used,
        "n_logs_consistent_across_runs": logs_consistent,
        "n_logs_in_matches_frozen": n_logs_in == EXPECTED_N_LOGS_IN,
        "n_logs_used_matches_frozen": n_logs_used == EXPECTED_N_LOGS_USED,
        "methods": list(predictions_all.columns),
        "sub_manifests": manifests,
        "input_hashes_all_match": all(
            m["input_hashes_all_match"] for m in manifests.values()
        ),
        "predictions_sha256": digest(predictions_all),
        "n_failures_total": sum(len(r.failures) for r in (baseline_result, temporal_result, shuffled_result)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Option A construct-level CATE benchmark")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target", default=PRIMARY_TARGET, choices=[PRIMARY_TARGET, SENSITIVITY_TARGET])
    parser.add_argument("--placebo", action="store_true", help="permute the target within year")
    parser.add_argument("--manifest", default=None, help="path to write the single combined run manifest")
    parser.add_argument("--show-folds", action="store_true")
    args = parser.parse_args()

    bench = load_benchmark(data_dir=args.data_dir)
    print(f"Loaded {len(bench)} evaluation rows, "
          f"{bench['TreatmentLessonConstructId'].nunique()} treatment constructs, "
          f"years {sorted(bench['Year'].unique())}")

    if args.placebo:
        bench = placebo_targets(bench, target=args.target)
        print("PLACEBO RUN: target permuted within year. These numbers are a control, not a result.")

    if args.show_folds:
        print("\n--- Folds ---")
        print(describe_folds(bench, leave_one_treatment_out(bench)).to_string(index=False))

    logs = pd.read_csv(os.path.join(args.data_dir, "checkins_lessons_checkouts_training.csv"))
    print(f"Logs loaded: {len(logs)} rows (expected n_logs_in={EXPECTED_N_LOGS_IN}: "
          f"{'OK' if len(logs) == EXPECTED_N_LOGS_IN else 'MISMATCH'})")

    builders = default_feature_builders(seed=42)

    baseline_result = run_protocol(
        build_baseline_methods(), bench=bench, target=args.target, data_dir=args.data_dir,
    )
    temporal_result = run_protocol(
        build_temporal_methods(), bench=bench, target=args.target, data_dir=args.data_dir,
        feature_builder=builders["temporal"], logs=logs, reference_method="TEMPORAL_PRIMARY",
    )
    shuffled_result = run_protocol(
        build_shuffled_methods(), bench=bench, target=args.target, data_dir=args.data_dir,
        feature_builder=builders["shuffled"], logs=logs, reference_method="NON_TEMPORAL_SAME_FEATURES",
    )

    print(f"n_logs_used from run: {temporal_result.manifest['n_logs_used']} "
          f"(expected {EXPECTED_N_LOGS_USED}: "
          f"{'OK' if temporal_result.manifest['n_logs_used'] == EXPECTED_N_LOGS_USED else 'MISMATCH'})")

    all_failures = baseline_result.failures + temporal_result.failures + shuffled_result.failures
    hashes_ok = (
        baseline_result.manifest["input_hashes_all_match"]
        and temporal_result.manifest["input_hashes_all_match"]
        and shuffled_result.manifest["input_hashes_all_match"]
    )

    predictions_all = pd.concat(
        [baseline_result.predictions, temporal_result.predictions, shuffled_result.predictions],
        axis=1,
    )
    if len(predictions_all.columns) != len(set(predictions_all.columns)):
        raise ValueError("duplicate method names across baseline/temporal/shuffled runs")
    n_methods = len(predictions_all.columns)
    fully_predicted = int((predictions_all.notna().all(axis=1)).sum())
    print(f"\n{n_methods} methods produced predictions; "
          f"{fully_predicted}/{len(bench)} rows have predictions from all {n_methods} methods")

    if all_failures or not hashes_ok:
        print(f"\n{len(all_failures)} failure(s), input_hashes_all_match={hashes_ok}: "
              f"skipping final combined table until these are resolved.")
        if all_failures:
            print(pd.DataFrame(all_failures).to_string(index=False))
        return 1

    label = "SENSITIVITY" if args.target == SENSITIVITY_TARGET else "PRIMARY"
    combined_results = pd.concat(
        [baseline_result.results_table, temporal_result.results_table, shuffled_result.results_table],
        ignore_index=True,
    ).sort_values("MAE", na_position="last").reset_index(drop=True)

    print(f"\n--- Out-of-fold results ({label} target: {args.target}) ---")
    print(combined_results.to_string(index=False))

    y_true = bench[args.target].to_numpy(dtype=float)
    clusters = bench[FOLD_KEY].to_numpy()
    pairwise_ci = combined_paired_ci(predictions_all, y_true, clusters)
    print(f"\n--- Paired MAE differences across all {n_methods} methods (cluster-bootstrap CI) ---")
    print(pairwise_ci.to_string(index=False))

    manifest = build_combined_manifest(baseline_result, temporal_result, shuffled_result, predictions_all)
    print(f"\nCombined manifest: n_logs_in={manifest['n_logs_in']}, n_logs_used={manifest['n_logs_used']}, "
          f"input_hashes_all_match={manifest['input_hashes_all_match']}, "
          f"predictions_sha256={manifest['predictions_sha256']}")

    if args.manifest:
        import json
        with open(args.manifest, "w", encoding="utf-8") as handle:
            json.dump(
                {"manifest": manifest, "pairwise_ci": pairwise_ci.to_dict(orient="records")},
                handle, indent=2, ensure_ascii=False,
            )
        print(f"Manifest written to {args.manifest}")

    return 0 if hashes_ok and not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())