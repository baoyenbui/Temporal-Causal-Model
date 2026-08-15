"""Entry point for the Option A benchmark.

Runs whichever methods are wired up and prints the frozen result table. With only the
three feature-free baselines registered it answers one question: how hard is this
benchmark to beat? Every later method is judged against that answer.

    python run_option_a.py
    python run_option_a.py --target ate_p_1__     # sensitivity analysis
    python run_option_a.py --placebo              # permuted-target control
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.option_a.benchmark import PRIMARY_TARGET, SENSITIVITY_TARGET, load_benchmark
from src.option_a.folds import describe_folds, leave_one_treatment_out
from src.option_a.methods import BASELINE_METHODS_WITHOUT_FEATURES
from src.option_a.protocol import placebo_targets, run_protocol, write_manifest

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build_methods():
    """Register the methods for this run.

    Bao Yen: append NON_TEMPORAL_SAME_FEATURES and TEMPORAL_PRIMARY here once they exist in
    src/methods_student.py, and pass a fitted FeatureBuilder to run_protocol.
    """
    return [cls() for cls in BASELINE_METHODS_WITHOUT_FEATURES]


def main() -> int:
    parser = argparse.ArgumentParser(description="Option A construct-level CATE benchmark")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target", default=PRIMARY_TARGET, choices=[PRIMARY_TARGET, SENSITIVITY_TARGET])
    parser.add_argument("--placebo", action="store_true", help="permute the target within year")
    parser.add_argument("--manifest", default=None, help="path to write the run manifest")
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

    methods = build_methods()
    result = run_protocol(methods, bench=bench, target=args.target, data_dir=args.data_dir)

    label = "SENSITIVITY" if args.target == SENSITIVITY_TARGET else "PRIMARY"
    print(f"\n--- Out-of-fold results ({label} target: {args.target}) ---")
    print(result.results_table.to_string(index=False))

    if len(result.paired_differences):
        print("\n--- Paired MAE differences ---")
        print(result.paired_differences.to_string(index=False))
    else:
        print("\nNo paired differences: the reference method is not registered yet.")

    if result.failures:
        print(f"\n--- {len(result.failures)} failure record(s) ---")
        print(pd.DataFrame(result.failures).to_string(index=False))
    else:
        print("\nNo failures: every row received an out-of-fold prediction from every method.")

    hashes_ok = result.manifest["input_hashes_all_match"]
    print(f"\nInput hashes match frozen snapshot: {hashes_ok}")
    print(f"Predictions SHA-256: {result.manifest['predictions_sha256'][:16]}...")
    print(f"Runtime: {result.manifest['runtime_seconds']}s")

    if args.manifest:
        write_manifest(result, args.manifest)
        print(f"Manifest written to {args.manifest}")

    return 0 if hashes_ok and not result.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
