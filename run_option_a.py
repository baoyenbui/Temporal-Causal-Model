import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.option_a.benchmark import PRIMARY_TARGET, SENSITIVITY_TARGET, load_benchmark
from src.option_a.folds import describe_folds, leave_one_treatment_out
from src.option_a.methods import BASELINE_METHODS_WITHOUT_FEATURES
from src.option_a.protocol import placebo_targets, run_protocol, write_manifest
from src.option_a.features import FakeFeatureBuilder

from src.non_temporal import NonTemporalSameFeatures

try:
    from src.temporal import TemporalPrimary
except ImportError:
    class TemporalPrimary:
        name = "TEMPORAL_PRIMARY"
        requires_features = True

        def __init__(self, n_estimators: int = 200, seed: int = 42):
            self.n_estimators = n_estimators
            self.seed = seed
            self.model = None
            self.scaler = None
            self.feature_names = None

        def fit(self, train_rows, X_train, target):
            if X_train is None:
                raise ValueError("TEMPORAL_PRIMARY requires features")
            self.feature_names = list(X_train.columns)
            X = X_train[self.feature_names].fillna(0.0).values.astype(float)
            y = train_rows[target].values.astype(float)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=-1,
            )
            self.model.fit(X_scaled, y)

        def predict(self, test_rows, X_test):
            if X_test is None:
                raise ValueError("TEMPORAL_PRIMARY requires features")
            X = X_test[self.feature_names].fillna(0.0).values.astype(float)
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build_methods():
    return [
        *[cls() for cls in BASELINE_METHODS_WITHOUT_FEATURES],
        NonTemporalSameFeatures(seed=42),
        TemporalPrimary(seed=42),
    ]


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
    feature_builder = FakeFeatureBuilder(n_features=8, seed=42)
    result = run_protocol(
        methods,
        bench=bench,
        target=args.target,
        data_dir=args.data_dir,
        feature_builder=feature_builder,
    )

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
    print(f"Feature builder used: {result.manifest['feature_builder']}")

    if args.manifest:
        write_manifest(result, args.manifest)
        print(f"Manifest written to {args.manifest}")

    return 0 if hashes_ok and not result.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())