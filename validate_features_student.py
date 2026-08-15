"""Throwaway validation harness for src/features_student.py. NOT part of the frozen
protocol, NOT the official TEMPORAL_PRIMARY method (that is Bao Yen's, in
src/methods_student.py + run_option_a.py's build_methods()).

This exists only to answer, with real data: does ConstructYearFeatures wire into
run_protocol without leakage errors, and roughly how far is a trivial model on top of it
from the TRAIN_MEAN baseline? Run it, read the number, then delete or ignore it -- it should
not be imported by anything under src/option_a/, tests/, or run_option_a.py.

    python validate_features_student.py
"""

import time

import pandas as pd
from sklearn.linear_model import Ridge

from src.features_student import ConstructYearFeatures
from src.option_a.benchmark import PRIMARY_TARGET, filter_pre_outcome, load_benchmark
from src.option_a.methods import BASELINE_METHODS_WITHOUT_FEATURES
from src.option_a.protocol import run_protocol


class RidgeOnConstructYearFeatures:
    """Minimal probe method, defined here (not in src/methods_student.py) so this stays
    entirely outside Bao Yen's and the PI's files."""

    name = "PROBE_RIDGE_CONSTRUCT_YEAR"
    requires_features = True

    def __init__(self) -> None:
        self.model = Ridge(alpha=1.0)

    def get_config(self) -> dict:
        return {"alpha": self.model.alpha}

    def fit(self, train_rows, X_train, target) -> None:
        self.model.fit(X_train.to_numpy(), train_rows[target].to_numpy())

    def predict(self, test_rows, X_test):
        return self.model.predict(X_test.to_numpy())


def main() -> None:
    print("Loading real logs (data/checkins_lessons_checkouts_training.csv)...")
    logs = pd.read_csv("data/checkins_lessons_checkouts_training.csv")

    n_before = len(logs)
    logs = filter_pre_outcome(logs)
    n_after = len(logs)
    print(f"filter_pre_outcome: {n_before} rows in, {n_after} pre-outcome rows kept "
          f"({n_before - n_after} outcome/other rows dropped)")
    print(f"Type values remaining: {sorted(logs['Type'].unique())}")

    bench = load_benchmark(data_dir="data")

    methods = [cls() for cls in BASELINE_METHODS_WITHOUT_FEATURES] + [RidgeOnConstructYearFeatures()]

    digests = []
    for run_i in range(2):
        start = time.time()
        result = run_protocol(
            methods,
            bench=bench,
            feature_builder=ConstructYearFeatures(),
            logs=logs,
            target=PRIMARY_TARGET,
        )
        elapsed = time.time() - start
        digests.append(result.manifest["predictions_sha256"])

        print(f"\n=== Run {run_i + 1} (elapsed {elapsed:.1f}s) ===")
        print(result.results_table.to_string(index=False))
        if result.failures:
            print(f"\n{len(result.failures)} failure(s):")
            print(pd.DataFrame(result.failures).to_string(index=False))
        print(f"predictions_sha256: {result.manifest['predictions_sha256']}")

    print(f"\npredictions_sha256 identical across the 2 runs: {digests[0] == digests[1]}")


if __name__ == "__main__":
    main()
