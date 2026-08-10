"""Contract tests for the Option A shared layer.

These are the tests both students code against. If a change here goes red, the change is
to the frozen design and needs a dated PI decision, not a fix to the test.

    python -m pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.option_a.benchmark import (
    EXPECTED_COLUMNS,
    FORBIDDEN_MODEL_INPUTS,
    N_BENCHMARK_ROWS,
    PRIMARY_TARGET,
    SENSITIVITY_TARGET,
    assert_no_forbidden_inputs,
    load_benchmark,
    parse_control_lessons,
    verify_input_hashes,
)
from src.option_a.folds import FOLD_KEY, assert_folds_partition, leave_one_treatment_out
from src.option_a.methods import ZeroEffect, TrainMean, YearStratifiedMean
from src.option_a.metrics import (
    cluster_bootstrap_ci,
    mae,
    paired_difference_ci,
    rmse,
    sign_agreement,
    sign_class,
    spearman,
)
from src.option_a.protocol import placebo_targets, run_protocol

DATA_DIR = "data"


@pytest.fixture(scope="module")
def bench():
    return load_benchmark(data_dir=DATA_DIR)


# --- sample contract -------------------------------------------------------------------

def test_benchmark_has_exactly_88_rows(bench):
    assert len(bench) == N_BENCHMARK_ROWS


def test_benchmark_schema_is_frozen(bench):
    for column in EXPECTED_COLUMNS:
        assert column in bench.columns


def test_both_targets_are_complete(bench):
    assert not bench[PRIMARY_TARGET].isna().any()
    assert not bench[SENSITIVITY_TARGET].isna().any()


def test_fifteen_treatment_constructs(bench):
    assert bench[FOLD_KEY].nunique() == 15


def test_input_hashes_match_frozen_snapshot():
    report = verify_input_hashes(DATA_DIR)
    mismatched = [name for name, entry in report.items() if not entry["match"]]
    assert mismatched == [], f"data files differ from the frozen snapshot: {mismatched}"


def test_multi_control_rows_keep_every_control_construct():
    assert parse_control_lessons("{2930,2931}") == [2930, 2931]
    assert parse_control_lessons("{3119}") == [3119]


def test_empty_comparator_is_a_failure_not_an_empty_list():
    with pytest.raises(ValueError):
        parse_control_lessons("{}")


def test_wrong_row_count_fails_closed(tmp_path, bench):
    short = bench.head(50)[EXPECTED_COLUMNS]
    path = tmp_path / "construct_experiments_ates_test.csv"
    short.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 88"):
        load_benchmark(data_dir=str(tmp_path), strict_hash=False)


def test_alias_column_fails_closed(tmp_path, bench):
    aliased = bench[EXPECTED_COLUMNS].rename(columns={"ate_k_1__": "ate_k_1"})
    path = tmp_path / "construct_experiments_ates_test.csv"
    aliased.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_benchmark(data_dir=str(tmp_path), strict_hash=False)


def test_forbidden_inputs_are_rejected():
    leaky = pd.DataFrame({"ate_k_1__": [0.1], "some_feature": [1.0]})
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_inputs(leaky)

    clean = pd.DataFrame({"some_feature": [1.0]})
    assert_no_forbidden_inputs(clean)


def test_user_counts_are_forbidden_inputs():
    assert "ControlUsersCount" in FORBIDDEN_MODEL_INPUTS
    assert "TreatmentUsersCount" in FORBIDDEN_MODEL_INPUTS


# --- folds -----------------------------------------------------------------------------

def test_fifteen_folds(bench):
    assert len(leave_one_treatment_out(bench)) == 15


def test_folds_partition_every_row_exactly_once(bench):
    folds = leave_one_treatment_out(bench)
    assert_folds_partition(folds, len(bench))
    assert sum(f.n_test for f in folds) == N_BENCHMARK_ROWS


def test_held_out_construct_never_appears_in_its_own_training_set(bench):
    for fold in leave_one_treatment_out(bench):
        train_constructs = set(bench.iloc[fold.train_idx][FOLD_KEY])
        assert fold.held_out_construct not in train_constructs


def test_folds_are_deterministic(bench):
    first = leave_one_treatment_out(bench)
    second = leave_one_treatment_out(bench)
    for a, b in zip(first, second):
        assert a.name == b.name
        assert np.array_equal(a.test_idx, b.test_idx)


def test_fold_sizes_are_small_and_uneven(bench):
    sizes = sorted(f.n_test for f in leave_one_treatment_out(bench))
    assert sizes[0] == 3 and sizes[-1] == 13


# --- metrics ---------------------------------------------------------------------------

def test_mae_and_rmse_are_zero_for_perfect_prediction():
    y = np.array([0.1, -0.2, 0.0, 0.5])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_mae_is_the_mean_absolute_difference():
    assert mae([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)


def test_exact_zero_is_its_own_sign_class():
    assert list(sign_class([-0.3, 0.0, 0.4])) == [-1.0, 0.0, 1.0]


def test_tiny_nonzero_value_is_not_absorbed_into_the_zero_class():
    assert sign_class([1e-12])[0] == 1.0


def test_sign_agreement_counts_all_three_classes():
    assert sign_agreement([-1.0, 0.0, 1.0], [-0.5, 0.0, 0.2]) == pytest.approx(1.0)
    assert sign_agreement([-1.0, 0.0, 1.0], [0.5, 0.0, 0.2]) == pytest.approx(2 / 3)


def test_spearman_is_nan_for_a_constant_prediction():
    assert np.isnan(spearman([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]))


def test_cluster_bootstrap_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    p = y + rng.normal(scale=0.2, size=60)
    clusters = np.repeat(np.arange(15), 4)
    assert cluster_bootstrap_ci(y, p, clusters) == cluster_bootstrap_ci(y, p, clusters)


def test_cluster_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    p = y + rng.normal(scale=0.3, size=60)
    clusters = np.repeat(np.arange(15), 4)
    point, lo, hi = cluster_bootstrap_ci(y, p, clusters)
    assert lo <= point <= hi


def test_paired_difference_is_negative_when_the_first_method_is_better():
    y = np.array([1.0, 2.0, 3.0, 4.0] * 4)
    good = y + 0.05
    poor = y + 0.9
    clusters = np.repeat(np.arange(4), 4)
    point, lo, hi = paired_difference_ci(y, good, poor, clusters)
    assert point < 0 and hi < 0


def test_degenerate_single_cluster_still_returns_a_number():
    y = np.array([0.1, 0.2, 0.3])
    point, lo, hi = cluster_bootstrap_ci(y, y, np.zeros(3))
    assert point == 0.0 and lo == 0.0 and hi == 0.0


# --- baselines and protocol ------------------------------------------------------------

def test_zero_effect_predicts_zero(bench):
    method = ZeroEffect()
    method.fit(bench, None, PRIMARY_TARGET)
    assert np.all(method.predict(bench, None) == 0.0)


def test_train_mean_uses_training_rows_only(bench):
    train, test = bench.iloc[:60], bench.iloc[60:]
    method = TrainMean()
    method.fit(train, None, PRIMARY_TARGET)
    assert method.predict(test, None)[0] == pytest.approx(train[PRIMARY_TARGET].mean())


def test_predicting_before_fitting_raises(bench):
    with pytest.raises(RuntimeError):
        TrainMean().predict(bench, None)


def test_year_stratified_mean_falls_back_and_counts_it(bench):
    train = bench[bench["Year"] != 5]
    test = bench[bench["Year"] == 5]
    method = YearStratifiedMean()
    method.fit(train, None, PRIMARY_TARGET)
    predicted = method.predict(test, None)
    assert method.n_fallback == len(test)
    assert predicted[0] == pytest.approx(train[PRIMARY_TARGET].mean())


def test_protocol_predicts_every_row_for_every_baseline(bench):
    result = run_protocol(
        [ZeroEffect(), TrainMean(), YearStratifiedMean()], bench=bench, data_dir=DATA_DIR
    )
    assert not result.failures
    assert not result.predictions.isna().any().any()
    assert (result.results_table["n_predicted"] == N_BENCHMARK_ROWS).all()


def test_protocol_is_reproducible(bench):
    a = run_protocol([ZeroEffect(), TrainMean()], bench=bench, data_dir=DATA_DIR)
    b = run_protocol([ZeroEffect(), TrainMean()], bench=bench, data_dir=DATA_DIR)
    assert a.manifest["predictions_sha256"] == b.manifest["predictions_sha256"]
    assert a.manifest["results_sha256"] == b.manifest["results_sha256"]


def test_a_failing_method_is_recorded_not_swallowed(bench):
    class Broken:
        name = "BROKEN"
        requires_features = False

        def fit(self, train_rows, X_train, target):
            raise RuntimeError("deliberate failure")

        def predict(self, test_rows, X_test):
            return np.zeros(len(test_rows))

    result = run_protocol([ZeroEffect(), Broken()], bench=bench, data_dir=DATA_DIR)
    assert len(result.failures) == 15
    assert result.predictions["BROKEN"].isna().all()
    assert result.predictions["ZERO_EFFECT"].notna().all()
    broken_row = result.results_table.set_index("method").loc["BROKEN"]
    assert broken_row["n_missing"] == N_BENCHMARK_ROWS


def test_a_method_returning_the_wrong_length_is_a_failure(bench):
    class WrongLength:
        name = "WRONG_LENGTH"
        requires_features = False

        def fit(self, train_rows, X_train, target):
            pass

        def predict(self, test_rows, X_test):
            return np.zeros(len(test_rows) + 1)

    result = run_protocol([WrongLength()], bench=bench, data_dir=DATA_DIR)
    assert len(result.failures) == 15


def test_placebo_preserves_the_within_year_value_multiset(bench):
    permuted = placebo_targets(bench)
    for year in bench["Year"].unique():
        original = sorted(bench[bench["Year"] == year][PRIMARY_TARGET])
        shuffled = sorted(permuted[permuted["Year"] == year][PRIMARY_TARGET])
        assert original == pytest.approx(shuffled)


def test_manifest_records_what_is_needed_to_reproduce_the_run(bench):
    result = run_protocol([ZeroEffect()], bench=bench, data_dir=DATA_DIR)
    for key in (
        "commit",
        "target",
        "n_folds",
        "methods",
        "bootstrap",
        "input_hashes",
        "predictions_sha256",
    ):
        assert key in result.manifest
    assert result.manifest["n_folds"] == 15
    assert result.manifest["input_hashes_all_match"] is True
