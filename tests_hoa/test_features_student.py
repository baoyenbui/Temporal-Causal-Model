"""Tests for src/features_student.py (Nguyen Xuan Hoa's slot).

Lives outside tests/ deliberately: tests/ is owned by the PI per CONTRACT.md and is not to
be edited by students. This directory is discovered automatically by a bare
`python -m pytest -q` since the repo has no pytest.ini/pyproject.toml restricting
`testpaths`.
"""

import numpy as np
import pandas as pd
import pytest

from src.features_student import ConstructYearFeatures
from src.option_a.benchmark import FORBIDDEN_MODEL_INPUTS


def make_logs(n_users=20, n_constructs=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2022-02-01")
    for user in range(n_users):
        t = start + pd.Timedelta(days=int(rng.integers(0, 30)))
        for _ in range(rng.integers(5, 15)):
            construct = int(rng.integers(100, 100 + n_constructs))
            t = t + pd.Timedelta(seconds=int(rng.integers(5, 250)))
            rows.append(
                {
                    "UserId": user,
                    "ConstructId": construct,
                    "IsCorrect": int(rng.random() < (0.3 + 0.1 * (construct % 3))),
                    "Timestamp": t,
                }
            )
    return pd.DataFrame(rows)


def make_bench_rows(n=12, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment_constructs = [10, 11, 12]
    question_constructs = list(range(100, 106))
    years = [5, 6, 7, 8]
    return pd.DataFrame(
        {
            "TreatmentLessonConstructId": rng.choice(treatment_constructs, size=n),
            "QuestionConstructId": rng.choice(question_constructs, size=n),
            "Year": rng.choice(years, size=n),
            "ate_k_1__": rng.normal(size=n),
            "ate_p_1__": rng.normal(size=n),
            "ControlUsersCount": rng.integers(10, 100, size=n),
            "TreatmentUsersCount": rng.integers(10, 100, size=n),
        }
    )


@pytest.fixture
def logs():
    return make_logs()


@pytest.fixture
def bench_rows():
    return make_bench_rows()


def test_fit_returns_self_and_transform_shape_matches_input(logs, bench_rows):
    builder = ConstructYearFeatures()
    train = bench_rows.iloc[:8]
    test = bench_rows.iloc[8:]

    fitted = builder.fit(train, logs)
    assert fitted is builder  # fit returns self, per FeatureBuilder contract

    X_train = builder.transform(train, logs)
    X_test = builder.transform(test, logs)

    assert len(X_train) == len(train)
    assert len(X_test) == len(test)


def test_no_forbidden_columns_in_output(logs, bench_rows):
    builder = ConstructYearFeatures().fit(bench_rows, logs)
    features = builder.transform(bench_rows, logs)
    assert set(features.columns) & FORBIDDEN_MODEL_INPUTS == set()


def test_no_nulls_in_output(logs, bench_rows):
    builder = ConstructYearFeatures().fit(bench_rows, logs)
    features = builder.transform(bench_rows, logs)
    assert not features.isna().any().any()


def test_check_output_rejects_length_mismatch(logs, bench_rows):
    builder = ConstructYearFeatures().fit(bench_rows, logs)
    features = builder.transform(bench_rows, logs)
    with pytest.raises(ValueError, match="misaligned"):
        builder.check_output(features.iloc[:-1], bench_rows)


def test_unseen_question_construct_falls_back_without_nan(logs, bench_rows):
    """A QuestionConstructId with zero interactions in logs must not produce NaN."""
    builder = ConstructYearFeatures().fit(bench_rows, logs)
    novel = bench_rows.copy()
    novel["QuestionConstructId"] = 999999  # never appears in `logs`
    features = builder.transform(novel, logs)
    assert not features.isna().any().any()
    # falls back to the fitted global mean, so every row gets the identical fallback value
    assert features["q_construct_difficulty"].nunique() == 1


def test_unseen_year_in_test_gets_all_zero_year_columns(logs, bench_rows):
    train = bench_rows[bench_rows["Year"] != 8]
    test = bench_rows[bench_rows["Year"] == 8]
    if test.empty:
        pytest.skip("fixture did not sample Year 8; not a property of the code under test")

    builder = ConstructYearFeatures().fit(train, logs)
    features = builder.transform(test, logs)

    year_cols = [c for c in features.columns if c.startswith("year_")]
    assert "year_8" not in year_cols  # fit never saw Year 8
    assert not features.isna().any().any()
    assert (features[year_cols] == 0).all(axis=None)


def test_fit_ignores_test_rows(logs, bench_rows):
    """Fitting on the same train_rows must give identical output regardless of what the
    disjoint test rows look like -- fit must not have peeked at them."""
    train = bench_rows.iloc[:8]
    test_a = bench_rows.iloc[8:]
    test_b = test_a.copy()
    test_b["Year"] = 12345  # a value fit() never saw; should not affect the fitted state

    builder_a = ConstructYearFeatures().fit(train, logs)
    out_a = builder_a.transform(train, logs)

    builder_b = ConstructYearFeatures().fit(train, logs)
    _ = builder_b.transform(test_b, logs)  # transform on the weird test rows first
    out_b = builder_b.transform(train, logs)  # then re-transform the same train rows

    pd.testing.assert_frame_equal(out_a, out_b)


def test_transform_is_deterministic(logs, bench_rows):
    builder = ConstructYearFeatures().fit(bench_rows, logs)
    first = builder.transform(bench_rows, logs)
    second = builder.transform(bench_rows, logs)
    pd.testing.assert_frame_equal(first, second)


def test_fit_without_logs_raises(bench_rows):
    with pytest.raises(ValueError, match="logs"):
        ConstructYearFeatures().fit(bench_rows, logs=None)


def test_transform_before_fit_raises(logs, bench_rows):
    builder = ConstructYearFeatures()
    with pytest.raises(RuntimeError, match="before fit"):
        builder.transform(bench_rows, logs)
