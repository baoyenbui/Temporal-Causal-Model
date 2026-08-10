# Option A working contract

This branch exists so that the two student workstreams can run at the same time instead of
one waiting on the other. `src/option_a/` owns the evaluation sample, the folds, and the
metrics. Neither student needs to guess what the other will produce, because both code
against the interfaces below.

Base commit: `eb7cd1d4cafd9f0c0e4a6b2b6545296107efe269`.

## What is already done

The three baselines that need no features are implemented and tested. They run today:

```bash
python run_option_a.py --show-folds
python -m pytest tests/ -q
```

That means the project already has a result table. The one remaining question is whether a
feature-driven method beats it.

## What must not change

| Frozen | Value |
|---|---|
| Evaluation sample | All 88 rows of `construct_experiments_ates_test.csv`. No row is dropped for effect size, sign, model error, or reachability. |
| Primary target | `ate_k_1__` |
| Sensitivity target | `ate_p_1__`, never a replacement after results are seen |
| Folds | Leave-one-`TreatmentLessonConstructId`-out, 15 folds |
| Primary metric | Row-level MAE |
| Uncertainty | Cluster bootstrap over treatment construct, 2000 replicates, seed 42 |

Changing any of these needs a dated PI decision. If a test in `tests/` goes red, the change
is to the frozen design, not to the test.

## Who owns what

| Owner | Files | Interface to implement |
|---|---|---|
| Nguyen Xuan Hoa | `src/preprocessing.ipynb`, `src/causal_discovery.py`, `src/features_student.py` | Subclass `FeatureBuilder`: `fit(train_rows, logs)` then `transform(rows, logs)` |
| Bao Yen | `src/methods_student.py`, `run_option_a.py` registration block | Implement `Method`: `fit(train_rows, X_train, target)` then `predict(test_rows, X_test)` |
| PI | `src/option_a/**`, `tests/**` | Owns the shared layer; students do not edit it |

Nobody edits `src/option_a/`. If it blocks you, say so and it gets changed once, for
everyone, with the tests updated in the same move.

## Why the interface has this shape

`fit` only ever receives training rows. That is what makes fold-local preprocessing the
default rather than something to remember. The leakage currently in the notebook is hard to
reproduce through this interface: `question_difficulty` and `student_cluster` are computed
from `IsCorrect` across the whole dataset, and every `StandardScaler` is fitted on the whole
dataset, so a held-out row's own outcome reaches the model that scores it.

The runtime guard in `assert_no_forbidden_inputs` rejects any feature matrix carrying the
A/B targets, the treatment and control user counts, or the checkout cell counts.

One thing the guard cannot check: an aggregate computed from the interaction logs that is
keyed by a construct held out in the current fold. Logs are pre-checkout, so using them is
not automatically leakage, but any construct-keyed statistic must be recomputed per fold.
Agree that rule with the PI and record it here before the frozen run.

## Wiring in your work

Bao Yen, in `run_option_a.py`:

```python
from src.methods_student import NonTemporalSameFeatures, TemporalPrimary
from src.features_student import ConstructYearFeatures

def build_methods():
    return [cls() for cls in BASELINE_METHODS_WITHOUT_FEATURES] + [
        NonTemporalSameFeatures(),
        TemporalPrimary(),
    ]
```

then pass `feature_builder=ConstructYearFeatures()` and `logs=...` to `run_protocol`. Set
`requires_features = True` on any method that consumes `X_train` / `X_test`.

Once `TEMPORAL_PRIMARY` is registered, the paired-difference table populates itself.

## Known structural gap

`aggregate_time_series` in `src/causal_discovery.py:129` pools every student into shared
calendar-day buckets, so the current graph carries no construct and no year dimension.
Nothing in that pipeline can emit a prediction for a `(TreatmentLessonConstructId,
QuestionConstructId, Year)` row. `TEMPORAL_PRIMARY` needs a construct-keyed representation
before it can exist at all. This is the largest remaining piece of work and it is shared
between both students.

## Reading the current numbers

Preliminary, not a frozen run. Primary target, out of fold:

| Method | MAE | 95% CI |
|---|---|---|
| TRAIN_MEAN | 0.1345 | 0.0966 to 0.1848 |
| ZERO_EFFECT | 0.1365 | 0.1004 to 0.1865 |
| YEAR_STRATIFIED_MEAN | 0.1371 | 0.1000 to 0.1845 |

Three things follow, and all three matter more than the ranking:

The bar is roughly MAE 0.134, and the target's standard deviation is 0.190. Predicting a
single constant for all 88 rows gives 0.132. There is very little variance available to
explain.

The interval is about 0.09 wide, close to two thirds of the point estimate, because there
are only 15 resampling clusters. An improvement smaller than roughly 0.03 will not separate
from noise no matter how the method is built.

Two reported figures are artifacts and must not be quoted as findings. `ZERO_EFFECT` scores
0.023 on sign agreement because only 2 of 88 rows have an exactly zero effect and exact zero
is its own class. `TRAIN_MEAN` scores -0.465 on Spearman because holding out a construct with
large effects lowers that fold's training mean, which induces anti-correlation mechanically.

## Falsification hooks

```bash
python run_option_a.py --placebo        # target permuted within year
```

On the placebo run `YEAR_STRATIFIED_MEAN` reaches MAE 0.1278, better than any method achieves
on the real target. Read that as a noise-floor measurement rather than a defect: for
feature-free baselines, real and permuted targets are indistinguishable at this sample size.
The check becomes decisive once `TEMPORAL_PRIMARY` exists. If it scores comparably on
permuted targets, claims that the method reads experimental effect are blocked.

The negative control, independently circular-shifting within-student order before feature
extraction, needs the feature layer and belongs to Nguyen Xuan Hoa.

## A null result is a result

The protocol reports paired differences whether or not the interval excludes zero. An
unfavourable outcome does not license a change of method, sample, or target. A clean
benchmark plus an honest negative finding is publishable; a favourable number obtained by
moving the target is not.
