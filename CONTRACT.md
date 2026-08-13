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

That means the project already has a preliminary result table for the three feature-free
baselines. The remaining gates are a leakage-safe feature representation, two prespecified
feature-driven methods, the temporal-order negative control, and a frozen comparison with
paired uncertainty. None of those gates is replaced by beating one baseline point estimate.

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

The shared contract was merged to `origin/main` at
`7cf32b360bd2354dd2dfbfe09aa640986422c3fe`. Both student branches start from that exact
integration point and proceed in parallel. They do not add commits to the already merged
`pi/option-a-contract` branch.

| Owner | Branch | Files | Interface to implement |
|---|---|---|---|
| Nguyen Xuan Hoa | `student/hoa/option-a-features` | `src/preprocessing.ipynb`, `src/causal_discovery.py`, `src/features_student.py`, `tests/test_preprocessing_contract.py` | Subclass `FeatureBuilder`: `fit(train_rows, logs)` then `transform(rows, logs)`; implement the prespecified temporal-order negative control |
| Bao Yen | `student/yen/option-a-methods` | `src/methods_student.py`, the registration block in `run_option_a.py`, `tests/test_evaluation_metrics.py` | Implement continuous-outcome `Method` classes: `fit(train_rows, X_train, target)` then `predict(test_rows, X_test)` |
| PI/Codex | shared contract branch | `src/option_a/**`, `tests/test_option_a_contract.py` | Own and audit the shared layer; students do not edit it |

Nobody edits `src/option_a/` in a student PR. If it blocks either workstream, report the
same failing contract test to the PI; the shared layer is then changed once for everyone.
The two students cross-review each other's bounded PRs before PI merge approval.

## Why the interface has this shape

`fit` only receives training benchmark rows, and the protocol deep-copies a fresh builder
and fresh method from their unfitted prototypes in every fold. That makes fold-local state
the default. The leakage currently in the notebook is still prohibited:
`question_difficulty` and `student_cluster` are computed from `IsCorrect` across the whole
interaction table, and every `StandardScaler` is fitted on the whole table. These quantities
cannot enter a held-out treatment-construct prediction unless the data source, cutoff and
fold-local construction rule make them available before the applicable outcome.

The runtime guard in `assert_no_forbidden_inputs` rejects any feature matrix carrying the
A/B targets, the treatment and control user counts, or the checkout cell counts.

Interaction logs are filtered once, before fold construction. Only `Checkin`,
`CheckinRetry`, and `Lesson` rows are permitted as pre-outcome inputs. `Checkout` and
`CheckoutRetry` rows are outcomes from which `ate_k_1__` is calculated and must never reach
`FeatureBuilder`; an unknown `Type` stops the run for PI classification. Construct-keyed
statistics may use only the filtered pre-outcome table. Fold-locality does not rescue a
question-construct statistic computed from checkout rows because question constructs do
not align with the treatment-construct fold key. The run manifest records `n_logs_in` and
`n_logs_used`; on the frozen snapshot these must be 641,490 and 563,117 respectively.

Each student component must expose a JSON-serializable `get_config()` containing its
constructor settings. Toy or random features may appear only in deterministic unit-test
fixtures. They must not be used to select a method, tune hyperparameters or report results.

## Wiring in your work

Bao Yen, in the registration block of `run_option_a.py`:

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
Both feature-driven methods predict the continuous A/B target. The non-temporal comparator
must use the same permissible input information and the same model-selection budget as the
temporal method; the intended contrast is temporal order, not a classifier-versus-regressor
or unequal-feature comparison.

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

The valid out-of-fold feature-free reference is approximately MAE 0.134, and the target's
sample standard deviation is 0.190. A constant selected using all 88 target values gives an
in-sample MAE of approximately 0.132, but that value has seen the held-out targets and is a
descriptive quantity rather than a valid comparator.

The marginal intervals are wide because the evaluation has only 15 treatment-construct
clusters. Their width does not define a minimum required improvement. The prespecified
decision evidence is the paired cluster-bootstrap difference between `TEMPORAL_PRIMARY`
and each baseline, reported whether or not its interval contains zero.

Two reported figures are artifacts and must not be quoted as findings. `ZERO_EFFECT` scores
0.023 on sign agreement because only 2 of 88 rows have an exactly zero effect and exact zero
is its own class. `TRAIN_MEAN` scores -0.465 on Spearman because holding out a construct with
large effects lowers that fold's training mean, which induces anti-correlation mechanically.

## Falsification hooks

```bash
python run_option_a.py --placebo        # target permuted within year
```

On the one prespecified placebo run, `YEAR_STRATIFIED_MEAN` reaches MAE 0.1278. This is one
negative-control realization, not an estimated noise floor or a permutation distribution.
Once `TEMPORAL_PRIMARY` exists, comparable performance on the real and permuted targets
blocks a claim that the model reads signal specific to the published experimental target.

The negative control, independently circular-shifting within-student order before feature
extraction, needs the feature layer and belongs to Nguyen Xuan Hoa.

## A null result must remain visible

The protocol reports paired differences whether or not the interval excludes zero. An
unfavourable outcome does not license a change of method, sample or target. A clean
benchmark plus an honest negative finding can support a scientifically transparent report,
but it does not by itself establish FAIR submission readiness or acceptance potential.
