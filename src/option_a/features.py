"""Feature construction contract. Owned by Nguyen Xuan Hoa.

The protocol calls `fit` with training rows only, then `transform` for each side of the
fold separately. That shape is deliberate: it makes fold-local preprocessing the default
and makes the leakage currently in the notebook awkward to reproduce.

The bug this replaces: `question_difficulty` and `student_cluster` in
`src/preprocessing.ipynb` are computed from `IsCorrect` across the whole dataset, and every
`StandardScaler` is fitted on the whole dataset. Anything of that kind belongs inside `fit`.
"""

from typing import Optional

import pandas as pd

from src.option_a.benchmark import assert_no_forbidden_inputs


class FeatureBuilder:
    """Turn benchmark rows into a model-ready matrix.

    Subclass this and implement `fit` and `transform`. Both receive the interaction logs;
    what differs is that `fit` may only look at training rows.

    Rules the protocol enforces at runtime:

    - `transform` returns one row per input row, in the same order.
    - No column may come from `FORBIDDEN_MODEL_INPUTS`: the A/B targets, the treatment and
      control user counts, the checkout cell counts, or anything derived from them.
    - Every fitted quantity is estimated in `fit` from training rows only.

    One thing the protocol cannot check for you: a statistic computed from `logs` that
    touches a construct held out in the current fold. Interaction logs are pre-checkout, so
    using them is not automatically leakage, but any aggregate keyed by a construct must be
    recomputed per fold. Agree the exact rule with the PI before the frozen run, and write
    it down in CONTRACT.md rather than leaving it implicit in code.
    """

    name = "FeatureBuilder"

    def fit(self, train_rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> "FeatureBuilder":
        raise NotImplementedError(
            "FeatureBuilder.fit is Nguyen Xuan Hoa's slot. Fit every scaler, encoder and "
            "aggregate here, using train_rows only."
        )

    def transform(self, rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        raise NotImplementedError(
            "FeatureBuilder.transform is Nguyen Xuan Hoa's slot. Return one row per input "
            "row, in the same order, using only quantities fitted in fit()."
        )

    def check_output(self, features: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Validate a transform result. Call this at the end of your `transform`."""
        if len(features) != len(rows):
            raise ValueError(
                f"{self.name}.transform returned {len(features)} rows for {len(rows)} input rows; "
                f"predictions would be misaligned"
            )
        assert_no_forbidden_inputs(features, where=f"{self.name}.transform output")
        if features.isna().any().any():
            bad = features.columns[features.isna().any()].tolist()
            raise ValueError(
                f"{self.name}.transform produced nulls in {bad}. Impute inside fit/transform "
                f"so that the imputation value itself is fold-local."
            )
        return features


class NoFeatures(FeatureBuilder):
    """Placeholder used while only the three feature-free baselines are wired up."""

    name = "NoFeatures"

    def fit(self, train_rows, logs=None):
        return self

    def transform(self, rows, logs=None):
        return pd.DataFrame(index=range(len(rows)))
