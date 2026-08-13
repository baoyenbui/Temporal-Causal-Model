"""The method interface, plus the three baselines that need no features at all.

`ZERO_EFFECT`, `TRAIN_MEAN` and `YEAR_STRATIFIED_MEAN` depend only on the 88-row table, so
they are implemented here and are runnable before either student's work lands. That gives
the project a real results table on day one and leaves exactly one open question: whether
a feature-driven method beats them.

`NON_TEMPORAL_SAME_FEATURES` and `TEMPORAL_PRIMARY` are Bao Yen's, in `src/methods_student.py`.
"""

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Method(Protocol):
    """Contract every Option A method implements.

    `fit` sees training rows only. `predict` sees the held-out rows only. Anything learned
    from the data — a mean, a scaler, an encoder, a difficulty table — is estimated inside
    `fit`, so a held-out treatment construct cannot influence its own prediction.

    Set `requires_features = True` to receive `X_train` / `X_test` from the FeatureBuilder.
    Doing so also makes `get_config()` mandatory: the protocol refuses to run a
    feature-driven method whose configuration it cannot record, because a manifest that
    omits the configuration cannot reproduce the run. Return a small JSON-serializable dict
    of the settings chosen before fitting, never fitted state:

        def get_config(self):
            return {"model": "ridge", "alpha": 1.0}

    A scikit-learn-style `get_params(deep=False)` is accepted instead. The three
    feature-free baselines below are exempt, since they have nothing to configure.
    """

    name: str
    requires_features: bool

    def fit(
        self,
        train_rows: pd.DataFrame,
        X_train: Optional[pd.DataFrame],
        target: str,
    ) -> None: ...

    def predict(
        self,
        test_rows: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
    ) -> np.ndarray: ...

    def get_config(self) -> dict:
        """Required when `requires_features` is True; see the class docstring."""
        ...


class ZeroEffect:
    """Predict no effect for every row.

    The reference point that matters most: a method that cannot beat it has not shown that
    it carries any information about effect magnitude.
    """

    name = "ZERO_EFFECT"
    requires_features = False

    def fit(self, train_rows, X_train, target) -> None:
        pass

    def predict(self, test_rows, X_test) -> np.ndarray:
        return np.zeros(len(test_rows), dtype=float)


class TrainMean:
    """Predict the training-fold mean of the target."""

    name = "TRAIN_MEAN"
    requires_features = False

    def __init__(self) -> None:
        self.value_: Optional[float] = None

    def fit(self, train_rows, X_train, target) -> None:
        self.value_ = float(train_rows[target].mean())

    def predict(self, test_rows, X_test) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("TRAIN_MEAN.predict called before fit")
        return np.full(len(test_rows), self.value_, dtype=float)


class YearStratifiedMean:
    """Predict the training-fold mean for the row's year, falling back to the global mean.

    Year 5 has 2 rows and Year 10 has 9 in the full sample, so some folds leave a year with
    no training support. Every fallback is counted rather than hidden, because a baseline
    that is silently the global mean most of the time is not a year-stratified baseline.
    """

    name = "YEAR_STRATIFIED_MEAN"
    requires_features = False

    def __init__(self) -> None:
        self.year_means_: Optional[dict] = None
        self.global_mean_: Optional[float] = None
        self.n_fallback = 0
        self.fallback_years: list = []

    def fit(self, train_rows, X_train, target) -> None:
        self.year_means_ = train_rows.groupby("Year")[target].mean().to_dict()
        self.global_mean_ = float(train_rows[target].mean())

    def predict(self, test_rows, X_test) -> np.ndarray:
        if self.year_means_ is None or self.global_mean_ is None:
            raise RuntimeError("YEAR_STRATIFIED_MEAN.predict called before fit")

        predictions = []
        for year in test_rows["Year"].to_numpy():
            if year in self.year_means_:
                predictions.append(float(self.year_means_[year]))
            else:
                self.n_fallback += 1
                self.fallback_years.append(int(year))
                predictions.append(self.global_mean_)
        return np.asarray(predictions, dtype=float)


BASELINE_METHODS_WITHOUT_FEATURES = (ZeroEffect, TrainMean, YearStratifiedMean)
