from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Method(Protocol):
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


class ZeroEffect:
    name = "ZERO_EFFECT"
    requires_features = False

    def fit(self, train_rows, X_train, target) -> None:
        pass

    def predict(self, test_rows, X_test) -> np.ndarray:
        return np.zeros(len(test_rows), dtype=float)


class TrainMean:
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