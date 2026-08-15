from typing import Optional

import numpy as np
import pandas as pd

from src.option_a.benchmark import assert_no_forbidden_inputs


class FeatureBuilder:
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
        if len(features) != len(rows):
            raise ValueError(
                f"{self.name}.transform returned {len(features)} rows for {len(rows)} input rows; "
                f"predictions would be misaligned"
            )
        if features.shape[1] == 0:
            raise ValueError(
                f"{self.name}.transform returned 0 columns for {len(rows)} input rows. "
                f"Check assert_no_forbidden_inputs in src/option_a/benchmark.py -- if it "
                f"filters columns instead of raising, it may be dropping every feature here."
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
    name = "NoFeatures"

    def fit(self, train_rows, logs=None):
        return self

    def transform(self, rows, logs=None):
        return pd.DataFrame(index=range(len(rows)))


class FakeFeatureBuilder(FeatureBuilder):
    name = "FakeFeatureBuilder"

    def __init__(self, n_features: int = 8, seed: int = 42):
        self.n_features = n_features
        self.seed = seed
        self.feature_names = [f"fake_f{i}" for i in range(n_features)]
        self._fitted = False

    def fit(self, train_rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> "FakeFeatureBuilder":
        self._fitted = True
        return self

    def transform(self, rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FakeFeatureBuilder.transform called before fit")

        rng = np.random.default_rng(self.seed + len(rows))
        data = rng.normal(size=(len(rows), self.n_features))
        features = pd.DataFrame(data, columns=self.feature_names)
        return self.check_output(features, rows)