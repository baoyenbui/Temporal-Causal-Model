import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.option_a.methods import Method


class NonTemporalSameFeatures(Method):
    name = "NON_TEMPORAL_SAME_FEATURES"
    requires_features = True
    feature_variant = "shuffled"

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None
        self.scaler = None
        self.feature_names: Optional[List[str]] = None

    def get_config(self) -> Dict[str, Any]:
        return {
            "class": type(self).__name__,
            "n_estimators": self.n_estimators,
            "seed": self.seed,
            "feature_variant": self.feature_variant,
        }

    def fit(self, train_rows: pd.DataFrame, X_train: Optional[pd.DataFrame], target: str) -> None:
        if X_train is None:
            raise ValueError("NON_TEMPORAL_SAME_FEATURES requires features")
        self.feature_names = list(X_train.columns)
        X_raw = X_train[self.feature_names].fillna(0.0).values.astype(float)
        y = train_rows[target].values.astype(float)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

    def predict(self, test_rows: pd.DataFrame, X_test: Optional[pd.DataFrame]) -> np.ndarray:
        if X_test is None:
            raise ValueError("NON_TEMPORAL_SAME_FEATURES requires features")
        X_raw = X_test[self.feature_names].fillna(0.0).values.astype(float)
        X_scaled = self.scaler.transform(X_raw)
        return self.model.predict(X_scaled)