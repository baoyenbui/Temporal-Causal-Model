import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.option_a.methods import Method

try:
    from lingam import DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False


TARGET_SENTINEL = "__TARGET__"


class NonTemporalSameFeatures(Method):
    name = "NON_TEMPORAL_SAME_FEATURES"
    requires_features = True

    def __init__(
        self,
        n_bootstrap: int = 20,
        stability_threshold: float = 0.6,
        n_estimators: int = 200,
        use_lingam: bool = True,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.n_estimators = n_estimators
        self.use_lingam = use_lingam and LINGAM_AVAILABLE
        self.seed = seed

        self.model = None
        self.scaler = None
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.stable_edges: Optional[pd.DataFrame] = None
        self.all_edges: Optional[pd.DataFrame] = None

    def _bootstrap_directlingam(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        empty = pd.DataFrame(
            columns=["source", "target", "stability", "mean_strength", "std_strength", "n_occurrences"]
        )
        if not self.use_lingam or len(names) < 2 or len(X) < 10:
            return empty, empty

        rng = np.random.default_rng(self.seed)
        n = len(X)
        edge_counts: Dict[Tuple[str, str], int] = {}
        edge_strengths: Dict[Tuple[str, str], list] = {}
        successful = 0

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, n)
            X_boot = X[idx]
            try:
                model = DirectLiNGAM()
                model.fit(X_boot)
            except Exception:
                continue

            successful += 1
            adj = model.adjacency_matrix_

            for i_row in range(adj.shape[0]):
                for j_col in range(adj.shape[1]):
                    if i_row == j_col:
                        continue
                    w = adj[i_row, j_col]
                    if abs(w) > 1e-6:
                        key = (names[j_col], names[i_row])
                        edge_counts[key] = edge_counts.get(key, 0) + 1
                        edge_strengths.setdefault(key, []).append(float(w))

        if successful == 0:
            return empty, empty

        rows = []
        for key, count in edge_counts.items():
            source, target = key
            rows.append({
                "source": source,
                "target": target,
                "stability": count / successful,
                "mean_strength": float(np.mean(edge_strengths[key])),
                "std_strength": float(np.std(edge_strengths[key])),
                "n_occurrences": count,
            })

        edges_df = pd.DataFrame(rows)
        if len(edges_df) == 0:
            return edges_df, edges_df

        edges_df = edges_df.sort_values("stability", ascending=False).reset_index(drop=True)
        stable_edges_df = edges_df[edges_df["stability"] >= self.stability_threshold].reset_index(drop=True)
        return edges_df, stable_edges_df

    def fit(self, train_rows: pd.DataFrame, X_train: Optional[pd.DataFrame], target: str) -> None:
        if X_train is None:
            raise ValueError("NON_TEMPORAL_SAME_FEATURES requires features")

        self.feature_names = list(X_train.columns)
        X_raw = X_train[self.feature_names].fillna(0.0).values.astype(float)
        y = train_rows[target].values.astype(float)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        combined_names = self.feature_names + [TARGET_SENTINEL]
        combined_X = np.column_stack([X_scaled, y_scaled])

        self.all_edges, self.stable_edges = self._bootstrap_directlingam(combined_X, combined_names)

        causal_parents: List[str] = []
        if self.stable_edges is not None and len(self.stable_edges):
            causal_parents = self.stable_edges.loc[
                self.stable_edges["target"] == TARGET_SENTINEL, "source"
            ].tolist()

        if causal_parents:
            self.selected_features = causal_parents
        else:
            self.selected_features = list(self.feature_names)
            print(f"NonTemporalSameFeatures: no stable DirectLiNGAM edge into target at "
                  f"stability_threshold={self.stability_threshold}; falling back to all "
                  f"{len(self.feature_names)} feature(s).")

        print(f"NonTemporalSameFeatures: using {len(self.selected_features)}/{len(self.feature_names)} "
              f"feature(s) selected as stable DirectLiNGAM causal parents of target: {self.selected_features}")

        selected_idx = [self.feature_names.index(f) for f in self.selected_features]
        X_selected = X_scaled[:, selected_idx]

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X_selected, y)

    def predict(self, test_rows: pd.DataFrame, X_test: Optional[pd.DataFrame]) -> np.ndarray:
        if X_test is None:
            raise ValueError("NON_TEMPORAL_SAME_FEATURES requires features")

        X_raw = X_test[self.feature_names].fillna(0.0).values.astype(float)
        X_scaled = self.scaler.transform(X_raw)
        selected_idx = [self.feature_names.index(f) for f in self.selected_features]
        X_selected = X_scaled[:, selected_idx]
        return self.model.predict(X_selected)