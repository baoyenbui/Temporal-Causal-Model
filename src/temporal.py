import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.causal_discovery import Layer1TemporalConstruction, Layer2StructureLearning, COUNT_COLUMNS
from src.option_a.benchmark import PRIMARY_TARGET, FORBIDDEN_MODEL_INPUTS, assert_no_outcome_rows
from src.option_a.features import FeatureBuilder
from src.option_a.folds import FOLD_KEY
from src.option_a.methods import Method


FEATURE_LABELS = {
    "avg_success": "Accuracy Rate",
    "success_trend": "Trending Up or Down",
    "avg_difficulty": "Question Difficulty",
    "difficulty_std": "Difficulty Swings",
    "difficulty_range": "Difficulty Gap",
    "recent5_correct_rate": "Recent Accuracy",
    "max_streak": "Longest Correct Streak",
    "attempts_mean": "Average Tries per Question",
    "attempts_max": "Most Tries on One Question",
    "pct_multi_attempt": "Retry Rate",
    "avg_response_time": "Response Time (Average Pace)",
    "response_time_std": "Response Time Swings",
    "median_response_time": "Response Time (Typical Pace)",
}

MIN_STABILITY = 0.30
HIGH_STABILITY = 0.45
MIN_STRENGTH = 0.12

PRE_OUTCOME_LOG_TYPES = {"Checkin", "CheckinRetry", "Lesson"}


def circular_shift_logs_per_user(logs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in logs.groupby("UserId", sort=False):
        g = g.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        n = len(g)
        if n > 1:
            shift = int(rng.integers(1, n))
            shifted = g.iloc[np.roll(np.arange(n), shift)].reset_index(drop=True)
            shifted["Timestamp"] = g["Timestamp"].to_numpy()
            parts.append(shifted)
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else logs.iloc[0:0].copy()


def _build_numeric_columns(base_columns: List[str], prefixes: Tuple[str, ...]) -> List[str]:
    return [f"{prefix}_{col}" for col in base_columns for prefix in prefixes]


class TemporalFeatureBuilder(FeatureBuilder):
    KEY_COLUMNS = ["TreatmentLessonConstructId", "QuestionConstructId", "Year"]
    LOOKUP_KEY_COLUMNS = ["QuestionConstructId", "Year"]
    BASE_COLUMNS = [
        "success_rate",
        "success_trend",
    ]
    SPLIT_PREFIXES = ("recent", "historical", "delta")
    NUMERIC_COLUMNS = _build_numeric_columns(BASE_COLUMNS, SPLIT_PREFIXES)

    def __init__(
        self,
        preserve_order: bool = True,
        seed: int = 42,
        window_size: int = 50,
        step_size: int = 50,
        max_response_time_seconds: float = 300.0,
    ):
        self.preserve_order = preserve_order
        self.seed = seed
        self.name = "TEMPORAL_FEATURES" if preserve_order else "SHUFFLED_FEATURES"
        self.window_size = window_size
        self.step_size = step_size
        self.max_response_time_seconds = max_response_time_seconds

        self.key_features_: Optional[Dict[Tuple, Dict[str, float]]] = None
        self.question_fallback_: Optional[Dict[Any, Dict[str, float]]] = None
        self.global_fallback_: Optional[Dict[str, float]] = None
        self.scaler_: Optional[StandardScaler] = None

    def get_config(self) -> Dict[str, Any]:
        return {
            "class": type(self).__name__,
            "preserve_order": self.preserve_order,
            "seed": self.seed,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "max_response_time_seconds": self.max_response_time_seconds,
            "lookup_key": self.LOOKUP_KEY_COLUMNS,
            "base_columns": self.BASE_COLUMNS,
        }

    def _split_recent_historical(self, g: pd.DataFrame) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if "window_start_time" not in g.columns or g["window_start_time"].isna().all():
            return stats
        median_time = g["window_start_time"].median()
        recent_mask = g["window_start_time"] > median_time
        historical_mask = ~recent_mask
        for col in self.BASE_COLUMNS:
            if col not in g.columns:
                continue
            recent_series = g.loc[recent_mask, col].dropna()
            historical_series = g.loc[historical_mask, col].dropna()
            recent_val = float(recent_series.mean()) if len(recent_series) else None
            historical_val = float(historical_series.mean()) if len(historical_series) else None
            if recent_val is not None:
                stats[f"recent_{col}"] = recent_val
            if historical_val is not None:
                stats[f"historical_{col}"] = historical_val
            if recent_val is not None and historical_val is not None:
                stats[f"delta_{col}"] = recent_val - historical_val
        return stats

    def fit(self, train_rows: pd.DataFrame, logs: Optional[pd.DataFrame]) -> "TemporalFeatureBuilder":
        if logs is None:
            raise ValueError(f"{self.name}.fit requires interaction logs (logs=None)")
        assert_no_outcome_rows(logs, where=f"{self.name}.fit logs")
        missing = set(self.KEY_COLUMNS) - set(train_rows.columns)
        if missing:
            raise ValueError(f"{self.name}.fit: train_rows missing {sorted(missing)}")

        event_logs = logs if self.preserve_order else circular_shift_logs_per_user(logs, seed=self.seed)

        layer1 = Layer1TemporalConstruction(
            window_size=self.window_size,
            step_size=self.step_size,
            max_response_time_seconds=self.max_response_time_seconds,
        )
        layer1_df = layer1.build_layer1(event_logs)

        mapping_df = build_construct_to_experiment_mapping(
            train_rows[self.KEY_COLUMNS].drop_duplicates()
        )

        self.key_features_ = {}
        self.question_fallback_ = {}
        if len(layer1_df) > 0:
            layer2 = Layer2StructureLearning()
            keyed = layer2.attach_experimental_keys(
                layer1_df,
                mapping_df,
                construct_col_in_layer1="dominant_construct",
                construct_col_in_mapping="ConstructId",
            )
            if "ambiguous_construct_mapping" in keyed.columns:
                n_ambiguous = int(keyed["ambiguous_construct_mapping"].sum())
                if n_ambiguous:
                    print(
                        f"{self.name}.fit: dropping {n_ambiguous}/{len(keyed)} keyed rows flagged "
                        f"ambiguous_construct_mapping before aggregating key/question statistics"
                    )
                keyed = keyed[~keyed["ambiguous_construct_mapping"]]
            if len(keyed) == 0:
                raise ValueError(
                    f"{self.name}.fit: no unambiguous keyed rows remain after filtering; "
                    f"cannot build key_features_ or question_fallback_"
                )
            if set(self.LOOKUP_KEY_COLUMNS).issubset(keyed.columns):
                for key, g in keyed.groupby(self.LOOKUP_KEY_COLUMNS, sort=False):
                    stats = self._split_recent_historical(g)
                    if stats:
                        self.key_features_[tuple(key)] = stats
            if "QuestionConstructId" in keyed.columns:
                for q, g in keyed.groupby("QuestionConstructId", sort=False):
                    stats = self._split_recent_historical(g)
                    if stats:
                        self.question_fallback_[q] = stats

        n_train_rows = len(train_rows)
        n_key_covered = sum(
            1
            for _, row in train_rows[self.LOOKUP_KEY_COLUMNS].iterrows()
            if (row["QuestionConstructId"], row["Year"]) in self.key_features_
        )
        print(
            f"{self.name}.fit: {len(self.key_features_)} (QuestionConstructId, Year) key(s) with direct stats, "
            f"{len(self.question_fallback_)} QuestionConstructId fallback(s); "
            f"{n_key_covered}/{n_train_rows} train rows have a direct key match"
        )

        if self.key_features_:
            df = pd.DataFrame(self.key_features_.values())
            print(f"{self.name} feature variance:\n{df.var()}")

        pooled_stats = list(self.key_features_.values()) + list(self.question_fallback_.values())
        if pooled_stats:
            fallback_df = pd.DataFrame(pooled_stats)
            self.global_fallback_ = {
                col: float(fallback_df[col].mean())
                if col in fallback_df.columns and fallback_df[col].notna().any()
                else 0.0
                for col in self.NUMERIC_COLUMNS
            }
        else:
            self.global_fallback_ = {col: 0.0 for col in self.NUMERIC_COLUMNS}

        numeric = self._numeric_features(train_rows).fillna(0.0)
        self.scaler_ = StandardScaler().fit(numeric.to_numpy())
        return self

    def transform(self, rows: pd.DataFrame, logs: Optional[pd.DataFrame]) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError(f"{self.name}.transform called before fit()")
        missing = set(self.KEY_COLUMNS) - set(rows.columns)
        if missing:
            raise ValueError(f"{self.name}.transform: rows missing {sorted(missing)}")

        numeric = self._numeric_features(rows).fillna(0.0)
        scaled = self.scaler_.transform(numeric.to_numpy())
        features = pd.DataFrame(scaled, columns=self.NUMERIC_COLUMNS, index=rows.index)
        features = features.reset_index(drop=True)
        return self.check_output(features, rows)

    def _numeric_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in rows[self.LOOKUP_KEY_COLUMNS].iterrows():
            q, year = row["QuestionConstructId"], row["Year"]
            stats = (self.key_features_ or {}).get((q, year))
            if stats is None:
                stats = (self.question_fallback_ or {}).get(q, {})
            record = {col: stats.get(col, self.global_fallback_[col]) for col in self.NUMERIC_COLUMNS}
            records.append(record)
        return pd.DataFrame(records, index=rows.index)[self.NUMERIC_COLUMNS]

    def coverage_report(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.key_features_ is None and self.question_fallback_ is None:
            raise RuntimeError(f"{self.name}.coverage_report called before fit()")

        records = []
        for idx, row in rows[self.LOOKUP_KEY_COLUMNS].iterrows():
            q, year = row["QuestionConstructId"], row["Year"]
            if (q, year) in (self.key_features_ or {}):
                level = "key_year"
            elif q in (self.question_fallback_ or {}):
                level = "question_only"
            else:
                level = "global"
            records.append(
                {
                    "row_index": idx,
                    "QuestionConstructId": q,
                    "Year": year,
                    "coverage_level": level,
                }
            )
        df = pd.DataFrame(records)
        summary = (
            df["coverage_level"]
            .value_counts()
            .reindex(["key_year", "question_only", "global"], fill_value=0)
        )
        print(
            f"{self.name} coverage: key_year={int(summary['key_year'])}, "
            f"question_only={int(summary['question_only'])}, "
            f"global={int(summary['global'])} (total={len(df)})"
        )
        return df


def default_feature_builders(seed: int = 42) -> Dict[str, FeatureBuilder]:
    return {
        "temporal": TemporalFeatureBuilder(preserve_order=True, seed=seed),
        "shuffled": TemporalFeatureBuilder(preserve_order=False, seed=seed + 1),
    }


class TemporalPrimary(Method):
    name = "TEMPORAL_PRIMARY"
    requires_features = True
    feature_variant = "temporal"

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
            raise ValueError("TEMPORAL_PRIMARY requires features")
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
            raise ValueError("TEMPORAL_PRIMARY requires features")
        X_raw = X_test[self.feature_names].fillna(0.0).values.astype(float)
        X_scaled = self.scaler.transform(X_raw)
        return self.model.predict(X_scaled)


def humanize_feature_name(name: str) -> str:
    match = re.match(r"^CTRL_cluster_(\d+)_ratio$", name)
    if match:
        return f"Learner Group #{match.group(1)} Share"
    return FEATURE_LABELS.get(name, name)


def filter_pre_outcome_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    if "Type" not in raw_logs.columns:
        raise ValueError(
            "filter_pre_outcome_logs: raw_logs has no 'Type' column. "
            "Check preprocessing.ipynb / clean_main_df."
        )
    before = len(raw_logs)
    present_types = set(raw_logs["Type"].dropna().unique())
    unknown_types = present_types - PRE_OUTCOME_LOG_TYPES - {"Checkout", "CheckoutRetry"}
    if unknown_types:
        print(f"NOTE: unrecognized Type value(s) {unknown_types} dropped.")
    filtered = raw_logs[raw_logs["Type"].isin(PRE_OUTCOME_LOG_TYPES)].copy()
    after = len(filtered)
    print(
        f"filter_pre_outcome_logs: {before} -> {after} pre-outcome logs "
        f"({before - after} dropped)"
    )
    if after == 0:
        raise ValueError("filter_pre_outcome_logs dropped everything.")
    return filtered


def filter_graph_by_stability(
    causal_graph: dict,
    edge_frequency: dict,
    min_stability: float = MIN_STABILITY,
    min_strength: float = MIN_STRENGTH,
) -> dict:
    filtered = {target: [] for target in causal_graph}
    for target, edges in causal_graph.items():
        for edge in edges:
            key = (edge["source"], target, edge["lag"])
            stab = edge_frequency.get(key, 0.0)
            strength = abs(edge.get("strength", 0.0))
            if stab >= min_stability and strength >= min_strength:
                new_edge = dict(edge)
                new_edge["stability"] = stab
                filtered[target].append(new_edge)
    return {t: e for t, e in filtered.items() if e}


def plot_causal_graph(
    causal_graph: dict,
    output_path: str = "output/causal_graph.png",
    high_stability: float = HIGH_STABILITY,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    G = nx.MultiDiGraph()
    for target, sources in causal_graph.items():
        for edge in sources:
            strength = edge.get("strength", 0.0)
            stab = edge.get("stability", 0.0)
            source_label = humanize_feature_name(edge["source"])
            target_label = humanize_feature_name(target)
            G.add_edge(
                source_label,
                target_label,
                key=edge["lag"],
                lag=edge["lag"],
                strength=strength,
                stability=stab,
                p_value=edge.get("p_value", 1.0),
            )
    print(f"Total edges plotted: {len(G.edges())}")
    if len(G.nodes()) == 0:
        print("No edges to plot")
        return

    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    plt.figure(figsize=(20, 14))
    node_colors = ["#FF6B6B" if G.in_degree(n) == 0 else "#4ECDC4" for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_size=3200, node_color=node_colors, alpha=0.95, edgecolors="black", linewidths=1.5
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    solid_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("stability", 0) >= high_stability]
    dashed_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("stability", 0) < high_stability]

    def draw_edges(edge_list, style):
        if not edge_list:
            return
        widths = [abs(d["strength"]) * 8 for _, _, d in edge_list]
        colors = ["#E74C3C" if d["strength"] < 0 else "#3498DB" for _, _, d in edge_list]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in edge_list],
            edge_color=colors,
            width=widths,
            arrows=True,
            arrowsize=22,
            alpha=0.85,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            style=style,
        )

    draw_edges(solid_edges, "solid")
    draw_edges(dashed_edges, "dashed")

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        line = f"lag {d['lag']}d | stab={d.get('stability', 0):.2f} | str={abs(d['strength']):.2f}"
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += f"\n{line}"
        else:
            edge_labels[(u, v)] = line
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9),
    )

    legend_handles = [
        Line2D([0], [0], color="#3498DB", lw=3, label="Increases target"),
        Line2D([0], [0], color="#E74C3C", lw=3, label="Decreases target"),
        Line2D([0], [0], color="gray", lw=2, linestyle="-", label=f"Stability ≥ {high_stability:.2f}"),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=2,
            linestyle="--",
            label=f"{MIN_STABILITY:.2f} ≤ Stability < {high_stability:.2f}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF6B6B",
            markersize=14,
            label="Root variable",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4ECDC4",
            markersize=14,
            label="Downstream variable",
        ),
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=11, frameon=True, framealpha=0.9)
    note = (
        f"Edges shown have bootstrap stability ≥ {MIN_STABILITY:.2f}. "
        f"Solid = stability ≥ {high_stability:.2f}; dashed = lower stability."
    )
    plt.figtext(
        0.5,
        0.01,
        note,
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="gray"),
    )
    plt.title(
        f"Temporal Causal Graph ({len(G.edges())} edges, stability ≥ {MIN_STABILITY:.2f})",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved graph: {output_path}")


def select_reporting_key(causal_graphs: dict, per_entity_agg: dict):
    scored = []
    for key, graph in causal_graphs.items():
        if key not in per_entity_agg:
            continue
        total_edges = sum(len(sources) for sources in graph.values())
        scored.append((total_edges, key))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def resolve_experiment_table_path(explicit_path: str = None) -> str:
    EXPERIMENT_TABLE_PATH = "data/construct_experiments_input_test.csv"
    EXPERIMENT_TABLE_COLUMNS = {"TreatmentLessonConstructId", "QuestionConstructId", "Year"}

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.append(EXPERIMENT_TABLE_PATH)
    data_dir = "data"
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.lower().endswith((".csv", ".xlsx")):
                candidates.append(os.path.join(data_dir, fname))
    checked = []
    for path in candidates:
        if path in checked:
            continue
        checked.append(path)
        if not os.path.exists(path):
            continue
        try:
            df = (
                pd.read_csv(path, nrows=5)
                if path.lower().endswith(".csv")
                else pd.read_excel(path, nrows=5)
            )
        except Exception as e:
            print(f"  (skipping {path}: {type(e).__name__}: {e})")
            continue
        if EXPERIMENT_TABLE_COLUMNS.issubset(set(df.columns)):
            print(f"Found experiment design table at: {path}")
            return path
    raise FileNotFoundError(
        f"Could not find a file with columns {sorted(EXPERIMENT_TABLE_COLUMNS)}. Checked: {checked}."
    )


def build_construct_to_experiment_mapping(experiments_df: pd.DataFrame) -> pd.DataFrame:
    required = {"TreatmentLessonConstructId", "QuestionConstructId", "Year"}
    missing = required - set(experiments_df.columns)
    if missing:
        raise ValueError(f"Experiment table missing columns: {missing}")

    treatment_rows = experiments_df.copy()
    treatment_rows["ConstructId"] = treatment_rows["TreatmentLessonConstructId"]
    treatment_rows["role"] = "treatment"

    question_rows = experiments_df.copy()
    question_rows["ConstructId"] = question_rows["QuestionConstructId"]
    question_rows["role"] = "question"

    mapping_df = pd.concat([treatment_rows, question_rows], ignore_index=True)
    role_counts = mapping_df.groupby("ConstructId")["role"].nunique()
    dual_role_ids = sorted(role_counts[role_counts > 1].index.tolist())
    print(
        f"build_construct_to_experiment_mapping: {len(experiments_df)} conditions -> "
        f"{len(mapping_df)} rows ({len(dual_role_ids)} dual-role ConstructIds)"
    )
    if dual_role_ids:
        print(
            f"  dual-role ConstructId(s): {dual_role_ids[:20]}"
            f"{' ...' if len(dual_role_ids) > 20 else ''}"
        )
    return mapping_df


class TemporalArchitecture:
    def __init__(
        self,
        window_size: int = 50,
        step_size: int = 50,
        max_response_time_seconds: float = 300.0,
        max_lag: int = 2,
        alpha: float = 0.1,
        pc_alpha: float = 0.2,
        corr_threshold: float = 0.97,
        min_effect: float = 0.12,
    ):
        self.layer1_builder = Layer1TemporalConstruction(
            window_size=window_size,
            step_size=step_size,
            max_response_time_seconds=max_response_time_seconds,
        )
        self.layer2_builder = Layer2StructureLearning(
            max_lag=max_lag,
            alpha=alpha,
            pc_alpha=pc_alpha,
            corr_threshold=corr_threshold,
            min_effect=min_effect,
        )

    def run(
        self,
        raw_logs: pd.DataFrame,
        mapping_path: str = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], dict, Optional[tuple], dict, dict, dict]:
        raw_logs = filter_pre_outcome_logs(raw_logs)

        layer1_df = self.layer1_builder.build_layer1(raw_logs)

        resolved_experiment_path = resolve_experiment_table_path(mapping_path)
        experiments_df = (
            pd.read_csv(resolved_experiment_path)
            if resolved_experiment_path.lower().endswith(".csv")
            else pd.read_excel(resolved_experiment_path)
        )
        mapping_df = build_construct_to_experiment_mapping(experiments_df)

        keyed_layer1_df = self.layer2_builder.attach_experimental_keys(
            layer1_df=layer1_df,
            mapping_df=mapping_df,
            construct_col_in_layer1="dominant_construct",
            construct_col_in_mapping="ConstructId",
        )

        if "role" in keyed_layer1_df.columns:
            print(f"Keyed windows by role: {keyed_layer1_df['role'].value_counts().to_dict()}")

        causal_graphs, per_entity_agg, key_status = self.layer2_builder.build_layer2(
            keyed_layer1_df, run_stability_selection=True
        )

        n_ok = sum(1 for v in key_status.values() if v["status"] == "ok")
        print(f"\nExperimental keys: {len(key_status)} total, {n_ok} with enough temporal density")

        if len(causal_graphs) == 0:
            print("No experimental key produced a usable causal graph.")
            return layer1_df, None, {}, None, {}, key_status, keyed_layer1_df

        reporting_key = select_reporting_key(causal_graphs, per_entity_agg)
        if reporting_key is None:
            print("No key has both causal graph and aggregated series.")
            return layer1_df, None, causal_graphs, None, {}, key_status, keyed_layer1_df

        raw_graph = causal_graphs[reporting_key]
        agg_df = per_entity_agg[reporting_key]
        print(f"\nReporting key: {reporting_key}")

        edge_frequency = {}
        if reporting_key in key_status and "stability" in key_status[reporting_key]:
            edge_frequency = key_status[reporting_key]["stability"].get("edge_frequency", {})

        filtered_graph = filter_graph_by_stability(
            raw_graph, edge_frequency, min_stability=MIN_STABILITY, min_strength=MIN_STRENGTH
        )
        n_filtered = sum(len(v) for v in filtered_graph.values())
        print(f"Edges after stability filter (≥ {MIN_STABILITY}): {n_filtered}")

        plot_causal_graph(filtered_graph)

        return (
            layer1_df,
            agg_df,
            causal_graphs,
            reporting_key,
            filtered_graph,
            key_status,
            keyed_layer1_df,
        )