import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Any

from src.causal_discovery import Layer1TemporalConstruction, Layer2StructureLearning
from src.option_a.features import FeatureBuilder
from src.option_a.benchmark import PRIMARY_TARGET
from src.option_a.folds import FOLD_KEY


FEATURE_LABELS = {
    'avg_success': "Accuracy Rate",
    'success_trend': "Trending Up or Down",
    'avg_difficulty': "Question Difficulty",
    'difficulty_std': "Difficulty Swings",
    'difficulty_range': "Difficulty Gap",
    'recent5_correct_rate': "Recent Accuracy",
    'max_streak': "Longest Correct Streak",
    'attempts_mean': "Average Tries per Question",
    'attempts_max': "Most Tries on One Question",
    'pct_multi_attempt': "Retry Rate",
    'avg_response_time': "Response Time (Average Pace)",
    'response_time_std': "Response Time Swings",
    'median_response_time': "Response Time (Typical Pace)",
}

MIN_STABILITY = 0.30
HIGH_STABILITY = 0.45
MIN_STRENGTH = 0.12

PRE_OUTCOME_LOG_TYPES = {"Checkin", "CheckinRetry", "Lesson"}


# ============================================================
# Feature builder dùng cho benchmark (protocol.py). Fake/placeholder,
# KHÔNG gọi Layer1/Layer2 ở đây vì chưa xác định được join key giữa
# bench (row-level) và raw_logs (log-level) an toàn cho fold-local fit.
# Bản thật cần thay _select_columns() bằng Layer1TemporalConstruction
# + Layer2StructureLearning, miễn là fit() chỉ được nhìn train_rows/logs.
# ============================================================
class TemporalFeatureBuilder(FeatureBuilder):
    def __init__(self, preserve_order: bool = True, seed: int = 42):
        self.preserve_order = preserve_order
        self.seed = seed
        self.name = "TEMPORAL_FEATURES" if preserve_order else "SHUFFLED_FEATURES"
        self.selected_columns: Optional[List[str]] = None

    def fit(self, train_rows: pd.DataFrame, logs: Optional[pd.DataFrame]) -> None:
        numeric_cols = train_rows.select_dtypes(include=[np.number]).columns.tolist()
        self.selected_columns = [c for c in numeric_cols if c not in (PRIMARY_TARGET, FOLD_KEY)]

    def transform(self, rows: pd.DataFrame, logs: Optional[pd.DataFrame]) -> pd.DataFrame:
        if self.selected_columns is None:
            raise RuntimeError("TemporalFeatureBuilder.transform called before fit()")
        X = rows[self.selected_columns].copy()
        if not self.preserve_order:
            rng = np.random.default_rng(self.seed)
            for col in X.columns:
                X[col] = rng.permutation(X[col].to_numpy())
        return X


def default_feature_builders(seed: int = 42) -> Dict[str, FeatureBuilder]:
    return {
        "temporal": TemporalFeatureBuilder(preserve_order=True, seed=seed),
        "shuffled": TemporalFeatureBuilder(preserve_order=False, seed=seed),
    }


# ============================================================
# Từ đây trở xuống là pipeline khám phá / report (đồ thị nhân quả,
# vẽ hình). KHÔNG được protocol.py gọi trong vòng lặp benchmark,
# chỉ chạy riêng để ra hình cho báo cáo/chuyên đề.
# ============================================================

def humanize_feature_name(name: str) -> str:
    match = re.match(r'^CTRL_cluster_(\d+)_ratio$', name)
    if match:
        return f"Learner Group #{match.group(1)} Share"
    return FEATURE_LABELS.get(name, name)


def filter_pre_outcome_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    if 'Type' not in raw_logs.columns:
        raise ValueError(
            "filter_pre_outcome_logs: raw_logs has no 'Type' column. "
            "Check preprocessing.ipynb / clean_main_df."
        )
    before = len(raw_logs)
    present_types = set(raw_logs['Type'].dropna().unique())
    unknown_types = present_types - PRE_OUTCOME_LOG_TYPES - {"Checkout", "CheckoutRetry"}
    if unknown_types:
        print(f"NOTE: unrecognized Type value(s) {unknown_types} dropped.")
    filtered = raw_logs[raw_logs['Type'].isin(PRE_OUTCOME_LOG_TYPES)].copy()
    after = len(filtered)
    print(f"filter_pre_outcome_logs: {before} -> {after} pre-outcome logs "
          f"({before - after} dropped)")
    if after == 0:
        raise ValueError("filter_pre_outcome_logs dropped everything.")
    return filtered


def filter_graph_by_stability(causal_graph: dict, edge_frequency: dict,
                              min_stability: float = MIN_STABILITY,
                              min_strength: float = MIN_STRENGTH) -> dict:
    filtered = {target: [] for target in causal_graph}
    for target, edges in causal_graph.items():
        for edge in edges:
            key = (edge['source'], target, edge['lag'])
            stab = edge_frequency.get(key, 0.0)
            strength = abs(edge.get('strength', 0.0))
            if stab >= min_stability and strength >= min_strength:
                new_edge = dict(edge)
                new_edge['stability'] = stab
                filtered[target].append(new_edge)
    return {t: e for t, e in filtered.items() if e}


def plot_causal_graph(causal_graph: dict, output_path: str = "output/causal_graph.png",
                      high_stability: float = HIGH_STABILITY):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    G = nx.MultiDiGraph()
    for target, sources in causal_graph.items():
        for edge in sources:
            strength = edge.get('strength', 0.0)
            stab = edge.get('stability', 0.0)
            source_label = humanize_feature_name(edge['source'])
            target_label = humanize_feature_name(target)
            G.add_edge(
                source_label, target_label,
                key=edge['lag'],
                lag=edge['lag'],
                strength=strength,
                stability=stab,
                p_value=edge.get('p_value', 1.0)
            )
    print(f"Total edges plotted: {len(G.edges())}")
    if len(G.nodes()) == 0:
        print("No edges to plot")
        return

    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    plt.figure(figsize=(20, 14))
    node_colors = ['#FF6B6B' if G.in_degree(n) == 0 else '#4ECDC4' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3200, node_color=node_colors,
                           alpha=0.95, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    solid_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('stability', 0) >= high_stability]
    dashed_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('stability', 0) < high_stability]

    def draw_edges(edge_list, style):
        if not edge_list:
            return
        widths = [abs(d['strength']) * 8 for _, _, d in edge_list]
        colors = ['#E74C3C' if d['strength'] < 0 else '#3498DB' for _, _, d in edge_list]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for u, v, _ in edge_list],
            edge_color=colors, width=widths,
            arrows=True, arrowsize=22, alpha=0.85,
            arrowstyle='-|>', connectionstyle='arc3,rad=0.15',
            style=style
        )

    draw_edges(solid_edges, 'solid')
    draw_edges(dashed_edges, 'dashed')

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        line = f"lag {d['lag']}d | stab={d.get('stability', 0):.2f} | str={abs(d['strength']):.2f}"
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += f"\n{line}"
        else:
            edge_labels[(u, v)] = line
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=7, font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9)
    )

    legend_handles = [
        Line2D([0], [0], color='#3498DB', lw=3, label='Increases target'),
        Line2D([0], [0], color='#E74C3C', lw=3, label='Decreases target'),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label=f'Stability ≥ {high_stability:.2f}'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label=f'{MIN_STABILITY:.2f} ≤ Stability < {high_stability:.2f}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=14, label='Root variable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=14, label='Downstream variable'),
    ]
    plt.legend(handles=legend_handles, loc='upper left', fontsize=11, frameon=True, framealpha=0.9)
    note = (f"Edges shown have bootstrap stability ≥ {MIN_STABILITY:.2f}. "
            f"Solid = stability ≥ {high_stability:.2f}; dashed = lower stability.")
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='gray'))
    plt.title(f"Temporal Causal Graph ({len(G.edges())} edges, stability ≥ {MIN_STABILITY:.2f})",
              fontsize=16, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
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
            df = pd.read_csv(path, nrows=5) if path.lower().endswith(".csv") else pd.read_excel(path, nrows=5)
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
    both_roles = mapping_df.groupby("ConstructId")["role"].nunique()
    n_dual_role = int((both_roles > 1).sum())
    print(f"build_construct_to_experiment_mapping: {len(experiments_df)} conditions -> "
          f"{len(mapping_df)} rows ({n_dual_role} dual-role ConstructIds)")
    return mapping_df


class TemporalArchitecture:
    # Diagnostics/report pipeline. KHÔNG dùng để cấp feature cho benchmark
    # (xem TemporalFeatureBuilder ở trên). Giữ nguyên để ra đồ thị nhân quả
    # phục vụ báo cáo, chạy độc lập ngoài vòng lặp protocol.run_protocol.
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
            construct_col_in_layer1='dominant_construct',
            construct_col_in_mapping='ConstructId'
        )

        if 'role' in keyed_layer1_df.columns:
            print(f"Keyed windows by role: {keyed_layer1_df['role'].value_counts().to_dict()}")

        causal_graphs, per_entity_agg, key_status = self.layer2_builder.build_layer2(
            keyed_layer1_df,
            run_stability_selection=True
        )

        n_ok = sum(1 for v in key_status.values() if v['status'] == 'ok')
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
        if reporting_key in key_status and 'stability' in key_status[reporting_key]:
            edge_frequency = key_status[reporting_key]['stability'].get('edge_frequency', {})

        filtered_graph = filter_graph_by_stability(
            raw_graph, edge_frequency,
            min_stability=MIN_STABILITY,
            min_strength=MIN_STRENGTH
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