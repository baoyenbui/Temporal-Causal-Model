import sys
import os
import re
import nbformat
from nbformat import read
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.causal_discovery import Layer1TemporalConstruction, Layer2StructureLearning
from src.cf import Layer3CounterfactualExplanation
from src.evaluation import run_full_evaluation

clean_main_df: pd.DataFrame

with open("src/preprocessing.ipynb", "r", encoding="utf-8") as f:
    notebook = read(f, as_version=4)

exec_globals = globals()
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            print(f"preprocessing.ipynb cell {i} failed: {type(e).__name__}: {e}")
            raise

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


def humanize_feature_name(name: str) -> str:
    match = re.match(r'^CTRL_cluster_(\d+)_ratio$', name)
    if match:
        return f"Learner Group #{match.group(1)} Share"
    return FEATURE_LABELS.get(name, name)


def plot_causal_graph(causal_graph: dict, output_path: str = "output/causal_graph.png", min_strength: float = 0.12):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    G = nx.MultiDiGraph()
    for target, sources in causal_graph.items():
        for edge in sources:
            p = edge['p_value']
            strength = edge.get('strength', 0.0)
            if abs(strength) < min_strength:
                continue
            weight = max(1.0 - min(p, 0.999999), 0.1)
            source_label = humanize_feature_name(edge['source'])
            target_label = humanize_feature_name(target)
            G.add_edge(source_label, target_label, key=edge['lag'], lag=edge['lag'], weight=weight, p_value=p, strength=strength)
    if len(G.nodes()) == 0:
        return
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    plt.figure(figsize=(20, 14))
    node_colors = ['#FF6B6B' if G.in_degree(n) == 0 else '#4ECDC4' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3200, node_color=node_colors, alpha=0.95, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    edge_widths = [abs(d['strength']) * 8 for u, v, d in G.edges(data=True)]
    edge_colors = ['#E74C3C' if d['strength'] < 0 else '#3498DB' for u, v, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                          arrows=True, arrowsize=22, alpha=0.82,
                          arrowstyle='-|>', connectionstyle='arc3,rad=0.15')
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        line = f"lag {d['lag']}d: p={d['p_value']:.2e}, strength={abs(d['strength']):.2f}"
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += f"\n{line}"
        else:
            edge_labels[(u, v)] = line
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_size=7, font_weight='bold',
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9))
    legend_handles = [
        Line2D([0], [0], color='#3498DB', lw=3, label='Increases target'),
        Line2D([0], [0], color='#E74C3C', lw=3, label='Decreases target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=14, label='Root variable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=14, label='Downstream variable'),
    ]
    plt.legend(handles=legend_handles, loc='upper left', fontsize=11, frameon=True, framealpha=0.9)
    plt.title(f"Temporal Causal Graph of Learning Behavior ({len(G.edges())} relationships found)",
              fontsize=16, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def main():
    raw_logs = clean_main_df.copy()
    layer1_builder = Layer1TemporalConstruction(window_size=50, step_size=50, max_response_time_seconds=300.0)
    layer1_df = layer1_builder.build_layer1(raw_logs)
    layer2_builder = Layer2StructureLearning(max_lag=2, alpha=0.15, pc_alpha=0.2, corr_threshold=0.97, min_effect=0.10)
    agg_df, causal_graph = layer2_builder.build_layer2(layer1_df)
    plot_causal_graph(causal_graph)

    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    actionable_vars = layer3.identify_actionable_variables()

    print("Actionable variables:", actionable_vars)
    print("Parents of success_trend:", [e['source'] for e in causal_graph.get('success_trend', [])])
    print("Parents of avg_success:", [e['source'] for e in causal_graph.get('avg_success', [])])
    print("Parents of max_streak:", [e['source'] for e in causal_graph.get('max_streak', [])])

    cf_target = 'success_trend'
    current_val = layer3.get_current_state(cf_target)
    target_step = layer3.get_max_step_delta(cf_target)
    if not np.isfinite(target_step) or target_step <= 0:
        target_step = float(agg_df[cf_target].std())
    cf_threshold = current_val + target_step

    cf_results = layer3.generate_counterfactual(
        target=cf_target,
        direction='increase',
        threshold=cf_threshold,
        current_value=current_val,
        max_hops=2,
        top_k=3
    )

    if cf_results:
        print(f"Counterfactuals to push '{cf_target}' from {current_val:.3f} up to >= {cf_threshold:.3f}:")
        for r in cf_results:
            print(f"  Change '{r['source_variable']}' from {r['current_source_value']:.3f} to "
                  f"{r['proposed_source_value']:.3f} (delta={r['delta']:+.3f}) via path [{r['causal_path']}], "
                  f"lag={r['lag_days']}d")
            print(f"    cf_score={r['cf_score']:.3f} | estimated_target_change={r['estimated_target_change']:+.3f} | "
                  f"proximity={r['proximity']:.3f} | sparsity={r['sparsity']} | "
                  f"stability={r['stability']:.3f} | flip_success_rate={r['flip_success_rate']:.3f} | "
                  f"causal_plausibility={r['causal_plausibility']:.3f}")
    else:
        print(f"No reliable counterfactuals found for '{cf_target}'")

    student_cf_df = layer3.generate_student_counterfactuals(
        layer1_df=layer1_df,
        target=cf_target,
        direction='increase',
        n_users=5,
        threshold_quantile=0.75
    )
    print(student_cf_df.to_string(index=False))

    evaluation_results = run_full_evaluation(layer1_df, agg_df, causal_graph)

    return layer1_df, agg_df, causal_graph, cf_results, student_cf_df, evaluation_results


if __name__ == "__main__":
    main()