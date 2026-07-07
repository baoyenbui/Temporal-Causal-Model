import sys
import os
import nbformat
from nbformat import read
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

clean_main_df: pd.DataFrame  # type: ignore

with open("src/preprocessing.ipynb", "r", encoding="utf-8") as f:
    notebook = read(f, as_version=4)

for cell in notebook.cells:
    if cell.cell_type == "code":
        try:
            exec(cell.source)
        except Exception:
            pass

sys.path.append(os.path.dirname(__file__))
from src.causal_discovery import Layer1TemporalConstruction, Layer2StructureLearning


def plot_causal_graph(stable_edges: dict, output_path: str = "output/causal_graph.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    G = nx.DiGraph()
    
    for edge_key, freq in stable_edges.items():
        source, target_lag = edge_key.split('->')
        target, lag = target_lag.split('@lag')
        G.add_edge(source, target, lag=int(lag), weight=freq)
    
    if len(G.nodes()) == 0:
        print("[!] No stable edges to plot")
        return
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    node_colors = ['#FF6B6B' if G.in_degree(n) == 0 else '#4ECDC4' for n in G.nodes()]
    
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): f"lag{d['lag']}" for u, v, d in G.edges(data=True)}
    edge_weights = [d['weight'] * 3 for u, v, d in G.edges(data=True)]
    
    nx.draw_networkx_edges(G, pos, edge_color='#2C3E50', width=edge_weights, 
                          arrows=True, arrowsize=30, alpha=0.8,
                          arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8, bbox=dict(boxstyle='round', alpha=0.3))
    
    plt.title("Temporal Causal Graph (Stable Edges)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {output_path}")


def main():
    raw_logs = clean_main_df.copy()
    print(f"Raw logs: {len(raw_logs)} rows, {len(raw_logs.columns)} cols")

    layer1_builder = Layer1TemporalConstruction(window_size=10, step_size=5)
    layer1_df = layer1_builder.build_layer1(raw_logs)
    print(f"Layer 1: {len(layer1_df)} windows")

    layer2_builder = Layer2StructureLearning(
        max_lag=2,
        alpha=0.05,
        bootstrap_runs=100,
        bootstrap_threshold=0.1
    )

    agg_df = layer2_builder.aggregate_time_series(layer1_df)
    agg_df = layer2_builder.add_latent_proxies(agg_df)
    print(f"Aggregated: {len(agg_df)} time buckets")

    X_scaled, feature_df = layer2_builder.prepare_timeseries_matrix(agg_df)
    print(f"Features: {X_scaled.shape[1]} vars, {X_scaled.shape[0]} time points")

    stable_edges = layer2_builder.bootstrap_stability(feature_df)
    print(f"Stable edges: {len(stable_edges)}")

    for edge_key, freq in stable_edges.items():
        print(f"    {edge_key}: {freq:.3f}")

    plot_causal_graph(stable_edges)

    return layer1_df, agg_df, stable_edges


if __name__ == "__main__":
    main()