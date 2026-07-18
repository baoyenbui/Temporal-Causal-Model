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

clean_main_df: pd.DataFrame

with open("src/preprocessing.ipynb", "r", encoding="utf-8") as f:
    notebook = read(f, as_version=4)

for cell in notebook.cells:
    if cell.cell_type == "code":
        try:
            exec(cell.source)
        except Exception:
            pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.causal_discovery import Layer1TemporalConstruction, Layer2StructureLearning


def plot_causal_graph(causal_graph: dict, output_path: str = "output/causal_graph.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    G = nx.MultiDiGraph()

    for target, sources in causal_graph.items():
        for edge in sources:
            p = edge['p_value']
            if round(p, 4) == 0:
                continue
            weight = max(1.0 - p, 0.1)
            G.add_edge(edge['source'], target, key=edge['lag'], lag=edge['lag'], weight=weight, p_value=p)

    print(f"Total edges plotted: {len(G.edges())}")

    if len(G.nodes()) == 0:
        print("No edges to plot")
        return

    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)

    plt.figure(figsize=(20, 14))
    node_colors = ['#FF6B6B' if G.in_degree(n) == 0 else '#4ECDC4' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=2600, node_color=node_colors, alpha=0.95, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    edge_widths = [d['weight'] * 5 for u, v, d in G.edges(data=True)]
    edge_colors = ['#E74C3C' if d['p_value'] < 0.01 else '#3498DB' for u, v, d in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                          arrows=True, arrowsize=22, alpha=0.82,
                          arrowstyle='-|>', connectionstyle='arc3,rad=0.15')

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        line = f"lag{d['lag']}: {d['p_value']:.3f}"
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += f"\n{line}"
        else:
            edge_labels[(u, v)] = line

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_size=7, font_weight='bold',
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9))

    plt.title(f"Temporal Causal Graph ({len(G.edges())} edges)",
              fontsize=18, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {output_path}")


def main():
    raw_logs = clean_main_df.copy()
    layer1_builder = Layer1TemporalConstruction(window_size=50, step_size=50)
    layer1_df = layer1_builder.build_layer1(raw_logs)
    layer2_builder = Layer2StructureLearning(max_lag=2, alpha=0.05)
    agg_df, causal_graph = layer2_builder.build_layer2(layer1_df)
    total_edges = sum(len(sources) for sources in causal_graph.values())
    print(f"Total edges discovered: {total_edges}")
    for target, sources in causal_graph.items():
        for edge in sources:
            print(f"  {edge['source']} -> {target} @ lag {edge['lag']}, p={edge['p_value']:.4f}, strength={edge['strength']:.3f}")
    plot_causal_graph(causal_graph)
    return layer1_df, agg_df, causal_graph


if __name__ == "__main__":
    main()