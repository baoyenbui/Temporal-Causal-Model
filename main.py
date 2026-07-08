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



def plot_causal_graph(causal_graph: dict, output_path: str = "output/causal_graph.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    G = nx.DiGraph()
    
    for target, sources in causal_graph.items():
        if 'CTRL' in target:
            continue
        for edge in sources:
            if 'CTRL' in edge['source']:
                continue
            edge_key = f"{edge['source']}->{target}@lag{edge['lag']}"
            weight = 1.0 - edge['p_value']
            G.add_edge(edge['source'], target, lag=edge['lag'], weight=weight, p_value=edge['p_value'])
    
    if len(G.nodes()) == 0:
        print("No edges to plot")
        return
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    node_colors = ['#FF6B6B' if G.in_degree(n) == 0 else '#4ECDC4' for n in G.nodes()]
    
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): f"{d['p_value']:.3f}" for u, v, d in G.edges(data=True)}
    edge_widths = [d['weight'] * 5 + 0.5 for u, v, d in G.edges(data=True)]
    
    nx.draw_networkx_edges(G, pos, edge_color='#2C3E50', width=edge_widths, 
                          arrows=True, arrowsize=30, alpha=0.8,
                          arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8, font_weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.title("Temporal Causal Graph (PCMCI)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {output_path}")



def main():
    raw_logs = clean_main_df.copy()


    layer1_builder = Layer1TemporalConstruction(window_size=10, step_size=5)
    layer1_df = layer1_builder.build_layer1(raw_logs)


    layer2_builder = Layer2StructureLearning(
        max_lag=2,
        alpha=0.01
    )


    agg_df, causal_graph = layer2_builder.build_layer2(layer1_df)
    
    total_edges = sum(len(sources) for sources in causal_graph.values())
    print(f"Total edges discovered: {total_edges}")
    
    for target, sources in causal_graph.items():
        for edge in sources:
            print(f"  {edge['source']} -> {target} @ lag {edge['lag']}, p={edge['p_value']:.4f}")


    plot_causal_graph(causal_graph)


    return layer1_df, agg_df, causal_graph



if __name__ == "__main__":
    main()