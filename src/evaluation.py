from collections import defaultdict
import numpy as np
import pandas as pd

from src.causal_discovery import Layer2StructureLearning
from src.cf import Layer3CounterfactualExplanation

LAYER2_KWARGS = dict(max_lag=2, alpha=0.1, pc_alpha=0.2, corr_threshold=0.97, min_effect=0.12)


def bootstrap_causal_graph(layer1_df: pd.DataFrame, n_bootstrap: int = 20, seed: int = 42, verbose: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = layer1_df['user_id'].unique()
    edge_counts = defaultdict(int)
    edge_strengths = defaultdict(list)
    successful_runs = 0

    for i in range(n_bootstrap):
        sampled_users = rng.choice(user_ids, size=len(user_ids), replace=True)
        resampled = pd.concat(
            [layer1_df[layer1_df['user_id'] == uid] for uid in sampled_users],
            ignore_index=True
        )
        builder = Layer2StructureLearning(**LAYER2_KWARGS)
        try:
            _, causal_graph = builder.build_layer2(resampled)
        except Exception as e:
            if verbose:
                print(f"Bootstrap iteration {i + 1}/{n_bootstrap}: failed ({type(e).__name__}: {e}), skipping")
            continue
        successful_runs += 1
        seen_this_run = set()
        for target, edges in causal_graph.items():
            for edge in edges:
                key = (edge['source'], target, edge['lag'])
                if key in seen_this_run:
                    continue
                seen_this_run.add(key)
                edge_counts[key] += 1
                edge_strengths[key].append(edge['strength'])

    rows = []
    for key, count in edge_counts.items():
        source, target, lag = key
        strengths = edge_strengths[key]
        rows.append({
            'source': source,
            'target': target,
            'lag': lag,
            'stability': count / successful_runs if successful_runs else 0.0,
            'mean_strength': float(np.mean(strengths)),
            'std_strength': float(np.std(strengths)),
            'n_occurrences': count
        })

    result_df = pd.DataFrame(rows).sort_values('stability', ascending=False).reset_index(drop=True)
    result_df.attrs['successful_runs'] = successful_runs
    result_df.attrs['n_bootstrap'] = n_bootstrap
    return result_df


def evaluate_layer3_directions(
    causal_graph: dict,
    agg_df: pd.DataFrame,
    n_targets: int = 6,
    seed: int = 42,
    verbose: bool = False
) -> pd.DataFrame:
    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    actionable_vars = layer3.identify_actionable_variables(verbose=False)
    all_nodes = list(layer3.dag.nodes())

    reachable_targets = []
    for node in all_nodes:
        paths = layer3.get_valid_causal_paths(node, actionable_vars, max_hops=2)
        if paths:
            reachable_targets.append(node)

    n_pairs_tested = len(reachable_targets) * 2

    rows = []
    for target in reachable_targets:
        for direction in ('increase', 'decrease'):
            current_value = layer3.get_current_state(target)
            step = layer3.get_max_step_delta(target)
            if not np.isfinite(step) or step <= 0:
                step = float(agg_df[target].std())
            direction_sign = 1.0 if direction == 'increase' else -1.0
            threshold = current_value + direction_sign * step

            results = layer3.generate_counterfactual(
                target=target, direction=direction, threshold=threshold,
                current_value=current_value, top_k=3
            )
            for r in results:
                expected_sign = 1 if direction == 'increase' else -1
                actual_sign = int(np.sign(r['estimated_target_change']))
                direction_ok = (actual_sign == expected_sign) or (actual_sign == 0)
                rows.append({
                    'target': target,
                    'direction': direction,
                    'source_variable': r['source_variable'],
                    'estimated_target_change': r['estimated_target_change'],
                    'direction_correct': direction_ok,
                    'flip_success_rate': r['flip_success_rate'],
                    'cf_score': r['cf_score']
                })

    df = pd.DataFrame(rows)
    df.attrs['n_pairs_tested'] = n_pairs_tested
    df.attrs['n_reachable_targets'] = len(reachable_targets)
    return df


def build_results_table(
    stability_df: pd.DataFrame,
    layer3_df: pd.DataFrame,
    stability_threshold: float = 0.5
) -> pd.DataFrame:
    metrics = []

    successful_runs = stability_df.attrs.get('successful_runs', None) if stability_df is not None else None
    n_bootstrap = stability_df.attrs.get('n_bootstrap', None) if stability_df is not None else None
    total_edges = len(stability_df) if stability_df is not None else 0
    stable_edges_df = stability_df[stability_df['stability'] >= stability_threshold] if total_edges > 0 else stability_df
    n_stable = len(stable_edges_df) if total_edges > 0 else 0

    metrics.append(('Bootstrap runs successful', f"{successful_runs}/{n_bootstrap}" if successful_runs is not None else "n/a"))
    metrics.append(('Total edges discovered (pooled)', total_edges))
    metrics.append((f'Edges with stability >= {stability_threshold:.0%}', n_stable))
    metrics.append(('Edge stability rate', f"{n_stable / total_edges:.1%}" if total_edges > 0 else "n/a"))
    metrics.append(('Mean edge stability', f"{stability_df['stability'].mean():.3f}" if total_edges > 0 else "n/a"))

    n_pairs_tested = layer3_df.attrs.get('n_pairs_tested', None) if layer3_df is not None else None
    n_reachable = layer3_df.attrs.get('n_reachable_targets', None) if layer3_df is not None else None
    n_results = len(layer3_df) if layer3_df is not None else 0
    n_wrong = int((~layer3_df['direction_correct']).sum()) if n_results > 0 else 0

    metrics.append(('DAG nodes reachable via an actionable path', n_reachable if n_reachable is not None else "n/a"))
    metrics.append(('(Target, direction) pairs tested', n_pairs_tested if n_pairs_tested is not None else "n/a"))
    metrics.append(('Pairs yielding >=1 reliable counterfactual', n_results if n_results > 0 else 0))
    metrics.append(('Coverage (reliable CF / pairs tested)',
                     f"{n_results / n_pairs_tested:.1%}" if n_pairs_tested else "n/a"))
    metrics.append(('Wrong-direction results (should be 0)', n_wrong))
    metrics.append(('Mean flip_success_rate', f"{layer3_df['flip_success_rate'].mean():.3f}" if n_results > 0 else "n/a"))
    metrics.append(('Median flip_success_rate', f"{layer3_df['flip_success_rate'].median():.3f}" if n_results > 0 else "n/a"))
    metrics.append(('Mean cf_score', f"{layer3_df['cf_score'].mean():.3f}" if n_results > 0 else "n/a"))
    metrics.append(('Median cf_score', f"{layer3_df['cf_score'].median():.3f}" if n_results > 0 else "n/a"))

    return pd.DataFrame(metrics, columns=['Metric', 'Value'])


def run_full_evaluation(
    layer1_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    causal_graph: dict,
    n_bootstrap: int = 20,
    n_layer3_targets: int = 6,
    stability_threshold: float = 0.5
) -> dict:
    stability_df = bootstrap_causal_graph(layer1_df, n_bootstrap=n_bootstrap)
    layer3_df = evaluate_layer3_directions(causal_graph, agg_df, n_targets=n_layer3_targets)
    results_table = build_results_table(stability_df, layer3_df, stability_threshold=stability_threshold)

    print(results_table.to_string(index=False))

    return {'stability': stability_df, 'layer3': layer3_df, 'results_table': results_table}