from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.causal_discovery import Layer2StructureLearning
from src.cf import Layer3CounterfactualExplanation

LAYER2_KWARGS = dict(max_lag=2, alpha=0.1, pc_alpha=0.2, corr_threshold=0.97, min_effect=0.12)

RESPONSE_TIME_FALLBACK_COLS = ('avg_response_time', 'median_response_time', 'response_time_std')


def classify_direction(estimated_target_change: float, direction: str) -> str:
    actual_sign = int(np.sign(estimated_target_change))
    expected_sign = 1 if direction == 'increase' else -1
    if actual_sign == 0:
        return 'zero_effect'
    return 'correct' if actual_sign == expected_sign else 'wrong_direction'


def summarize_pair_coverage(rows_df: pd.DataFrame, n_pairs_tested: int) -> Dict:
    if n_pairs_tested == 0:
        return {'n_pairs_covered': 0, 'coverage': float('nan')}
    if len(rows_df) == 0:
        return {'n_pairs_covered': 0, 'coverage': 0.0}
    n_covered = rows_df[['target', 'direction']].drop_duplicates().shape[0]
    return {'n_pairs_covered': n_covered, 'coverage': n_covered / n_pairs_tested}


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


def _default_threshold(layer3: Layer3CounterfactualExplanation, agg_df: pd.DataFrame, target: str, direction: str,
                        feasible_fraction: float = 0.7):
    current_value = layer3.get_current_state(target)

    sim_results = layer3.simulate_max_feasible_effect(target, direction)
    best_feasible = next((r for r in sim_results if r['moves_correct_direction']), None)
    if best_feasible is not None:
        threshold = current_value + feasible_fraction * best_feasible['achievable_change']
        return current_value, threshold

    step = layer3.get_max_step_delta(target)
    if not np.isfinite(step) or step <= 0:
        step = float(agg_df[target].std())
    direction_sign = 1.0 if direction == 'increase' else -1.0
    threshold = current_value + direction_sign * step
    return current_value, threshold


def evaluate_layer3_directions(
    causal_graph: dict,
    agg_df: pd.DataFrame,
    n_targets: int = 6,
    seed: int = 42,
    max_hops: int = 2,
    verbose: bool = False
) -> pd.DataFrame:
    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    actionable_vars = layer3.identify_actionable_variables(verbose=False)
    all_nodes = list(layer3.dag.nodes())

    reachable_targets = []
    for node in all_nodes:
        paths = layer3.get_valid_causal_paths(node, actionable_vars, max_hops=max_hops)
        if paths:
            reachable_targets.append(node)

    n_pairs_tested = len(reachable_targets) * 2

    rows = []
    for target in reachable_targets:
        for direction in ('increase', 'decrease'):
            current_value, threshold = _default_threshold(layer3, agg_df, target, direction)
            results = layer3.generate_counterfactual(
                target=target, direction=direction, threshold=threshold,
                current_value=current_value, max_hops=max_hops, top_k=3
            )
            for r in results:
                status = classify_direction(r['estimated_target_change'], direction)
                rows.append({
                    'target': target,
                    'direction': direction,
                    'source_variable': r['source_variable'],
                    'estimated_target_change': r['estimated_target_change'],
                    'status': status,
                    'direction_correct': status == 'correct',
                    'flip_success_rate': r['flip_success_rate'],
                    'cf_score': r['cf_score']
                })

    df = pd.DataFrame(rows)
    coverage_info = summarize_pair_coverage(df, n_pairs_tested)
    df.attrs['n_pairs_tested'] = n_pairs_tested
    df.attrs['n_reachable_targets'] = len(reachable_targets)
    df.attrs['n_pairs_covered'] = coverage_info['n_pairs_covered']
    df.attrs['coverage'] = coverage_info['coverage']
    return df


def build_target_diagnostic_table(
    causal_graph: dict,
    agg_df: pd.DataFrame,
    layer1_df: pd.DataFrame,
    max_hops: int = 2
) -> pd.DataFrame:
    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    targets = list(layer3.dag.nodes())

    rows = []
    for target in targets:
        for direction in ('increase', 'decrease'):
            current_value, threshold = _default_threshold(layer3, agg_df, target, direction)
            candidates, _ = layer3.get_candidates(
                target, direction, threshold, current_value, max_hops, verbose=False
            )
            n_candidates = len(candidates)
            n_zero = sum(1 for c in candidates if classify_direction(c['estimated_target_change'], direction) == 'zero_effect')
            zero_rate = n_zero / n_candidates if n_candidates else float('nan')
            n_infeasible = sum(1 for c in candidates if not c.get('reaches_threshold_before_noise', True))
            bound_infeasible_rate = n_infeasible / n_candidates if n_candidates else float('nan')

            target_col = layer3._layer1_col(target)
            if target_col in layer1_df.columns:
                if target_col in RESPONSE_TIME_FALLBACK_COLS:
                    missing_rate = float((layer1_df[target_col] == 0.0).mean())
                else:
                    missing_rate = float(layer1_df[target_col].isna().mean())
            else:
                missing_rate = float('nan')

            direct_parents = sorted({e['source'] for e in causal_graph.get(target, [])})
            expected_meaning = (f"reachable via: {', '.join(direct_parents)}"
                                 if direct_parents else "no direct causal parent in this graph")

            rows.append({
                'target': target,
                'direction': direction,
                'zero_rate': zero_rate,
                'bound_infeasible_rate': bound_infeasible_rate,
                'missing_rate': missing_rate,
                'n_feasible_paths': n_candidates,
                'expected_meaning': expected_meaning
            })

    diagnostic_df = pd.DataFrame(rows)

    print("Response-time feature fallback check (windows with <=5 valid observations after outlier capping):")
    for col in RESPONSE_TIME_FALLBACK_COLS:
        if col in layer1_df.columns:
            n_fallback = int((layer1_df[col] == 0.0).sum())
            print(f"  '{col}': {n_fallback}/{len(layer1_df)} windows ({n_fallback / len(layer1_df):.1%}) fell back to 0.0")

    return diagnostic_df


SELECTION_METHODS = ('CURRENT_WEIGHTED_SCORE', 'RANDOM_FEASIBLE', 'MIN_STANDARDIZED_CHANGE', 'MAX_PREDICTED_MOVE')


def _select_by_method(method: str, candidates, rng):
    if method == 'CURRENT_WEIGHTED_SCORE':
        return Layer3CounterfactualExplanation.select_current_weighted_score(candidates)
    if method == 'RANDOM_FEASIBLE':
        return Layer3CounterfactualExplanation.select_random_feasible(candidates, rng)
    if method == 'MIN_STANDARDIZED_CHANGE':
        return Layer3CounterfactualExplanation.select_min_standardized_change(candidates)
    if method == 'MAX_PREDICTED_MOVE':
        return Layer3CounterfactualExplanation.select_max_predicted_move(candidates)
    raise ValueError(f"Unknown selection method '{method}'")


def run_baseline_comparison(
    causal_graph: dict,
    agg_df: pd.DataFrame,
    max_hops: int = 2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    targets = list(layer3.dag.nodes())
    directions = ('increase', 'decrease')
    rng = np.random.default_rng(seed)

    detail_rows = []
    for target in targets:
        for direction in directions:
            current_value, threshold = _default_threshold(layer3, agg_df, target, direction)
            candidates, _ = layer3.get_candidates(
                target, direction, threshold, current_value, max_hops, verbose=False
            )
            has_candidates = len(candidates) > 0

            for method in SELECTION_METHODS:
                selected = _select_by_method(method, candidates, rng)
                if selected is None:
                    detail_rows.append({
                        'method': method, 'target': target, 'direction': direction,
                        'has_candidates': has_candidates, 'status': 'no_candidate',
                        'flip_success_rate': np.nan, 'cf_score': np.nan,
                        'proximity': np.nan, 'estimated_target_change': np.nan
                    })
                    continue
                status = classify_direction(selected['estimated_target_change'], direction)
                detail_rows.append({
                    'method': method, 'target': target, 'direction': direction,
                    'has_candidates': has_candidates, 'status': status,
                    'flip_success_rate': selected['flip_success_rate'], 'cf_score': selected['cf_score'],
                    'proximity': selected['proximity'], 'estimated_target_change': selected['estimated_target_change']
                })

    detail_df = pd.DataFrame(detail_rows)
    n_pairs_tested = len(targets) * len(directions)

    summary_rows = []
    for method in SELECTION_METHODS:
        sub = detail_df[detail_df['method'] == method]
        covered = sub[sub['status'] != 'no_candidate']
        n_covered = covered[['target', 'direction']].drop_duplicates().shape[0]
        n_total = len(covered)
        n_correct = int((covered['status'] == 'correct').sum())
        n_wrong = int((covered['status'] == 'wrong_direction').sum())
        n_zero = int((covered['status'] == 'zero_effect').sum())
        proximity_clean = covered['proximity'].replace([np.inf, -np.inf], np.nan)
        summary_rows.append({
            'method': method,
            'coverage': n_covered / n_pairs_tested if n_pairs_tested else float('nan'),
            'pct_correct_direction': n_correct / n_total if n_total else float('nan'),
            'pct_wrong_direction': n_wrong / n_total if n_total else float('nan'),
            'pct_zero_effect': n_zero / n_total if n_total else float('nan'),
            'mean_flip_success_rate': covered['flip_success_rate'].mean() if n_total else float('nan'),
            'mean_cf_score': covered['cf_score'].mean() if n_total else float('nan'),
            'mean_proximity': proximity_clean.mean() if n_total else float('nan')
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail_df


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
    n_pairs_covered = layer3_df.attrs.get('n_pairs_covered', None) if layer3_df is not None else None
    coverage = layer3_df.attrs.get('coverage', None) if layer3_df is not None else None
    n_reachable = layer3_df.attrs.get('n_reachable_targets', None) if layer3_df is not None else None
    n_candidate_rows = len(layer3_df) if layer3_df is not None else 0

    n_wrong = int((layer3_df['status'] == 'wrong_direction').sum()) if n_candidate_rows > 0 else 0
    n_zero = int((layer3_df['status'] == 'zero_effect').sum()) if n_candidate_rows > 0 else 0
    n_correct = int((layer3_df['status'] == 'correct').sum()) if n_candidate_rows > 0 else 0

    metrics.append(('DAG nodes reachable via an actionable path', n_reachable if n_reachable is not None else "n/a"))
    metrics.append(('(Target, direction) pairs tested', n_pairs_tested if n_pairs_tested is not None else "n/a"))
    metrics.append(('Pairs covered (>=1 candidate found, unique pairs)', n_pairs_covered if n_pairs_covered is not None else "n/a"))
    metrics.append(('Coverage (unique pairs covered / pairs tested)',
                     f"{coverage:.1%}" if coverage is not None else "n/a"))
    metrics.append(('Total candidate results returned (rows, not pairs)', n_candidate_rows))
    metrics.append(('  of which: correct direction', n_correct))
    metrics.append(('  of which: zero estimated effect', n_zero))
    metrics.append(('  of which: wrong direction (should be 0)', n_wrong))
    metrics.append(('Mean flip_success_rate', f"{layer3_df['flip_success_rate'].mean():.3f}" if n_candidate_rows > 0 else "n/a"))
    metrics.append(('Median flip_success_rate', f"{layer3_df['flip_success_rate'].median():.3f}" if n_candidate_rows > 0 else "n/a"))
    metrics.append(('Mean cf_score', f"{layer3_df['cf_score'].mean():.3f}" if n_candidate_rows > 0 else "n/a"))
    metrics.append(('Median cf_score', f"{layer3_df['cf_score'].median():.3f}" if n_candidate_rows > 0 else "n/a"))

    return pd.DataFrame(metrics, columns=['Metric', 'Value'])


def run_full_evaluation(
    layer1_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    causal_graph: dict,
    n_bootstrap: int = 20,
    n_layer3_targets: int = 6,
    stability_threshold: float = 0.5,
    max_hops: int = 2
) -> dict:
    stability_df = bootstrap_causal_graph(layer1_df, n_bootstrap=n_bootstrap)
    layer3_df = evaluate_layer3_directions(causal_graph, agg_df, n_targets=n_layer3_targets, max_hops=max_hops)
    results_table = build_results_table(stability_df, layer3_df, stability_threshold=stability_threshold)

    print(results_table.to_string(index=False))

    stable_edges = stability_df[stability_df['stability'] >= stability_threshold].sort_values('stability', ascending=False)
    print(f"\nEdges with stability >= {stability_threshold:.0%} (safe to report as headline findings):")
    if len(stable_edges) > 0:
        print(stable_edges[['source', 'target', 'lag', 'stability', 'mean_strength']].to_string(index=False))
    else:
        print("  (none)")

    print("\n--- Target diagnostic table ---")
    diagnostic_df = build_target_diagnostic_table(causal_graph, agg_df, layer1_df, max_hops=max_hops)
    print(diagnostic_df.to_string(index=False))

    print("\n--- Baseline comparison (same candidate set, different selection method) ---")
    baseline_summary_df, baseline_detail_df = run_baseline_comparison(causal_graph, agg_df, max_hops=max_hops)
    print(baseline_summary_df.to_string(index=False))

    return {
        'stability': stability_df,
        'layer3': layer3_df,
        'results_table': results_table,
        'target_diagnostic': diagnostic_df,
        'baseline_summary': baseline_summary_df,
        'baseline_detail': baseline_detail_df
    }