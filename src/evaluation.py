from collections import defaultdict
import numpy as np
import pandas as pd

from src.causal_discovery import Layer2StructureLearning
from src.cf import Layer3CounterfactualExplanation

LAYER2_KWARGS = dict(max_lag=2, alpha=0.1, pc_alpha=0.2, corr_threshold=0.97, min_effect=0.12)


def bootstrap_causal_graph(layer1_df: pd.DataFrame, n_bootstrap: int = 20, seed: int = 42) -> pd.DataFrame:
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

    print(f"\nCausal graph bootstrap: {successful_runs}/{n_bootstrap} successful runs "
          f"(resampling students with replacement)")

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
    if len(result_df) == 0:
        print("No edges recovered in any bootstrap run.")
        return result_df

    print("Edge stability (fraction of bootstrap runs where the edge reappeared):")
    for _, row in result_df.iterrows():
        print(f"  {row['source']} -> {row['target']} @ lag {row['lag']}: "
              f"stability={row['stability']:.0%}, mean_strength={row['mean_strength']:.3f} "
              f"(+/-{row['std_strength']:.3f})")

    low_stability = result_df[result_df['stability'] < 0.5]
    if len(low_stability) > 0:
        print(f"\nWarning: {len(low_stability)} edge(s) appear in under half of bootstrap runs "
              f"and should be treated as unreliable for the thesis's headline results:")
        for _, row in low_stability.iterrows():
            print(f"  {row['source']} -> {row['target']} @ lag {row['lag']} (stability={row['stability']:.0%})")

    return result_df


def evaluate_layer3_directions(
    causal_graph: dict,
    agg_df: pd.DataFrame,
    n_targets: int = 6,
    seed: int = 42
) -> pd.DataFrame:
    layer3 = Layer3CounterfactualExplanation(causal_graph=causal_graph, agg_df=agg_df)
    all_nodes = list(layer3.dag.nodes())

    rng = np.random.default_rng(seed)
    targets = list(all_nodes)
    rng.shuffle(targets)
    targets = targets[:n_targets]

    rows = []
    for target in targets:
        for direction in ('increase', 'decrease'):
            current_value = float(agg_df[target].quantile(0.5))
            threshold = float(agg_df[target].quantile(0.9 if direction == 'increase' else 0.1))
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
    print(f"\nLayer 3 evaluation ran on {len(targets)} sampled target(s): {targets}")

    if len(df) == 0:
        print("No Layer 3 counterfactuals were generated for the sampled targets/directions "
              "(either no actionable path exists, or all candidates were discarded as unreliable).")
        return df

    n_wrong_direction = int((~df['direction_correct']).sum())
    print(f"Direction sanity check: {n_wrong_direction}/{len(df)} results moved the target "
          f"in the WRONG direction relative to what was requested.")
    if n_wrong_direction > 0:
        print("These should NOT appear if the sign-flip fix is working; investigate immediately if any are listed:")
        for _, row in df[~df['direction_correct']].iterrows():
            print(f"  target={row['target']}, direction={row['direction']}, "
                  f"source={row['source_variable']}, estimated_target_change={row['estimated_target_change']:+.3f}")

    print(f"flip_success_rate summary across {len(df)} results: "
          f"mean={df['flip_success_rate'].mean():.3f}, median={df['flip_success_rate'].median():.3f}, "
          f"min={df['flip_success_rate'].min():.3f}, max={df['flip_success_rate'].max():.3f}")
    print(f"cf_score summary: mean={df['cf_score'].mean():.3f}, "
          f"median={df['cf_score'].median():.3f}")

    return df


def run_full_evaluation(layer1_df: pd.DataFrame, agg_df: pd.DataFrame, causal_graph: dict,
                         n_bootstrap: int = 20, n_layer3_targets: int = 6) -> dict:
    print("=" * 60)
    print("EVALUATION 1/2: Causal graph stability (bootstrap over students)")
    print("=" * 60)
    stability_df = bootstrap_causal_graph(layer1_df, n_bootstrap=n_bootstrap)

    print("\n" + "=" * 60)
    print("EVALUATION 2/2: Layer 3 counterfactual quality")
    print("=" * 60)
    layer3_df = evaluate_layer3_directions(causal_graph, agg_df, n_targets=n_layer3_targets)

    return {'stability': stability_df, 'layer3': layer3_df}