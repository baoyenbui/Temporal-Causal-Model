import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional


RAW_COLUMN_MAP = {
    'avg_difficulty': 'question_difficulty',
    'difficulty_std': 'question_difficulty',
    'difficulty_range': 'question_difficulty',
    'avg_response_time': 'time_delta',
    'median_response_time': 'time_delta',
    'response_time_std': 'time_delta',
    'attempts_mean': 'attempts_on_same_question',
    'attempts_max': 'attempts_on_same_question',
    'pct_multi_attempt': 'attempts_on_same_question',
    'avg_success': 'IsCorrect',
    'success_trend': 'IsCorrect',
    'recent5_correct_rate': 'consecutive_correct',
    'max_streak': 'consecutive_correct',
}

LAYER1_TO_GRAPH_FEATURE = {'success_rate': 'avg_success'}
GRAPH_FEATURE_TO_LAYER1 = {v: k for k, v in LAYER1_TO_GRAPH_FEATURE.items()}

DEFAULT_NON_ACTIONABLE = {
    'avg_success', 'success_trend', 'recent5_correct_rate', 'max_streak', 'attempts_max'
}

DEFAULT_CF_SCORE_WEIGHTS = {
    'proximity': 0.20,
    'sparsity': 0.20,
    'effect_snr': 0.20,
    'flip_success_rate': 0.25,
    'causal_plausibility': 0.15
}


class Layer3CounterfactualExplanation:
    def __init__(
        self,
        causal_graph: Dict[str, List[Dict]],
        agg_df: pd.DataFrame,
        actionable_features: Optional[List[str]] = None,
        non_actionable_features: Optional[List[str]] = None,
        lower_bound_quantile: float = 0.01,
        upper_bound_quantile: float = 0.99,
        step_delta_quantile: float = 0.95,
        current_state_window: int = 7
    ):
        self.causal_graph = causal_graph
        self.agg_df = agg_df
        self.actionable_features = set(actionable_features) if actionable_features is not None else None
        self.non_actionable_features = set(non_actionable_features) if non_actionable_features is not None else set()
        self.lower_bound_quantile = lower_bound_quantile
        self.upper_bound_quantile = upper_bound_quantile
        self.step_delta_quantile = step_delta_quantile
        self.current_state_window = current_state_window
        self.dag = self._build_dag()

    @staticmethod
    def _layer1_col(feature: str) -> str:
        return GRAPH_FEATURE_TO_LAYER1.get(feature, feature)

    def get_current_state(self, feature: str, window: Optional[int] = None) -> float:
        w = window if window is not None else self.current_state_window
        if w and w > 1:
            tail = self.agg_df[feature].tail(w)
            return float(tail.median())
        return float(self.agg_df[feature].iloc[-1])

    def _build_dag(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for target, edges in self.causal_graph.items():
            G.add_node(target)
            for edge in edges:
                source = edge['source']
                lag = edge['lag']
                strength = edge['strength']
                G.add_node(source)
                if G.has_edge(source, target):
                    if lag < G[source][target]['lag']:
                        G[source][target]['lag'] = lag
                        G[source][target]['strength'] = strength
                else:
                    G.add_edge(source, target, lag=lag, strength=strength)
        return G

    def identify_actionable_variables(self, verbose: bool = True) -> List[str]:
        all_nodes = set(self.dag.nodes())
        control_vars = {n for n in all_nodes if n.startswith('CTRL_')}
        sinks = {n for n in all_nodes if self.dag.out_degree(n) == 0}

        if self.actionable_features is not None:
            candidates = set(self.actionable_features) & all_nodes
        else:
            candidates = all_nodes - control_vars - DEFAULT_NON_ACTIONABLE

        candidates -= control_vars
        candidates -= sinks
        candidates -= self.non_actionable_features

        candidates = sorted(candidates)
        excluded = sorted(all_nodes - set(candidates))
        if verbose:
            print(f"Layer 3 actionable variables: {candidates}")
            print(f"Layer 3 excluded (control / pure-consequence / non-actionable): {excluded}")
        return candidates

    def get_valid_range(self, feature: str) -> Tuple[float, float]:
        series = self.agg_df[feature].dropna()
        lo = float(series.quantile(self.lower_bound_quantile))
        hi = float(series.quantile(self.upper_bound_quantile))
        return lo, hi

    def get_max_step_delta(self, feature: str) -> float:
        diffs = self.agg_df[feature].diff().dropna().abs()
        if len(diffs) == 0:
            return float('inf')
        step = float(diffs.quantile(self.step_delta_quantile))
        return step if step > 0 else float('inf')

    def get_valid_range_from_windows(self, layer1_df: pd.DataFrame, feature: str) -> Tuple[float, float]:
        series = layer1_df[feature].dropna()
        lo = float(series.quantile(self.lower_bound_quantile))
        hi = float(series.quantile(self.upper_bound_quantile))
        return lo, hi

    def get_max_step_delta_from_windows(self, layer1_df: pd.DataFrame, feature: str) -> float:
        ordered = layer1_df.sort_values(['user_id', 'window_start_time'])
        diffs = ordered.groupby('user_id')[feature].diff().dropna().abs()
        if len(diffs) == 0:
            return float('inf')
        step = float(diffs.quantile(self.step_delta_quantile))
        return step if step > 0 else float('inf')

    def raw_effect(self, standardized_effect: float, source: str, target: str) -> float:
        std_source = float(self.agg_df[source].std())
        std_target = float(self.agg_df[target].std())
        if std_source <= 1e-9:
            return 0.0
        return standardized_effect * (std_target / std_source)

    def compute_cf_score(self, metrics: Dict, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or DEFAULT_CF_SCORE_WEIGHTS
        proximity = metrics['proximity']
        proximity_score = 1.0 / (1.0 + proximity) if np.isfinite(proximity) else 0.0
        sparsity_score = 1.0 / max(metrics['sparsity'], 1)
        snr = metrics['effect_snr']
        snr_score = snr / (1.0 + snr) if np.isfinite(snr) and snr >= 0 else 0.0
        flip = metrics['flip_success_rate']
        if not metrics.get('reaches_threshold_before_noise', True):
            flip = 0.0
        score = (
            w['proximity'] * proximity_score +
            w['sparsity'] * sparsity_score +
            w['effect_snr'] * snr_score +
            w['flip_success_rate'] * flip +
            w['causal_plausibility'] * metrics['causal_plausibility']
        )
        return float(np.clip(score, 0.0, 1.0))

    def get_valid_causal_paths(self, target: str, actionable_vars: List[str], max_hops: int = 2) -> List[Dict]:
        paths = []
        for var in actionable_vars:
            if var == target or not nx.has_path(self.dag, var, target):
                continue
            for path in nx.all_simple_paths(self.dag, var, target, cutoff=max_hops):
                if len(path) - 1 > max_hops:
                    continue
                cumulative_lag = 0
                cumulative_strength = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    edge = self.dag[u][v]
                    cumulative_lag += edge['lag']
                    cumulative_strength *= edge['strength']
                raw_estimated_effect = self.raw_effect(cumulative_strength, var, target)
                paths.append({
                    'source': var,
                    'path': path,
                    'hops': len(path) - 1,
                    'total_lag': cumulative_lag,
                    'estimated_effect_standardized': cumulative_strength,
                    'estimated_effect': raw_estimated_effect
                })
        paths.sort(key=lambda p: (p['hops'], -abs(p['estimated_effect_standardized'])))
        return paths

    def apply_intervention_to_window(self, window_data: pd.DataFrame, feature: str, delta: float, layer1_builder) -> Dict:
        raw_col = RAW_COLUMN_MAP.get(feature)
        if raw_col is None or raw_col not in window_data.columns:
            raise ValueError(f"No raw-column mapping available for feature '{feature}'; cannot intervene at the window level.")

        intervened = window_data.copy()

        if raw_col == 'question_difficulty':
            intervened[raw_col] = (intervened[raw_col] + delta).clip(lower=0)
        elif raw_col == 'time_delta':
            mask = intervened[raw_col] > 0
            intervened.loc[mask, raw_col] = (intervened.loc[mask, raw_col] + delta).clip(lower=0.1)
        elif raw_col == 'attempts_on_same_question':
            intervened[raw_col] = (intervened[raw_col] + delta).round().clip(lower=1)
        elif raw_col in ('consecutive_correct',):
            intervened[raw_col] = (intervened[raw_col] + delta).clip(lower=0)
        elif raw_col == 'IsCorrect':
            flip_prob = min(abs(delta), 1.0)
            rng = np.random.default_rng(0)
            flip_mask = rng.random(len(intervened)) < flip_prob
            direction = 1 if delta > 0 else -1
            if direction > 0:
                intervened.loc[flip_mask & (intervened[raw_col] == 0), raw_col] = 1
            else:
                intervened.loc[flip_mask & (intervened[raw_col] == 1), raw_col] = 0

        return layer1_builder.extract_window_features(intervened)

    def evaluate_counterfactual(
        self,
        current_target_value: float,
        delta: float,
        estimated_effect: float,
        hops: int,
        source: str,
        threshold: float,
        direction: str,
        source_std: float,
        source_max_step: float,
        n_trials: int = 200,
        seed: int = 42
    ) -> Dict:
        sparsity = 1

        proximity = abs(delta) / source_std if source_std and source_std > 0 else float('nan')

        if np.isfinite(source_max_step) and source_max_step > 0:
            noise_std = source_max_step / 1.645
        else:
            noise_std = abs(delta) * 0.1 + 1e-6

        rng = np.random.default_rng(seed)
        noisy_deltas = rng.normal(delta, noise_std, n_trials)
        simulated_target = current_target_value + noisy_deltas * estimated_effect

        projected_value = current_target_value + delta * estimated_effect
        reaches_threshold_before_noise = (
            projected_value >= threshold if direction == 'increase' else projected_value <= threshold
        )

        if not reaches_threshold_before_noise:
            flip_success_rate = 0.0
        else:
            if direction == 'increase':
                flips = simulated_target >= threshold
            else:
                flips = simulated_target <= threshold
            flip_success_rate = float(np.mean(flips))

        outcome_spread = np.std(noisy_deltas * estimated_effect)
        expected_move = abs(delta * estimated_effect) + 1e-6
        effect_snr = float(expected_move / (outcome_spread + 1e-6))

        actionable_vars = self.identify_actionable_variables(verbose=False)
        causal_plausibility = (1.0 / hops) * (1.0 if source in actionable_vars else 0.0)

        return {
            'proximity': proximity,
            'sparsity': sparsity,
            'effect_snr': effect_snr,
            'flip_success_rate': flip_success_rate,
            'causal_plausibility': causal_plausibility,
            'reaches_threshold_before_noise': reaches_threshold_before_noise
        }

    def get_candidates(
        self,
        target: str,
        direction: str,
        threshold: float,
        current_value: Optional[float] = None,
        max_hops: int = 2,
        verbose: bool = True,
        min_flip_success_rate: float = 0.0
    ) -> Tuple[List[Dict], Optional[float]]:
        if direction not in ('increase', 'decrease'):
            raise ValueError("direction must be 'increase' or 'decrease'")

        if target not in self.dag.nodes():
            if verbose:
                print(f"'{target}' is not part of the discovered causal graph; no counterfactual can be generated.")
            return [], current_value

        if current_value is None:
            current_value = self.get_current_state(target)

        gap = threshold - current_value
        if (direction == 'increase' and gap <= 0) or (direction == 'decrease' and gap >= 0):
            if verbose:
                print(f"'{target}' already satisfies the '{direction}' goal relative to threshold={threshold:.3f} "
                      f"(current={current_value:.3f}); no intervention needed.")
            return [], current_value

        actionable_vars = self.identify_actionable_variables(verbose=False)
        paths = self.get_valid_causal_paths(target, actionable_vars, max_hops=max_hops)

        if not paths:
            if verbose:
                print(f"No actionable causal path leads to '{target}' within {max_hops} hop(s); "
                      f"the discovered DAG offers no valid lever for this outcome.")
            return [], current_value

        candidates = []
        for path_info in paths:
            effect = path_info['estimated_effect']
            if abs(effect) < 1e-9:
                continue

            source = path_info['source']
            required_delta = gap / effect

            lo, hi = self.get_valid_range(source)
            max_step = self.get_max_step_delta(source)
            source_current = self.get_current_state(source)

            if source_current < lo or source_current > hi:
                if verbose:
                    print(f"Skipping '{source}': its current value {source_current:.3f} already lies outside the "
                          f"typical historical range [{lo:.3f}, {hi:.3f}]; clipping from this baseline would not "
                          f"reflect the intended intervention.")
                continue

            clipped_delta = float(np.clip(required_delta, -max_step, max_step))
            proposed_value = float(np.clip(source_current + clipped_delta, lo, hi))
            actual_delta = proposed_value - source_current

            if abs(actual_delta) < 1e-9:
                continue

            metrics = self.evaluate_counterfactual(
                current_target_value=current_value,
                delta=actual_delta,
                estimated_effect=effect,
                hops=path_info['hops'],
                source=source,
                threshold=threshold,
                direction=direction,
                source_std=self.agg_df[source].std(),
                source_max_step=max_step
            )

            cf_score = self.compute_cf_score(metrics)

            if verbose and not metrics['reaches_threshold_before_noise']:
                print(f"Note: '{source}' is bound-constrained — within the realistic step limit "
                      f"(max_step={max_step:.3f}), the best deterministic change only moves the target to "
                      f"{current_value + actual_delta * effect:.3f}, short of threshold={threshold:.3f}. flip_success_rate here reflects "
                      f"structural infeasibility under realistic bounds, not just noise sensitivity.")

            candidates.append({
                'target': target,
                'direction': direction,
                'source_variable': source,
                'causal_path': ' -> '.join(path_info['path']),
                'lag_days': path_info['total_lag'],
                'current_source_value': source_current,
                'proposed_source_value': proposed_value,
                'delta': actual_delta,
                'estimated_target_change': actual_delta * effect,
                'reaches_threshold_before_noise': metrics['reaches_threshold_before_noise'],
                'valid_range': (lo, hi),
                'max_step_delta': max_step,
                **{k: v for k, v in metrics.items() if k != 'reaches_threshold_before_noise'},
                'cf_score': cf_score
            })

        if verbose:
            for c in candidates:
                if c['proximity'] > 2.0:
                    print(f"Caution: proposed change to '{c['source_variable']}' is {c['proximity']:.1f}x its typical "
                          f"day-to-day standard deviation — a large, less realistic intervention.")

        if min_flip_success_rate > 0:
            filtered = [c for c in candidates if c['flip_success_rate'] >= min_flip_success_rate]
            if verbose and len(filtered) < len(candidates):
                print(f"Hard filter: dropped {len(candidates) - len(filtered)} candidate(s) with "
                      f"flip_success_rate < {min_flip_success_rate}")
            candidates = filtered

        return candidates, current_value

    @staticmethod
    def _filter_reliable(candidates: List[Dict], reliability_cutoff: float) -> List[Dict]:
        if not candidates or reliability_cutoff <= 0:
            return candidates
        reliable = [c for c in candidates if c['flip_success_rate'] >= reliability_cutoff]
        return reliable if reliable else candidates

    @staticmethod
    def select_current_weighted_score(candidates: List[Dict], reliability_cutoff: float = 0.5) -> Optional[Dict]:
        if not candidates:
            return None
        pool = Layer3CounterfactualExplanation._filter_reliable(candidates, reliability_cutoff)
        ranked = sorted(pool, key=lambda r: (
            0 if r['flip_success_rate'] >= reliability_cutoff else 1,
            -r['cf_score']
        ))
        return ranked[0]

    @staticmethod
    def select_random_feasible(candidates: List[Dict], rng: np.random.Generator, reliability_cutoff: float = 0.0) -> Optional[Dict]:
        if not candidates:
            return None
        pool = Layer3CounterfactualExplanation._filter_reliable(candidates, reliability_cutoff)
        idx = int(rng.integers(0, len(pool)))
        return pool[idx]

    @staticmethod
    def select_min_standardized_change(candidates: List[Dict], reliability_cutoff: float = 0.0) -> Optional[Dict]:
        if not candidates:
            return None
        pool = Layer3CounterfactualExplanation._filter_reliable(candidates, reliability_cutoff)
        finite = [c for c in pool if np.isfinite(c['proximity'])]
        pool = finite if finite else pool
        return min(pool, key=lambda c: c['proximity'])

    @staticmethod
    def select_max_predicted_move(candidates: List[Dict], reliability_cutoff: float = 0.0) -> Optional[Dict]:
        if not candidates:
            return None
        pool = Layer3CounterfactualExplanation._filter_reliable(candidates, reliability_cutoff)
        return max(pool, key=lambda c: abs(c['estimated_target_change']))

    def generate_counterfactual(
        self,
        target: str,
        direction: str,
        threshold: float,
        current_value: Optional[float] = None,
        max_hops: int = 2,
        top_k: int = 3
    ) -> List[Dict]:
        candidates, _ = self.get_candidates(target, direction, threshold, current_value, max_hops, verbose=True)
        if not candidates:
            return []
        reliability_cutoff = 0.5
        ranked = sorted(candidates, key=lambda r: (
            0 if r['flip_success_rate'] >= reliability_cutoff else 1,
            -r['cf_score']
        ))
        return ranked[:top_k]

    def simulate_max_feasible_effect(
        self,
        target: str,
        direction: str,
        max_hops: int = 2,
        current_value: Optional[float] = None
    ) -> List[Dict]:
        if direction not in ('increase', 'decrease'):
            raise ValueError("direction must be 'increase' or 'decrease'")

        if target not in self.dag.nodes():
            return []

        if current_value is None:
            current_value = self.get_current_state(target)

        actionable_vars = self.identify_actionable_variables(verbose=False)
        paths = self.get_valid_causal_paths(target, actionable_vars, max_hops=max_hops)
        if not paths:
            return []

        wanted_sign = 1.0 if direction == 'increase' else -1.0
        results = []
        for path_info in paths:
            source = path_info['source']
            effect = path_info['estimated_effect']
            if abs(effect) < 1e-9:
                continue

            lo, hi = self.get_valid_range(source)
            max_step = self.get_max_step_delta(source)
            source_current = self.get_current_state(source)

            if source_current < lo or source_current > hi:
                continue
            if not np.isfinite(max_step) or max_step <= 0:
                continue

            push_sign = wanted_sign if effect > 0 else -wanted_sign
            raw_delta = push_sign * max_step
            proposed_value = float(np.clip(source_current + raw_delta, lo, hi))
            actual_delta = proposed_value - source_current
            if abs(actual_delta) < 1e-9:
                continue

            achievable_target_value = current_value + actual_delta * effect
            achievable_change = achievable_target_value - current_value

            results.append({
                'target': target,
                'direction': direction,
                'source_variable': source,
                'causal_path': ' -> '.join(path_info['path']),
                'lag_days': path_info['total_lag'],
                'current_source_value': source_current,
                'max_feasible_source_value': proposed_value,
                'max_feasible_delta': actual_delta,
                'current_target_value': current_value,
                'achievable_target_value': achievable_target_value,
                'achievable_change': achievable_change,
                'moves_correct_direction': (achievable_change > 0) == (wanted_sign > 0)
            })

        results.sort(key=lambda r: -abs(r['achievable_change']))
        return results

    def generate_student_counterfactuals(
        self,
        layer1_df: pd.DataFrame,
        target: str = 'max_streak',
        direction: str = 'increase',
        n_users: int = 5,
        threshold_quantile: float = 0.60,
        selection: str = 'lowest',
        max_hops: int = 2
    ) -> pd.DataFrame:
        target_col = self._layer1_col(target)
        if target_col not in layer1_df.columns:
            raise ValueError(f"'{target}' (layer1 column '{target_col}') not found in layer1_df")

        user_stats = layer1_df.groupby('user_id')[target_col].mean().rename('current_value').reset_index()
        threshold = float(layer1_df[target_col].quantile(threshold_quantile))
        user_stats = user_stats.sort_values('current_value', ascending=(selection == 'lowest'))
        selected_users = user_stats.head(n_users)

        actionable_vars = self.identify_actionable_variables(verbose=False)
        paths = self.get_valid_causal_paths(target, actionable_vars, max_hops=max_hops)

        print(f"Selected {len(selected_users)} students (lowest '{target}') to test counterfactual push toward "
              f"class threshold={threshold:.3f}")

        rows = []
        for _, urow in selected_users.iterrows():
            user_id = urow['user_id']
            current_value = float(urow['current_value'])
            user_windows = layer1_df[layer1_df['user_id'] == user_id].sort_values('window_start_time')

            gap = threshold - current_value
            goal_met = (direction == 'increase' and gap <= 0) or (direction == 'decrease' and gap >= 0)

            if goal_met or not paths:
                rows.append({
                    'user_id': user_id, 'target': target, 'current_value': current_value,
                    'threshold': threshold, 'status': 'no_intervention_needed' if goal_met else 'no_causal_path',
                    'cf_score': np.nan
                })
                continue

            best = None
            for path_info in paths:
                effect = path_info['estimated_effect']
                if abs(effect) < 1e-9:
                    continue

                source = path_info['source']
                source_col = self._layer1_col(source)
                if source_col not in user_windows.columns:
                    continue

                window_count = min(self.current_state_window, len(user_windows))
                source_current = float(user_windows[source_col].tail(window_count).median())
                lo, hi = self.get_valid_range_from_windows(layer1_df, source_col)
                max_step = self.get_max_step_delta_from_windows(layer1_df, source_col)

                if source_current < lo or source_current > hi:
                    continue

                required_delta = gap / effect
                clipped_delta = float(np.clip(required_delta, -max_step, max_step))
                proposed_value = float(np.clip(source_current + clipped_delta, lo, hi))
                actual_delta = proposed_value - source_current
                if abs(actual_delta) < 1e-9:
                    continue

                metrics = self.evaluate_counterfactual(
                    current_target_value=current_value,
                    delta=actual_delta,
                    estimated_effect=effect,
                    hops=path_info['hops'],
                    source=source,
                    threshold=threshold,
                    direction=direction,
                    source_std=layer1_df[source_col].std(),
                    source_max_step=max_step
                )
                cf_score = self.compute_cf_score(metrics)

                candidate = {
                    'user_id': user_id,
                    'target': target,
                    'current_value': current_value,
                    'threshold': threshold,
                    'source_variable': source,
                    'causal_path': ' -> '.join(path_info['path']),
                    'lag_days': path_info['total_lag'],
                    'current_source_value': source_current,
                    'proposed_source_value': proposed_value,
                    'delta': actual_delta,
                    'estimated_target_change': actual_delta * effect,
                    'reaches_threshold_before_noise': metrics['reaches_threshold_before_noise'],
                    **{k: v for k, v in metrics.items() if k != 'reaches_threshold_before_noise'},
                    'cf_score': cf_score,
                    'status': 'ok' if metrics['reaches_threshold_before_noise'] else 'bound_constrained_infeasible'
                }
                if best is None or cf_score > best['cf_score']:
                    best = candidate

            rows.append(best if best is not None else {
                'user_id': user_id, 'target': target, 'current_value': current_value,
                'threshold': threshold, 'status': 'no_valid_intervention_within_bounds', 'cf_score': np.nan
            })

        return pd.DataFrame(rows)


def run_layer3_pipeline(
    causal_graph: Dict[str, List[Dict]],
    agg_df: pd.DataFrame,
    target: str,
    direction: str,
    threshold: float,
    actionable_features: Optional[List[str]] = None,
    non_actionable_features: Optional[List[str]] = None,
    max_hops: int = 2,
    top_k: int = 3
) -> List[Dict]:
    layer3 = Layer3CounterfactualExplanation(
        causal_graph=causal_graph,
        agg_df=agg_df,
        actionable_features=actionable_features,
        non_actionable_features=non_actionable_features
    )
    results = layer3.generate_counterfactual(
        target=target,
        direction=direction,
        threshold=threshold,
        max_hops=max_hops,
        top_k=top_k
    )
    return results