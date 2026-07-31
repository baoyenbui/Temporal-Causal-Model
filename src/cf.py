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
    'stability': 0.20,
    'flip_success_rate': 0.25,
    'causal_plausibility': 0.15
}

MIN_RELIABLE_FLIP_RATE = 0.10


class Layer3CounterfactualExplanation:
    def __init__(
        self,
        causal_graph: Dict[str, List[Dict]],
        agg_df: pd.DataFrame,
        actionable_features: Optional[List[str]] = None,
        non_actionable_features: Optional[List[str]] = None,
        lower_bound_quantile: float = 0.01,
        upper_bound_quantile: float = 0.99,
        step_delta_quantile: float = 0.95
    ):
        self.causal_graph = causal_graph
        self.agg_df = agg_df
        self.actionable_features = set(actionable_features) if actionable_features is not None else None
        self.non_actionable_features = set(non_actionable_features) if non_actionable_features is not None else set()
        self.lower_bound_quantile = lower_bound_quantile
        self.upper_bound_quantile = upper_bound_quantile
        self.step_delta_quantile = step_delta_quantile
        self.dag = self._build_dag()

    @staticmethod
    def _layer1_col(feature: str) -> str:
        return GRAPH_FEATURE_TO_LAYER1.get(feature, feature)

    def _build_dag(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for target, edges in self.causal_graph.items():
            G.add_node(target)
            for edge in edges:
                source = edge['source']
                lag = edge['lag']
                strength = edge['strength']
                p_value = edge.get('p_value', edge.get('q_value', 1.0))
                G.add_node(source)
                if G.has_edge(source, target):
                    if lag < G[source][target]['lag']:
                        G[source][target]['lag'] = lag
                        G[source][target]['strength'] = strength
                        G[source][target]['p_value'] = p_value
                else:
                    G.add_edge(source, target, lag=lag, strength=strength, p_value=p_value)
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

    def raw_effect(self, path: List[str], standardized_effect: float, source: str, target: str) -> float:
        std_source = float(self.agg_df[source].std())
        std_target = float(self.agg_df[target].std())
        if std_source <= 1e-9:
            return 0.0
        return standardized_effect * (std_target / std_source)

    def path_plausibility(self, path: List[str]) -> float:
        confidence = 1.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.dag[u][v]
            p_val = edge.get('p_value', 1.0)
            confidence *= max(1.0 - p_val, 0.0)
        return float(np.clip(confidence, 0.0, 1.0))

    def compute_cf_score(self, metrics: Dict, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or DEFAULT_CF_SCORE_WEIGHTS
        proximity = metrics['proximity']
        proximity_score = 1.0 / (1.0 + proximity) if np.isfinite(proximity) else 0.0
        sparsity_score = 1.0 / max(metrics['sparsity'], 1)
        score = (
            w['proximity'] * proximity_score +
            w['sparsity'] * sparsity_score +
            w['stability'] * metrics['stability'] +
            w['flip_success_rate'] * metrics['flip_success_rate'] +
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
                raw_estimated_effect = self.raw_effect(path, cumulative_strength, var, target)
                paths.append({
                    'source': var,
                    'path': path,
                    'hops': len(path) - 1,
                    'total_lag': cumulative_lag,
                    'estimated_effect_standardized': cumulative_strength,
                    'estimated_effect': raw_estimated_effect,
                    'plausibility': self.path_plausibility(path)
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
        plausibility: float,
        source_std: float,
        source_max_step: float,
        threshold: float,
        direction: str,
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

        if direction == 'increase':
            flips = simulated_target >= threshold
        else:
            flips = simulated_target <= threshold
        flip_success_rate = float(np.mean(flips))

        outcome_spread = np.std(noisy_deltas * estimated_effect)
        expected_move = abs(delta * estimated_effect) + 1e-6
        stability = float(np.clip(1.0 - outcome_spread / expected_move, 0.0, 1.0))

        return {
            'proximity': proximity,
            'sparsity': sparsity,
            'stability': stability,
            'flip_success_rate': flip_success_rate,
            'causal_plausibility': plausibility
        }

    def _resolve_intervention(self, source: str, required_delta: float, direction_sign: float) -> Optional[Dict]:
        lo, hi = self.get_valid_range(source)
        max_step = self.get_max_step_delta(source)
        source_current = float(self.agg_df[source].iloc[-1])

        if not (lo <= source_current <= hi):
            print(f"Skipping path via '{source}': current value {source_current:.3f} is already outside "
                  f"its typical range [{lo:.3f}, {hi:.3f}] (looks like an outlier day); "
                  f"clipping to range would misrepresent the direction of change.")
            return None

        clipped_delta = float(np.clip(required_delta, -max_step, max_step))
        proposed_value = float(np.clip(source_current + clipped_delta, lo, hi))
        actual_delta = proposed_value - source_current

        if abs(actual_delta) < 1e-9:
            return None

        required_sign = np.sign(required_delta) if required_delta != 0 else direction_sign
        actual_sign = np.sign(actual_delta)
        if required_sign != 0 and actual_sign != 0 and actual_sign != required_sign:
            print(f"Skipping path via '{source}': range clipping reversed the intended direction "
                  f"(wanted {'increase' if required_sign > 0 else 'decrease'}, "
                  f"got {'increase' if actual_sign > 0 else 'decrease'}); "
                  f"'{source}' is too close to its historical extreme to move further that way.")
            return None

        return {
            'source_current': source_current,
            'proposed_value': proposed_value,
            'actual_delta': actual_delta,
            'valid_range': (lo, hi),
            'max_step': max_step
        }

    def generate_counterfactual(
        self,
        target: str,
        direction: str,
        threshold: float,
        current_value: Optional[float] = None,
        max_hops: int = 2,
        top_k: int = 3
    ) -> List[Dict]:
        if direction not in ('increase', 'decrease'):
            raise ValueError("direction must be 'increase' or 'decrease'")

        if target not in self.dag.nodes():
            print(f"'{target}' is not part of the discovered causal graph; no counterfactual can be generated.")
            return []

        if current_value is None:
            current_value = float(self.agg_df[target].iloc[-1])

        gap = threshold - current_value
        direction_sign = 1.0 if direction == 'increase' else -1.0
        if (direction == 'increase' and gap <= 0) or (direction == 'decrease' and gap >= 0):
            print(f"'{target}' already satisfies the '{direction}' goal relative to threshold={threshold:.3f} "
                  f"(current={current_value:.3f}); no intervention needed.")
            return []

        actionable_vars = self.identify_actionable_variables(verbose=False)
        paths = self.get_valid_causal_paths(target, actionable_vars, max_hops=max_hops)

        if not paths:
            print(f"No actionable causal path leads to '{target}' within {max_hops} hop(s); "
                  f"the discovered DAG offers no valid lever for this outcome.")
            return []

        results = []
        for path_info in paths:
            effect = path_info['estimated_effect']
            if abs(effect) < 1e-9:
                continue

            source = path_info['source']
            required_delta = gap / effect

            resolved = self._resolve_intervention(source, required_delta, direction_sign)
            if resolved is None:
                continue

            actual_delta = resolved['actual_delta']

            metrics = self.evaluate_counterfactual(
                current_target_value=current_value,
                delta=actual_delta,
                estimated_effect=effect,
                plausibility=path_info['plausibility'],
                source_std=self.agg_df[source].std(),
                source_max_step=resolved['max_step'],
                threshold=threshold,
                direction=direction
            )

            if metrics['flip_success_rate'] < MIN_RELIABLE_FLIP_RATE:
                print(f"Discarding path via '{source}': flip_success_rate={metrics['flip_success_rate']:.3f} "
                      f"is below the {MIN_RELIABLE_FLIP_RATE:.0%} reliability floor; "
                      f"this intervention essentially never reaches the goal under realistic noise.")
                continue

            cf_score = self.compute_cf_score(metrics)

            results.append({
                'target': target,
                'source_variable': source,
                'causal_path': ' -> '.join(path_info['path']),
                'lag_days': path_info['total_lag'],
                'current_source_value': resolved['source_current'],
                'proposed_source_value': resolved['proposed_value'],
                'delta': actual_delta,
                'estimated_target_change': actual_delta * effect,
                'valid_range': resolved['valid_range'],
                'max_step_delta': resolved['max_step'],
                **metrics,
                'cf_score': cf_score
            })

        results.sort(key=lambda r: -r['cf_score'])
        for r in results:
            if r['proximity'] > 2.0:
                print(f"Caution: proposed change to '{r['source_variable']}' is {r['proximity']:.1f}x its typical "
                      f"day-to-day standard deviation — a large, less realistic intervention.")
        return results[:top_k]

    def generate_student_counterfactuals(
        self,
        layer1_df: pd.DataFrame,
        target: str = 'max_streak',
        direction: str = 'increase',
        n_users: int = 5,
        threshold_quantile: float = 0.75,
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
        direction_sign = 1.0 if direction == 'increase' else -1.0

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

                source_current = float(user_windows[source_col].iloc[-1])
                lo, hi = self.get_valid_range_from_windows(layer1_df, source_col)
                max_step = self.get_max_step_delta_from_windows(layer1_df, source_col)

                if not (lo <= source_current <= hi):
                    continue

                required_delta = gap / effect
                clipped_delta = float(np.clip(required_delta, -max_step, max_step))
                proposed_value = float(np.clip(source_current + clipped_delta, lo, hi))
                actual_delta = proposed_value - source_current
                if abs(actual_delta) < 1e-9:
                    continue

                required_sign = np.sign(required_delta) if required_delta != 0 else direction_sign
                actual_sign = np.sign(actual_delta)
                if required_sign != 0 and actual_sign != 0 and actual_sign != required_sign:
                    continue

                metrics = self.evaluate_counterfactual(
                    current_target_value=current_value,
                    delta=actual_delta,
                    estimated_effect=effect,
                    plausibility=path_info['plausibility'],
                    source_std=layer1_df[source_col].std(),
                    source_max_step=max_step,
                    threshold=threshold,
                    direction=direction
                )

                if metrics['flip_success_rate'] < MIN_RELIABLE_FLIP_RATE:
                    continue

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
                    **metrics,
                    'cf_score': cf_score,
                    'status': 'ok'
                }
                if best is None or cf_score > best['cf_score']:
                    best = candidate

            rows.append(best if best is not None else {
                'user_id': user_id, 'target': target, 'current_value': current_value,
                'threshold': threshold, 'status': 'no_reliable_intervention_within_bounds', 'cf_score': np.nan
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