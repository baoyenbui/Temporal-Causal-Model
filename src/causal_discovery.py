import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.data_loader import load_data
from sklearn.preprocessing import StandardScaler
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame

class Layer1TemporalConstruction:
    def __init__(self, window_size: int = 15, step_size: int = 5):
        self.window_size = window_size
        self.step_size = step_size

    def sort_by_time(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        df = raw_logs.copy()

        if 'step_index_scaled' in df.columns:
            df = df.sort_values('step_index_scaled').reset_index(drop=True)
            df['timestamp'] = df['step_index_scaled']
        else:
            df = df.reset_index(drop=True)
            df['timestamp'] = df.index

        return df

    def create_sliding_windows(self, sorted_df: pd.DataFrame) -> List[Dict]:
        windows_list = []
        n = len(sorted_df)

        if n < self.window_size:
            return windows_list

        for start_idx in range(0, n - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window_data = sorted_df.iloc[start_idx:end_idx].copy()

            window_info = {
                'step_index': start_idx,
                'window_start_time': window_data['timestamp'].min(),
                'window_end_time': window_data['timestamp'].max(),
                'cluster_id': window_data['student_cluster'].iloc[0] if 'student_cluster' in window_data.columns else -1
            }

            windows_list.append({
                'window_info': window_info,
                'window_data': window_data
            })

        return windows_list

    def extract_window_features(self, window_data: pd.DataFrame) -> Dict:
        features = {}
        features['success_rate'] = window_data['is_correct'].mean()

        if 'question_difficulty' in window_data.columns:
            features['avg_difficulty'] = window_data['question_difficulty'].mean()
            features['difficulty_std'] = window_data['question_difficulty'].std()
            features['difficulty_range'] = window_data['question_difficulty'].max() - window_data['question_difficulty'].min()

        if 'time_delta' in window_data.columns:
            features['avg_response_time'] = window_data['time_delta'].mean()
            features['response_time_std'] = window_data['time_delta'].std()

        features['engagement_proxy'] = len(window_data)

        if 'engagement_proxy' in window_data.columns:
            features['avg_engagement_proxy'] = window_data['engagement_proxy'].mean()

        return features

    def process_chunk(self, windows_list: List[Dict], chunk_size: int = 10000) -> pd.DataFrame:
        chunk_rows = []

        for i, window_item in enumerate(windows_list):
            window_info = window_item['window_info']
            window_data = window_item['window_data']

            features = self.extract_window_features(window_data)
            row = {**window_info, **features}
            chunk_rows.append(row)

        chunk_df = pd.DataFrame(chunk_rows)
        return chunk_df

    def build_layer1(self, raw_logs: pd.DataFrame, student_clusters: pd.DataFrame = None) -> pd.DataFrame:
        sorted_df = self.sort_by_time(raw_logs)
        windows_list = self.create_sliding_windows(sorted_df)
        layer1_df = self.process_chunk(windows_list)
        layer1_df = layer1_df.reset_index(drop=True)

        return layer1_df


class Layer2StructureLearning:
    def __init__(self, max_lag: int = 5, alpha: float = 0.05, bootstrap_runs: int = 100, bootstrap_threshold: float = 0.7):
        self.max_lag = max_lag
        self.alpha = alpha
        self.bootstrap_runs = bootstrap_runs
        self.bootstrap_threshold = bootstrap_threshold

    def aggregate_time_series(self, layer1_df: pd.DataFrame, bucket_size: int = 10000) -> pd.DataFrame:
        layer1_df = layer1_df.copy()
        layer1_df = layer1_df.sort_values('step_index')

        layer1_df['time_bucket'] = (layer1_df['step_index'] // bucket_size) * bucket_size

        agg_df = layer1_df.groupby('time_bucket').agg(
            avg_success=('success_rate', 'mean'),
            avg_engagement=('engagement_proxy', 'mean'),
            avg_difficulty=('avg_difficulty', 'mean'),
            total_windows=('step_index', 'count'),
            avg_response_time=('avg_response_time', 'mean')
        ).reset_index()

        agg_df['rolling_success_mean'] = agg_df['avg_success'].rolling(window=5, min_periods=1).mean()
        agg_df['rolling_success_std'] = agg_df['avg_success'].rolling(window=5, min_periods=1).std()
        agg_df['rolling_engagement_trend'] = agg_df['avg_engagement'].rolling(window=5, min_periods=1).mean()

        if 'cluster_id' in layer1_df.columns:
            cluster_dist = layer1_df.groupby(['time_bucket', 'cluster_id']).size().unstack(fill_value=0)
            cluster_dist.columns = [f'cluster_{c}_ratio' for c in cluster_dist.columns]
            cluster_dist = cluster_dist.div(cluster_dist.sum(axis=1), axis=0).fillna(0)
            agg_df = agg_df.merge(cluster_dist, on='time_bucket', how='left')

        agg_df = agg_df.fillna(0)

        return agg_df

    def prepare_timeseries_matrix(self, agg_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        exclude_cols = ['time_bucket', 'avg_engagement', 'engagement_trend']
        feature_cols = [col for col in agg_df.columns if col not in exclude_cols]
        
        X = agg_df[feature_cols].values
        stds = np.std(X, axis=0)
        mask = stds > 1e-8
        feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
        X = X[:, mask]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, pd.DataFrame(X_scaled, columns=feature_cols)

    def run_pcmci(self, feature_df: pd.DataFrame) -> Dict:
        n_obs, n_features = feature_df.shape
        feature_names = feature_df.columns.tolist()
        max_lag_safe = max(1, min(self.max_lag, (n_obs // n_features) - 1))

        if max_lag_safe < 1 or n_obs < 20:
            return {feature: [] for feature in feature_names}

        dataframe = DataFrame(data=feature_df.values, var_names=feature_names)
        cond_ind_test = ParCorr(significance='analytic')   # Đổi thành 'analytic'

        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=max_lag_safe, pc_alpha=0.05)

        p_matrix = results['p_matrix']
        val_matrix = results['val_matrix']

        causal_graph = {}
        for i, target in enumerate(feature_names):
            causal_graph[target] = []
            for j, source in enumerate(feature_names):
                if i == j:
                    continue
                for lag in range(1, max_lag_safe + 1):
                    p_val = p_matrix[j, i, lag]
                    if p_val < self.alpha:
                        causal_graph[target].append({
                            'source': source,
                            'lag': lag,
                            'p_value': p_val
                        })
        return causal_graph

    def bootstrap_stability(self, feature_df: pd.DataFrame) -> Dict:
        edge_counts = {}
        n_samples = len(feature_df)

        for run in range(self.bootstrap_runs):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = feature_df.iloc[indices]

            if len(X_boot) < self.max_lag + 10:
                continue

            causal_graph = self.run_pcmci(X_boot)

            for target, sources in causal_graph.items():
                for edge in sources:
                    edge_key = f"{edge['source']}->{target}@lag{edge['lag']}"
                    edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        stable_edges = {}

        for edge_key, count in edge_counts.items():
            frequency = count / self.bootstrap_runs
            if frequency >= self.bootstrap_threshold:
                stable_edges[edge_key] = frequency

        return stable_edges

    def add_latent_proxies(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        agg_df['ability_proxy'] = agg_df['avg_success'] * agg_df['avg_engagement']

        if 'avg_difficulty' in agg_df.columns:
            agg_df['learning_efficiency'] = agg_df['avg_success'] / (agg_df['avg_difficulty'] + 0.01)

        engagement_cols = [col for col in agg_df.columns if 'engagement' in col.lower()]
        if len(engagement_cols) > 0:
            agg_df['engagement_trend'] = agg_df[engagement_cols].diff().fillna(0).mean(axis=1)

        success_cols = [col for col in agg_df.columns if 'success' in col.lower()]
        if len(success_cols) > 0:
            agg_df['success_trend'] = agg_df[success_cols].diff().fillna(0).mean(axis=1)

        return agg_df

    def build_layer2(self, layer1_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        agg_df = self.aggregate_time_series(layer1_df)
        agg_df = self.add_latent_proxies(agg_df)
        X_scaled, feature_df = self.prepare_timeseries_matrix(agg_df)
        causal_graph = self.run_pcmci(feature_df)
        stable_edges = self.bootstrap_stability(feature_df)

        return agg_df, causal_graph, stable_edges


def run_layer1_pipeline(
    base_path: str = 'data',
    window_size: int = 15,
    step_size: int = 5
) -> pd.DataFrame:
    data = load_data(base_path)
    raw_logs = data['training_data']

    layer1_builder = Layer1TemporalConstruction(window_size=window_size, step_size=step_size)
    layer1_df = layer1_builder.build_layer1(raw_logs)

    return layer1_df


def run_layer2_pipeline(
    layer1_df: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    bootstrap_runs: int = 100,
    bootstrap_threshold: float = 0.7
) -> Tuple[pd.DataFrame, Dict, Dict]:
    layer2_builder = Layer2StructureLearning(
        max_lag=max_lag,
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        bootstrap_threshold=bootstrap_threshold
    )

    agg_df, causal_graph, stable_edges = layer2_builder.build_layer2(layer1_df)

    return agg_df, causal_graph, stable_edges


def run_full_pipeline(
    base_path: str = 'data',
    window_size: int = 15,
    step_size: int = 5,
    max_lag: int = 5,
    alpha: float = 0.05,
    bootstrap_runs: int = 100,
    bootstrap_threshold: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    layer1_df = run_layer1_pipeline(
        base_path=base_path,
        window_size=window_size,
        step_size=step_size
    )

    agg_df, causal_graph, stable_edges = run_layer2_pipeline(
        layer1_df=layer1_df,
        max_lag=max_lag,
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        bootstrap_threshold=bootstrap_threshold
    )

    return layer1_df, agg_df, stable_edges