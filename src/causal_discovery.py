import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.data_loader import load_data
from sklearn.preprocessing import StandardScaler
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class Layer1TemporalConstruction:
    def __init__(self, window_size: int = 50, step_size: int = 50):
        self.window_size = window_size
        self.step_size = step_size
        if self.step_size < self.window_size:
            overlap_ratio = 1.0 - (self.step_size / self.window_size)
            print(f"Warning: windows overlap by {overlap_ratio:.0%} (step_size={self.step_size} < window_size={self.window_size}). "
                  f"This makes consecutive layer1 rows share raw data, which can artificially deflate PCMCI p-values. "
                  f"Set step_size=window_size for non-overlapping windows.")

    def sort_by_time(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        required = {'UserId', 'Timestamp'}
        missing = required - set(raw_logs.columns)
        if missing:
            raise ValueError(f"Missing required columns for per-student windowing: {missing}. "
                              f"Make sure preprocessing keeps 'UserId' and 'Timestamp' in clean_main_df.")
        df = raw_logs.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values(['UserId', 'Timestamp'], kind='mergesort').reset_index(drop=True)
        return df

    def create_sliding_windows(self, sorted_df: pd.DataFrame) -> List[Dict]:
        windows_list = []

        for user_id, user_df in sorted_df.groupby('UserId', sort=False):
            user_df = user_df.reset_index(drop=True)
            n = len(user_df)

            if n < self.window_size:
                continue

            for start_idx in range(0, n - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window_data = user_df.iloc[start_idx:end_idx]

                window_info = {
                    'user_id': user_id,
                    'window_start_time': window_data['Timestamp'].min(),
                    'window_end_time': window_data['Timestamp'].max(),
                    'cluster_id': window_data['student_cluster'].iloc[0] if 'student_cluster' in window_data.columns else -1
                }

                windows_list.append({
                    'window_info': window_info,
                    'window_data': window_data
                })

        return windows_list

    def extract_window_features(self, window_data: pd.DataFrame) -> Dict:
        features = {}

        features['success_rate'] = (window_data['IsCorrect'] == 1).mean()

        features['avg_difficulty'] = window_data['question_difficulty'].mean()
        features['difficulty_std'] = window_data['question_difficulty'].std()

        features['consecutive_correct_mean'] = window_data['consecutive_correct'].mean()
        features['attempts_mean'] = window_data['attempts_on_same_question'].mean()

        non_zero = window_data[window_data['time_delta'] > 0]
        if len(non_zero) > 5:
            features['avg_response_time'] = non_zero['time_delta'].mean()
        else:
            features['avg_response_time'] = 0.0

        return features

    def process_chunk(self, windows_list: List[Dict]) -> pd.DataFrame:
        chunk_rows = []
        for window_item in windows_list:
            window_info = window_item['window_info']
            window_data = window_item['window_data']
            features = self.extract_window_features(window_data)
            row = {**window_info, **features}
            chunk_rows.append(row)
        return pd.DataFrame(chunk_rows)

    def build_layer1(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        sorted_df = self.sort_by_time(raw_logs)
        windows_list = self.create_sliding_windows(sorted_df)
        layer1_df = self.process_chunk(windows_list)
        return layer1_df.reset_index(drop=True)


class Layer2StructureLearning:
    def __init__(self, max_lag: int = 2, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha

    def aggregate_time_series(self, layer1_df: pd.DataFrame, bucket_freq: str = 'D') -> pd.DataFrame:
        layer1_df = layer1_df.copy()
        layer1_df = layer1_df.sort_values('window_start_time')

        layer1_df['time_bucket'] = layer1_df['window_start_time'].dt.floor(bucket_freq)

        agg_df = layer1_df.groupby('time_bucket').agg(
            avg_success=('success_rate', 'mean'),
            avg_difficulty=('avg_difficulty', 'mean'),
            difficulty_std=('difficulty_std', 'mean'),
            consecutive_correct_mean=('consecutive_correct_mean', 'mean'),
            attempts_mean=('attempts_mean', 'mean'),
            avg_response_time=('avg_response_time', 'mean'),
            total_windows=('window_start_time', 'count')
        ).reset_index()

        if 'cluster_id' in layer1_df.columns:
            cluster_dist = layer1_df.groupby(['time_bucket', 'cluster_id']).size().unstack(fill_value=0)
            cluster_dist.columns = [f'CTRL_cluster_{c}_ratio' for c in cluster_dist.columns]
            cluster_dist = cluster_dist.div(cluster_dist.sum(axis=1), axis=0).fillna(0)
            agg_df = agg_df.merge(cluster_dist, on='time_bucket', how='left')

        agg_df = agg_df.fillna(0)

        print(f"Aggregated into {len(agg_df)} real-calendar-time buckets (freq={bucket_freq}), "
              f"spanning {agg_df['time_bucket'].min()} to {agg_df['time_bucket'].max()}")

        return agg_df

    def prepare_timeseries_matrix(self, agg_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        exclude_cols = ['time_bucket', 'total_windows']
        feature_cols = [col for col in agg_df.columns
                        if not any(excl in col for excl in exclude_cols)]

        X = agg_df[feature_cols].values

        stds = np.std(X, axis=0)
        mask = stds > 1e-6
        feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]
        X = X[:, mask]

        if len(X) == 0:
            return np.array([]), pd.DataFrame()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        feature_df = pd.DataFrame(X_scaled, columns=feature_cols)
        feature_df = self.drop_redundant_features(feature_df)

        return feature_df.values, feature_df

    def drop_redundant_features(self, feature_df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
        corr_matrix = feature_df.corr().abs()
        to_drop = set()
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_i, col_j = cols[i], cols[j]
                if col_j in to_drop or col_i in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] > corr_threshold:
                    print(f"Dropping '{col_j}': contemporaneous correlation with '{col_i}' is "
                          f"{corr_matrix.loc[col_i, col_j]:.3f} (> {corr_threshold}), likely a shared-definition redundancy")
                    to_drop.add(col_j)
        return feature_df.drop(columns=list(to_drop))

    def check_stationarity(self, feature_df: pd.DataFrame, alpha: float = 0.05) -> List[str]:
        if not STATSMODELS_AVAILABLE:
            print("statsmodels not installed, skipping stationarity check (pip install statsmodels to enable it)")
            return []
        non_stationary = []
        for col in feature_df.columns:
            try:
                adf_p = adfuller(feature_df[col].values, autolag='AIC')[1]
            except Exception:
                adf_p = np.nan
            is_stationary = (not np.isnan(adf_p)) and adf_p < alpha
            print(f"ADF test - {col}: p={adf_p:.4f} ({'stationary' if is_stationary else 'NON-stationary'})")
            if not is_stationary:
                non_stationary.append(col)
        return non_stationary

    def run_pcmci(self, feature_df: pd.DataFrame) -> Tuple[Dict, np.ndarray, List[str]]:
        n_obs, n_features = feature_df.shape
        feature_names = feature_df.columns.tolist()

        max_lag_safe = max(1, min(self.max_lag, n_obs // 8))

        if max_lag_safe < 1 or n_obs < 15:
            return {feature: [] for feature in feature_names}, np.array([]), []

        dataframe = DataFrame(data=feature_df.values, var_names=feature_names)
        cond_ind_test = ParCorr(significance='analytic')

        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=max_lag_safe, pc_alpha=self.alpha)

        p_matrix = results['p_matrix']
        val_matrix = results['val_matrix']

        q_matrix = pcmci.get_corrected_pvalues(
            p_matrix=p_matrix,
            fdr_method='fdr_bh',
            exclude_contemporaneous=True,
            tau_min=0,
            tau_max=max_lag_safe,
        )

        n_tests = n_features * (n_features - 1) * max_lag_safe
        print(f"PCMCI n_obs={n_obs}, FDR (fdr_bh) correction applied over m={n_tests} lagged tests "
              f"(with large n, even a weak partial correlation can reach a tiny p-value; check 'strength' below, not just p)")

        causal_graph = {}
        for i, target in enumerate(feature_names):
            causal_graph[target] = []
            for j, source in enumerate(feature_names):
                if i == j:
                    continue
                for lag in range(1, max_lag_safe + 1):
                    p_val = p_matrix[j, i, lag]
                    q_val = q_matrix[j, i, lag]
                    if q_val < self.alpha:
                        causal_graph[target].append({
                            'source': source,
                            'lag': lag,
                            'p_value': p_val,
                            'p_value_fdr': q_val,
                            'strength': val_matrix[j, i, lag]
                        })
        return causal_graph, p_matrix, feature_names

    def build_layer2(self, layer1_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        agg_df = self.aggregate_time_series(layer1_df)
        X_scaled, feature_df = self.prepare_timeseries_matrix(agg_df)

        non_stationary_cols = self.check_stationarity(feature_df)
        if non_stationary_cols:
            print(f"Differencing all features once because these are non-stationary: {non_stationary_cols}")
            feature_df = feature_df.diff().dropna().reset_index(drop=True)

        causal_graph, _, _ = self.run_pcmci(feature_df)

        return agg_df, causal_graph


def run_layer1_pipeline(
    raw_logs: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 50
) -> pd.DataFrame:
    layer1_builder = Layer1TemporalConstruction(window_size=window_size, step_size=step_size)
    layer1_df = layer1_builder.build_layer1(raw_logs)
    return layer1_df


def run_layer2_pipeline(
    layer1_df: pd.DataFrame,
    max_lag: int = 2,
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, Dict]:
    layer2_builder = Layer2StructureLearning(
        max_lag=max_lag,
        alpha=alpha
    )
    agg_df, causal_graph = layer2_builder.build_layer2(layer1_df)
    return agg_df, causal_graph


def run_full_pipeline(
    raw_logs: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 50,
    max_lag: int = 2,
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    layer1_df = run_layer1_pipeline(
        raw_logs=raw_logs,
        window_size=window_size,
        step_size=step_size
    )
    agg_df, causal_graph = run_layer2_pipeline(
        layer1_df=layer1_df,
        max_lag=max_lag,
        alpha=alpha
    )
    return layer1_df, agg_df, causal_graph