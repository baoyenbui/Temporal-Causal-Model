import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterable, Any
from sklearn.preprocessing import StandardScaler
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


FORBIDDEN_MODEL_INPUTS = {
    "IsCorrect",
    "is_correct",
}

OUTCOME_DERIVED_FEATURES = {
    "success_rate", "success_trend", "recent5_correct_rate", "max_streak",
    "avg_success",
}

COUNT_COLUMNS = {"total_windows", "response_time_n_obs", "n_students"}

FORBIDDEN_LOG_TYPES = {
    "Checkout",
    "CheckoutRetry",
}


def assert_pre_outcome_only(logs: pd.DataFrame) -> None:
    if "Type" not in logs.columns:
        return
    leaked_types = set(logs["Type"].dropna().unique()) & FORBIDDEN_LOG_TYPES
    if leaked_types:
        raise AssertionError(
            f"assert_pre_outcome_only: found forbidden log type(s) {leaked_types} in logs "
            f"passed to causal_discovery.py."
        )


class Layer1TemporalConstruction:
    def __init__(
        self,
        window_size: int = 50,
        step_size: int = 50,
        max_response_time_seconds: float = 300.0,
        min_dominant_share: Optional[float] = None,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.max_response_time_seconds = max_response_time_seconds
        self.min_dominant_share = min_dominant_share
        if self.step_size < self.window_size:
            overlap_ratio = 1.0 - (self.step_size / self.window_size)
            print(
                f"Warning: windows overlap by {overlap_ratio:.0%} "
                f"(step_size={self.step_size} < window_size={self.window_size})."
            )

    def sort_by_time(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        required = {"UserId", "Timestamp", "ConstructId"}
        missing = required - set(raw_logs.columns)
        if missing:
            raise ValueError(f"Missing required columns for per-student windowing: {missing}.")
        assert_pre_outcome_only(raw_logs)
        df = raw_logs.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(["UserId", "Timestamp"], kind="mergesort").reset_index(drop=True)
        df["time_delta"] = df.groupby("UserId")["Timestamp"].diff().dt.total_seconds()
        return df

    def create_sliding_windows(self, sorted_df: pd.DataFrame) -> List[Dict]:
        windows_list = []
        n_low_purity_dropped = 0
        for user_id, user_df in sorted_df.groupby("UserId", sort=False):
            user_df = user_df.reset_index(drop=True)
            n = len(user_df)
            if n < self.window_size:
                continue
            for start_idx in range(0, n - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                window_data = user_df.iloc[start_idx:end_idx]
                window_end_time = window_data["Timestamp"].max()
                construct_counts = window_data["ConstructId"].value_counts()
                dominant_construct = construct_counts.idxmax()
                dominant_construct_share = construct_counts.max() / len(window_data)

                if self.min_dominant_share is not None and dominant_construct_share < self.min_dominant_share:
                    n_low_purity_dropped += 1
                    continue

                window_info = {
                    "user_id": user_id,
                    "window_start_time": window_data["Timestamp"].min(),
                    "window_end_time": window_end_time,
                    "year": window_end_time.year,
                    "cluster_id": window_data["student_cluster"].iloc[0]
                    if "student_cluster" in window_data.columns
                    else -1,
                    "dominant_construct": dominant_construct,
                    "dominant_construct_share": dominant_construct_share,
                    "n_constructs": window_data["ConstructId"].nunique(),
                    "construct_distribution": construct_counts.to_dict(),
                }
                windows_list.append({"window_info": window_info, "window_data": window_data})
        if self.min_dominant_share is not None and n_low_purity_dropped:
            print(f"min_dominant_share={self.min_dominant_share}: dropped {n_low_purity_dropped} window(s).")
        return windows_list

    def extract_window_features(self, window_data: pd.DataFrame) -> Dict:
        features = {}

        if "IsCorrect" in window_data.columns:
            correct = pd.to_numeric(window_data["IsCorrect"], errors="coerce")
            valid_correct = correct.dropna()
            n_valid = len(valid_correct)
        else:
            correct = pd.Series(dtype=float)
            valid_correct = correct
            n_valid = 0

        features["success_rate"] = valid_correct.mean() if n_valid else np.nan
        features["n_correct_obs"] = n_valid

        if n_valid > 1 and valid_correct.std() > 1e-9:
            idx = np.arange(n_valid)
            slope = np.polyfit(idx, valid_correct.values, 1)[0]
        else:
            slope = 0.0 if n_valid else np.nan
        features["success_trend"] = slope

        if "question_difficulty" in window_data.columns:
            diff = pd.to_numeric(window_data["question_difficulty"], errors="coerce").dropna()
            if len(diff):
                features["avg_difficulty"] = diff.mean()
                features["difficulty_std"] = diff.std()
                features["difficulty_range"] = diff.max() - diff.min()
            else:
                features["avg_difficulty"] = np.nan
                features["difficulty_std"] = np.nan
                features["difficulty_range"] = np.nan
        else:
            features["avg_difficulty"] = np.nan
            features["difficulty_std"] = np.nan
            features["difficulty_range"] = np.nan

        if "consecutive_correct" in window_data.columns:
            features["recent5_correct_rate"] = window_data["consecutive_correct"].mean()
            features["max_streak"] = window_data["consecutive_correct"].max()
        else:
            features["recent5_correct_rate"] = np.nan
            features["max_streak"] = np.nan

        if "attempts_on_same_question" in window_data.columns:
            features["attempts_mean"] = window_data["attempts_on_same_question"].mean()
            features["attempts_max"] = window_data["attempts_on_same_question"].max()
            features["pct_multi_attempt"] = (window_data["attempts_on_same_question"] > 1).mean()
        else:
            features["attempts_mean"] = np.nan
            features["attempts_max"] = np.nan
            features["pct_multi_attempt"] = np.nan

        if "time_delta" in window_data.columns:
            non_zero = window_data[window_data["time_delta"] > 0]
            capped = non_zero[non_zero["time_delta"] <= self.max_response_time_seconds]
        else:
            capped = window_data.iloc[0:0]
        features["response_time_n_obs"] = len(capped)
        if len(capped) > 5:
            features["avg_response_time"] = capped["time_delta"].mean()
            features["response_time_std"] = capped["time_delta"].std()
            features["median_response_time"] = capped["time_delta"].median()
        else:
            features["avg_response_time"] = np.nan
            features["response_time_std"] = np.nan
            features["median_response_time"] = np.nan

        return features

    def process_chunk(self, windows_list: List[Dict]) -> pd.DataFrame:
        chunk_rows = []
        for window_item in windows_list:
            window_info = window_item["window_info"]
            window_data = window_item["window_data"]
            features = self.extract_window_features(window_data)
            row = {**window_info, **features}
            chunk_rows.append(row)
        df = pd.DataFrame(chunk_rows)
        if len(df):
            n_fallback = int(df["avg_response_time"].isna().sum())
            print(
                f"{n_fallback}/{len(df)} windows have insufficient response-time coverage "
                f"(<=5 valid samples) and carry NaN for avg_response_time/response_time_std/median_response_time."
            )
            if "n_correct_obs" in df.columns:
                n_no_correct = int((df["n_correct_obs"] == 0).sum())
                if n_no_correct:
                    print(
                        f"{n_no_correct}/{len(df)} windows have NO IsCorrect observations "
                        f"and carry NaN for success_rate/success_trend."
                    )
        return df

    def build_layer1(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        sorted_df = self.sort_by_time(raw_logs)
        windows_list = self.create_sliding_windows(sorted_df)
        layer1_df = self.process_chunk(windows_list)
        return layer1_df.reset_index(drop=True)


class Layer2StructureLearning:
    AGG_SPEC = dict(
        avg_success=("success_rate", "mean"),
        success_trend=("success_trend", "mean"),
        avg_difficulty=("avg_difficulty", "mean"),
        difficulty_std=("difficulty_std", "mean"),
        difficulty_range=("difficulty_range", "mean"),
        recent5_correct_rate=("recent5_correct_rate", "mean"),
        max_streak=("max_streak", "mean"),
        attempts_mean=("attempts_mean", "mean"),
        attempts_max=("attempts_max", "mean"),
        pct_multi_attempt=("pct_multi_attempt", "mean"),
        avg_response_time=("avg_response_time", "mean"),
        response_time_std=("response_time_std", "mean"),
        median_response_time=("median_response_time", "mean"),
        response_time_n_obs=("response_time_n_obs", "sum"),
        total_windows=("window_start_time", "count"),
        n_students=("user_id", "nunique"),
    )

    REQUIRED_EXPERIMENTAL_KEYS = {
        "TreatmentLessonConstructId",
        "QuestionConstructId",
        "Year",
    }

    def __init__(
        self,
        max_lag: int = 2,
        alpha: float = 0.1,
        pc_alpha: float = 0.2,
        corr_threshold: float = 0.97,
        min_effect: float = 0.12,
    ):
        self.max_lag = max_lag
        self.alpha = alpha
        self.pc_alpha = pc_alpha
        self.corr_threshold = corr_threshold
        self.min_effect = min_effect

    def _require_experimental_keys(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_EXPERIMENTAL_KEYS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing experimental keys {missing}. "
                f"Join Layer 1 to the PI-confirmed ConstructId -> Treatment/Question mapping first."
            )

    def attach_experimental_keys(
        self,
        layer1_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        construct_col_in_layer1: str = "dominant_construct",
        construct_col_in_mapping: str = "ConstructId",
    ) -> pd.DataFrame:
        required_mapping = self.REQUIRED_EXPERIMENTAL_KEYS | {construct_col_in_mapping}
        missing = required_mapping - set(mapping_df.columns)
        if missing:
            raise ValueError(f"mapping_df missing columns: {missing}")
        if construct_col_in_layer1 not in layer1_df.columns:
            raise ValueError(f"layer1_df missing column: {construct_col_in_layer1}")

        has_role = "role" in mapping_df.columns
        merge_cols = list(required_mapping) + (["role"] if has_role else [])

        keyed = layer1_df.merge(
            mapping_df[merge_cols],
            left_on=construct_col_in_layer1,
            right_on=construct_col_in_mapping,
            how="inner",
        )

        if len(keyed) == 0:
            raise ValueError("attach_experimental_keys produced 0 rows. Check mapping coverage.")

        if has_role:
            question_side = mapping_df[mapping_df["role"] == "question"]
            dup_counts = question_side.groupby(construct_col_in_mapping).size()
            ambiguous_ids = set(dup_counts[dup_counts > 1].index)
            keyed["ambiguous_construct_mapping"] = (keyed["role"] == "question") & keyed[
                construct_col_in_layer1
            ].isin(ambiguous_ids)
            if ambiguous_ids:
                print(
                    f"WARNING: {len(ambiguous_ids)} question-role ConstructId(s) map to more than one "
                    f"(Treatment, Question, Year) row in mapping_df, e.g. {sorted(ambiguous_ids)[:5]}. "
                    f"Windows keyed through one of these question-side matches are flagged "
                    f"'ambiguous_construct_mapping' and excluded downstream, since the log alone can't "
                    f"tell which treatment context they belong to. Treatment-side fan-out (one treatment "
                    f"mapped to many questions) is expected and is NOT flagged ambiguous."
                )
        else:
            dup_counts = mapping_df.groupby(construct_col_in_mapping).size()
            ambiguous_ids = set(dup_counts[dup_counts > 1].index)
            keyed["ambiguous_construct_mapping"] = keyed[construct_col_in_layer1].isin(ambiguous_ids)
            if ambiguous_ids:
                print(
                    f"WARNING: {len(ambiguous_ids)} ConstructId(s) map to more than one "
                    f"(Treatment, Question, Year) row in mapping_df, e.g. {sorted(ambiguous_ids)[:5]}. "
                    f"mapping_df has no 'role' column, so ambiguity is flagged conservatively on ANY "
                    f"duplicate ConstructId (may over-exclude legitimate treatment fan-out)."
                )

        n_ambiguous_rows = int(keyed["ambiguous_construct_mapping"].sum())
        print(
            f"attach_experimental_keys: {len(layer1_df)} Layer 1 windows -> {len(keyed)} keyed rows "
            f"({keyed[construct_col_in_layer1].nunique()} distinct dominant_construct values matched "
            f"out of {layer1_df[construct_col_in_layer1].nunique()} present in Layer 1; "
            f"{n_ambiguous_rows}/{len(keyed)} ({n_ambiguous_rows / len(keyed):.1%}) keyed row(s) "
            f"flagged ambiguous_construct_mapping)"
        )

        self._require_experimental_keys(keyed)
        return keyed.reset_index(drop=True)

    def _regularize_time_index(
        self, agg: pd.DataFrame, bucket_freq: str = "D", max_gap_to_interpolate: int = 2
    ) -> Tuple[pd.DataFrame, Dict]:
        agg = agg.sort_values("time_bucket").reset_index(drop=True)
        n_before = len(agg)

        full_index = pd.date_range(agg["time_bucket"].min(), agg["time_bucket"].max(), freq=bucket_freq)
        reg = agg.set_index("time_bucket").reindex(full_index)
        reg.index.name = "time_bucket"

        count_cols = [c for c in COUNT_COLUMNS if c in reg.columns]
        if count_cols:
            reg[count_cols] = reg[count_cols].fillna(0)

        feature_cols = [c for c in reg.columns if c not in COUNT_COLUMNS]
        is_na = reg[feature_cols].isna().any(axis=1) if feature_cols else pd.Series(False, index=reg.index)
        n_missing_total = int(is_na.sum())

        n_interpolated = 0
        if n_missing_total and feature_cols:
            run_id = (~is_na).cumsum()
            run_lengths = is_na.groupby(run_id).transform("sum")
            interpolatable = is_na & (run_lengths <= max_gap_to_interpolate)
            n_interpolated = int(interpolatable.sum())
            if n_interpolated:
                candidate = reg[feature_cols].interpolate(method="linear", limit_area="inside")
                for col in feature_cols:
                    fill_here = interpolatable & reg[col].isna()
                    reg.loc[fill_here, col] = candidate.loc[fill_here, col]

        reg = reg.reset_index()
        n_unrecoverable = int(reg[feature_cols].isna().any(axis=1).sum()) if feature_cols else 0

        stats = {
            "n_original_buckets": n_before,
            "n_regular_buckets": len(reg),
            "n_missing_buckets": n_missing_total,
            "n_interpolated_buckets": n_interpolated,
            "n_unrecoverable_missing_buckets": n_unrecoverable,
        }
        if n_missing_total:
            print(
                f"  time regularization: {n_before} observed buckets -> {len(reg)} regular {bucket_freq}-buckets "
                f"({n_missing_total} gap bucket(s), {n_interpolated} interpolated, {n_unrecoverable} left as NaN)"
            )
        return reg, stats

    def _fill_remaining_gaps(
        self, agg: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
        unfillable_cols = []
        n_filled = {}
        for col in feature_cols:
            if col not in agg.columns:
                continue
            n_missing = int(agg[col].isna().sum())
            if n_missing == 0:
                continue
            entity_median = agg[col].median()
            if pd.isna(entity_median):
                unfillable_cols.append(col)
                continue
            agg[col] = agg[col].fillna(entity_median)
            n_filled[col] = n_missing
        if n_filled:
            print(f"  filled remaining gaps with this entity's own median: {n_filled}")
        if unfillable_cols:
            print(f"  entity has NO valid data in column(s) {unfillable_cols} -- entity will be excluded")
        return agg, unfillable_cols, n_filled

    def aggregate_time_series_by_experimental_key(
        self,
        keyed_layer1_df: pd.DataFrame,
        bucket_freq: str = "D",
        held_out_user_ids: Optional[Iterable] = None,
        min_obs_per_entity: int = 30,
        max_gap_to_interpolate: int = 2,
    ) -> Tuple[Dict[Tuple, pd.DataFrame], Dict[Tuple, Dict]]:
        self._require_experimental_keys(keyed_layer1_df)
        required_extra = {"window_start_time", "user_id"}
        missing = required_extra - set(keyed_layer1_df.columns)
        if missing:
            raise ValueError(f"aggregate_time_series_by_experimental_key: missing {missing}")

        df = keyed_layer1_df.copy()
        if held_out_user_ids is not None:
            df = df[~df["user_id"].isin(held_out_user_ids)]

        df["time_bucket"] = df["window_start_time"].dt.floor(bucket_freq)
        key_cols = ["TreatmentLessonConstructId", "QuestionConstructId", "Year"]
        has_ambiguous_flag = "ambiguous_construct_mapping" in df.columns

        per_entity = {}
        key_status = {}
        for key, g in df.groupby(key_cols):
            n_students_total = g["user_id"].nunique()
            n_ambiguous = int(g["ambiguous_construct_mapping"].sum()) if has_ambiguous_flag else 0

            if n_ambiguous:
                key_status[key] = {
                    "n_buckets": 0,
                    "n_students_total": int(n_students_total),
                    "regularity": {},
                    "unfillable_columns": [],
                    "n_gap_filled": {},
                    "status": "ambiguous_construct_mapping",
                    "n_ambiguous_windows": n_ambiguous,
                    "n_windows_total": int(len(g)),
                }
                continue

            agg = g.groupby("time_bucket").agg(**self.AGG_SPEC).reset_index()
            agg, regularity_stats = self._regularize_time_index(
                agg, bucket_freq=bucket_freq, max_gap_to_interpolate=max_gap_to_interpolate
            )
            feature_cols = [c for c in agg.columns if c not in COUNT_COLUMNS and c != "time_bucket"]
            agg, unfillable_cols, n_filled = self._fill_remaining_gaps(agg, feature_cols)

            n_buckets = len(agg)
            status_entry = {
                "n_buckets": n_buckets,
                "n_students_total": int(n_students_total),
                "regularity": regularity_stats,
                "unfillable_columns": unfillable_cols,
                "n_gap_filled": n_filled,
            }
            if n_buckets >= min_obs_per_entity and not unfillable_cols:
                per_entity[key] = agg
                status_entry["status"] = "ok"
            elif unfillable_cols:
                status_entry["status"] = "unfillable_columns"
            else:
                status_entry["status"] = "insufficient_temporal_history"
            key_status[key] = status_entry

        n_ok = sum(1 for v in key_status.values() if v["status"] == "ok")
        n_ambiguous_keys = sum(1 for v in key_status.values() if v["status"] == "ambiguous_construct_mapping")
        print(
            f"{n_ok}/{len(key_status)} (Treatment, Question, Year) triples "
            f"have >= {min_obs_per_entity} regular {bucket_freq}-buckets with no unfillable columns "
            f"and got a temporal model"
        )
        if n_ambiguous_keys:
            print(
                f"{n_ambiguous_keys}/{len(key_status)} triples excluded for ambiguous ConstructId mapping "
                f"(cannot disambiguate from event data alone)"
            )
        n_bad = len(key_status) - n_ok
        if n_bad:
            print(f"{n_bad} triples excluded (status != 'ok'); caller must emit a failure record per row for these.")

        return per_entity, key_status

    def check_experimental_key_sparsity(
        self, keyed_layer1_df: pd.DataFrame, bucket_freq: str = "D", min_obs: int = 30
    ) -> pd.Series:
        self._require_experimental_keys(keyed_layer1_df)
        df = keyed_layer1_df.copy()
        df["time_bucket"] = df["window_start_time"].dt.floor(bucket_freq)
        counts = (
            df.groupby(["TreatmentLessonConstructId", "QuestionConstructId", "Year"])["time_bucket"]
            .nunique()
            .sort_values(ascending=False)
        )
        n_ok = (counts >= min_obs).sum()
        print(counts.describe())
        print(
            f"{n_ok}/{len(counts)} (Treatment, Question, Year) triples have >= {min_obs} {bucket_freq}-buckets "
            f"(observed buckets only, not the regularized elapsed-period count)"
        )
        return counts

    def prepare_timeseries_matrix(
        self, agg_df: pd.DataFrame, feature_contract: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        exclude_cols = ["time_bucket", "total_windows", "response_time_n_obs", "n_students"]

        if feature_contract is not None:
            feature_cols = [c for c in feature_contract if c in agg_df.columns]
            missing = set(feature_contract) - set(feature_cols)
            if missing:
                print(f"NOTE: feature_contract names not present in agg_df, skipped: {missing}")
        else:
            feature_cols = [col for col in agg_df.columns if not any(excl in col for excl in exclude_cols)]

        leaked = set(feature_cols) & FORBIDDEN_MODEL_INPUTS
        if leaked:
            raise AssertionError(f"Forbidden target-derived columns leaked into feature matrix: {leaked}")

        outcome_derived_present = set(feature_cols) & OUTCOME_DERIVED_FEATURES
        if outcome_derived_present:
            print(f"NOTE: feature matrix includes outcome-derived columns {sorted(outcome_derived_present)}.")

        if agg_df[feature_cols].isna().any().any():
            bad = agg_df[feature_cols].columns[agg_df[feature_cols].isna().any()].tolist()
            raise ValueError(
                f"prepare_timeseries_matrix received NaN in {bad} after aggregation. "
                f"This entity should have been excluded upstream in aggregate_time_series_by_experimental_key."
            )

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

    def drop_redundant_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        corr_matrix = feature_df.corr().abs()
        to_drop = set()
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_i, col_j = cols[i], cols[j]
                if col_j in to_drop or col_i in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] > self.corr_threshold:
                    print(
                        f"Dropping '{col_j}': correlation with '{col_i}' is "
                        f"{corr_matrix.loc[col_i, col_j]:.3f} (> {self.corr_threshold})"
                    )
                    to_drop.add(col_j)
        return feature_df.drop(columns=list(to_drop))

    def check_stationarity(self, feature_df: pd.DataFrame, alpha: float = 0.05) -> List[str]:
        if not STATSMODELS_AVAILABLE:
            print("statsmodels not installed, skipping stationarity check")
            return []
        non_stationary = []
        for col in feature_df.columns:
            try:
                adf_p = adfuller(feature_df[col].values, autolag="AIC")[1]
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
        cond_ind_test = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=max_lag_safe, pc_alpha=self.pc_alpha, fdr_method="fdr_bh")
        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]
        q_matrix = p_matrix
        causal_graph = {feature: [] for feature in feature_names}
        for i, target in enumerate(feature_names):
            for j, source in enumerate(feature_names):
                if i == j:
                    continue
                if source.startswith("CTRL_") and target.startswith("CTRL_"):
                    continue
                for lag in range(1, max_lag_safe + 1):
                    q_val = q_matrix[j, i, lag]
                    strength = val_matrix[j, i, lag]
                    if q_val >= self.alpha:
                        continue
                    if abs(strength) < self.min_effect:
                        continue
                    causal_graph[target].append(
                        {
                            "source": source,
                            "lag": lag,
                            "p_value": q_val,
                            "q_value": q_val,
                            "strength": strength,
                            "n_obs": n_obs,
                        }
                    )
        return causal_graph, p_matrix, feature_names

    def run_pcmci_stability_selection(
        self,
        feature_df: pd.DataFrame,
        n_bootstrap: int = 100,
        block_size: int = 5,
        stability_threshold: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Dict:
        rng = np.random.default_rng(random_state)
        n_obs = len(feature_df)
        if n_obs < 15:
            return {"stable_graph": {}, "edge_frequency": {}, "n_bootstrap": 0}

        edge_counts: Dict[Tuple[str, str, int], int] = {}
        successful_runs = 0
        n_blocks = max(1, n_obs // block_size)

        for _ in range(n_bootstrap):
            block_starts = rng.integers(0, max(1, n_obs - block_size + 1), size=n_blocks)
            idx = np.concatenate([np.arange(s, min(s + block_size, n_obs)) for s in block_starts])
            if len(idx) < 15:
                continue
            resampled = feature_df.iloc[idx].reset_index(drop=True)
            try:
                causal_graph, _, _ = self.run_pcmci(resampled)
            except Exception:
                continue
            successful_runs += 1
            for target, edges in causal_graph.items():
                for edge in edges:
                    key = (edge["source"], target, edge["lag"])
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        if successful_runs == 0:
            return {"stable_graph": {}, "edge_frequency": {}, "n_bootstrap": 0}

        edge_frequency = {k: v / successful_runs for k, v in edge_counts.items()}
        stable_graph: Dict[str, List[Dict]] = {}
        for (source, target, lag), freq in edge_frequency.items():
            if freq >= stability_threshold:
                stable_graph.setdefault(target, []).append(
                    {"source": source, "lag": lag, "stability": freq}
                )

        print(
            f"stability selection: {successful_runs}/{n_bootstrap} bootstrap runs succeeded, "
            f"{sum(len(v) for v in stable_graph.values())} edge(s) >= {stability_threshold:.0%} stable "
            f"out of {len(edge_frequency)} edge(s) ever seen"
        )

        return {
            "stable_graph": stable_graph,
            "edge_frequency": edge_frequency,
            "n_bootstrap": successful_runs,
        }

    def run_lag_sensitivity(
        self, feature_df: pd.DataFrame, lag_candidates: Optional[List[int]] = None
    ) -> Dict[int, Dict]:
        if lag_candidates is None:
            lag_candidates = [1, 2, 3, 5, 7]
        results = {}
        original_max_lag = self.max_lag
        try:
            for lag in lag_candidates:
                self.max_lag = lag
                causal_graph, p_matrix, feature_names = self.run_pcmci(feature_df)
                results[lag] = {"causal_graph": causal_graph}
        finally:
            self.max_lag = original_max_lag
        return results

    def run_pcmci_per_entity(
        self,
        per_entity_agg: Dict[Tuple, pd.DataFrame],
        feature_contract: Optional[List[str]] = None,
        run_stability_selection: bool = False,
        stability_kwargs: Optional[Dict] = None,
    ) -> Tuple[Dict[Tuple, Dict], Dict[Tuple, Dict]]:
        causal_graphs: Dict[Tuple, Dict] = {}
        stability_results: Dict[Tuple, Dict] = {}
        for entity_key, agg_df in per_entity_agg.items():
            print(f"Running PCMCI for experimental key: {entity_key}")
            X_scaled, feature_df = self.prepare_timeseries_matrix(agg_df, feature_contract=feature_contract)
            if feature_df.empty:
                continue
            non_stationary_cols = self.check_stationarity(feature_df)
            if non_stationary_cols:
                feature_df = feature_df.copy()
                feature_df[non_stationary_cols] = feature_df[non_stationary_cols].diff()
                feature_df = feature_df.dropna().reset_index(drop=True)
            causal_graph, _, _ = self.run_pcmci(feature_df)
            causal_graphs[entity_key] = causal_graph
            if run_stability_selection:
                kwargs = stability_kwargs or {}
                stability_results[entity_key] = self.run_pcmci_stability_selection(feature_df, **kwargs)
        return causal_graphs, stability_results

    def build_layer2(
        self,
        keyed_layer1_df: pd.DataFrame,
        held_out_user_ids: Optional[Iterable] = None,
        bucket_freq: str = "D",
        min_obs_per_entity: int = 30,
        max_gap_to_interpolate: int = 2,
        feature_contract: Optional[List[str]] = None,
        run_stability_selection: bool = False,
    ) -> Tuple[Dict[Tuple, Dict], Dict[Tuple, pd.DataFrame], Dict[Tuple, Dict]]:
        self._require_experimental_keys(keyed_layer1_df)
        per_entity_agg, key_status = self.aggregate_time_series_by_experimental_key(
            keyed_layer1_df,
            bucket_freq=bucket_freq,
            held_out_user_ids=held_out_user_ids,
            min_obs_per_entity=min_obs_per_entity,
            max_gap_to_interpolate=max_gap_to_interpolate,
        )
        causal_graphs, stability_results = self.run_pcmci_per_entity(
            per_entity_agg,
            feature_contract=feature_contract,
            run_stability_selection=run_stability_selection,
        )
        for key, agg_df in per_entity_agg.items():
            if key not in key_status:
                continue
            if "n_students" in agg_df.columns:
                key_status[key]["n_students_avg"] = float(agg_df["n_students"].mean())
            if key in stability_results:
                key_status[key]["stability"] = stability_results[key]
        return causal_graphs, per_entity_agg, key_status

    def predict_for_key(
        self, causal_graphs: Dict[Tuple, Dict], treatment_id: Any, question_id: Any, year: int
    ) -> Optional[Dict]:
        key = (treatment_id, question_id, year)
        return causal_graphs.get(key)


def run_layer1_pipeline(
    raw_logs: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 50,
    min_dominant_share: Optional[float] = None,
) -> pd.DataFrame:
    layer1_builder = Layer1TemporalConstruction(
        window_size=window_size, step_size=step_size, min_dominant_share=min_dominant_share
    )
    return layer1_builder.build_layer1(raw_logs)


def run_full_pipeline(
    raw_logs: pd.DataFrame,
    mapping_df: pd.DataFrame,
    window_size: int = 50,
    step_size: int = 50,
    max_lag: int = 2,
    alpha: float = 0.1,
    pc_alpha: float = 0.2,
    corr_threshold: float = 0.97,
    min_effect: float = 0.12,
    held_out_user_ids: Optional[Iterable] = None,
    bucket_freq: str = "D",
    min_obs_per_entity: int = 30,
    max_gap_to_interpolate: int = 2,
    feature_contract: Optional[List[str]] = None,
    min_dominant_share: Optional[float] = None,
    run_stability_selection: bool = False,
) -> Tuple[Dict[Tuple, Dict], Dict[Tuple, pd.DataFrame], pd.DataFrame, Dict[Tuple, Dict]]:
    layer1_df = run_layer1_pipeline(
        raw_logs, window_size=window_size, step_size=step_size, min_dominant_share=min_dominant_share
    )
    layer2 = Layer2StructureLearning(
        max_lag=max_lag,
        alpha=alpha,
        pc_alpha=pc_alpha,
        corr_threshold=corr_threshold,
        min_effect=min_effect,
    )
    keyed_layer1_df = layer2.attach_experimental_keys(layer1_df, mapping_df)
    causal_graphs, per_entity_agg, key_status = layer2.build_layer2(
        keyed_layer1_df,
        held_out_user_ids=held_out_user_ids,
        bucket_freq=bucket_freq,
        min_obs_per_entity=min_obs_per_entity,
        max_gap_to_interpolate=max_gap_to_interpolate,
        feature_contract=feature_contract,
        run_stability_selection=run_stability_selection,
    )
    return causal_graphs, per_entity_agg, keyed_layer1_df, key_status


def run_layer2_pipeline(
    keyed_layer1_df: pd.DataFrame,
    max_lag: int = 2,
    alpha: float = 0.1,
    pc_alpha: float = 0.2,
    corr_threshold: float = 0.97,
    min_effect: float = 0.12,
    held_out_user_ids: Optional[Iterable] = None,
    bucket_freq: str = "D",
    min_obs_per_entity: int = 30,
    max_gap_to_interpolate: int = 2,
    feature_contract: Optional[List[str]] = None,
    run_stability_selection: bool = False,
) -> Tuple[Dict[Tuple, Dict], Dict[Tuple, pd.DataFrame], Dict[Tuple, Dict]]:
    layer2 = Layer2StructureLearning(
        max_lag=max_lag,
        alpha=alpha,
        pc_alpha=pc_alpha,
        corr_threshold=corr_threshold,
        min_effect=min_effect,
    )
    return layer2.build_layer2(
        keyed_layer1_df,
        held_out_user_ids=held_out_user_ids,
        bucket_freq=bucket_freq,
        min_obs_per_entity=min_obs_per_entity,
        max_gap_to_interpolate=max_gap_to_interpolate,
        feature_contract=feature_contract,
        run_stability_selection=run_stability_selection,
    )