"""Feature builder for Option A. Owned by Nguyen Xuan Hoa.

Implements the `FeatureBuilder` contract in `src/option_a/features.py`. See that file and
`CONTRACT.md` for the rules this must follow; the short version:

- `fit(train_rows, logs)` may only look at `train_rows` (this fold's training slice of the
  88-row benchmark). Every scaler, lookup table or fallback value is estimated here.
- `transform(rows, logs)` only applies what `fit` already learned. It must not compute
  anything new from `rows` itself (that would let a held-out row's own attributes leak into
  its own prediction).
- No column in the FORBIDDEN_MODEL_INPUTS set, no NaNs. `check_output()` enforces both.

Phase 1 (this file): a small, deliberately unoptimized feature set, to get a
leakage-safe feature pipeline wired end to end before spending effort on MAE.
`aggregate_time_series` in `src/causal_discovery.py` is NOT touched here (separate, larger
piece of work noted in CONTRACT.md's "Known structural gap").
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.option_a.features import FeatureBuilder

QUESTION_CONSTRUCT_KEY = "QuestionConstructId"
YEAR_KEY = "Year"

# Interaction logs are pre-checkout (CONTRACT.md), so a construct-keyed statistic computed
# from the full logs table is not target leakage by itself: it does not depend on any row's
# A/B outcome. What CONTRACT.md flags as the open question is narrower -- whether a
# construct-keyed aggregate should additionally exclude interactions tied to the fold's
# held-out TreatmentLessonConstructId on validity grounds, not a leakage grounds. This file
# takes the permissive reading (use the full logs table) and calls it out here rather than
# deciding it silently, per CONTRACT.md's "agree the exact rule with the PI and record it
# here before the frozen run." Not yet agreed -- do not treat Phase 1 numbers as frozen.
MAX_RESPONSE_TIME_SECONDS = 300.0


def _response_time_by_construct(logs: pd.DataFrame) -> pd.Series:
    """Mean seconds since the student's previous interaction, averaged per ConstructId.

    Approximates response time from consecutive-interaction gaps (logs have no explicit
    duration column). Non-positive and implausibly long gaps (session breaks) are dropped
    before averaging, same cap convention used elsewhere in this codebase.
    """
    df = logs[["UserId", "ConstructId", "Timestamp"]].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values(["UserId", "Timestamp"], kind="mergesort")
    df["time_delta"] = df.groupby("UserId")["Timestamp"].diff().dt.total_seconds()
    valid = df[(df["time_delta"] > 0) & (df["time_delta"] <= MAX_RESPONSE_TIME_SECONDS)]
    return valid.groupby("ConstructId")["time_delta"].mean()


class ConstructYearFeatures(FeatureBuilder):
    """Historical per-construct difficulty/volume/response-time, plus a Year indicator.

    Every per-construct statistic is recomputed inside `fit` on each call (once per fold),
    rather than cached at the class or module level, so nothing depends on call order across
    folds. Missing constructs (no interactions found in `logs`) fall back to the fold's
    global mean, itself estimated inside `fit`.
    """

    name = "ConstructYearFeatures"

    NUMERIC_COLUMNS = [
        "q_construct_difficulty",
        "q_construct_log_n_interactions",
        "q_construct_avg_response_time",
    ]

    def __init__(self) -> None:
        self.difficulty_by_construct_: Optional[pd.Series] = None
        self.log_n_interactions_by_construct_: Optional[pd.Series] = None
        self.response_time_by_construct_: Optional[pd.Series] = None
        self.global_fallback_: Optional[Dict[str, float]] = None
        self.years_seen_: Optional[List] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, train_rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> "ConstructYearFeatures":
        if logs is None:
            raise ValueError(f"{self.name}.fit requires interaction logs (logs=None)")
        required_log_cols = {"UserId", "ConstructId", "Timestamp", "IsCorrect"}
        missing = required_log_cols - set(logs.columns)
        if missing:
            raise ValueError(f"{self.name}.fit: logs is missing required columns {sorted(missing)}")
        if QUESTION_CONSTRUCT_KEY not in train_rows.columns or YEAR_KEY not in train_rows.columns:
            raise ValueError(
                f"{self.name}.fit: train_rows is missing '{QUESTION_CONSTRUCT_KEY}' or '{YEAR_KEY}'"
            )

        by_construct = logs.groupby("ConstructId")
        self.difficulty_by_construct_ = 1.0 - by_construct["IsCorrect"].mean()
        self.log_n_interactions_by_construct_ = np.log1p(by_construct.size())
        self.response_time_by_construct_ = _response_time_by_construct(logs)

        self.global_fallback_ = {
            "q_construct_difficulty": float(self.difficulty_by_construct_.mean()),
            "q_construct_log_n_interactions": float(self.log_n_interactions_by_construct_.mean()),
            "q_construct_avg_response_time": float(self.response_time_by_construct_.mean()),
        }

        self.years_seen_ = sorted(train_rows[YEAR_KEY].unique().tolist())

        numeric = self._numeric_features(train_rows)
        self.scaler_ = StandardScaler().fit(numeric.to_numpy())

        return self

    def transform(self, rows: pd.DataFrame, logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError(f"{self.name}.transform called before fit")

        numeric = self._numeric_features(rows)
        scaled = self.scaler_.transform(numeric.to_numpy())
        features = pd.DataFrame(scaled, columns=self.NUMERIC_COLUMNS, index=rows.index)

        for year in self.years_seen_:
            features[f"year_{year}"] = (rows[YEAR_KEY] == year).astype(float).to_numpy()

        features = features.reset_index(drop=True)
        return self.check_output(features, rows)

    def _numeric_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Raw (unscaled) per-row construct statistics, missing constructs imputed."""
        construct_ids = rows[QUESTION_CONSTRUCT_KEY]

        difficulty = construct_ids.map(self.difficulty_by_construct_)
        log_n = construct_ids.map(self.log_n_interactions_by_construct_)
        response_time = construct_ids.map(self.response_time_by_construct_)

        difficulty = difficulty.fillna(self.global_fallback_["q_construct_difficulty"])
        log_n = log_n.fillna(self.global_fallback_["q_construct_log_n_interactions"])
        response_time = response_time.fillna(self.global_fallback_["q_construct_avg_response_time"])

        return pd.DataFrame(
            {
                "q_construct_difficulty": difficulty.to_numpy(),
                "q_construct_log_n_interactions": log_n.to_numpy(),
                "q_construct_avg_response_time": response_time.to_numpy(),
            },
            index=rows.index,
        )
