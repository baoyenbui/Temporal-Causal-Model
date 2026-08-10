"""Option A: construct-level CATE evaluation against the frozen CausalEdu A/B benchmark.

This package is the shared contract between the two student workstreams. It owns the
evaluation sample, the folds, and the metrics, so that feature construction and method
implementation can proceed in parallel without either side guessing the other's shape.

Nothing in this package may be changed without a dated PI decision. See CONTRACT.md.
"""

from src.option_a.benchmark import (
    PRIMARY_TARGET,
    SENSITIVITY_TARGET,
    FORBIDDEN_MODEL_INPUTS,
    N_BENCHMARK_ROWS,
    load_benchmark,
    verify_input_hashes,
)
from src.option_a.folds import leave_one_treatment_out, Fold
from src.option_a.metrics import (
    mae,
    rmse,
    spearman,
    sign_agreement,
    cluster_bootstrap_ci,
    paired_difference_ci,
)
from src.option_a.methods import Method, ZeroEffect, TrainMean, YearStratifiedMean
from src.option_a.features import FeatureBuilder

__all__ = [
    "PRIMARY_TARGET",
    "SENSITIVITY_TARGET",
    "FORBIDDEN_MODEL_INPUTS",
    "N_BENCHMARK_ROWS",
    "load_benchmark",
    "verify_input_hashes",
    "leave_one_treatment_out",
    "Fold",
    "mae",
    "rmse",
    "spearman",
    "sign_agreement",
    "cluster_bootstrap_ci",
    "paired_difference_ci",
    "Method",
    "ZeroEffect",
    "TrainMean",
    "YearStratifiedMean",
    "FeatureBuilder",
]
