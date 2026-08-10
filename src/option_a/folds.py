"""Leave-one-TreatmentLessonConstructId-out folds.

Fifteen treatment constructs generate fifteen folds. Fold sizes run from 3 to 13 rows, so
per-fold error is noisy by construction; that is a property of the frozen sample, not
something to smooth away by regrouping.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

FOLD_KEY = "TreatmentLessonConstructId"


@dataclass(frozen=True)
class Fold:
    name: str
    held_out_construct: int
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def leave_one_treatment_out(bench: pd.DataFrame) -> List[Fold]:
    """Build one fold per treatment construct, in ascending construct order.

    Ordering is deterministic so that two runs on the same data produce identical folds
    without depending on row order or on a random seed.
    """
    if FOLD_KEY not in bench.columns:
        raise ValueError(f"Benchmark table is missing the fold key '{FOLD_KEY}'")

    constructs = sorted(bench[FOLD_KEY].unique())
    positions = np.arange(len(bench))

    folds: List[Fold] = []
    for construct in constructs:
        mask = (bench[FOLD_KEY] == construct).to_numpy()
        test_idx = positions[mask]
        train_idx = positions[~mask]
        if len(train_idx) == 0:
            raise ValueError(f"Fold '{construct}' would have an empty training set")
        folds.append(
            Fold(
                name=f"LOTO_{construct}",
                held_out_construct=int(construct),
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )

    assert_folds_partition(folds, len(bench))
    return folds


def assert_folds_partition(folds: List[Fold], n_rows: int) -> None:
    """Every row is held out exactly once, and no row is in both sides of a fold."""
    seen: List[int] = []
    for fold in folds:
        overlap = set(fold.train_idx) & set(fold.test_idx)
        if overlap:
            raise ValueError(f"Fold {fold.name} has {len(overlap)} row(s) in both train and test")
        seen.extend(fold.test_idx.tolist())

    if sorted(seen) != list(range(n_rows)):
        counts = pd.Series(seen).value_counts()
        missing = sorted(set(range(n_rows)) - set(seen))
        repeated = sorted(counts[counts > 1].index.tolist())
        raise ValueError(
            f"Folds do not partition the {n_rows} evaluation rows. "
            f"missing={missing}, held out more than once={repeated}"
        )


def describe_folds(bench: pd.DataFrame, folds: List[Fold]) -> pd.DataFrame:
    """Fold sizes and year coverage, for the record rather than for tuning."""
    rows = []
    for fold in folds:
        test_rows = bench.iloc[fold.test_idx]
        train_years = set(bench.iloc[fold.train_idx]["Year"].unique())
        test_years = sorted(test_rows["Year"].unique())
        rows.append(
            {
                "fold": fold.name,
                "held_out_construct": fold.held_out_construct,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "test_years": test_years,
                "years_absent_from_train": sorted(set(test_years) - train_years),
            }
        )
    return pd.DataFrame(rows)
