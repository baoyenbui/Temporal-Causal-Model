"""The frozen 88-row evaluation sample.

This module is the only authorised way to load the Option A evaluation sample. It exists
because the notebook path currently reaches the A/B table through a left join on
`construct_experiments_input_test.csv` (45 rows), which silently reduces the sample to 57
rows and duplicates 13 of them. The benchmark table is the left table here, always.
"""

import hashlib
import os
from typing import Dict, List

import pandas as pd

PRIMARY_TARGET = "ate_k_1__"
SENSITIVITY_TARGET = "ate_p_1__"

N_BENCHMARK_ROWS = 88

BENCHMARK_FILE = "construct_experiments_ates_test.csv"

EXPECTED_COLUMNS = [
    "TreatmentLessonConstructId",
    "QuestionConstructId",
    "Year",
    "ControlLessonConstructIds",
    "ControlUsersCount",
    "TreatmentUsersCount",
    "ate_p_1__",
    "ate_k_1__",
]

# Aliases without the trailing double underscore existed in earlier notebook code and
# silently produced an empty analysis. Seeing one is a schema failure, not a warning.
FORBIDDEN_ALIASES = ["ate_p_1", "ate_k_1"]

# A model input drawn from any of these leaks the outcome, or information only available
# after the outcome, into a prediction for that row.
FORBIDDEN_MODEL_INPUTS = frozenset(
    {
        "ate_p_1__",
        "ate_k_1__",
        "ControlUsersCount",
        "TreatmentUsersCount",
        "n00",
        "n01",
        "n10",
        "n11",
        "p010_m",
        "k010_m",
        "ATE_difference",
        "TotalUsersCount",
        "ProportionTreatment",
    }
)

FROZEN_INPUT_SHA256: Dict[str, str] = {
    "checkin_to_checkout.csv": "0E9A4A461AF6EC2027FBFCAC3858D0FAB49526E6BDE436526C47234165EA1C7D",
    "checkins_lessons_checkouts_training.csv": "B5076BBABC6A3B9B8DB6F0C4FEC3F67E9EED27213D4209A0FBC7E5DCCAE13D94",
    "construct_experiments_ates_test.csv": "791107AA8FD5969F544CB4B2E4DF28EBF22CDAE9A675A3F02CA1DEE8EE8969F3",
    "construct_experiments_input_test.csv": "9C547DC3114A25E2B884AC2E37116EC7A2FE0282595901BEEA4D91E940D2C265",
    "construct_prerequisites_test.csv": "846161CC0EE708F3CA5B515A880F5E6DBFACDE66FBD03D6FB54E4F8DFF5C32B4",
    "constructs_input_test.csv": "A7482206822F687EC79BD80A3E1E0BDB5108217FEFAE3BA79B370154163099B2",
    "student_metadata.csv": "6D16E1B875F4EF0AA4996CB8145C16CEF37A243163450D815285BD7B8607B6F8",
    "subject_metadata.csv": "72C4B16936BAEE70A25D5B8B05753E4BDDAB06F3690343D8055C8A5808587EC8",
    "topic_pathway_metadata.csv": "35FE8A90501992BA1D070A09634D51CE70D7DBAFE325B65EBF22BBD140C890F2",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_input_hashes(data_dir: str = "data") -> Dict[str, Dict[str, object]]:
    """Compare every frozen input against its recorded SHA-256.

    Returned per file: the expected hash, the observed hash, and whether they agree.
    A mismatch means the data version changed and no result from this run may be
    compared against an earlier one.
    """
    report: Dict[str, Dict[str, object]] = {}
    for filename, expected in FROZEN_INPUT_SHA256.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            report[filename] = {"expected": expected, "observed": None, "match": False}
            continue
        observed = sha256_file(path)
        report[filename] = {
            "expected": expected,
            "observed": observed,
            "match": observed == expected,
        }
    return report


def parse_control_lessons(value: object) -> List[int]:
    """Parse the published `{a,b}` control-lesson set without inventing a comparator.

    A row with several control lessons keeps all of them. Reducing such a row to a single
    control construct would change the comparator defined by the source experiment.
    """
    if pd.isna(value):
        raise ValueError("ControlLessonConstructIds is null; the row has no published comparator")
    text = str(value).strip().strip("{}").strip()
    if not text:
        raise ValueError("ControlLessonConstructIds is empty; the row has no published comparator")
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_benchmark(data_dir: str = "data", strict_hash: bool = True) -> pd.DataFrame:
    """Load the 88-row evaluation sample, failing closed on any contract violation.

    No row may be dropped for effect size, sign, model error, or graph reachability. If a
    row cannot be parsed, that is a reported preprocessing failure, not a silent removal,
    so this function raises rather than returning a shorter table.
    """
    path = os.path.join(data_dir, BENCHMARK_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark table not found at {path}")

    if strict_hash:
        observed = sha256_file(path)
        expected = FROZEN_INPUT_SHA256[BENCHMARK_FILE]
        if observed != expected:
            raise ValueError(
                f"{BENCHMARK_FILE} does not match the frozen snapshot.\n"
                f"  expected {expected}\n  observed {observed}\n"
                f"Results from a different data version are not comparable to the frozen protocol."
            )

    bench = pd.read_csv(path)

    if list(bench.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Benchmark schema mismatch.\n  expected {EXPECTED_COLUMNS}\n  observed {list(bench.columns)}"
        )

    for alias in FORBIDDEN_ALIASES:
        if alias in bench.columns:
            raise ValueError(
                f"Column '{alias}' is the wrong target name; the materialised columns are "
                f"'{PRIMARY_TARGET}' and '{SENSITIVITY_TARGET}'. Failing closed instead of "
                f"running an empty analysis."
            )

    if len(bench) != N_BENCHMARK_ROWS:
        raise ValueError(
            f"Expected exactly {N_BENCHMARK_ROWS} evaluation rows, found {len(bench)}. "
            f"The evaluation sample is frozen; rows may not be added or dropped."
        )

    for target in (PRIMARY_TARGET, SENSITIVITY_TARGET):
        if bench[target].isna().any():
            n_null = int(bench[target].isna().sum())
            raise ValueError(f"{n_null} row(s) have a null '{target}'; every row needs a target value")

    bench = bench.copy()
    bench["ControlLessonConstructIds_parsed"] = bench["ControlLessonConstructIds"].apply(
        parse_control_lessons
    )
    bench["row_id"] = bench.index

    return bench


def assert_no_forbidden_inputs(features: pd.DataFrame, where: str = "feature matrix") -> None:
    """Reject a feature matrix that carries outcome or post-outcome information."""
    offending = sorted(set(features.columns) & FORBIDDEN_MODEL_INPUTS)
    if offending:
        raise ValueError(
            f"{where} contains forbidden model inputs {offending}. These are the A/B outcome, "
            f"the treatment/control user counts, or a descendant of the checkout outcome, and "
            f"none of them is available before the row's outcome."
        )
