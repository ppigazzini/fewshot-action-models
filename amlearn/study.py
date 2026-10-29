"""The study file: the counts a learning curve publishes, and the check that reads them back.

A study is written by ``tools/curve.py --json`` and read back by ``tools/curve.py --check``,
which reruns the curve and requires the same counts. It is therefore a wire format between
two versions of this code, and its shape is declared once, here, as closed records: a key
the format does not name is a defect, not something to skip.

valgebra checks the shape, the integer type of every count (never a ``bool``, which
subclasses ``int``), and each count's range, and it reports every failure at once with its
path. The rules between two fields of a row, which no single annotation carries, are plain
Python in `check_rows`.
"""

import json
from typing import Annotated, Literal

from annotated_types import Ge
from typing_extensions import TypedDict

from amlearn._valgebra import NOT_BOOL, ValidationError, Validator

type Count = Annotated[int, NOT_BOOL, Ge(0)]
type Positive = Annotated[int, NOT_BOOL, Ge(1)]

FORMAT = 2


class Row(TypedDict, closed=True):
    """One point of the curve for one seed.

    The planning counts come first; then the true positives, false positives and false
    negatives of `amlearn.metrics`: precondition and effect literals against the reference,
    applicability of ground actions on the test states, and the atoms an action applied in
    both models changes there.
    """

    demos: Positive
    seed: Count
    transitions: Count
    unexplained: Count
    actions_learned: Count
    problems: Positive
    solved: Count
    valid: Count
    limit: Count
    reference_solved: Count
    pre_tp: Count
    pre_fp: Count
    pre_fn: Count
    eff_tp: Count
    eff_fp: Count
    eff_fn: Count
    app_tp: Count
    app_fp: Count
    app_fn: Count
    change_tp: Count
    change_fp: Count
    change_fn: Count


class Study(TypedDict, closed=True):
    """A learning curve over seeds, with the settings that produced it."""

    format: Literal[2]
    benchmark: str
    learner: Literal["sam", "optimistic", "pisam"]
    planner: Literal["bfs"]
    max_nodes: Positive
    test_problems: Positive
    train_sizes: list[Positive]
    test_sizes: list[Positive]
    demonstrator: Literal["shortest", "walk"]
    walk_steps: Count
    probe_steps: Count
    noise_permille: Count
    masked_permille: Count
    rows: list[Row]


#: The settings a study records, each compared by `differences`.
SETTINGS = (
    "format",
    "benchmark",
    "learner",
    "planner",
    "max_nodes",
    "test_problems",
    "train_sizes",
    "test_sizes",
    "demonstrator",
    "walk_steps",
    "probe_steps",
    "noise_permille",
    "masked_permille",
)


STUDY = Validator(Study)


class StudyError(ValueError):
    """A study file that is malformed or contradicts itself."""


def check_rows(study: Study) -> list[str]:
    """Return every rule a row breaks between two of its fields, or with another row."""
    problems: list[str] = []
    first: dict[tuple[int, int], int] = {}
    for i, row in enumerate(study["rows"]):
        at = f"rows[{i}]"
        point = (row["seed"], row["demos"])
        if point in first:
            problems.append(f"{at}: seed {point[0]}, demos {point[1]} repeats rows[{first[point]}]")
        first.setdefault(point, i)
        if row["problems"] != study["test_problems"]:
            problems.append(f"{at}: problems {row['problems']} != {study['test_problems']}")
        if not row["valid"] <= row["solved"] <= row["problems"]:
            problems.append(f"{at}: needs valid <= solved <= problems")
        if row["limit"] > row["problems"] - row["solved"]:
            problems.append(f"{at}: more limit hits than unsolved problems")
        if row["reference_solved"] > row["problems"]:
            problems.append(f"{at}: reference_solved exceeds problems")
        if row["valid"] > row["reference_solved"]:
            problems.append(f"{at}: more valid plans than problems the reference solves")
    return problems


def loads(text: str) -> Study:
    """Parse and check a study, reporting every failure rather than the first."""
    try:
        study = STUDY.load(text)
    except ValidationError as exc:
        raise StudyError(str(exc)) from exc
    broken = check_rows(study)
    if broken:
        raise StudyError("; ".join(broken))
    return study


def dumps(study: Study) -> str:
    """Serialize a study, refusing one this code should never have built."""
    try:
        STUDY.validate(study)
    except ValidationError as exc:
        raise StudyError(str(exc)) from exc
    broken = check_rows(study)
    if broken:
        raise StudyError("; ".join(broken))
    return json.dumps(study, indent=1) + "\n"


def differences(expected: Study, actual: Study) -> list[str]:
    """Return one line per setting or count that differs."""
    out = [
        f"{key}: {expected[key]!r} != {actual[key]!r}"
        for key in SETTINGS
        if expected[key] != actual[key]
    ]
    if len(expected["rows"]) != len(actual["rows"]):
        out.append(f"rows: {len(expected['rows'])} != {len(actual['rows'])}")
        return out
    for i, (old, new) in enumerate(zip(expected["rows"], actual["rows"], strict=True)):
        out.extend(f"rows[{i}].{k}: {old[k]} != {new[k]}" for k in old if old[k] != new[k])
    return out
