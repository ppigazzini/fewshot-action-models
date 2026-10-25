import os
import subprocess
import sys
from pathlib import Path

import pytest

from amlearn.curve import rows_for_seed
from amlearn.domains import BENCHMARKS
from amlearn.study import differences, loads


def test_rows_are_a_function_of_the_seed_alone():
    bench = BENCHMARKS["blocksworld"]
    first = rows_for_seed(bench, 0, [1, 3], test_problems=8, max_nodes=50_000)
    again = rows_for_seed(bench, 0, [1, 3], test_problems=8, max_nodes=50_000)
    assert first == again


def test_a_learned_model_is_never_scored_above_its_own_solved_count():
    bench = BENCHMARKS["blocksworld"]
    for row in rows_for_seed(bench, 1, [1, 2, 5], test_problems=8, max_nodes=50_000):
        assert row["valid"] == row["solved"]
        assert row["solved"] <= row["reference_solved"]


@pytest.mark.slow
def test_the_published_blocksworld_study_still_reproduces():
    path = Path(__file__).parent.parent / "docs" / "studies" / "blocksworld.json"
    expected = loads(path.read_text(encoding="utf-8"))
    demos = sorted({r["demos"] for r in expected["rows"]})
    rows = []
    for seed in sorted({r["seed"] for r in expected["rows"]}):
        rows += rows_for_seed(
            bench=BENCHMARKS[expected["benchmark"]],
            seed=seed,
            demos=demos,
            test_problems=expected["test_problems"],
            max_nodes=expected["max_nodes"],
        )
    assert differences(expected, expected | {"rows": rows}) == []


def test_a_study_does_not_depend_on_the_hash_seed(tmp_path):

    root = Path(__file__).parent.parent
    outputs = []
    for hash_seed in ("0", "4242"):
        target = tmp_path / f"study-{hash_seed}.json"
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        subprocess.run(  # noqa: S603 - the interpreter running the tests, on a committed tool
            [
                sys.executable,
                str(root / "tools" / "curve.py"),
                "blocksworld",
                "--demos",
                "1,3",
                "--seeds",
                "2",
                "--test",
                "6",
                "--json",
                str(target),
            ],
            check=True,
            env=env,
            capture_output=True,
        )
        outputs.append(target.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]
