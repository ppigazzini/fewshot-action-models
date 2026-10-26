import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from amlearn.curve import rows_for_seed
from amlearn.domains import BENCHMARKS
from amlearn.study import differences, loads

ROOT = Path(__file__).parent.parent


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


def test_a_plan_counts_as_valid_only_where_the_reference_solves_within_the_bound():
    # At 40 nodes the reference exhausts the bound on every test problem of this seed, and
    # the learned model, whose search is narrower, still solves four.
    row = rows_for_seed(BENCHMARKS["blocksworld"], 1, [2], test_problems=6, max_nodes=40)[0]
    assert row["reference_solved"] == row["valid"] == 0
    assert row["solved"] == 4
    assert row["limit"] == 2


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


class TestTool:
    TOOL = (sys.executable, str(Path(__file__).parent.parent / "tools" / "curve.py"))

    def test_a_curve_no_reference_plan_scores_prints_without_a_score(self):
        args = ["blocksworld", "--demos", "1", "--seeds", "1", "--test", "2", "--max-nodes", "1"]
        out = subprocess.run([*self.TOOL, *args], capture_output=True, text=True, check=True)
        assert out.stdout.splitlines()[-1] == "| 1 | 0.0 | 0.0 | 0.0 (0-0) | n/a | n/a |"

    def test_check_fails_on_a_count_the_code_no_longer_produces(self, tmp_path):
        study = json.loads((ROOT / "docs" / "studies" / "gripper.json").read_text())
        study["rows"][3]["transitions"] += 1
        changed = tmp_path / "gripper.json"
        changed.write_text(json.dumps(study))
        out = subprocess.run(
            [*self.TOOL, "gripper", "--check", str(changed)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 1
        assert "rows[3].transitions:" in out.stderr

    def test_a_study_of_another_benchmark_is_refused(self):
        study = str(ROOT / "docs" / "studies" / "gripper.json")
        out = subprocess.run(
            [*self.TOOL, "logistics", "--check", study], capture_output=True, text=True, check=False
        )
        assert out.returncode == 2
        assert "is a gripper study, not logistics" in out.stderr

    @pytest.mark.parametrize("option", ["--test", "--seeds", "--max-nodes", "--demos"])
    def test_a_count_below_one_is_refused(self, option):
        out = subprocess.run(
            [*self.TOOL, "blocksworld", option, "0"], capture_output=True, text=True, check=False
        )
        assert out.returncode == 2
        assert "0 is not a positive count" in out.stderr


def test_demonstrations_do_not_depend_on_the_hash_seed(bench, tmp_path):
    written = []
    for hash_seed in ("0", "4242"):
        target = tmp_path / f"{bench.name}-{hash_seed}.jsonl"
        tool = [sys.executable, str(ROOT / "tools" / "learn.py"), bench.name, "--demos", "5"]
        subprocess.run(
            [*tool, "--write-traces", str(target)],
            check=True,
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            capture_output=True,
        )
        written.append(target.read_text(encoding="utf-8"))
    assert written[0] == written[1]


def test_a_study_does_not_depend_on_the_hash_seed(tmp_path):

    root = Path(__file__).parent.parent
    outputs = []
    for hash_seed in ("0", "4242"):
        target = tmp_path / f"study-{hash_seed}.json"
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        subprocess.run(
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
