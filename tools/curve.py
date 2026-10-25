"""Measure the learning curve of a benchmark, print it, and write or check the study file.

    uv run python tools/curve.py blocksworld                     # print the table
    uv run python tools/curve.py blocksworld --json docs/studies/blocksworld.json
    uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json

``--check`` reruns the curve with the settings the file records and exits 1 if any count
differs: a published number is one this code still produces.
"""

import argparse
import sys
from pathlib import Path

from amlearn.curve import rows_for_seed
from amlearn.domains import BENCHMARKS
from amlearn.study import FORMAT, Row, Study, differences, dumps, loads


def measure(  # noqa: PLR0913, PLR0917 - the settings a study records
    benchmark: str,
    demos: list[int],
    seeds: list[int],
    test_problems: int,
    max_nodes: int,
    learner: str = "sam",
) -> Study:
    """Run the curve for every seed and return it as a study."""
    bench = BENCHMARKS[benchmark]
    rows: list[Row] = []
    for seed in seeds:
        rows += rows_for_seed(
            bench, seed, demos, test_problems=test_problems, max_nodes=max_nodes, learner=learner
        )
    return Study(
        format=FORMAT,
        benchmark=benchmark,
        learner="sam" if learner == "sam" else "optimistic",
        planner="bfs",
        max_nodes=max_nodes,
        test_problems=test_problems,
        rows=rows,
    )


def _pct(values: list[float]) -> str:
    lo, hi, mean = min(values), max(values), sum(values) / len(values)
    return f"{100 * mean:.1f} ({100 * lo:.0f}-{100 * hi:.0f})"


def table(study: Study) -> str:
    """Render a study as a Markdown table: seed mean, with the per-seed range in brackets."""
    seeds = sorted({r["seed"] for r in study["rows"]})
    lines = [
        (
            f"{study['benchmark']}, {study['learner']}: {len(seeds)} seeds, "
            f"{study['test_problems']} test problems"
            f" per seed, BFS bound {study['max_nodes']} nodes"
        ),
        "",
        "| demos | transitions | actions | solved % | valid % of solved | score vs reference % |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({r["demos"] for r in study["rows"]}):
        rows = [r for r in study["rows"] if r["demos"] == n]
        trans = sum(r["transitions"] for r in rows) / len(rows)
        acts = sum(r["actions_learned"] for r in rows) / len(rows)
        solved = _pct([r["solved"] / r["problems"] for r in rows])
        solved_rows = [r for r in rows if r["solved"]]
        valid = _pct([r["valid"] / r["solved"] for r in solved_rows]) if solved_rows else "n/a"
        score = _pct([r["valid"] / r["reference_solved"] for r in rows if r["reference_solved"]])
        lines.append(f"| {n} | {trans:.1f} | {acts:.1f} | {solved} | {valid} | {score} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Parse arguments, measure, and print, write or check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS))
    parser.add_argument("--demos", default="1,2,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--test", type=int, default=50, help="test problems per seed")
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--learner", choices=["sam", "optimistic"], default="sam")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", type=Path, help="write the study file")
    out.add_argument("--check", type=Path, help="rerun a study file and compare")
    args = parser.parse_args(argv)

    if args.check:
        expected = loads(args.check.read_text(encoding="utf-8"))
        rows = expected["rows"]
        demos = sorted({r["demos"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        actual = measure(
            expected["benchmark"],
            demos,
            seeds,
            expected["test_problems"],
            expected["max_nodes"],
            expected["learner"],
        )
        diff = differences(expected, actual)
        print(table(actual))
        for line in diff:
            print(line, file=sys.stderr)
        print(f"\n{len(diff)} difference(s) against {args.check}")
        return 1 if diff else 0

    demos = sorted({int(d) for d in args.demos.split(",")})
    study = measure(
        args.benchmark, demos, list(range(args.seeds)), args.test, args.max_nodes, args.learner
    )
    print(table(study))
    if args.json:
        args.json.write_text(dumps(study), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
