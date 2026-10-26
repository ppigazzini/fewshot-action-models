"""Measure the learning curve of a benchmark, print it, and write or check the study file.

    uv run python tools/curve.py blocksworld                     # print the table
    uv run python tools/curve.py blocksworld --json docs/studies/blocksworld.json
    uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json

``--check`` reruns the curve with the settings the file records and exits 1 if any count
differs: a published number is one this code still produces.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from amlearn.curve import rows_for_seed
from amlearn.domains import BENCHMARKS
from amlearn.study import FORMAT, Row, Study, differences, dumps, loads


def measure(  # noqa: PLR0913 - the settings a study records
    benchmark: str,
    demos: list[int],
    seeds: list[int],
    *,
    test_problems: int,
    max_nodes: int,
    learner: str,
    train_sizes: list[int],
    test_sizes: list[int],
    demonstrator: str,
    walk_steps: int,
    probe_steps: int,
) -> Study:
    """Run the curve for every seed and return it as a study."""
    bench = BENCHMARKS[benchmark]
    rows: list[Row] = []
    for seed in seeds:
        rows += rows_for_seed(
            bench,
            seed,
            demos,
            test_problems=test_problems,
            max_nodes=max_nodes,
            learner=learner,
            train_sizes=train_sizes,
            test_sizes=test_sizes,
            demonstrator=demonstrator,
            walk_steps=walk_steps,
            probe_steps=probe_steps,
        )
    return Study(
        format=FORMAT,
        benchmark=benchmark,
        learner="sam" if learner == "sam" else "optimistic",
        planner="bfs",
        max_nodes=max_nodes,
        test_problems=test_problems,
        train_sizes=train_sizes,
        test_sizes=test_sizes,
        demonstrator="walk" if demonstrator == "walk" else "shortest",
        walk_steps=walk_steps,
        probe_steps=probe_steps,
        rows=rows,
    )


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"{value} is not a positive count"
        raise argparse.ArgumentTypeError(msg)
    return value


def _counts(text: str) -> list[int]:
    return sorted({_positive(part) for part in text.split(",")})


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
        scored = [r["valid"] / r["reference_solved"] for r in rows if r["reference_solved"]]
        score = _pct(scored) if scored else "n/a"
        lines.append(f"| {n} | {trans:.1f} | {acts:.1f} | {solved} | {valid} | {score} |")
    lines += [
        "",
        "Precision / recall %, counts pooled over seeds:",
        "",
        "| demos | preconditions | effects | applicability | changes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({r["demos"] for r in study["rows"]}):
        rows = [r for r in study["rows"] if r["demos"] == n]
        cells = [_pooled(rows, counts) for counts in FAMILIES]
        lines.append(f"| {n} | {' | '.join(cells)} |")
    return "\n".join(lines)


#: The model-quality counts of a row, family by family, in the order the table prints them.
FAMILIES: tuple[Callable[[Row], tuple[int, int, int]], ...] = (
    lambda r: (r["pre_tp"], r["pre_fp"], r["pre_fn"]),
    lambda r: (r["eff_tp"], r["eff_fp"], r["eff_fn"]),
    lambda r: (r["app_tp"], r["app_fp"], r["app_fn"]),
    lambda r: (r["change_tp"], r["change_fp"], r["change_fn"]),
)


def _pooled(rows: list[Row], counts: Callable[[Row], tuple[int, int, int]]) -> str:
    """Return the pooled precision and recall of one family of counts, as percentages."""
    tp, fp, fn = (sum(column) for column in zip(*map(counts, rows), strict=True))
    precision = f"{100 * tp / (tp + fp):.1f}" if tp + fp else "n/a"
    recall = f"{100 * tp / (tp + fn):.1f}" if tp + fn else "n/a"
    return f"{precision} / {recall}"


def main(argv: list[str]) -> int:
    """Parse arguments, measure, and print, write or check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS))
    parser.add_argument("--demos", type=_counts, default="1,2,5,10,20,50")
    parser.add_argument("--seeds", type=_positive, default=3)
    parser.add_argument("--test", type=_positive, default=50, help="test problems per seed")
    parser.add_argument("--max-nodes", type=_positive, default=200_000)
    parser.add_argument("--learner", choices=["sam", "optimistic"], default="sam")
    parser.add_argument("--train-sizes", type=_counts, help="default: the benchmark's")
    parser.add_argument("--test-sizes", type=_counts, help="default: the benchmark's")
    parser.add_argument("--demonstrator", choices=["shortest", "walk"], default="shortest")
    parser.add_argument("--walk-steps", type=_positive, default=10, help="steps per walk shown")
    parser.add_argument("--probe-steps", type=_positive, default=10, help="steps per test walk")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", type=Path, help="write the study file")
    out.add_argument("--check", type=Path, help="rerun a study file and compare")
    args = parser.parse_args(argv)

    if args.check:
        expected = loads(args.check.read_text(encoding="utf-8"))
        if expected["benchmark"] != args.benchmark:
            parser.error(f"{args.check} is a {expected['benchmark']} study, not {args.benchmark}")
        rows = expected["rows"]
        demos = sorted({r["demos"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        actual = measure(
            expected["benchmark"],
            demos,
            seeds,
            test_problems=expected["test_problems"],
            max_nodes=expected["max_nodes"],
            learner=expected["learner"],
            train_sizes=expected["train_sizes"],
            test_sizes=expected["test_sizes"],
            demonstrator=expected["demonstrator"],
            walk_steps=expected["walk_steps"],
            probe_steps=expected["probe_steps"],
        )
        diff = differences(expected, actual)
        print(table(actual))
        for line in diff:
            print(line, file=sys.stderr)
        print(f"\n{len(diff)} difference(s) against {args.check}")
        return 1 if diff else 0

    bench = BENCHMARKS[args.benchmark]
    study = measure(
        args.benchmark,
        args.demos,
        list(range(args.seeds)),
        test_problems=args.test,
        max_nodes=args.max_nodes,
        learner=args.learner,
        train_sizes=args.train_sizes or list(bench.train_sizes),
        test_sizes=args.test_sizes or list(bench.test_sizes),
        demonstrator=args.demonstrator,
        walk_steps=args.walk_steps if args.demonstrator == "walk" else 0,
        probe_steps=args.probe_steps,
    )
    print(table(study))
    if args.json:
        args.json.write_text(dumps(study), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
