"""Learn a model from demonstrations and print it as PDDL, beside the reference it replaces.

uv run python tools/learn.py blocksworld --demos 10 --seed 0
"""

import argparse
import random
import sys

from amlearn.domains import BENCHMARKS, draw
from amlearn.learn import Signature
from amlearn.learn.sam import learn
from amlearn.pddl import write_domain
from amlearn.traces import demonstrate


def main(argv: list[str]) -> int:
    """Draw training problems, demonstrate them, learn, and print the learned domain."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS))
    parser.add_argument("--demos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    bench = BENCHMARKS[args.benchmark]
    rng = random.Random(f"{bench.name}:train:{args.seed}")
    traces = [demonstrate(bench.domain, p) for p in draw(bench, rng, args.demos, bench.train_sizes)]
    model = learn(Signature.of(bench.domain), traces)
    print(f"; learned from {args.demos} demonstrations, {sum(map(len, traces))} transitions")
    print(write_domain(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
