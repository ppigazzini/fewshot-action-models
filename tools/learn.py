"""Learn a model from demonstrations and print it as PDDL, beside the reference it replaces.

uv run python tools/learn.py blocksworld --demos 10 --seed 0
uv run python tools/learn.py --domain mine.pddl --traces mine.jsonl

``--domain`` reads the signature from a PDDL domain: its predicates and each action's
parameters. Preconditions and effects in the file are not read by the learner.
"""

import argparse
import random
import sys
from pathlib import Path

from amlearn.domains import BENCHMARKS, draw
from amlearn.learn import Signature
from amlearn.learn.sam import learn
from amlearn.pddl import PddlError, load_domain, write_domain
from amlearn.tracefile import TraceFileError, TraceSchema
from amlearn.traces import demonstrate


def main(argv: list[str]) -> int:
    """Draw training problems, demonstrate them, learn, and print the learned domain."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", nargs="?", choices=sorted(BENCHMARKS))
    parser.add_argument("--domain", type=Path, help="read the signature from a PDDL domain")
    parser.add_argument("--demos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--traces", type=Path, help="learn from a trace file instead")
    source.add_argument("--write-traces", type=Path, help="also write the demonstrations")
    args = parser.parse_args(argv)
    if (args.benchmark is None) == (args.domain is None):
        parser.error("name exactly one of a benchmark and --domain")
    if args.domain and not args.traces:
        parser.error("--domain needs --traces: a domain file draws no problems to demonstrate")

    if args.domain:
        try:
            signature = Signature.of(load_domain(args.domain))
        except (OSError, PddlError) as exc:
            print(f"{args.domain}: {exc}", file=sys.stderr)
            return 1
    else:
        signature = Signature.of(BENCHMARKS[args.benchmark].domain)
    schema = TraceSchema.of(signature)
    if args.traces:
        try:
            traces = schema.load(args.traces)
        except (OSError, TraceFileError) as exc:
            print(f"{args.traces}:\n{exc}", file=sys.stderr)
            return 1
    else:
        bench = BENCHMARKS[args.benchmark]
        rng = random.Random(f"{bench.name}:train:{args.seed}")
        problems = draw(bench, rng, args.demos, bench.train_sizes)
        traces = [demonstrate(bench.domain, p) for p in problems]
        if args.write_traces:
            schema.dump(args.write_traces, traces)
    model = learn(signature, traces)
    print(f"; learned from {len(traces)} demonstrations, {sum(map(len, traces))} transitions")
    print(write_domain(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
