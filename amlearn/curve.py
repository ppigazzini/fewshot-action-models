"""The learning curve: plan quality of a learned model against the number of demonstrations.

For each seed, one pool of training problems and one set of test problems are drawn. The
model for ``n`` demonstrations learns from the first ``n`` traces of the pool, so the
curve is nested: each point sees every trace the previous point saw, and a dip between two
points is the learner's, not a different draw's.

Every number a row carries is a count. Ratios are computed where they are printed, so the
study file compares exactly across platforms and a rounding rule is never part of a result.
"""

import random
from collections.abc import Sequence

from amlearn.domains import Benchmark, draw
from amlearn.learn import Signature, optimistic, sam
from amlearn.plan.bfs import bfs
from amlearn.strips import Domain
from amlearn.study import Row
from amlearn.traces import demonstrate
from amlearn.validate import validate


def rows_for_seed(  # noqa: PLR0913 - the measurement's settings, each recorded in the study
    bench: Benchmark,
    seed: int,
    demos: Sequence[int],
    *,
    test_problems: int,
    max_nodes: int,
    learner: str = "sam",
) -> list[Row]:
    """Return one row per demonstration count for one seed.

    A plan counts as valid only on a test problem the reference model itself solves within
    the node bound, so the score is a fraction of the reference's own and never exceeds it.
    """
    learn = sam.learn if learner == "sam" else optimistic.learn
    train_rng = random.Random(f"{bench.name}:train:{seed}")
    test_rng = random.Random(f"{bench.name}:test:{seed}")
    pool = draw(bench, train_rng, max(demos), bench.train_sizes)
    tests = draw(bench, test_rng, test_problems, bench.test_sizes)
    traces = [demonstrate(bench.domain, p, max_nodes=max_nodes) for p in pool]
    reference_ok = [bfs(bench.domain, p, max_nodes=max_nodes).plan is not None for p in tests]
    reference_solved = sum(reference_ok)
    signature = Signature.of(bench.domain)
    rows: list[Row] = []
    previous: Domain | None = None
    solved = valid = limit = 0
    for n in demos:
        model = learn(signature, traces[:n])
        # Traces that show nothing new leave the model as it was, and an equal model searches
        # the same: two rows in three of every published study repeat the row before.
        if model != previous:
            solved = valid = limit = 0
            for problem, ok in zip(tests, reference_ok, strict=True):
                outcome = bfs(model, problem, max_nodes=max_nodes)
                limit += outcome.status == "limit"
                if outcome.plan is None:
                    continue
                solved += 1
                valid += ok and validate(bench.domain, problem, outcome.plan).valid
        previous = model
        rows.append(
            Row(
                demos=n,
                seed=seed,
                transitions=sum(len(t) for t in traces[:n]),
                actions_learned=len(model.actions),
                problems=len(tests),
                solved=solved,
                valid=valid,
                limit=limit,
                reference_solved=reference_solved,
            )
        )
    return rows
