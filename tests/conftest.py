import random
from collections.abc import Callable

import pytest

from amlearn.domains import BENCHMARKS, Benchmark
from amlearn.strips import Action, Domain, Problem


@pytest.fixture(params=sorted(BENCHMARKS))
def bench(request) -> Benchmark:
    return BENCHMARKS[request.param]


@pytest.fixture
def blocks() -> Benchmark:
    return BENCHMARKS["blocksworld"]


@pytest.fixture
def rng() -> random.Random:
    return random.Random("tests")


@pytest.fixture
def guarded() -> tuple[Domain, Problem]:
    """``make`` needs ``blocked`` false, and only ``unblock`` makes it so: the shortest plan is
    unblock then make, and any shorter one breaks the negative precondition."""
    make = Action(
        "make", ("?x",), pre_neg=frozenset({("blocked", "?x")}), add=frozenset({("made", "?x")})
    )
    unblock = Action("unblock", ("?x",), delete=frozenset({("blocked", "?x")}))
    domain = Domain("guarded", {"blocked": 1, "made": 1}, {"make": make, "unblock": unblock})
    problem = Problem("p", ("o",), frozenset({("blocked", "o")}), frozenset({("made", "o")}))
    return domain, problem


@pytest.fixture
def chain() -> Domain:
    """A token stepping along ``next``: one reachable state per position."""
    step = Action(
        "step",
        ("?a", "?b"),
        pre_pos=frozenset({("at", "?a"), ("next", "?a", "?b")}),
        add=frozenset({("at", "?b")}),
        delete=frozenset({("at", "?a")}),
    )
    return Domain("chain", {"at": 1, "next": 2}, {"step": step})


@pytest.fixture
def walk_to() -> Callable[[str], Problem]:
    """Return the chain problem with goal ``at(goal)``: positions o1, o2, o3 are reachable,
    o4 never is."""

    def problem(goal: str) -> Problem:
        init = {("at", "o1"), ("next", "o1", "o2"), ("next", "o2", "o3")}
        objects = ("o1", "o2", "o3", "o4")
        return Problem("walk", objects, frozenset(init), frozenset({("at", goal)}))

    return problem
