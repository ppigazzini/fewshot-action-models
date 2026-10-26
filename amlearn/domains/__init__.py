"""Reference domains and the random problems each benchmark draws.

A reference domain is the model a learned one is scored against. It is read from the PDDL
file beside this module, so the text a reader sees is the model the validator runs.

Every generator takes a `random.Random` and draws from sorted sequences only: a problem must
not depend on set iteration order, or a published number would change with
``PYTHONHASHSEED``.
"""

import functools
import random
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from amlearn.pddl import load_domain
from amlearn.strips import Atom, Domain, Problem

_HERE = Path(__file__).parent

type Generator = Callable[[random.Random, int], Problem]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """A reference domain, its problem generator, and the sizes it trains and tests on.

    Test sizes include the largest training size and go one beyond it, so a model is scored
    on instances larger than any it saw demonstrated as well as on ones as large.
    """

    name: str
    domain: Domain
    generate: Generator
    train_sizes: tuple[int, ...]
    test_sizes: tuple[int, ...]
    #: The generator of test problems, where it differs from the training one.
    generate_test: Generator | None = None


def _towers(rng: random.Random, blocks: list[str]) -> list[list[str]]:
    order = blocks[:]
    rng.shuffle(order)
    towers: list[list[str]] = []
    for block in order:
        if not towers or rng.random() < 0.5:  # noqa: PLR2004
            towers.append([block])
        else:
            rng.choice(towers).append(block)
    return towers


def _resting(towers: list[list[str]]) -> set[Atom]:
    atoms: set[Atom] = set()
    for tower in towers:
        atoms.add(("ontable", tower[0]))
        atoms.update(("on", upper, lower) for lower, upper in pairwise(tower))
    return atoms


def blocksworld_problem(rng: random.Random, n: int) -> Problem:
    """Draw ``n`` blocks in random towers, and a different random arrangement as the goal."""
    blocks = [f"b{i}" for i in range(1, n + 1)]
    start = _towers(rng, blocks)
    init = _resting(start) | {("clear", t[-1]) for t in start} | {("handempty",)}
    while True:
        goal = _resting(_towers(rng, blocks))
        if not goal <= init:
            break
    return Problem(f"blocks-{n}", tuple(blocks), frozenset(init), frozenset(goal))


def gripper_problem(rng: random.Random, n: int) -> Problem:
    """Draw ``n`` balls spread over two rooms, the robot in one, and a different spread as goal."""
    rooms = ["rooma", "roomb"]
    grippers = ["left", "right"]
    balls = [f"ball{i}" for i in range(1, n + 1)]
    init: set[Atom] = {("room", r) for r in rooms} | {("gripper", g) for g in grippers}
    init |= {("ball", b) for b in balls} | {("free", g) for g in grippers}
    init.add(("at-robby", rng.choice(rooms)))
    init |= {("at", b, rng.choice(rooms)) for b in balls}
    while True:
        goal = {("at", b, rng.choice(rooms)) for b in balls}
        if not goal <= init:
            break
    objects = tuple(rooms + grippers + balls)
    return Problem(f"gripper-{n}", objects, frozenset(init), frozenset(goal))


def logistics_problem(rng: random.Random, n: int, *, away: float | None = None) -> Problem:
    """Draw ``n`` packages over two cities, each with an airport and a post office.

    One truck per city and one airplane start at random places of their own kind; each
    package starts and ends at random locations, and at least one must move. With ``away``
    a package's goal is in the other city with that probability and in its own otherwise,
    so ``away=0.1`` draws problems that rarely need the airplane.
    """
    cities = ["c1", "c2"]
    airports = {c: f"ap{c[1]}" for c in cities}
    offices = {c: f"po{c[1]}" for c in cities}
    trucks = {c: f"t{c[1]}" for c in cities}
    locations = sorted([*airports.values(), *offices.values()])
    packages = [f"p{i}" for i in range(1, n + 1)]
    init: set[Atom] = {("city", c) for c in cities} | {("airplane", "a1")}
    init |= {("location", loc) for loc in locations} | {("airport", a) for a in airports.values()}
    init |= {("obj", p) for p in packages} | {("truck", t) for t in trucks.values()}
    for c in cities:
        init |= {("in-city", airports[c], c), ("in-city", offices[c], c)}
        init.add(("at", trucks[c], rng.choice([airports[c], offices[c]])))
    init.add(("at", "a1", rng.choice(sorted(airports.values()))))
    start = {p: rng.choice(locations) for p in packages}
    init |= {("at", p, start[p]) for p in packages}
    city = {loc: c for c in cities for loc in (airports[c], offices[c])}

    def destination(p: str) -> str:
        if away is None:
            return rng.choice(locations)
        other = rng.random() < away
        return rng.choice([loc for loc in locations if (city[loc] != city[start[p]]) == other])

    while True:
        goal = {("at", p, destination(p)) for p in packages}
        if not goal <= init:
            break
    objects = tuple(cities + locations + sorted(trucks.values()) + ["a1"] + packages)
    return Problem(f"logistics-{n}", objects, frozenset(init), frozenset(goal))


BENCHMARKS: dict[str, Benchmark] = {
    "blocksworld": Benchmark(
        "blocksworld",
        load_domain(_HERE / "blocksworld.pddl"),
        blocksworld_problem,
        train_sizes=(3, 4),
        test_sizes=(4, 5),
    ),
    "logistics": Benchmark(
        "logistics",
        load_domain(_HERE / "logistics.pddl"),
        logistics_problem,
        train_sizes=(1, 2),
        test_sizes=(2, 3),
    ),
    # Demonstrations that rarely show the airplane, tests that need it on most problems:
    # a package changes city in 1 training goal in 10 and in 1 test goal in 2.
    "logistics-local": Benchmark(
        "logistics-local",
        load_domain(_HERE / "logistics.pddl"),
        functools.partial(logistics_problem, away=0.1),
        train_sizes=(1, 2),
        test_sizes=(2, 3),
        generate_test=logistics_problem,
    ),
    "gripper": Benchmark(
        "gripper",
        load_domain(_HERE / "gripper.pddl"),
        gripper_problem,
        train_sizes=(2, 3),
        test_sizes=(3, 4),
    ),
}


def draw(
    bench: Benchmark,
    rng: random.Random,
    count: int,
    sizes: tuple[int, ...],
    *,
    test: bool = False,
) -> list[Problem]:
    """Draw ``count`` problems, each size picked uniformly from ``sizes``.

    ``test`` draws from the benchmark's test generator, which is its training generator
    unless the benchmark names another.
    """
    generate = bench.generate_test if test and bench.generate_test else bench.generate
    return [generate(rng, rng.choice(sizes)) for _ in range(count)]
