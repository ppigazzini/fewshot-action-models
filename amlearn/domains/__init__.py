"""Reference domains and the random problems each benchmark draws.

A reference domain is the model a learned one is scored against. It is read from the PDDL
file beside this module, so the text a reader sees is the model the validator runs.

Every generator takes a `random.Random` and draws from sorted sequences only: a problem must
not depend on set iteration order, or a published number would change with
``PYTHONHASHSEED``.
"""

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


BENCHMARKS: dict[str, Benchmark] = {
    "blocksworld": Benchmark(
        "blocksworld",
        load_domain(_HERE / "blocksworld.pddl"),
        blocksworld_problem,
        train_sizes=(3, 4),
        test_sizes=(4, 5),
    ),
    "gripper": Benchmark(
        "gripper",
        load_domain(_HERE / "gripper.pddl"),
        gripper_problem,
        train_sizes=(2, 3),
        test_sizes=(3, 4),
    ),
}


def draw(bench: Benchmark, rng: random.Random, count: int, sizes: tuple[int, ...]) -> list[Problem]:
    """Draw ``count`` problems, each size picked uniformly from ``sizes``."""
    return [bench.generate(rng, rng.choice(sizes)) for _ in range(count)]
