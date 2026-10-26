"""The model-quality counts of Stern et al. (2025), beside the problem-solving ones of a curve.

Two families, each kept as true positives, false positives and false negatives summed over
actions, which is the cumulative form of the metrics: a study records counts, and the
precision and recall they give are computed where they are printed.

- Syntactic: the literals of each reference action's precondition and effect against the
  learned action's. A literal is an atom with its sign, so a negative precondition is a
  literal like any other, and an action the model lacks has every reference literal as a
  false negative.
- Predictive: on a set of states, which ground actions each model applies (applicability),
  and, where both apply one, which atoms each changes (changes). A safe model applies
  nothing the reference refuses and changes exactly what the reference changes, so both of
  its false-positive counts and its change false negatives are 0.

The states are those of random walks under the reference from each test problem: states
the environment reaches, not every assignment of the atoms.
"""

import random
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from amlearn.strips import Atom, Domain, GroundAction, GroundedAction, Problem, State
from amlearn.traces import random_walk


class Counts(NamedTuple):
    """True positives, false positives and false negatives of one comparison."""

    tp: int
    fp: int
    fn: int


def _compare(got: Iterable[object], want: Iterable[object]) -> Counts:
    got, want = set(got), set(want)
    return Counts(len(got & want), len(got - want), len(want - got))


def _total(parts: Iterable[Counts]) -> Counts:
    tp = fp = fn = 0
    for part in parts:
        tp, fp, fn = tp + part.tp, fp + part.fp, fn + part.fn
    return Counts(tp, fp, fn)


def _signed(pos: frozenset[Atom], neg: frozenset[Atom]) -> set[tuple[bool, Atom]]:
    return {(True, a) for a in pos} | {(False, a) for a in neg}


def syntactic(model: Domain, reference: Domain) -> tuple[Counts, Counts]:
    """Return the precondition and the effect counts of ``model`` against ``reference``."""
    pre: list[Counts] = []
    eff: list[Counts] = []
    for name in sorted(reference.actions):
        want = reference.actions[name]
        got = model.actions.get(name)
        pre.append(
            _compare(
                _signed(got.pre_pos, got.pre_neg) if got else (),
                _signed(want.pre_pos, want.pre_neg),
            )
        )
        eff.append(
            _compare(_signed(got.add, got.delete) if got else (), _signed(want.add, want.delete))
        )
    return _total(pre), _total(eff)


class Probe(NamedTuple):
    """A test state, with what the reference applies there and the atoms each one changes."""

    problem: Problem
    state: State
    truth: dict[GroundAction, frozenset[Atom]]


def probes(
    reference: Domain, problems: Iterable[Problem], length: int, rng: random.Random
) -> list[Probe]:
    """Return the states of a reference random walk of ``length`` steps from each problem.

    The reference's side of every comparison is computed here once, for all the models a
    seed learns.
    """
    out: list[Probe] = []
    for problem in problems:
        actions = reference.relevant(problem)
        walk = random_walk(reference, problem, length, rng)
        for state in [problem.init, *(step.next_state for step in walk)]:
            truth = {g.label: g.apply(state) ^ state for g in actions if g.applicable(state)}
            out.append(Probe(problem, state, truth))
    return out


def predictive(model: Domain, reference: Domain, cases: Sequence[Probe]) -> tuple[Counts, Counts]:
    """Return the applicability and the change counts of ``model`` on ``cases``.

    The model's groundings are pruned by the reference's static predicates: their atoms
    hold as in the initial state along every reference walk, so a grounding that fails one
    there is applicable in no test state. The model's own static predicates would not do: an
    atom only an unobserved action changes looks static to the model and changes on a walk.
    """
    static = reference.static_predicates()
    app: list[Counts] = []
    change: list[Counts] = []
    learned: list[GroundedAction] = []
    problem: Problem | None = None
    for case in cases:
        if case.problem is not problem:
            problem = case.problem
            learned = model.relevant(problem, static=static)
        got = {g.label: g for g in learned if g.applicable(case.state)}
        app.append(_compare(got, case.truth))
        change += [
            _compare(got[label].apply(case.state) ^ case.state, flips)
            for label, flips in case.truth.items()
            if label in got
        ]
    return _total(app), _total(change)
