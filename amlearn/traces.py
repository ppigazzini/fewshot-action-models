"""Demonstrations: fully observed state-action-state traces, produced under a reference model.

A demonstration is a shortest plan executed under the reference domain, every state
recorded. That makes a demonstration goal-directed, as a human one is, and so biased: an
action a shortest plan never takes is never observed, and the learned model omits it.
`observe` is the observer between the world and the learner, which may miss or hide atoms.
`random_walk` is the goal-independent contrast: a uniform choice among the applicable
actions, biased only toward states near the start and actions with many groundings.
"""

import itertools
import random
from collections.abc import Mapping

from amlearn.plan.bfs import bfs
from amlearn.strips import (
    Domain,
    GroundAction,
    Observed,
    Problem,
    State,
    StripsError,
    Trace,
    Transition,
)


def execute(domain: Domain, problem: Problem, plan: tuple[GroundAction, ...]) -> Trace:
    """Record every transition of ``plan`` under ``domain``, refusing an inapplicable step."""
    state = problem.init
    steps: list[Transition] = []
    for label in plan:
        grounded = domain.actions[label[0]].ground(label[1:])
        if not grounded.applicable(state):
            msg = f"step {len(steps)} of the plan is not applicable"
            raise StripsError(msg)
        succ = grounded.apply(state)
        steps.append(Transition(state, label, succ))
        state = succ
    return tuple(steps)


def demonstrate(domain: Domain, problem: Problem, *, max_nodes: int = 200_000) -> Trace:
    """Return a shortest plan's trace, or an empty trace when there is none within the bound."""
    outcome = bfs(domain, problem, max_nodes=max_nodes)
    if outcome.plan is None:
        return ()
    return execute(domain, problem, outcome.plan)


def random_walk(domain: Domain, problem: Problem, length: int, rng: random.Random) -> Trace:
    """Take up to ``length`` uniformly random applicable actions from the initial state."""
    actions = domain.relevant(problem)
    state = problem.init
    steps: list[Transition] = []
    for _ in range(length):
        options = [a for a in actions if a.applicable(state)]
        if not options:
            break
        chosen = rng.choice(options)
        succ = chosen.apply(state)
        steps.append(Transition(state, chosen.label, succ))
        state = succ
    return tuple(steps)


def observe(  # noqa: PLR0913 - the observer's two rates, and what it can see
    trace: Trace,
    rng: random.Random,
    *,
    predicates: Mapping[str, int],
    objects: tuple[str, ...],
    drop: float = 0.0,
    hide: float = 0.0,
) -> tuple[Observed, ...]:
    """Return ``trace`` as an observer saw it.

    The observer misses each true atom of a state with probability ``drop`` and reports it
    false, which is noise; and hides each atom over ``objects`` with probability ``hide``
    and reports it unknown, which is partial observability. Each state is seen once, so the
    step that ends in a state and the step that starts there see it alike, and a rate of 0
    draws nothing from ``rng``.
    """
    universe = [
        (name, *args)
        for name in sorted(predicates)
        for args in itertools.product(sorted(objects), repeat=predicates[name])
        if hide
    ]
    states = [trace[0].state, *(step.next_state for step in trace)] if trace else []
    seen: list[State] = []
    hidden: list[State] = []
    for state in states:
        kept = frozenset(a for a in sorted(state) if not (drop and rng.random() < drop))
        unknown = frozenset(a for a in universe if hide and rng.random() < hide)
        seen.append(kept - unknown)
        hidden.append(unknown)
    return tuple(
        Observed(Transition(seen[i], step.action, seen[i + 1]), hidden[i], hidden[i + 1])
        for i, step in enumerate(trace)
    )
