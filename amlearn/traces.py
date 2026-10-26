"""Demonstrations: fully observed state-action-state traces, produced under a reference model.

A demonstration is a shortest plan executed under the reference domain, every state
recorded. That makes a demonstration goal-directed, as a human one is, and so biased: an
action a shortest plan never takes is never observed, and the learned model omits it.
`random_walk` is the goal-independent contrast: a uniform choice among the applicable
actions, biased only toward states near the start and actions with many groundings.
"""

import random

from amlearn.plan.bfs import bfs
from amlearn.strips import Domain, GroundAction, Problem, StripsError, Trace, Transition


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
