"""Breadth-first search over ground states: shortest plans, and proofs of unsolvability."""

from collections import deque

from amlearn.plan import Outcome
from amlearn.strips import Domain, GroundAction, Problem, State


def bfs(domain: Domain, problem: Problem, *, max_nodes: int = 200_000) -> Outcome:
    """Return a shortest plan for ``problem`` under ``domain``.

    Successors are generated in the fixed order of `Domain.relevant`, so the plan returned
    among equally short ones is a function of the inputs alone.

    ``max_nodes`` bounds the states stored, not expanded: memory is what runs out first. The
    search returns ``limit`` when it stores the ``max_nodes``-th state and that state is not
    a goal, so ``unsolvable`` proves a problem with fewer than ``max_nodes`` reachable states.
    """
    actions = domain.relevant(problem)
    start = problem.init
    if problem.satisfied(start):
        return Outcome("solved", (), 0)
    parent: dict[State, tuple[State, GroundAction] | None] = {start: None}
    frontier = deque([start])
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        for action in actions:
            if not action.applicable(state):
                continue
            succ = action.apply(state)
            if succ in parent:
                continue
            parent[succ] = (state, action.label)
            if problem.satisfied(succ):
                return Outcome("solved", _unwind(parent, succ), expanded)
            if len(parent) >= max_nodes:
                return Outcome("limit", None, expanded)
            frontier.append(succ)
    return Outcome("unsolvable", None, expanded)


def _unwind(
    parent: dict[State, tuple[State, GroundAction] | None], state: State
) -> tuple[GroundAction, ...]:
    steps: list[GroundAction] = []
    link = parent[state]
    while link is not None:
        state, label = link
        steps.append(label)
        link = parent[state]
    return tuple(reversed(steps))
