"""Breadth-first search over ground states: shortest plans, and proofs of unsolvability."""

from collections import deque

from amlearn.plan import Outcome
from amlearn.strips import Atom, Domain, GroundAction, Problem

type _Step = tuple[int, int, int, int, GroundAction]


def _encode(domain: Domain, problem: Problem) -> tuple[list[_Step], int, int]:
    """Return the relevant actions, the initial state and the goal as masks of atom bits.

    Only the changing atoms get a bit. `Domain.relevant` checks every static literal against
    the initial state, and a static atom never changes, so leaving them out changes no
    successor; a logistics state drops from about 35 atoms to 6. A state is then an int:
    applicability is two masks, a successor two more, and its hash is the int itself, where
    a set of tuples is hashed element by element. The bits are a bijection on the changing
    atoms, so the states, their order and the plan are those of the atoms themselves.
    """
    static = domain.static_predicates()
    bit: dict[Atom, int] = {}

    def mask(atoms: frozenset[Atom]) -> int:
        out = 0
        for atom in atoms:
            if atom[0] not in static:
                out |= 1 << bit.setdefault(atom, len(bit))
        return out

    steps = [
        (mask(a.pre_pos), mask(a.pre_neg), ~mask(a.delete), mask(a.add), a.label)
        for a in domain.relevant(problem)
    ]
    start = mask(problem.init)
    # A static goal atom holds in every state or in none. One the initial state holds is
    # left out with the rest; one it lacks is kept, as a bit no state sets, so the search
    # exhausts and proves the goal unreachable.
    goal = mask(problem.goal)
    for atom in problem.goal - problem.init:
        if atom[0] in static:
            goal |= 1 << bit.setdefault(atom, len(bit))
    return steps, start, goal


def bfs(domain: Domain, problem: Problem, *, max_nodes: int = 200_000) -> Outcome:
    """Return a shortest plan for ``problem`` under ``domain``.

    Successors are generated in the fixed order of `Domain.relevant`, so the plan returned
    among equally short ones is a function of the inputs alone.

    ``max_nodes`` bounds the states stored, not expanded: memory is what runs out first. The
    search returns ``limit`` when it stores the ``max_nodes``-th state and that state is not
    a goal, so ``unsolvable`` proves a problem with fewer than ``max_nodes`` reachable states.
    """
    steps, start, goal = _encode(domain, problem)
    if start & goal == goal:
        return Outcome("solved", (), 0)
    parent: dict[int, tuple[int, GroundAction] | None] = {start: None}
    frontier = deque([start])
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        for pre, neg, keep, add, label in steps:
            if state & pre != pre or state & neg:
                continue
            succ = (state & keep) | add
            if succ in parent:
                continue
            parent[succ] = (state, label)
            if succ & goal == goal:
                return Outcome("solved", _unwind(parent, succ), expanded)
            if len(parent) >= max_nodes:
                return Outcome("limit", None, expanded)
            frontier.append(succ)
    return Outcome("unsolvable", None, expanded)


def _unwind(
    parent: dict[int, tuple[int, GroundAction] | None], state: int
) -> tuple[GroundAction, ...]:
    steps: list[GroundAction] = []
    link = parent[state]
    while link is not None:
        state, label = link
        steps.append(label)
        link = parent[state]
    return tuple(reversed(steps))
