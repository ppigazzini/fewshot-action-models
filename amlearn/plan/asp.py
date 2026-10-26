"""Answer-set planning with clingo, by incremental horizon.

The encoding is the textbook one for STRIPS with negative preconditions: one action per
step, frame axioms by default negation, delete before add. Ground actions come from
`Domain.relevant`, the same list breadth-first search expands, so the two planners differ
in search alone and a disagreement in plan length isolates a defect to one of them.

Atoms and actions enter the program as integers, numbered as the facts list them, and a
plan decodes through the list of labels. No name meets clingo's syntax, where an upper-case
first letter reads as a variable and a comma inside a quoted ``on(a,b)`` would be confused
with an argument boundary; tuples of quoted names would be as safe and ground a third
slower on test-size problems.
"""

import clingo

from amlearn.plan import Outcome
from amlearn.strips import Atom, Domain, GroundAction, Problem

_BASE = """
holds(F,0) :- init(F).
"""

_STEP = """
1 { occ(A,t) : action(A) } 1.
:- occ(A,t), pre(A,F), not holds(F,t-1).
:- occ(A,t), neg(A,F), holds(F,t-1).
holds(F,t) :- occ(A,t), add(A,F).
holds(F,t) :- holds(F,t-1), not deleted(F,t).
deleted(F,t) :- occ(A,t), del(A,F).
"""

_CHECK = """
#external query(t).
:- query(t), goal(F), not holds(F,t).
"""


def _facts(domain: Domain, problem: Problem) -> tuple[str, list[GroundAction]]:
    """Return the facts, and the labels by the number each action has in them."""
    lines: list[str] = []
    atoms: dict[Atom, int] = {}
    labels: list[GroundAction] = []

    def n(atom: Atom) -> int:
        return atoms.setdefault(atom, len(atoms))

    for act in domain.relevant(problem):
        a = len(labels)
        labels.append(act.label)
        lines.append(f"action({a}).")
        lines += [f"pre({a},{n(f)})." for f in sorted(act.pre_pos)]
        lines += [f"neg({a},{n(f)})." for f in sorted(act.pre_neg)]
        lines += [f"add({a},{n(f)})." for f in sorted(act.add)]
        lines += [f"del({a},{n(f)})." for f in sorted(act.delete)]
    lines += [f"init({n(f)})." for f in sorted(problem.init)]
    lines += [f"goal({n(f)})." for f in sorted(problem.goal)]
    return "\n".join(lines), labels


def asp(domain: Domain, problem: Problem, *, max_horizon: int = 30) -> Outcome:
    """Return a shortest plan of at most ``max_horizon`` steps, or ``limit``.

    The search never returns ``unsolvable``: an unsatisfiable horizon says nothing about
    the next one, and a plan-length bound that proves unsolvability needs the state count,
    which is what breadth-first search computes anyway.
    """
    facts, labels = _facts(domain, problem)
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], facts + _BASE)
    ctl.add("step", ["t"], _STEP)
    ctl.add("check", ["t"], _CHECK)
    ctl.ground([("base", []), ("check", [clingo.Number(0)])])
    for t in range(max_horizon + 1):
        if t > 0:
            ctl.ground([("step", [clingo.Number(t)]), ("check", [clingo.Number(t)])])
        query = clingo.Function("query", [clingo.Number(t)])
        ctl.assign_external(query, truth=True)
        found: list[tuple[int, GroundAction]] = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for sym in model.symbols(atoms=True):
                    if sym.name == "occ":
                        action, step = sym.arguments
                        found.append((step.number, labels[action.number]))
                break
        if found or (t == 0 and problem.satisfied(problem.init)):
            return Outcome("solved", tuple(label for _, label in sorted(found)), t)
        ctl.release_external(query)
    return Outcome("limit", None, max_horizon)
