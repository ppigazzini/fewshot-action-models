"""Answer-set planning with clingo, by incremental horizon.

The encoding is the textbook one for STRIPS with negative preconditions: one action per
step, frame axioms by default negation, delete before add. Ground actions come from
`Domain.relevant`, the same list breadth-first search expands, so the two planners differ
in search alone and a disagreement in plan length isolates a defect to one of them.

Atoms and actions enter the program as quoted strings: object names may start with an
upper-case letter, which clingo would read as a variable.
"""

import clingo

from amlearn.plan import Outcome
from amlearn.strips import Domain, GroundAction, Problem, render

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


def _q(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _facts(domain: Domain, problem: Problem) -> tuple[str, dict[str, GroundAction]]:
    lines: list[str] = []
    labels: dict[str, GroundAction] = {}
    for act in domain.relevant(problem):
        name = render(act.label)
        labels[name] = act.label
        a = _q(name)
        lines.append(f"action({a}).")
        lines += [f"pre({a},{_q(render(f))})." for f in sorted(act.pre_pos)]
        lines += [f"neg({a},{_q(render(f))})." for f in sorted(act.pre_neg)]
        lines += [f"add({a},{_q(render(f))})." for f in sorted(act.add)]
        lines += [f"del({a},{_q(render(f))})." for f in sorted(act.delete)]
    lines += [f"init({_q(render(f))})." for f in sorted(problem.init)]
    lines += [f"goal({_q(render(f))})." for f in sorted(problem.goal)]
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
                        name, step = sym.arguments
                        found.append((step.number, labels[name.string]))
                break
        if found or (t == 0 and problem.satisfied(problem.init)):
            return Outcome("solved", tuple(label for _, label in sorted(found)), t)
        ctl.release_external(query)
    return Outcome("limit", None, max_horizon)
