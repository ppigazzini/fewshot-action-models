"""Plan validation against a reference model: the only judge of a learned model's plans.

A learned model's own simulation of its plan proves nothing about the world; the plan is
valid only if it executes under the reference domain and reaches the goal there.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from amlearn.strips import Domain, GroundAction, Problem, State, is_injective, render


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a plan is valid, and where and why it fails if it is not.

    ``step`` is the 0-based index of the failing action, or ``len(plan)`` when every step
    executes and the goal does not hold.
    """

    valid: bool
    step: int | None = None
    reason: str = ""


def validate(domain: Domain, problem: Problem, plan: Sequence[GroundAction]) -> Verdict:
    """Execute ``plan`` from the initial state of ``problem`` under ``domain``."""
    objects = frozenset(problem.objects)
    state = problem.init
    for i, label in enumerate(plan):
        reason = _refusal(domain, objects, state, label)
        if reason:
            return Verdict(valid=False, step=i, reason=reason)
        state = domain.actions[label[0]].ground(label[1:]).apply(state)
    unmet = sorted(problem.goal - state)
    if unmet:
        reason = f"goal unmet: {', '.join(render(f) for f in unmet)}"
        return Verdict(valid=False, step=len(plan), reason=reason)
    return Verdict(valid=True)


def _refusal(domain: Domain, objects: frozenset[str], state: State, label: GroundAction) -> str:
    """Return why ``label`` cannot be taken in ``state``, or the empty string if it can."""
    if not label:
        return "an empty step names no action"
    name, args = label[0], label[1:]
    action = domain.actions.get(name)
    if action is None:
        return f"no action {name!r} in {domain.name}"
    if len(args) != len(action.params):
        return f"{render(label)}: {name} takes {len(action.params)} objects, got {len(args)}"
    unknown = [a for a in args if a not in objects]
    if unknown:
        return f"{render(label)}: unknown {unknown}"
    if not is_injective(args):
        return f"{render(label)}: repeated object"
    grounded = action.ground(args)
    why = [f"{render(f)} false" for f in sorted(grounded.pre_pos - state)]
    why += [f"{render(f)} true" for f in sorted(grounded.pre_neg & state)]
    return f"{render(label)}: {', '.join(why)}" if why else ""
