"""Plan validation against a reference model: the only judge of a learned model's plans.

A learned model's own simulation of its plan proves nothing about the world; the plan is
valid only if it executes under the reference domain and reaches the goal there.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from amlearn.strips import Domain, GroundAction, Problem, is_injective, render


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
    objects = set(problem.objects)
    state = problem.init
    for i, label in enumerate(plan):
        name, args = label[0], label[1:]
        action = domain.actions.get(name)
        if action is None:
            return Verdict(valid=False, step=i, reason=f"no action {name!r} in {domain.name}")
        unknown = [a for a in args if a not in objects]
        if unknown:
            return Verdict(valid=False, step=i, reason=f"{render(label)}: unknown {unknown}")
        if not is_injective(args):
            return Verdict(valid=False, step=i, reason=f"{render(label)}: repeated object")
        grounded = action.ground(args)
        missing = sorted(grounded.pre_pos - state)
        present = sorted(grounded.pre_neg & state)
        if missing or present:
            why = [f"{render(f)} false" for f in missing] + [f"{render(f)} true" for f in present]
            return Verdict(valid=False, step=i, reason=f"{render(label)}: {', '.join(why)}")
        state = grounded.apply(state)
    unmet = sorted(problem.goal - state)
    if unmet:
        reason = f"goal unmet: {', '.join(render(f) for f in unmet)}"
        return Verdict(valid=False, step=len(plan), reason=reason)
    return Verdict(valid=True)
