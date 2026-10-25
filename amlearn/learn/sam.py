"""Safe action-model learning (SAM) for lifted STRIPS from fully observed traces.

After Stern and Juba (2017) and Juba, Le and Stern (2021). For each action schema:

- the precondition starts as every literal, positive and negative, over the schema's
  parameters, and loses each literal some observed pre-state falsifies;
- the effects are exactly the changes observed: an atom false before and true after is an
  add effect, true before and false after a delete effect.

The model is **safe**: a plan valid under it is valid under the true model and reaches the
same states. A true effect never observed as a change was already in the state it would
have produced every time the action was seen, so the learned precondition keeps that
literal and the effect is a no-op wherever the learned model lets the action run. That
argument needs negative preconditions, deterministic STRIPS, full observability and
injective grounding; `amlearn.strips` fixes the last, and the property is tested on every
benchmark rather than assumed.

The price of safety is completeness. An unobserved action is absent from the model, and a
literal that held by coincidence in every observation stays a precondition, so a learned
model solves fewer problems than the true one. How many fewer, as a function of the number
of demonstrations, is the learning curve this package measures.

Whether some model meets the assumptions can be read off the traces: `unexplained` replays
each step under the learned model and names every atom it gets wrong.
"""

from collections.abc import Iterable

from amlearn.learn import Signature
from amlearn.strips import (
    Action,
    Atom,
    Domain,
    Trace,
    Transition,
    is_injective,
    lifted_atoms,
    render,
)


def learn(signature: Signature, traces: Iterable[Trace]) -> Domain:
    """Return the safe model of the actions observed in ``traces``.

    A transition whose action repeats an object is skipped: it lies outside the injective
    semantics the guarantee is stated for, and one of its ground atoms would lift to more
    than one schema atom.
    """
    candidates = {
        name: tuple(lifted_atoms(signature.predicates, params))
        for name, params in signature.actions.items()
    }
    pre_pos: dict[str, set[Atom]] = {}
    pre_neg: dict[str, set[Atom]] = {}
    add: dict[str, set[Atom]] = {}
    delete: dict[str, set[Atom]] = {}
    for trace in traces:
        for step in trace:
            name, objects = step.action[0], step.action[1:]
            if not is_injective(objects):
                continue
            params = signature.actions[name]
            if name not in pre_pos:
                pre_pos[name] = set(candidates[name])
                pre_neg[name] = set(candidates[name])
                add[name] = set()
                delete[name] = set()
            binding = dict(zip(params, objects, strict=True))
            for atom in candidates[name]:
                ground = (atom[0], *(binding[v] for v in atom[1:]))
                before = ground in step.state
                after = ground in step.next_state
                if before:
                    pre_neg[name].discard(atom)
                else:
                    pre_pos[name].discard(atom)
                if after and not before:
                    add[name].add(atom)
                if before and not after:
                    delete[name].add(atom)
    actions = {
        name: Action(
            name,
            signature.actions[name],
            frozenset(pre_pos[name]),
            frozenset(pre_neg[name]),
            frozenset(add[name]),
            frozenset(delete[name]),
        )
        for name in sorted(pre_pos)
    }
    return Domain(f"{signature.name}-sam", dict(signature.predicates), actions)


def unexplained(model: Domain, step: Transition) -> list[str]:
    """Return every atom ``model`` gets wrong replaying ``step``, one reason per atom.

    ``model`` is the one `learn` returns from traces that include ``step``. Then every step
    replays with an empty list exactly when some STRIPS model without constants reproduces
    all of them under injective grounding: the effects of any such model contain every
    change observed, so the learned model reproduces each step it learned from, and it is
    such a model itself. A reason therefore refutes the traces, not the learner: an atom
    changes that the action does not name, which no schema without constants can do, or an
    effect another step of the same action shows fails to happen here.

    A step that repeats an object is skipped, as `learn` skips it.
    """
    name, objects = step.action[0], step.action[1:]
    if not is_injective(objects):
        return []
    action = model.actions[name]
    replayed = action.ground(objects).apply(step.state)
    param = dict(zip(objects, action.params, strict=True))
    reasons: list[str] = []
    for atom in sorted(replayed ^ step.next_state):
        outside = sorted({o for o in atom[1:] if o not in param})
        if outside:
            reasons.append(
                f"{render(atom)} changes, but {render(step.action)} does not name"
                f" {' or '.join(outside)}"
            )
            continue
        lifted = render((atom[0], *(param[o] for o in atom[1:])))
        if atom in step.next_state:
            reasons.append(
                f"{render(atom)} still holds after {render(step.action)},"
                f" but another {name} step deletes {lifted}"
            )
        else:
            reasons.append(
                f"{render(atom)} is false after {render(step.action)},"
                f" but another {name} step adds {lifted}"
            )
    return reasons
