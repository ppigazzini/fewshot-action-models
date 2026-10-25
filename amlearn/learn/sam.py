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
"""

from collections.abc import Iterable

from amlearn.learn import Signature
from amlearn.strips import Action, Atom, Domain, Trace, is_injective, lifted_atoms


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
