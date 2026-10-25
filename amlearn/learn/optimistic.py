"""The unsafe contrast to SAM: observed effects, and no precondition at all.

This is the most permissive model consistent with what the traces show an action does:
every action is applicable everywhere. It solves at least every problem the safe model
solves, and the plans it finds are judged by the reference like any other. The learning
curve of the two side by side is what safety buys and what it costs.
"""

from collections.abc import Iterable

from amlearn.learn import Signature
from amlearn.learn.sam import learn as learn_sam
from amlearn.strips import Action, Domain, Trace


def learn(signature: Signature, traces: Iterable[Trace]) -> Domain:
    """Return SAM's effects for every observed action, with empty preconditions."""
    safe = learn_sam(signature, traces)
    actions = {
        name: Action(name, a.params, frozenset(), frozenset(), a.add, a.delete)
        for name, a in safe.actions.items()
    }
    return Domain(f"{signature.name}-optimistic", dict(signature.predicates), actions)
