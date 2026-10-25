"""Action-model learners: traces in, a `Domain` out.

A learner is given what the demonstrator knows without knowing the model: the predicates
with their arities and each action's name and parameter count. Preconditions and effects
are what it learns.
"""

from dataclasses import dataclass, field

from amlearn.strips import Domain


@dataclass(frozen=True, slots=True)
class Signature:
    """Predicates, and each action's parameter names, without preconditions or effects."""

    name: str
    predicates: dict[str, int]
    actions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, domain: Domain) -> Signature:
        """Strip ``domain`` to what a learner is allowed to know."""
        return cls(
            domain.name,
            dict(domain.predicates),
            {name: a.params for name, a in domain.actions.items()},
        )
