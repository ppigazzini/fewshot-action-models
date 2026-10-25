"""Every valgebra name the package uses, and the recipes it composes from them.

Nothing else in the package imports valgebra: a valgebra release that renames or moves a
name is one file's edit here, and `tests/test_boundaries.py` holds the boundary. The names
come from the ``valgebra`` package, whose ``__all__`` is the supported surface.

valgebra ships the minimal Boolean algebra and leaves derived forms to be composed; the
ones this package needs are written once, below, as the recipes of valgebra's own
documentation (`docs/04-algebra.md`, "Composition recipes").
"""

from typing import Annotated, Any, Literal

from annotated_types import Len
from valgebra import ValidationError, Validator, anything, complement, intersection, union

__all__ = [
    "NOT_BOOL",
    "ValidationError",
    "Validator",
    "anything",
    "complement",
    "constant",
    "each",
    "fixed",
    "implies",
    "intersection",
    "union",
]

#: Every value but a `bool`, which subclasses `int` and so lies inside a bare `int`.
NOT_BOOL = complement(bool)


def constant(value: object) -> Validator[Any]:
    """Return the set holding ``value`` alone, as a `Literal` built from a runtime value."""
    return Validator(Literal[value])  # ty: ignore[invalid-type-form]


def each(schema: object) -> Validator[Any]:
    """Return the lists whose every element lies in ``schema``."""
    return Validator(list[schema])  # ty: ignore[invalid-type-form]


def fixed(parts: list[object]) -> Validator[Any]:
    """Return the lists of exactly these elements, position by position.

    A one-element list literal is valgebra's homogeneous list, so a fixed length of one
    is a homogeneous list pinned to length one.
    """
    if not parts:
        return Validator([])
    if len(parts) == 1:
        return intersection(Annotated[list[Any], Len(1, 1)], each(parts[0]))
    return Validator(parts)


def implies(condition: object, then: object, otherwise: object = anything) -> Validator[Any]:
    """Return ``(condition & then) | (~condition & otherwise)``: the conditional recipe."""
    return union(intersection(condition, then), intersection(complement(condition), otherwise))
