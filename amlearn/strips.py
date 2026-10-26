"""Lifted STRIPS with negative preconditions: the representation every other module shares.

An atom is a tuple, predicate first: ``("on", "a", "b")``. In an action schema its arguments
are parameter names (``"?x"``); in a state they are objects. Schemas take no constants, so
every argument of a schema atom is a parameter, and lifting a ground atom back to a schema
atom is a lookup rather than a search.

Grounding is injective: distinct parameters take distinct objects. The safety guarantee of
`amlearn.learn.sam` holds under that semantics, and the planners and the validator use the
same one, so a learned model is never applied where its guarantee does not reach.
"""

import itertools
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

type Atom = tuple[str, ...]
type State = frozenset[Atom]
type GroundAction = tuple[str, ...]


class StripsError(ValueError):
    """A domain, problem or plan that breaks the representation's rules."""


@dataclass(frozen=True, slots=True)
class Action:
    """An action schema: parameters, a precondition of literals, add and delete effects."""

    name: str
    params: tuple[str, ...]
    pre_pos: frozenset[Atom] = frozenset()
    pre_neg: frozenset[Atom] = frozenset()
    add: frozenset[Atom] = frozenset()
    delete: frozenset[Atom] = frozenset()
    # Each literal set with its arguments as parameter positions, so grounding indexes the
    # objects instead of looking each argument up: twice as fast on a learned precondition
    # of 70 literals, which planning grounds for every binding of every test problem.
    _positions: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(set(self.params)) != len(self.params):
            msg = f"{self.name}: repeated parameter in {self.params}"
            raise StripsError(msg)
        bound = set(self.params)
        for atom in self.pre_pos | self.pre_neg | self.add | self.delete:
            unbound = [arg for arg in atom[1:] if arg not in bound]
            if unbound:
                msg = f"{self.name}: {render(atom)} uses {unbound}, which are not parameters"
                raise StripsError(msg)
        index = {p: i for i, p in enumerate(self.params)}
        positions = tuple(
            tuple((a[0], tuple(index[v] for v in a[1:])) for a in atoms)
            for atoms in (self.pre_pos, self.pre_neg, self.add, self.delete)
        )
        object.__setattr__(self, "_positions", positions)

    def ground(self, objects: tuple[str, ...]) -> GroundedAction:
        """Bind the parameters to ``objects`` in order."""
        if len(objects) != len(self.params):
            msg = f"{self.name} takes {len(self.params)} objects, got {len(objects)}"
            raise StripsError(msg)
        pre_pos, pre_neg, add, delete = (
            frozenset((name, *[objects[i] for i in at]) for name, at in atoms)
            for atoms in self._positions
        )
        return GroundedAction((self.name, *objects), pre_pos, pre_neg, add, delete)


@dataclass(frozen=True, slots=True)
class GroundedAction:
    """An action with its parameters bound: the unit the planners and the validator step on."""

    label: GroundAction
    pre_pos: frozenset[Atom]
    pre_neg: frozenset[Atom]
    add: frozenset[Atom]
    delete: frozenset[Atom]

    def applicable(self, state: State) -> bool:
        """Return whether every positive literal holds in ``state`` and no negative one does."""
        return self.pre_pos <= state and self.pre_neg.isdisjoint(state)

    def apply(self, state: State) -> State:
        """Return the successor state.

        Delete before add, as PDDL defines it: an atom both deleted and added holds after
        the step.
        """
        return (state - self.delete) | self.add


@dataclass(frozen=True, slots=True)
class Domain:
    """Predicates with their arities, and the action schemas by name."""

    name: str
    predicates: Mapping[str, int]
    actions: Mapping[str, Action] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for action in self.actions.values():
            for atom in action.pre_pos | action.pre_neg | action.add | action.delete:
                self.check_atom(atom, where=action.name)

    def check_atom(self, atom: Atom, *, where: str) -> None:
        """Raise if ``atom`` names an undeclared predicate or has the wrong arity."""
        arity = self.predicates.get(atom[0])
        if arity is None:
            msg = f"{where}: undeclared predicate {atom[0]!r}"
            raise StripsError(msg)
        if arity != len(atom) - 1:
            msg = f"{where}: {render(atom)} has {len(atom) - 1} arguments, {atom[0]} takes {arity}"
            raise StripsError(msg)

    def check_problem(self, problem: Problem) -> None:
        """Raise if an atom of ``problem`` names an undeclared predicate or has the wrong arity.

        Search would otherwise prove a misspelled goal unsolvable: an atom no action makes
        true and an atom no domain declares look the same to it. That every atom names listed
        objects needs no domain, and `Problem` holds it.
        """
        for part, atoms in (("init", problem.init), ("goal", problem.goal)):
            for atom in sorted(atoms):
                self.check_atom(atom, where=f"{problem.name} {part}")

    def grounded(self, objects: Iterable[str]) -> list[GroundedAction]:
        """Every injective grounding of every schema over ``objects``, in a fixed order.

        The order is the planners' expansion order, so it is sorted rather than left to set
        iteration: a plan must not depend on ``PYTHONHASHSEED``.
        """
        objs = sorted(objects)
        return [
            action.ground(binding)
            for name in sorted(self.actions)
            for action in (self.actions[name],)
            for binding in itertools.permutations(objs, len(action.params))
        ]

    def static_predicates(self) -> frozenset[str]:
        """Return the predicates no action adds or deletes: their atoms never change."""
        changed = {a[0] for act in self.actions.values() for a in act.add | act.delete}
        return frozenset(self.predicates) - changed

    def relevant(self, problem: Problem) -> list[GroundedAction]:
        """Return the groundings whose static preconditions hold in the initial state.

        A static atom never changes, so a grounding that fails one is never applicable;
        dropping it changes no search result and keeps `grounded`'s order. Gripper's pick
        over 5 balls has 504 groundings and 20 survive.

        Every planner and walk grounds through here, so the problem is checked against the
        domain here, once.
        """
        self.check_problem(problem)
        static = self.static_predicates()
        init = problem.init
        objs = sorted(problem.objects)
        out: list[GroundedAction] = []
        for name in sorted(self.actions):
            action = self.actions[name]
            # Narrow each parameter by its unary static literals before binding: the
            # product of the narrowed lists, in parameter order, is the subsequence of
            # `grounded`'s permutations that survives, in the same order, and it skips the
            # thousands of bindings a four-parameter action has over a dozen objects.
            choices = [
                [
                    o
                    for o in objs
                    if all(
                        (f[0], o) in init
                        for f in action.pre_pos
                        if f[0] in static and f[1:] == (p,)
                    )
                    and not any(
                        (f[0], o) in init
                        for f in action.pre_neg
                        if f[0] in static and f[1:] == (p,)
                    )
                ]
                for p in action.params
            ]
            for binding in itertools.product(*choices):
                if not is_injective(binding):
                    continue
                g = action.ground(binding)
                if all(f in init for f in g.pre_pos if f[0] in static) and not any(
                    f in init for f in g.pre_neg if f[0] in static
                ):
                    out.append(g)
        return out


@dataclass(frozen=True, slots=True)
class Problem:
    """Objects, an initial state and a conjunctive positive goal."""

    name: str
    objects: tuple[str, ...]
    init: State
    goal: frozenset[Atom]

    def __post_init__(self) -> None:
        # A repeated object makes `Domain.grounded` bind it to two parameters at once, which
        # is the one grounding the injective semantics excludes.
        if not is_injective(self.objects):
            repeated = sorted({o for o in self.objects if self.objects.count(o) > 1})
            msg = f"{self.name}: objects named more than once: {repeated}"
            raise StripsError(msg)
        objects = frozenset(self.objects)
        for part, atoms in (("init", self.init), ("goal", self.goal)):
            for atom in sorted(atoms):
                unknown = sorted(set(atom[1:]) - objects)
                if unknown:
                    msg = f"{self.name} {part}: {render(atom)} names {unknown}, not objects"
                    raise StripsError(msg)

    def satisfied(self, state: State) -> bool:
        """Return whether ``state`` reaches the goal."""
        return self.goal <= state


@dataclass(frozen=True, slots=True)
class Transition:
    """One fully observed step: the state, the ground action taken, the state that followed."""

    state: State
    action: GroundAction
    next_state: State


type Trace = tuple[Transition, ...]


def lifted_atoms(predicates: Mapping[str, int], params: tuple[str, ...]) -> Iterator[Atom]:
    """Yield every atom over ``params``, repeated variables included, in a fixed order."""
    for name in sorted(predicates):
        for args in itertools.product(params, repeat=predicates[name]):
            yield (name, *args)


def render(atom: Atom) -> str:
    """Render an atom as ``on(a,b)``, and a 0-ary one as ``handempty``."""
    return atom[0] if len(atom) == 1 else f"{atom[0]}({','.join(atom[1:])})"


def is_injective(objects: tuple[str, ...]) -> bool:
    """Return whether no object fills two parameters."""
    return len(set(objects)) == len(objects)
