"""Read and write the PDDL subset `amlearn.strips` represents.

The subset is untyped STRIPS with negative preconditions: a precondition is a conjunction
of atoms and negated atoms, an effect a conjunction of added and deleted atoms, a goal a
conjunction of atoms. Types, constants in schemas, conditional effects, quantifiers and
numeric fluents raise `PddlError` naming the construct, so a domain outside the subset
fails when it is read rather than being learned as something it is not.

Grounding is injective, and a PDDL planner binds two parameters to one object unless a
precondition forbids it, so the writer states the semantics: ``(not (= ?x ?y))`` for every
pair of parameters, under ``:equality``. The reader drops such an inequality, which the
semantics already implies, and refuses any other use of ``=``. The writer emits what the
reader accepts, so a learned model can be handed to any planner or validator that reads
PDDL and is grounded there as it is here.
"""

import itertools
from collections.abc import Iterable
from pathlib import Path

from amlearn.strips import Action, Atom, Domain, Problem, StripsError

type SExpr = str | list[SExpr]


class PddlError(StripsError):
    """PDDL text outside the supported subset, or malformed."""


def tokenize(text: str) -> list[str]:
    """Split PDDL text into parentheses and lower-cased symbols, dropping ``;`` comments."""
    lines = (line.split(";", 1)[0] for line in text.splitlines())
    return " ".join(lines).replace("(", " ( ").replace(")", " ) ").lower().split()


def parse(text: str) -> SExpr:
    """Parse one s-expression."""
    tokens = tokenize(text)
    expr, end = _parse(tokens, 0)
    if end != len(tokens):
        msg = f"trailing tokens after the expression: {tokens[end : end + 5]}"
        raise PddlError(msg)
    return expr


def _parse(tokens: list[str], i: int) -> tuple[SExpr, int]:
    if i >= len(tokens):
        msg = "unexpected end of input"
        raise PddlError(msg)
    tok = tokens[i]
    if tok == ")":
        msg = "unexpected ')'"
        raise PddlError(msg)
    if tok != "(":
        return tok, i + 1
    out: list[SExpr] = []
    i += 1
    while i < len(tokens) and tokens[i] != ")":
        expr, i = _parse(tokens, i)
        out.append(expr)
    if i >= len(tokens):
        msg = "unbalanced '('"
        raise PddlError(msg)
    return out, i + 1


def _sections(expr: SExpr, kind: str) -> tuple[str, dict[str, list[SExpr]]]:
    if not isinstance(expr, list) or len(expr) < 2 or expr[0] != "define":  # noqa: PLR2004
        msg = "expected (define ...)"
        raise PddlError(msg)
    head = expr[1]
    if not isinstance(head, list) or len(head) != 2 or head[0] != kind:  # noqa: PLR2004
        msg = f"expected ({kind} <name>)"
        raise PddlError(msg)
    sections: dict[str, list[SExpr]] = {}
    for item in expr[2:]:
        if not isinstance(item, list) or not item or not isinstance(item[0], str):
            msg = f"malformed section {item!r}"
            raise PddlError(msg)
        key = item[0]
        if key == ":action":
            sections.setdefault(key, []).append(item)
        elif key in sections:
            msg = f"{key} appears twice"
            raise PddlError(msg)
        else:
            sections[key] = item[1:]
    return _symbol(head[1]), sections


def _symbol(expr: SExpr) -> str:
    if not isinstance(expr, str):
        msg = f"expected a symbol, got {expr!r}"
        raise PddlError(msg)
    return expr


def _atom(expr: SExpr) -> Atom:
    if isinstance(expr, str):
        return (expr,)
    if expr and expr[0] in {"and", "or", "not", "forall", "exists", "when", "imply", "="}:
        msg = f"{expr[0]!r} is outside the supported subset"
        raise PddlError(msg)
    if not expr or not all(isinstance(e, str) for e in expr):
        msg = f"expected an atom, got {expr!r}"
        raise PddlError(msg)
    return tuple(_symbol(e) for e in expr)


def _literals(expr: SExpr) -> tuple[frozenset[Atom], frozenset[Atom]]:
    """Split a conjunction into its positive and its negated atoms."""
    parts: list[SExpr] = expr[1:] if isinstance(expr, list) and expr[:1] == ["and"] else [expr]
    pos: set[Atom] = set()
    neg: set[Atom] = set()
    for part in parts:
        if isinstance(part, list) and part[:1] == ["not"]:
            if len(part) != 2:  # noqa: PLR2004
                msg = f"malformed negation {part!r}"
                raise PddlError(msg)
            neg.add(_atom(part[1]))
        else:
            pos.add(_atom(part))
    return frozenset(pos), frozenset(neg)


def _action(item: list[SExpr]) -> Action:
    if not item[1:]:
        msg = "an :action without a name"
        raise PddlError(msg)
    name = _symbol(item[1])
    fields: dict[str, SExpr] = {}
    rest = item[2:]
    if len(rest) % 2:
        msg = f"{name}: a field without a value in {rest!r}"
        raise PddlError(msg)
    for key, value in zip(rest[::2], rest[1::2], strict=True):
        fields[_symbol(key)] = value
    unknown = set(fields) - {":parameters", ":precondition", ":effect"}
    if unknown:
        msg = f"{name}: unsupported action fields {sorted(unknown)}"
        raise PddlError(msg)
    raw_params = fields.get(":parameters", [])
    if not isinstance(raw_params, list) or "-" in raw_params:
        msg = f"{name}: typed or malformed parameters; the subset is untyped"
        raise PddlError(msg)
    params = tuple(_symbol(p) for p in raw_params)
    pre_pos, pre_neg = _literals(_drop_distinct(name, params, fields.get(":precondition", ["and"])))
    add, delete = _literals(fields.get(":effect", ["and"]))
    try:
        return Action(name, params, pre_pos, pre_neg, add, delete)
    except StripsError as exc:
        raise PddlError(str(exc)) from exc


def _drop_distinct(name: str, params: tuple[str, ...], expr: SExpr) -> SExpr:
    """Drop each ``(not (= ?a ?b))`` over two parameters from a precondition."""
    parts: list[SExpr] = expr[1:] if isinstance(expr, list) and expr[:1] == ["and"] else [expr]
    kept: list[SExpr] = []
    for part in parts:
        # A negation of anything but one expression is left to `_literals`, which refuses it:
        # read as an inequality, its other expressions would be dropped with it.
        negation = isinstance(part, list) and part[:1] == ["not"] and len(part) == 2  # noqa: PLR2004
        inner = part[1] if negation else None
        if not (isinstance(inner, list) and inner[:1] == ["="]):
            kept.append(part)
            continue
        args = inner[1:]
        if len(args) != 2 or args[0] == args[1] or not all(a in params for a in args):  # noqa: PLR2004
            msg = f"{name}: {part!r} is not an inequality of two parameters"
            raise PddlError(msg)
    return ["and", *kept]


def read_domain(text: str) -> Domain:
    """Read a domain."""
    name, sections = _sections(parse(text), "domain")
    predicates: dict[str, int] = {}
    for decl in sections.get(":predicates", []):
        atom = _atom(decl)
        if "-" in atom:
            msg = f"typed predicate {atom!r}; the subset is untyped"
            raise PddlError(msg)
        predicates[atom[0]] = len(atom) - 1
    if ":types" in sections or ":constants" in sections:
        msg = f"{name}: :types and :constants are outside the supported subset"
        raise PddlError(msg)
    actions = {}
    for item in sections.get(":action", []):
        if not isinstance(item, list):
            msg = f"malformed action {item!r}"
            raise PddlError(msg)
        action = _action(item)
        if action.name in actions:
            msg = f"{name}: action {action.name} defined twice"
            raise PddlError(msg)
        actions[action.name] = action
    try:
        return Domain(name, predicates, actions)
    except StripsError as exc:
        raise PddlError(str(exc)) from exc


def read_problem(text: str) -> Problem:
    """Read a problem."""
    name, sections = _sections(parse(text), "problem")
    objects = tuple(_symbol(o) for o in sections.get(":objects", []))
    if "-" in objects:
        msg = f"{name}: typed objects; the subset is untyped"
        raise PddlError(msg)
    init = frozenset(_atom(a) for a in sections.get(":init", []))
    goal_expr = sections.get(":goal", [["and"]])
    if len(goal_expr) != 1:
        msg = f"{name}: :goal holds {len(goal_expr)} expressions, not one"
        raise PddlError(msg)
    goal, negated = _literals(goal_expr[0])
    if negated:
        msg = f"{name}: negated goals are outside the supported subset"
        raise PddlError(msg)
    try:
        return Problem(name, objects, init, goal)
    except StripsError as exc:
        raise PddlError(str(exc)) from exc


def load_domain(path: Path) -> Domain:
    """Read a domain file."""
    return read_domain(path.read_text(encoding="utf-8"))


def _fmt(atom: Atom) -> str:
    return f"({' '.join(atom)})"


def _conj(pos: Iterable[Atom], neg: Iterable[Atom] = (), distinct: tuple[str, ...] = ()) -> str:
    parts = [_fmt(a) for a in sorted(pos)] + [f"(not {_fmt(a)})" for a in sorted(neg)]
    parts += [f"(not (= {a} {b}))" for a, b in itertools.combinations(distinct, 2)]
    return f"(and {' '.join(parts)})" if parts else "(and)"


def write_domain(domain: Domain) -> str:
    """Write a domain, literals sorted so equal models give equal text."""
    preds = " ".join(
        _fmt((p, *(f"?a{i}" for i in range(domain.predicates[p]))))
        for p in sorted(domain.predicates)
    )
    reqs = ":strips :negative-preconditions"
    if any(len(a.params) > 1 for a in domain.actions.values()):
        reqs += " :equality"
    lines = [
        f"(define (domain {domain.name})",
        f"  (:requirements {reqs})",
        f"  (:predicates {preds})",
    ]
    for name in sorted(domain.actions):
        a = domain.actions[name]
        lines += [
            f"  (:action {a.name}",
            f"    :parameters ({' '.join(a.params)})",
            f"    :precondition {_conj(a.pre_pos, a.pre_neg, a.params)}",
            f"    :effect {_conj(a.add, a.delete)})",
        ]
    lines.append(")")
    return "\n".join(lines) + "\n"


def write_problem(problem: Problem, domain_name: str) -> str:
    """Write a problem."""
    return "\n".join(
        [
            f"(define (problem {problem.name}) (:domain {domain_name})",
            f"  (:objects {' '.join(sorted(problem.objects))})",
            f"  (:init {' '.join(_fmt(a) for a in sorted(problem.init))})",
            f"  (:goal {_conj(problem.goal)}))",
            "",
        ]
    )
