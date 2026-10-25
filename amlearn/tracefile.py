"""Trace files: demonstrations recorded elsewhere, read against the domain's signature.

A trace file is JSON Lines, one trace per line::

    {"steps": [{"state": [["clear", "a"], ["handempty"]],
                "action": ["pick-up", "a"],
                "next_state": [["holding", "a"]]}, ...]}

An atom is its predicate followed by its objects, an action its name followed by its
objects, a state the list of atoms that hold. A simulator, a robot log or another planner
can write one; `tools/learn.py --traces` learns from it.

The schema is not written by hand: `TraceSchema.of` compiles it from the signature the
learner is given. An atom is one of the predicates with exactly its arity, an action one
of the actions with exactly its parameter count, and every object a non-empty string; a
step is a closed record, so a key the format does not name is refused. valgebra checks
each line in one pass and reports every failure with its path (``steps[3].state[2]``),
on every line, before anything is learned: a step whose atom has the wrong arity would
otherwise prune a true precondition and void the safety guarantee.

Two rules span more than one part of a file and are plain Python. Each step starts where
the step before it ended (`_check_continuity`). And one STRIPS model without constants
explains every step (`_check_strips`): the model SAM learns from the whole file replays
each one, or no such model exists and the safety guarantee, stated for traces such a
model produced, has nothing to stand on.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from annotated_types import MinLen

from amlearn._valgebra import (
    ValidationError,
    Validator,
    anything,
    constant,
    each,
    fixed,
    implies,
    intersection,
    union,
)
from amlearn.learn import Signature
from amlearn.learn.sam import learn, unexplained
from amlearn.strips import Atom, Trace, Transition

#: An object's name.
OBJECT = Validator(Annotated[str, MinLen(1)])

#: How many failures one error names before summarising the rest.
MAX_REPORTED = 20


class TraceFileError(ValueError):
    """A trace file that does not fit the signature or contradicts itself."""


def _shapes(arities: dict[str, int]) -> Validator[Any]:
    """Return the lists naming one of ``arities`` and exactly that many objects.

    Written as a known name followed by anything, met with "a list that starts with this
    name has this name's shape" for each name: a union of the shapes would decide the same
    set, but would report a misspelled name as the wrong arity of whichever name has the
    same length, and the conditional form names the unknown name instead.
    """
    names = union(*(constant(n) for n in sorted(arities)))
    clauses = [
        implies(Validator([constant(n), anything, ...]), fixed([constant(n), *[OBJECT] * k]))
        for n, k in sorted(arities.items())
    ]
    return intersection(Validator([names, anything, ...]), *clauses)


def _explain(item: dict[str, Any], raw: object, arities: dict[str, dict[str, int]]) -> str:
    """Return ``item``'s message, with the arity said where valgebra's rendering does not.

    A wrong arity fails one of the conditional clauses, which valgebra renders as the
    union it is; the arity is what the writer of the trace needs to read.
    """
    path = tuple(item.get("path") or ())
    value: Any = raw
    try:
        for part in path:
            value = value[part]
    except LookupError, TypeError:
        return str(item["message"])
    kind = "action" if path[-1:] == ("action",) else "atom"
    known = arities[kind]
    if item.get("code") == "union_error" and isinstance(value, list) and value[:1]:
        name = value[0]
        if name in known and len(value) - 1 != known[name]:
            where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path).lstrip(".")
            return f"at {where}: {name} takes {known[name]} objects, got {len(value) - 1}"
    return str(item["message"])


@dataclass(frozen=True, slots=True)
class TraceSchema:
    """The set of valid trace-file lines for one signature, compiled once."""

    signature: Signature
    line: Validator[Any]

    @property
    def arities(self) -> dict[str, dict[str, int]]:
        """Return each predicate's and each action's number of objects."""
        actions = {n: len(p) for n, p in self.signature.actions.items()}
        return {"atom": dict(self.signature.predicates), "action": actions}

    @classmethod
    def of(cls, signature: Signature) -> TraceSchema:
        """Compile the line schema from the predicates and action arities of ``signature``."""
        atom = _shapes(signature.predicates)
        action = _shapes({name: len(params) for name, params in signature.actions.items()})
        step = Validator({"state": each(atom), "action": action, "next_state": each(atom)})
        return cls(signature, Validator({"steps": each(step)}))

    def loads(self, text: str) -> list[Trace]:
        """Read every trace of a JSON Lines text, reporting every failure on every line."""
        failures: list[str] = []
        traces: list[Trace] = []
        numbers: list[int] = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = self.line.load(line)
            except ValidationError as exc:
                raw = _parsed(line)
                failures += [
                    f"line {number}: {_explain(item, raw, self.arities)}" for item in exc.errors
                ]
                continue
            trace = _trace(record["steps"])
            failures += [f"line {number}: {f}" for f in _check_continuity(trace)]
            traces.append(trace)
            numbers.append(number)
        failures += _check_strips(self.signature, numbers, traces)
        if failures:
            shown = failures[:MAX_REPORTED]
            more = len(failures) - len(shown)
            raise TraceFileError("\n".join(shown + ([f"(+{more} more)"] if more else [])))
        return traces

    def dumps(self, traces: Iterable[Trace]) -> str:
        """Write traces as JSON Lines, refusing one this signature does not admit."""
        lines = []
        for trace in traces:
            record = {"steps": [_step(t) for t in trace]}
            try:
                self.line.validate(record)
            except ValidationError as exc:
                raise TraceFileError(str(exc)) from exc
            lines.append(json.dumps(record, separators=(",", ":")))
        return "".join(line + "\n" for line in lines)

    def load(self, path: Path) -> list[Trace]:
        """Read a trace file."""
        return self.loads(path.read_text(encoding="utf-8"))

    def dump(self, path: Path, traces: Iterable[Trace]) -> None:
        """Write a trace file."""
        path.write_text(self.dumps(traces), encoding="utf-8")


def _parsed(line: str) -> object:
    """Return the JSON value of a line for a message, or None where it does not parse."""
    try:
        return json.loads(line)
    except ValueError:
        return None


def _atoms(raw: list[list[str]]) -> frozenset[Atom]:
    return frozenset(tuple(a) for a in raw)


def _trace(steps: list[dict[str, Any]]) -> Trace:
    return tuple(
        Transition(_atoms(s["state"]), tuple(s["action"]), _atoms(s["next_state"])) for s in steps
    )


def _step(t: Transition) -> dict[str, Any]:
    """Serialise a step, atoms sorted so equal traces give equal text."""
    return {
        "state": [list(a) for a in sorted(t.state)],
        "action": list(t.action),
        "next_state": [list(a) for a in sorted(t.next_state)],
    }


def _check_continuity(trace: Trace) -> list[str]:
    """Return every step that does not start in the state the step before it ended in."""
    return [
        f"steps[{i}].state differs from steps[{i - 1}].next_state"
        for i in range(1, len(trace))
        if trace[i].state != trace[i - 1].next_state
    ]


def _check_strips(signature: Signature, numbers: list[int], traces: list[Trace]) -> list[str]:
    """Return every step the model SAM learns from the whole file does not replay."""
    model = learn(signature, traces)
    return [
        f"line {number}: steps[{i}]: {reason}"
        for number, trace in zip(numbers, traces, strict=True)
        for i, step in enumerate(trace)
        for reason in unexplained(model, step)
    ]
