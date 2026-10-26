"""Check this SAM and its syntactic counts against the authors' SAM and AMLGym's metrics.

    uv run --python 3.14 --with amlgym==1.0.13 python tools/crosscheck.py

For each benchmark and 1, 3 and 10 demonstrations of seed 0's pool, the authors' SAM
(`sam_learning`, run through AMLGym) learns from the same traces, and AMLGym scores this
package's model against the reference. AMLGym reads typed domains only, so both sides are
written with one type for every variable. Two differences of form are set aside before the
models are compared: the authors' SAM writes injective grounding as inequalities, which
this package's writer states too; and it does not enumerate a literal whose arguments repeat
a parameter, such as ``on(?x,?x)``, which this SAM keeps as a candidate.

AMLGym and torch are not dependencies of the package; ``--with`` installs them for this
command alone, on a CPython torch has wheels for. Exit 1 on any disagreement.
"""

import os
import random
import re
import tempfile
import warnings
from pathlib import Path

from amlgym.algorithms import get_algorithm  # ty: ignore[unresolved-import]
from amlgym.metrics import syntactic_precision, syntactic_recall  # ty: ignore[unresolved-import]
from amlgym.util.SimpleDomainReader import SimpleDomainReader  # ty: ignore[unresolved-import]

from amlearn.domains import BENCHMARKS, draw
from amlearn.learn import Signature, sam
from amlearn.metrics import syntactic
from amlearn.pddl import write_domain
from amlearn.strips import Atom, Domain, Trace
from amlearn.traces import demonstrate

INEQUALITY = re.compile(r" \(not \(= \?[^\s()]+ \?[^\s()]+\)\)")


def typed(text: str) -> str:
    """Return a written domain with every variable of one type, and no inequalities."""
    text = INEQUALITY.sub("", text).replace(" :equality", "")
    head, *actions = text.replace("  (:predicates", "  (:types obj)\n  (:predicates").split(
        "(:action"
    )
    head = re.sub(r"\((\S+)((?: \?\w+)+)\)", r"(\1\2 - obj)", head)
    text = head + "".join("(:action" + part for part in actions)
    return re.sub(r":parameters \(([^)]+)\)", r":parameters (\1 - obj)", text)


def trajectory(trace: Trace) -> str:
    """Return ``trace`` in AMLGym's trajectory format."""

    def state(atoms: frozenset[Atom]) -> str:
        return "(:state " + " ".join(f"({' '.join(a)})" for a in sorted(atoms)) + ")"

    lines = ["(:trajectory", state(trace[0].state)]
    for step in trace:
        lines += [f"(:action ({' '.join(step.action)}))", state(step.next_state)]
    return "\n".join([*lines, ")"]) + "\n"


def literals(path: Path) -> dict[str, list[frozenset[str]]]:
    """Return each action's signed literals as AMLGym reads them, in the form both write."""

    def keep(literal: str) -> bool:
        args = literal[literal.index("(") + 1 : -1].split(",") if "(" in literal else []
        return not literal.startswith("=(") and len(set(args)) == len(args)

    model = SimpleDomainReader(input_file=str(path))
    return {
        op.operator_name.replace("_", "-"): [
            frozenset(filter(keep, part))
            for part in (op.precs_pos, op.precs_neg, op.eff_pos, op.eff_neg)
        ]
        for op in model.operators
    }


def means(model: Domain, reference: Domain) -> tuple[float, float]:
    """Return this package's mean per-action syntactic precision and recall."""
    precision, recall = [], []
    for name in sorted(reference.actions):
        one = {name: model.actions[name]} if name in model.actions else {}
        pre, eff = syntactic(
            Domain(model.name, model.predicates, one),
            Domain(reference.name, reference.predicates, {name: reference.actions[name]}),
        )
        tp, fp, fn = pre.tp + eff.tp, pre.fp + eff.fp, pre.fn + eff.fn
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / (tp + fn) if tp + fn else 1.0)
    return sum(precision) / len(precision), sum(recall) / len(recall)


def main() -> int:
    """Learn, compare and print one line per model; exit 1 on any disagreement."""
    warnings.filterwarnings("ignore")
    work = Path(tempfile.mkdtemp())
    os.chdir(work)
    failures = total = 0
    for name, bench in sorted(BENCHMARKS.items()):
        domain = work / f"{name}.pddl"
        domain.write_text(typed(write_domain(bench.domain)))
        pool = draw(bench, random.Random(f"{name}:train:0"), 10, bench.train_sizes)
        traces = [demonstrate(bench.domain, p) for p in pool]
        for n in (1, 3, 10):
            model = sam.learn(Signature.of(bench.domain), traces[:n])
            ours = work / f"{name}-{n}.pddl"
            ours.write_text(typed(write_domain(model)))
            paths = []
            for i, trace in enumerate(traces[:n]):
                path = work / f"{name}-{n}-{i}.traj"
                path.write_text(trajectory(trace))
                paths.append(str(path))
            theirs = work / f"{name}-{n}-theirs.pddl"
            theirs.write_text(get_algorithm("SAM").learn(str(domain), paths))
            a, b = literals(ours), literals(theirs)
            same = all(a[k] == b.get(k) for k in a if any(a[k]))
            precision, recall = means(model, bench.domain)
            amlgym = syntactic_precision(str(ours), str(domain))["mean"]
            amlgym_recall = syntactic_recall(str(ours), str(domain))["mean"]
            agree = (round(precision, 2), round(recall, 2)) == (amlgym, amlgym_recall)
            total += 1
            failures += not (same and agree)
            shown = f"{n} demonstration{'s' * (n > 1)}"
            verdict = "equal" if same else "DIFFERENT"
            print(
                f"{name}, {shown}: model {verdict} to the authors' SAM; precision / recall"
                f" {precision:.2f} / {recall:.2f}, AMLGym {amlgym:.2f} / {amlgym_recall:.2f}"
            )
    print(f"\n{total - failures} of {total} models agree with both")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
