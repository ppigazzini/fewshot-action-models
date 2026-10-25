# Using the package

## Install

```bash
uv sync
```

Python 3.14 or later, as `requires-python` in `pyproject.toml` says. The runtime dependencies are clingo (the second
planner), valgebra with annotated-types and typing-extensions (the study file check).

## Learn a model

```bash
uv run python tools/learn.py blocksworld --demos 5 --seed 0
```

Draws 5 training problems, demonstrates each with a shortest plan under the reference
domain, learns from the traces and prints the learned domain as PDDL. The same seed draws
the same problems as `tools/curve.py`, so the printed model is the one a curve row scores.

## Learn from recorded traces

Demonstrations recorded elsewhere — a simulator, a robot log, another planner — are read
from a trace file: JSON Lines, one trace per line, each step a state, an action and the
state after it.

```json
{"steps": [{"state": [["clear", "a"], ["handempty"], ["ontable", "a"]],
            "action": ["pick-up", "a"],
            "next_state": [["holding", "a"]]}]}
```

```bash
uv run python tools/learn.py blocksworld --demos 5 --write-traces demos.jsonl
uv run python tools/learn.py blocksworld --traces demos.jsonl
```

The benchmark names the signature the file is read against; `amlearn.tracefile.TraceSchema`
compiles a valgebra schema from it, so an atom must be one of the predicates with exactly
its arity, an action one of the actions with exactly its parameter count, an object a
non-empty string, and a step carries no key the format does not name. Every failure on
every line is reported before anything is learned, with its line and path:

```
line 1: at steps[0].action: stack takes 2 objects, got 1
line 1: at steps[0].state[0][1]: expected length >= 1, got '' [too_short]
line 2: expected valid JSON, got expected value at line 1 column 12 [json_invalid]
```

Each step must also start in the state the step before it ended in. A step with the wrong
arity would otherwise prune a true precondition and void the safety guarantee.

And one STRIPS model without constants must explain every step: the model SAM learns from
the whole file replays each one. A step it does not replay is reported with the atom it
gets wrong:

```
line 2: steps[3]: clear(zz) changes, but stack(b2,b1) does not name zz
line 4: steps[4]: holding(b1) is false after pick-up(b1), but another pick-up step adds holding(?x)
```

The first is a change no action schema without constants makes; the second, an atom
dropped from one recorded state, which the other pick-up steps contradict. The safety guarantee is stated for traces such a model
produced, so a file no such model explains is refused.

## Run a learning curve

```bash
uv run python tools/curve.py blocksworld --demos 1,2,3,5,10,20,50 --seeds 5 --test 50
```

Per seed: one pool of training problems, one set of 50 test problems, and one learned model
per demonstration count, each from the first `n` traces of the pool. Columns:

| column | meaning |
|---|---|
| transitions | observed steps in the first `n` demonstrations |
| actions | action schemas observed at least once, so present in the model |
| solved % | test problems for which BFS finds a plan under the learned model |
| valid % of solved | of those plans, the share valid under the reference model |
| score vs reference % | valid plans over the test problems the reference model solves |

Each cell is the mean over seeds with the per-seed minimum and maximum in brackets.

## Publish and check a study

```bash
uv run python tools/curve.py blocksworld --demos 1,2,3,5,10,20,50 --seeds 5 --test 50 \
    --json docs/studies/blocksworld.json
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
```

A study file holds counts only. `--check` reads the settings it records, reruns the curve and
exits 1 on any count that differs; CI runs it on every published study.

## Use another domain

A domain in the supported PDDL subset (untyped STRIPS, negative preconditions, no constants
in schemas) is read with `amlearn.pddl.load_domain`. A benchmark adds a problem generator and
its train and test sizes to `amlearn.domains.BENCHMARKS`. A typed domain is rewritten with
type predicates, as gripper's `room`, `ball` and `gripper` are.
