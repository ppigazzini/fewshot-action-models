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
