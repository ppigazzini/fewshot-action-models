# fewshot-action-models

Learns the preconditions and effects of planning actions (a lifted STRIPS model) from a few
demonstrations, plans with the learned model, and scores every plan against an independent
reference model.

The learner is safe action-model learning (SAM): a plan valid under the learned model is
valid under the true one. The guarantee holds under stated assumptions — deterministic
STRIPS, full observability, injective grounding — and the suite tests it on every
benchmark instead of assuming it. What safety costs is completeness: a learned model solves
fewer problems than the reference. The learning curve measures how many fewer, against the
number of demonstrations.

## Layout

```
amlearn/strips.py        lifted STRIPS with negative preconditions; injective grounding
amlearn/pddl.py          the PDDL subset, read and written
amlearn/domains/         reference domains (PDDL) and their problem generators
amlearn/traces.py        demonstrations: shortest plans executed under the reference
amlearn/tracefile.py     trace files: recorded demonstrations, checked against the signature
amlearn/learn/sam.py     safe action-model learning
amlearn/learn/optimistic.py  the unsafe contrast: observed effects, no preconditions
amlearn/plan/bfs.py      breadth-first search: shortest plans, proofs of unsolvability
amlearn/plan/asp.py      answer-set planning with clingo: the cross-check
amlearn/validate.py      plan validation against the reference model
amlearn/curve.py         the learning curve; every number is a count
amlearn/study.py         the study file format, checked with valgebra
amlearn/_valgebra.py     the one module that imports valgebra, and its recipes
tools/                   the committed commands that produce published numbers
```

## Running

```bash
uv sync                                              # one lockfile; dev tools included
uv run python tools/learn.py blocksworld --demos 5   # print a learned model as PDDL
uv run python tools/learn.py blocksworld --traces demos.jsonl  # learn from recorded traces
uv run python tools/curve.py blocksworld             # the learning curve, 3 seeds
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-optimistic.json
uv run pytest
uv run ruff check && uv run ruff format --check && uv run ty check
uv run python tools/check_markdown.py
```

`--check` reruns a published study and fails on any count that differs.

## Results

Blocksworld, training on 3–4 blocks, testing on 4–5; both learners on the same 5 seeds of
50 test problems, BFS bound 20,000 nodes (`docs/04-results.md`). Each cell counts plans
over the 250 test problems, with the per-seed range of the share in brackets:

| demonstrations | SAM: valid plans | SAM: invalid plans | optimistic: plans found | optimistic: valid plans |
|---:|---:|---:|---:|---:|
| 1 | 70 (0–52 %) | 0 | 142 (22–86 %) | 0 |
| 3 | 201 (50–100 %) | 0 | 199 (72–86 %) | 0 |
| 10 | 250 (100 %) | 0 | 199 (72–86 %) | 0 |

The optimistic model, which keeps the observed effects and drops every precondition, finds
a plan for 199 of the 250 test problems from 3 demonstrations, and not one of them
executes under the reference. On Logistics (six actions, 1–2 packages in training, 2–3 in
test) SAM's plans are valid on 113 of 250 test problems at one demonstration (0 to 100 %
by seed) and on all 250 from two, every plan valid.

See [`docs/`](docs/README.md) for the method, the results and the development rules.

## Licence

Apache-2.0. See [`LICENSE`](LICENSE).
