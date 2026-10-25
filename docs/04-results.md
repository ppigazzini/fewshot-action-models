# Results

Every number below is printed by a committed command and recorded as counts in
[`studies/`](studies/); `tools/curve.py --check` reruns it.

## Blocksworld

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
```

Training problems of 3–4 blocks, test problems of 4–5 blocks (the largest training size and
one beyond); 5 seeds, 50 test problems per seed. Mean over seeds, per-seed range in brackets.

| demos | transitions | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.4 | 3.0 | 28.0 (0-52) | 100.0 (100-100) | 28.0 (0-52) |
| 2 | 10.4 | 3.8 | 74.4 (22-100) | 100.0 (100-100) | 74.4 (22-100) |
| 3 | 14.4 | 4.0 | 80.4 (50-100) | 100.0 (100-100) | 80.4 (50-100) |
| 5 | 22.4 | 4.0 | 90.0 (50-100) | 100.0 (100-100) | 90.0 (50-100) |
| 10 | 43.6 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 20 | 87.6 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 50 | 221.6 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |

- Every plan a learned model produced is valid under the reference: 100 % of solved at every
  point, as the safety guarantee requires.
- From 3 demonstrations all four actions are observed in every seed, yet the score is 80.4 %.
  With every action present, an unsolved test problem needs a grounding the learned
  precondition refuses and the reference admits: a literal that held in every observed
  pre-state stays in the model.
- 10 demonstrations (43.6 transitions on average) reach the reference score in all 5 seeds.

## What safety buys

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-optimistic.json
```

The optimistic baseline on the same benchmark: 3 seeds, 30 test problems per seed, BFS
bound 20,000 nodes (a model without preconditions makes every action applicable
everywhere, and an unsolvable problem then fills the bound).

| demos | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|
| 1 | 3.0 | 62.2 (27-83) | 0.0 (0-0) | 0.0 (0-0) |
| 2 | 3.7 | 61.1 (27-80) | 0.0 (0-0) | 0.0 (0-0) |
| 3 | 4.0 | 81.1 (77-87) | 0.0 (0-0) | 0.0 (0-0) |
| 5 | 4.0 | 81.1 (77-87) | 0.0 (0-0) | 0.0 (0-0) |
| 10 | 4.0 | 81.1 (77-87) | 0.0 (0-0) | 0.0 (0-0) |
| 20 | 4.0 | 81.1 (77-87) | 0.0 (0-0) | 0.0 (0-0) |

Every plan it finds breaks under the reference: without preconditions it stacks a block
it is not holding. The planner finds a plan for more problems than SAM does at one or
two demonstrations, and the plans are worthless; SAM's 28 % at one demonstration are all
valid.

## Gripper

```bash
uv run python tools/curve.py gripper --check docs/studies/gripper.json
```

Training problems of 2–3 balls, test problems of 3–4 balls; 3 seeds, 30 test problems per
seed. One demonstration reaches the reference score in all 3 seeds. A goal differs from the
initial state, so every plan moves a ball across rooms and uses all three actions: one
demonstration observes each of them. Gripper therefore checks the guarantee and the
planners; it does not produce a curve.

## Not measured

- Noisy or partially observed traces.
- Domains where a shortest plan rarely uses some action, which is where goal-directed
  demonstrations and random walks should part.
- A learner between the two extremes. Inductive logic programming (Popper) on the same
  traces is the planned comparison.
