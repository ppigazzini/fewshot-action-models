# Results

Every number below is printed by a committed command and recorded as counts in
[`studies/`](studies/); `tools/curve.py --check` reruns it.

## Blocksworld

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
```

Training problems of 3–4 blocks, test problems of 4–5 blocks (the largest training size and
one beyond); 5 seeds, 50 test problems per seed, BFS bound 20,000 nodes, which no search
reaches: `limit` is 0 in every row. Mean over seeds, per-seed range in brackets.

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

## Logistics

```bash
uv run python tools/curve.py logistics --check docs/studies/logistics.json
```

Six actions over two cities, each with an airport and a post office, one truck per city
and one airplane; training problems of 1–2 packages, test problems of 2–3; 5 seeds, 50
test problems per seed.

| demos | transitions | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.0 | 4.8 | 45.2 (0-100) | 100.0 (100-100) | 45.2 (0-100) |
| 2 | 16.4 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 3 | 25.2 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 5 | 38.0 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 10 | 72.6 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 20 | 146.0 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |

One demonstration observes 4.8 of the 6 actions on average, and each seed's score follows
what its demonstration leaves out (rows of `docs/studies/logistics.json` at 1
demonstration):

| seed | demonstration | actions | valid of 50 |
|---:|---|---:|---:|
| 0 | the airplane alone | 3 | 3 |
| 1 | one truck within a city | 3 | 0 |
| 2 | trucks and the airplane | 6 | 50 |
| 3 | trucks and the airplane | 6 | 50 |
| 4 | trucks that only drive to and unload at airports | 6 | 10 |

Seed 4 observes every action and still solves 10: drive-truck keeps `airport(?to)` and
unload-truck keeps `airport(?l)`, coincidences of its one demonstration, so its trucks
never reach a post office. Two demonstrations observe all six actions in every seed and
solve every test problem.

## What safety buys

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-optimistic.json
```

The optimistic baseline on the seeds, test problems and bound of the blocksworld study
above, so each row pairs with the row of the same demonstration count there.

| demos | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|
| 1 | 3.0 | 56.8 (22-86) | 0.0 (0-0) | 0.0 (0-0) |
| 2 | 3.8 | 68.4 (28-86) | 0.0 (0-0) | 0.0 (0-0) |
| 3 | 4.0 | 79.6 (72-86) | 0.0 (0-0) | 0.0 (0-0) |
| 5 | 4.0 | 79.6 (72-86) | 0.0 (0-0) | 0.0 (0-0) |
| 10 | 4.0 | 79.6 (72-86) | 0.0 (0-0) | 0.0 (0-0) |
| 20 | 4.0 | 79.6 (72-86) | 0.0 (0-0) | 0.0 (0-0) |
| 50 | 4.0 | 79.6 (72-86) | 0.0 (0-0) | 0.0 (0-0) |

Every plan it finds breaks under the reference. Without preconditions every goal atom is
one step from any state: put-down adds `ontable` and stack adds `on`, for a block the hand
does not hold as well as for one it does. It finds a plan for more problems than SAM only
at one demonstration, 56.8 % against 28.0 %, and those plans are worthless; SAM's 28.0 %
are all valid. From two demonstrations on SAM solves more: 74.4 % against 68.4 % at two,
80.4 % against 79.6 % at three.

Every problem it leaves unsolved is a bound hit, not a proof: `solved` plus `limit` is 50
in every row. With every action applicable in every state, the search fills 20,000 nodes
on 51 of the 250 test problems from 3 demonstrations on, problems SAM solves from 10.

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

- The optimistic baseline on Logistics. Without preconditions nothing narrows the 11,880
  bindings of `drive-truck` over a dozen objects, and every search state tries them all.
- Noisy or partially observed traces.
- Domains where a shortest plan rarely uses some action, which is where goal-directed
  demonstrations and random walks should part.
- A learner between the two extremes. Inductive logic programming (Popper) on the same
  traces is the planned comparison.
