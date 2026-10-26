# Results

Every number below is printed by a committed command and recorded as counts in
[`studies/`](studies/); `tools/curve.py --check` reruns it.

## Blocksworld

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
```

Training problems of 3–4 blocks, test problems of 4–5 blocks (the largest training size and
one beyond); 10 seeds, 50 test problems per seed, BFS bound 20,000 nodes, which no search
reaches: `limit` is 0 in every row. Mean over seeds, per-seed range in brackets.

| demos | transitions | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 | 3.3 | 42.2 (0-100) | 100.0 (100-100) | 42.2 (0-100) |
| 2 | 10.0 | 3.9 | 68.0 (22-100) | 100.0 (100-100) | 68.0 (22-100) |
| 3 | 14.8 | 4.0 | 76.8 (28-100) | 100.0 (100-100) | 76.8 (28-100) |
| 5 | 23.6 | 4.0 | 90.2 (50-100) | 100.0 (100-100) | 90.2 (50-100) |
| 10 | 46.6 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 20 | 92.6 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 50 | 229.4 | 4.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |

- Every plan a learned model produced is valid under the reference: 100 % of solved at every
  point, as the safety guarantee requires.
- From 3 demonstrations all four actions are observed in every seed, yet the score is
  76.8 %. With every action present, an unsolved test problem needs a grounding the
  learned precondition refuses and the reference admits: a literal that held in every
  observed pre-state stays in the model.
- 10 demonstrations (46.6 transitions on average) reach the reference score in all 10
  seeds.

Model quality, precision / recall % from counts pooled over the 10 seeds:

| demos | preconditions | effects | applicability | changes |
|---:|---:|---:|---:|---:|
| 1 | 28.6 / 82.2 | 100.0 / 82.8 | 100.0 / 70.1 | 100.0 / 100.0 |
| 2 | 29.5 / 98.9 | 100.0 / 97.8 | 100.0 / 88.5 | 100.0 / 100.0 |
| 3 | 29.4 / 100.0 | 100.0 / 100.0 | 100.0 / 91.8 | 100.0 / 100.0 |
| 5 | 29.8 / 100.0 | 100.0 / 100.0 | 100.0 / 97.3 | 100.0 / 100.0 |
| 10 | 30.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| 20 | 30.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| 50 | 30.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |

Seven in ten precondition literals SAM keeps are not in the reference's, and from 10
demonstrations none of them refuses an action the reference applies on any of the 5,500
test states: applicability recall is 100.0 %. Precision of applicability and of changes is
100.0 % at every point, as the guarantee requires. Below 10 demonstrations the score and
applicability recall rise together: 42.2 % and 70.1 % at one, 90.2 % and 97.3 % at five.

## Logistics

```bash
uv run python tools/curve.py logistics --check docs/studies/logistics.json
```

Six actions over two cities, each with an airport and a post office, one truck per city
and one airplane; training problems of 1–2 packages, test problems of 2–3; 10 seeds, 50
test problems per seed.

| demos | transitions | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.1 | 5.4 | 63.8 (0-100) | 100.0 (100-100) | 63.8 (0-100) |
| 2 | 16.8 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 3 | 25.1 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 5 | 38.7 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 10 | 74.4 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |
| 20 | 147.6 | 6.0 | 100.0 (100-100) | 100.0 (100-100) | 100.0 (100-100) |

One demonstration observes 5.4 of the 6 actions on average, and each seed's score follows
what its demonstration leaves out (rows of `docs/studies/logistics.json` at 1
demonstration):

| seed | demonstration | actions | valid of 50 |
|---:|---|---:|---:|
| 0 | the airplane alone | 3 | 3 |
| 1 | one truck within a city | 3 | 0 |
| 2 | trucks and the airplane | 6 | 50 |
| 3 | trucks and the airplane | 6 | 50 |
| 4 | trucks that only drive to and unload at airports | 6 | 10 |
| 5 | trucks and the airplane | 6 | 50 |
| 6 | trucks and the airplane | 6 | 50 |
| 7 | trucks that only load at and drive from airports | 6 | 6 |
| 8 | trucks and the airplane | 6 | 50 |
| 9 | trucks and the airplane | 6 | 50 |

Seeds 4 and 7 observe every action and still solve 10 and 6. Seed 4's drive-truck keeps
`airport(?to)` and its unload-truck `airport(?l)`, so its trucks never reach a post office;
seed 7's drive-truck keeps `airport(?from)` and its load-truck `airport(?l)`, so they never
load at one. Each literal is a coincidence of one demonstration. Two demonstrations
observe all six actions in every seed and solve every test problem.

From two demonstrations precondition precision is 10.6 % and applicability recall
100.0 %: nine in ten kept literals are not in the reference's, and none refuses an action
the reference applies on a test state.

## What safety buys

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-optimistic.json
```

The optimistic baseline on the seeds, test problems and bound of the blocksworld study
above, so each row pairs with the row of the same demonstration count there.

| demos | actions | solved % | valid % of solved | score vs reference % |
|---:|---:|---:|---:|---:|
| 1 | 3.3 | 62.4 (8-88) | 0.0 (0-0) | 0.0 (0-0) |
| 2 | 3.9 | 75.0 (28-88) | 0.0 (0-0) | 0.0 (0-0) |
| 3 | 4.0 | 80.6 (72-88) | 0.0 (0-0) | 0.0 (0-0) |
| 5 | 4.0 | 80.6 (72-88) | 0.0 (0-0) | 0.0 (0-0) |
| 10 | 4.0 | 80.6 (72-88) | 0.0 (0-0) | 0.0 (0-0) |
| 20 | 4.0 | 80.6 (72-88) | 0.0 (0-0) | 0.0 (0-0) |
| 50 | 4.0 | 80.6 (72-88) | 0.0 (0-0) | 0.0 (0-0) |

From 3 demonstrations its applicability precision is 6.6 % with recall 100.0 %: of the
ground actions it applies on a test state, 1 in 15 is applicable under the reference. Its
changes are those of the reference wherever both apply, precision and recall 100.0 %:
the effects are SAM's.

Every plan it finds breaks under the reference. Without preconditions every goal atom is
one step from any state: put-down adds `ontable` and stack adds `on`, for a block the hand
does not hold as well as for one it does. It finds a plan for more problems than SAM up to
three demonstrations, 62.4 % against 42.2 % at one, 75.0 % against 68.0 % at two and
80.6 % against 76.8 % at three, and not one of its plans is valid, where every SAM plan
is. From five demonstrations SAM solves more: 90.2 % against 80.6 %.

Every problem it leaves unsolved is a bound hit, not a proof, in every row but one: seed
7 learns two actions from its one demonstration, and its model proves 39 of its 50 test
problems unsolvable. With every action applicable in every state, the search fills 20,000
nodes on 97 of the 500 test problems from 3 demonstrations on, problems SAM solves from
10.

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
