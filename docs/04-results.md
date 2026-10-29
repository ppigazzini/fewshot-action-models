# Results

Every number below is printed by a committed command and recorded as counts in
[`studies/`](studies/); `tools/curve.py --check` reruns it.

## Blocksworld

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld.json
```

Training problems of 3–4 blocks, test problems of 4–5 blocks (the largest training size and
one beyond); 10 seeds, 50 test problems per seed, BFS bound 20,000 nodes, which no search
reaches: `limit` is 0 in every row. Mean over seeds, per-seed range in parentheses.

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

## An action demonstrations rarely show

```bash
uv run python tools/curve.py logistics-local --check docs/studies/logistics-local.json
```

Logistics with training problems whose package goals are drawn in the other city with
probability 0.1, and test problems from the logistics study's generator, where the
probability is 0.5, a problem whose goal already holds being drawn again in both
(`logistics-local` in `amlearn/domains/__init__.py`); 10 seeds of 50 test problems, drawn
by the benchmark's own seeds. Valid plans over 500 test problems, and actions observed on
average:

| demos | logistics: valid | logistics: actions | local: valid | local: actions |
|---:|---:|---:|---:|---:|
| 1 | 319 | 5.4 | 88 | 3.9 |
| 2 | 500 | 6.0 | 159 | 4.8 |
| 3 | 500 | 6.0 | 420 | 5.7 |
| 5 | 500 | 6.0 | 428 | 5.7 |
| 10 | 500 | 6.0 | 468 | 6.0 |
| 20 | 500 | 6.0 | 500 | 6.0 |

A demonstration that keeps its packages in their cities does not show the airplane: one
local demonstration observes all six actions in 3 of 10 seeds, three in 9. A model
without the airplane's actions solves only the test problems that leave every package in
its city, and the local curve needs 20 demonstrations to solve every test problem where
the logistics one needs 2. Every plan is valid.

## Demonstrations against random walks

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-walks.json
uv run python tools/curve.py logistics --check docs/studies/logistics-walks.json
```

The same seeds, training problems, test problems and bound as the two studies above, each
demonstration replaced by a random walk of 10 steps under the reference from the same
training problem. Valid plans over the 500 test problems, with the transitions observed:

| shown | blocksworld: shortest plans | blocksworld: walks | logistics: shortest plans | logistics: walks |
|---:|---:|---:|---:|---:|
| 1 | 211 (5.0) | 204 (10.0) | 319 (9.1) | 7 (10.0) |
| 2 | 340 (10.0) | 405 (20.0) | 500 (16.8) | 31 (20.0) |
| 3 | 384 (14.8) | 500 (30.0) | 500 (25.1) | 57 (30.0) |
| 5 | 451 (23.6) | 500 (50.0) | 500 (38.7) | 288 (50.0) |
| 10 | 500 (46.6) | 500 (100.0) | 500 (74.4) | 453 (100.0) |
| 20 | 500 (92.6) | 500 (200.0) | 500 (147.6) | 500 (200.0) |

On blocksworld a walk observes all four actions from the first one and three walks, 30
transitions, solve every test problem, where shortest plans need 10 demonstrations and
46.6 transitions. On logistics the order reverses: two shortest plans, 16.8 transitions,
solve every test problem, and walks need 20, 200 transitions. Inferred from the action
counts of the two studies: a uniform walk among the 5.1 actions the reference applies on
an average test state rarely carries a package to another city, so it observes the
airplane's actions late and keeps the coincidental truck preconditions longer. Every plan is valid in both studies, and no walk
model applies an action the reference refuses on a test state.

## Gripper

```bash
uv run python tools/curve.py gripper --check docs/studies/gripper.json
```

Training problems of 2–3 balls, test problems of 3–4 balls; 3 seeds, 30 test problems per
seed. One demonstration reaches the reference score in all 3 seeds. A goal differs from the
initial state, so every plan moves a ball across rooms and uses all three actions: one
demonstration observes each of them. Gripper therefore checks the guarantee and the
planners; it does not produce a curve.

## What an observer hides or misses

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-masked-pisam.json
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-masked-sam.json
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-noise.json
uv run python tools/curve.py logistics --check docs/studies/logistics-masked-pisam.json
```

The blocksworld study with an observer between the demonstrations and the learner
(`observe` in `amlearn/traces.py`): one that hides each atom of each state with probability
0.3, the higher rate of the experiments of Le, Juba and Stern (2024), and one that misses
each true atom with probability 0.05. Both masked studies see the same masks. Valid plans
over 500 test problems, and plans found:

| demos | fully observed | 30 % hidden, PI-SAM | 30 % hidden, SAM | 5 % missed, SAM |
|---:|---:|---:|---:|---:|
| 1 | 211 | 0 | 35 of 96 | 80 of 95 |
| 2 | 340 | 28 | 63 of 255 | 178 of 239 |
| 3 | 384 | 82 | 54 of 364 | 144 of 271 |
| 5 | 451 | 208 | 57 of 423 | 127 of 360 |
| 10 | 500 | 426 | 40 of 437 | 58 of 380 |
| 20 | 500 | 475 | 40 of 444 | 46 of 413 |
| 50 | 500 | 500 | 40 of 444 | 37 of 442 |

PI-SAM keeps the guarantee: every plan is valid, and on no test state does it apply an
action the reference refuses. It needs more demonstrations, because a literal is learned
from only where it is seen both before and after a step, which at 30 % masking is 0.7 × 0.7
= 49 % of the time; on logistics it solves all 500 test problems from 20 demonstrations,
and none from two. SAM reading a hidden atom as false has no guarantee left: from 10
demonstrations it finds plans for 437 test problems and 40 of them are valid. A missed atom
does the same, and more demonstrations make it worse, since each prunes another true
precondition: valid plans fall from 178 at two demonstrations to 37 at fifty.

The STRIPS check refuses those trace files: `unexplained` is above 0 in every row of the
two unsafe studies, and 0 in every PI-SAM row, as the check's theorem requires of steps
masked without noise.

## Larger test problems

```bash
uv run python tools/curve.py blocksworld --check docs/studies/blocksworld-large.json
uv run python tools/curve.py gripper --check docs/studies/gripper-large.json
uv run python tools/curve.py logistics --check docs/studies/logistics-large.json
```

The blocksworld, gripper and logistics studies with the same training problems and seeds,
and test problems of 6–7 blocks, 6–8 balls and 4–5 packages, sizes no demonstration shows;
BFS bound 200,000 nodes, under which the reference solves every test problem. Valid plans,
against the studies above:

| demos | blocksworld 4–5 | blocksworld 6–7 | logistics 2–3 | logistics 4–5 |
|---:|---:|---:|---:|---:|
| 1 | 211 of 500 | 180 of 500 | 319 of 500 | 309 of 500 |
| 2 | 340 | 302 | 500 | 500 |
| 3 | 384 | 345 | 500 | 500 |
| 5 | 451 | 440 | 500 | 500 |
| 10 | 500 | 500 | 500 | 500 |

A model learned from 3–4 blocks solves every 7-block test problem from 10 demonstrations,
and one from 1–2 packages every 5-package problem from 2. Inferred from the precondition
counts, which are the same at every test size: once a precondition holds no literal a
small demonstration made true by coincidence, the size of the problem does not matter to
it. Gripper solves all 90 test problems of 6–8 balls from one demonstration.
Every plan is valid, and no model applies an action the reference refuses on a test state.
These three studies take about 13 minutes and are checked weekly
(`.github/workflows/large.yml`).

## Not measured

- The optimistic baseline on Logistics. Without preconditions nothing narrows the 11,880
  bindings of `drive-truck` over a dozen objects, and every search state tries them all.
- A trace file with hidden atoms: the format records full observations, so PI-SAM learns
  from the observer of a study and not from a file.
- Negative demonstrations, failed actions. They alone narrow the upper boundary of a
  precondition's version space (Aineto and Scala 2024), so with demonstrations only, no
  learner between SAM and the optimistic model has evidence to choose a precondition.
