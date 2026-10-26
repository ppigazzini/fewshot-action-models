# Method

## Representation

Lifted STRIPS with negative preconditions (`amlearn/strips.py`). A precondition is a set
of atoms that must hold and a set that must not; an effect is a set of added and a set of
deleted atoms, applied delete first. Action schemas take no constants, so every argument
of a schema atom is a parameter.

Grounding is injective: distinct parameters take distinct objects. The planners, the
validator and the learner share that semantics, so a learned model is never run where its
guarantee does not reach.

## The learner

Safe action-model learning (SAM; Stern and Juba 2017, lifted by Juba, Le and Stern 2021),
in `amlearn/learn/sam.py`. The learner knows the predicates and each action's parameter
count; it learns the preconditions and effects. For each action schema:

- the precondition starts as every literal over the schema's parameters, positive and
  negative, and loses each literal that some observed pre-state falsifies;
- the effects are the observed changes: false before and true after is an add effect,
  true before and false after a delete effect.

**Safety.** A plan valid under the learned model is valid under the true model and reaches
the same states. A true effect never seen as a change was already in the state it would
produce every time the action was observed, so its literal stays in the learned
precondition and the effect is a no-op wherever the learned model lets the action run. The
argument needs deterministic STRIPS, full observability, negative preconditions and
injective grounding. `tests/test_sam.py` checks the property on every benchmark, from
demonstrations and from random walks, by validating each learned model's plans under the
reference and comparing the final states.

**SAM is the tightest sound model, and the optimistic model a complete one.** The models
consistent with the traces form a version space. Its lower boundaries, the literals true in
every observed pre-state and the union of the observed changes, are SAM's precondition and
effects, and give the sound model no other sound model contains (Aineto and Scala 2024,
Theorem 5). With demonstrations alone, none of them a failed action, the upper boundary of
a precondition stays the empty conjunction (rule RUP of their Theorem 2), so the tightest
complete model applies every action everywhere, its effects any consistent set (Theorem
6): the optimistic model is its member with the observed changes. Every subset of SAM's
precondition is consistent with the demonstrations, and nothing in them favours one over
another; only negative demonstrations narrow the upper boundary.

**Completeness is what safety costs.** An action never observed is absent from the model.
A literal true by coincidence in every observation stays a precondition. Both make a
learned model solve fewer problems than the reference, and the learning curve measures how
many fewer.

## Demonstrations

A demonstration is a shortest plan under the reference domain, executed with every state
recorded (`amlearn/traces.py`). Demonstrations are goal-directed, so an action no shortest
plan takes is never observed. Traces recorded elsewhere are read from a trace file
(`amlearn/tracefile.py`), checked against the signature before they reach the learner.
A file is refused when no STRIPS model without constants explains every step. The test is
exact: such a model contains every observed change among its effects, so the model SAM
learns replays each step whenever one exists, and it is one itself.
Random walks are the goal-independent contrast: the safety tests learn from them, and
`--demonstrator walk` measures a curve on them.

## Planning

Breadth-first search (`amlearn/plan/bfs.py`) is the measuring planner: it returns a
shortest plan, proves unsolvability by exhausting the reachable states, and reports a node
bound as `limit`, never as a proof. The answer-set planner (`amlearn/plan/asp.py`, clingo,
incremental horizon) exists to disagree with it: both return shortest plans from the same
grounding, so their lengths must be equal on every problem both solve.

Both plan over `Domain.relevant`: the injective groundings whose static preconditions hold
in the initial state. A static atom never changes, so the pruning changes no result.

## Scoring

A learned model is scored on test problems drawn from the largest training size and one
size beyond it. For each test problem BFS plans under the learned model; the plan is then
executed under the reference domain by `amlearn/validate.py`. A valid plan counts only on
a test problem the reference model itself solves within the node bound, and the score is
the number of such plans over the number of those problems: 100 % is the reference
model's own score, and no learned model can exceed it.

The reference solves every test problem in every published study, so the score there is
the solving ratio of Stern et al. (2025), and solved minus valid over the test problems is
their false-plan ratio: 0 in every SAM row, as the guarantee requires.

Each row also records the model-quality counts of Stern et al. (2025), as true positives,
false positives and false negatives summed over actions (`amlearn/metrics.py`):

| family | compared |
|---|---|
| preconditions | each reference action's precondition literals, an atom with its sign, against the learned action's; an action the model lacks counts every reference literal as a false negative |
| effects | the same for the add and delete effects |
| applicability | on each test state, the ground actions each model applies |
| changes | for a ground action both models apply on a test state, the atoms each changes |

The test states are those of a reference random walk of `probe_steps` steps from each test
problem's initial state, from a random stream of their own: 550 per seed for 50 test
problems and 10 steps. A safe model applies nothing the reference refuses and changes
exactly what the reference changes there, so its applicability false positives and its
change false positives and negatives are 0; every SAM row of every study records 0 for
all three, and 0 effect false positives.

Every number a study records is a count. Ratios are computed where they are printed, so a
study file compares exactly across platforms and no rounding rule is part of a result.
Precision and recall are printed from counts pooled over seeds.

## The unsafe contrast

`amlearn/learn/optimistic.py` keeps SAM's effects and drops every precondition: the most
permissive model consistent with what the traces show each action does. Without a search
bound it solves every problem SAM solves, since its actions apply wherever SAM's do and have
the same effects. Under a node bound it can solve fewer: every action applies in every
state, and the search can exhaust the bound where SAM's finds a plan. On blocksworld it
hits the bound on 97 of 500 test problems from 3 demonstrations on, all of which SAM
solves from 10 (`docs/04-results.md`). Its plans are judged by the same validator.

## Limits

- Fully observed, noise-free traces only. An atom missing from one observed pre-state
  removes a true precondition and voids the guarantee. A trace file no STRIPS model
  explains is refused; noise some STRIPS model still explains passes, an atom missing from
  every state of a trace for one. Partial observability is not handled; the safe learners
  of Le, Juba and Stern (2024) handle it.
- Untyped STRIPS only: no conditional effects, quantifiers, numeric fluents or constants in
  schemas. Safe learning with conditional effects (Mordoch, Scala, Stern and Juba 2024)
  and with numeric fluents (Mordoch, Juba and Stern 2023) holds under further assumptions.
- Small instances. Training stays at 3–4 blocks, 2–3 balls and 1–2 packages; the studies
  test up to 5 blocks, 4 balls and 3 packages, and the large ones up to 7 blocks, 8 balls
  and 5 packages, where BFS still exhausts every reachable state of the reference.
