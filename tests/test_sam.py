import random

import pytest

from amlearn.domains import draw
from amlearn.learn import Signature, optimistic
from amlearn.learn.sam import learn, learn_observed, unexplained
from amlearn.plan.bfs import bfs
from amlearn.strips import Action, Domain, Problem, StripsError, Transition, lifted_atoms
from amlearn.traces import demonstrate, execute, observe, random_walk
from amlearn.validate import validate


def _traces(bench, seed, n):
    rng = random.Random(f"{bench.name}:sam:{seed}")
    return [demonstrate(bench.domain, p) for p in draw(bench, rng, n, bench.train_sizes)]


class TestSafety:
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_every_plan_of_a_learned_model_is_valid_under_the_reference(self, bench, seed, n):
        model = learn(Signature.of(bench.domain), _traces(bench, seed, n))
        rng = random.Random(f"{bench.name}:safety:{seed}")
        for problem in draw(bench, rng, 15, bench.test_sizes):
            plan = bfs(model, problem).plan
            if plan is None:
                continue
            assert validate(bench.domain, problem, plan).valid
            assert execute(model, problem, plan)[-1:] == execute(bench.domain, problem, plan)[-1:]

    @pytest.mark.parametrize("seed", range(3))
    def test_random_walks_keep_the_guarantee(self, bench, seed):
        rng = random.Random(f"{bench.name}:walk:{seed}")
        walks = [
            random_walk(bench.domain, p, 15, rng) for p in draw(bench, rng, 3, bench.train_sizes)
        ]
        model = learn(Signature.of(bench.domain), walks)
        for problem in draw(bench, rng, 15, bench.test_sizes):
            plan = bfs(model, problem).plan
            if plan is not None:
                assert validate(bench.domain, problem, plan).valid


class TestWhatIsLearned:
    def test_a_learned_precondition_keeps_every_true_one(self, bench):
        model = learn(Signature.of(bench.domain), _traces(bench, 0, 20))
        for name, action in model.actions.items():
            assert bench.domain.actions[name].pre_pos <= action.pre_pos

    def test_effects_match_the_reference_once_every_change_is_seen(self, bench):
        model = learn(Signature.of(bench.domain), _traces(bench, 0, 50))
        assert set(model.actions) == set(bench.domain.actions)
        for name, action in model.actions.items():
            assert action.add == bench.domain.actions[name].add
            assert action.delete == bench.domain.actions[name].delete

    def test_more_traces_never_add_a_precondition(self, bench):
        traces = _traces(bench, 1, 20)
        few = learn(Signature.of(bench.domain), traces[:5])
        many = learn(Signature.of(bench.domain), traces)
        for name, action in few.actions.items():
            assert many.actions[name].pre_pos <= action.pre_pos
            assert many.actions[name].pre_neg <= action.pre_neg

    def test_an_unobserved_action_is_absent(self, blocks):
        state = frozenset({("clear", "a"), ("ontable", "a"), ("handempty",)})
        nxt = frozenset({("holding", "a")})
        model = learn(Signature.of(blocks.domain), [(Transition(state, ("pick-up", "a"), nxt),)])
        assert set(model.actions) == {"pick-up"}

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (("teleport", "a"), r"teleport\(a\): not an action of blocksworld"),
            (("stack", "a"), r"stack\(a\): stack takes 2 objects"),
        ],
    )
    def test_a_step_outside_the_signature_is_refused(self, blocks, action, expected):
        step = Transition(frozenset(), action, frozenset())
        with pytest.raises(StripsError, match=expected):
            learn(Signature.of(blocks.domain), [(step,)])

    def test_a_transition_that_repeats_an_object_is_skipped(self, blocks):
        step = Transition(frozenset(), ("stack", "a", "a"), frozenset({("on", "a", "a")}))
        model = learn(Signature.of(blocks.domain), [(step,)])
        assert model.actions == {}


class TestVersionSpace:
    """SAM's model is the lower boundaries of the version spaces of Aineto and Scala (2024).

    Their rule ULP shrinks a precondition's lower boundary to its intersection with each
    pre-state, and ULE grows an effect's by each observed change; both over signed literals,
    a step lifted by inverting its binding.
    """

    def test_sam_is_the_lower_boundary_of_every_version_space(self, bench):
        traces = _traces(bench, 0, 5)
        signature = Signature.of(bench.domain)
        pre: dict[str, set] = {}
        eff: dict[str, set] = {}
        for step in (s for trace in traces for s in trace):
            name, objects = step.action[0], step.action[1:]
            params = signature.actions[name]
            parameter = dict(zip(objects, params, strict=True))

            def lift(atoms, parameter=parameter):
                return {
                    (a[0], *(parameter[o] for o in a[1:]))
                    for a in atoms
                    if all(o in parameter for o in a[1:])
                }

            before, after = lift(step.state), lift(step.next_state)
            candidates = set(lifted_atoms(signature.predicates, params))
            state = {(True, a) for a in before} | {(False, a) for a in candidates - before}
            changes = {(True, a) for a in after - before} | {(False, a) for a in before - after}
            pre[name] = pre.get(name, state) & state
            eff[name] = eff.get(name, set()) | changes
        model = learn(signature, traces)
        assert set(model.actions) == set(pre)
        for name, action in model.actions.items():
            assert pre[name] == {(True, a) for a in action.pre_pos} | {
                (False, a) for a in action.pre_neg
            }
            assert eff[name] == {(True, a) for a in action.add} | {
                (False, a) for a in action.delete
            }


class TestPartialObservability:
    """PI-SAM: SAM's rules on the atoms seen both before and after a step."""

    @staticmethod
    def _observed(bench, seed, n, hide):
        rng = random.Random(f"{bench.name}:pisam:{seed}")
        problems = draw(bench, rng, n, bench.train_sizes)
        return [
            observe(
                demonstrate(bench.domain, p),
                rng,
                predicates=bench.domain.predicates,
                objects=p.objects,
                hide=hide,
            )
            for p in problems
        ]

    def test_with_nothing_hidden_it_is_sam(self, bench):
        observed = self._observed(bench, 0, 5, 0.0)
        steps = [tuple(o.step for o in t) for t in observed]
        signature = Signature.of(bench.domain)
        assert learn_observed(signature, observed) == learn(signature, steps)

    @pytest.mark.parametrize("hide", [0.1, 0.3])
    def test_every_plan_of_a_model_from_masked_traces_is_valid(self, bench, hide):
        for seed in range(2):
            model = learn_observed(Signature.of(bench.domain), self._observed(bench, seed, 5, hide))
            rng = random.Random(f"{bench.name}:pisam-test:{seed}")
            for problem in draw(bench, rng, 10, bench.test_sizes):
                plan = bfs(model, problem, max_nodes=50_000).plan
                if plan is not None:
                    assert validate(bench.domain, problem, plan).valid

    def test_reading_hidden_atoms_as_false_is_unsafe(self, blocks):
        # A hidden true precondition read as false prunes it from SAM's precondition.
        observed = self._observed(blocks, 0, 10, 0.3)
        model = learn(Signature.of(blocks.domain), [tuple(o.step for o in t) for t in observed])
        rng = random.Random("closed-world")
        invalid = 0
        for problem in draw(blocks, rng, 10, blocks.test_sizes):
            plan = bfs(model, problem, max_nodes=20_000).plan
            invalid += plan is not None and not validate(blocks.domain, problem, plan).valid
        assert invalid > 0

    def test_an_atom_the_observer_misses_is_refuted_by_the_check(self, blocks):
        rng = random.Random("noise")
        problems = draw(blocks, rng, 10, blocks.train_sizes)
        observed = [
            observe(
                demonstrate(blocks.domain, p),
                rng,
                predicates=blocks.domain.predicates,
                objects=p.objects,
                drop=0.05,
            )
            for p in problems
        ]
        steps = [tuple(o.step for o in t) for t in observed]
        model = learn(Signature.of(blocks.domain), steps)
        assert any(unexplained(model, step) for trace in steps for step in trace)

    def test_masked_steps_replay_under_their_model(self, bench):
        observed = self._observed(bench, 1, 5, 0.3)
        model = learn_observed(Signature.of(bench.domain), observed)
        steps = [o for t in observed for o in t]
        assert all(unexplained(model, o.step, o.hidden, o.next_hidden) == [] for o in steps)


class TestTheoremStepByStep:
    @pytest.mark.parametrize("seed", range(3))
    def test_a_random_walk_in_the_learned_model_is_mirrored_state_for_state(self, bench, seed):
        model = learn(Signature.of(bench.domain), _traces(bench, seed, 3))
        rng = random.Random(f"{bench.name}:mirror:{seed}")
        for problem in draw(bench, rng, 5, bench.test_sizes):
            walk = random_walk(model, problem, 12, rng)
            state = problem.init
            for step in walk:
                truth = bench.domain.actions[step.action[0]].ground(step.action[1:])
                assert truth.applicable(state)
                state = truth.apply(state)
                assert state == step.next_state


PICK_A = Transition(
    frozenset({("clear", "a"), ("ontable", "a"), ("handempty",)}),
    ("pick-up", "a"),
    frozenset({("holding", "a")}),
)
B_ON_TABLE = frozenset({("clear", "b"), ("ontable", "b"), ("handempty",)})


class TestReplay:
    @pytest.mark.parametrize("seed", range(3))
    def test_the_learned_model_replays_every_step_it_learned_from(self, bench, seed):
        rng = random.Random(f"{bench.name}:replay:{seed}")
        problems = draw(bench, rng, 3, bench.train_sizes)
        traces = _traces(bench, seed, 5) + [random_walk(bench.domain, p, 15, rng) for p in problems]
        model = learn(Signature.of(bench.domain), traces)
        assert [r for trace in traces for step in trace for r in unexplained(model, step)] == []

    def test_a_step_that_repeats_an_object_is_unexplained(self, blocks):
        step = Transition(frozenset(), ("stack", "a", "a"), frozenset({("on", "a", "a")}))
        model = learn(Signature.of(blocks.domain), [(PICK_A,)])
        assert unexplained(model, step) == [
            "stack(a,a) repeats a, which injective grounding never does"
        ]

    def test_a_change_the_action_does_not_name_is_unexplained(self, blocks):
        step = Transition(PICK_A.state, PICK_A.action, PICK_A.next_state | {("clear", "b")})
        model = learn(Signature.of(blocks.domain), [(step,)])
        assert unexplained(model, step) == ["clear(b) changes, but pick-up(a) does not name b"]

    def test_an_add_effect_another_step_shows_must_happen(self, blocks):
        failed = Transition(B_ON_TABLE, ("pick-up", "b"), frozenset())
        model = learn(Signature.of(blocks.domain), [(PICK_A,), (failed,)])
        assert unexplained(model, PICK_A) == []
        assert unexplained(model, failed) == [
            "holding(b) is false after pick-up(b), but another pick-up step adds holding(?x)"
        ]

    def test_a_delete_effect_another_step_shows_must_happen(self, blocks):
        kept = Transition(
            B_ON_TABLE, ("pick-up", "b"), frozenset({("holding", "b"), ("handempty",)})
        )
        model = learn(Signature.of(blocks.domain), [(PICK_A,), (kept,)])
        assert unexplained(model, kept) == [
            "handempty still holds after pick-up(b), but another pick-up step deletes handempty"
        ]


class TestNegativePreconditions:
    """A true delete effect never seen as a change is what the negative preconditions guard.

    The demonstration cleans an object that is not dirty, so ``not dirty(?x)`` holds in
    every observed pre-state of clean and stays in the learned precondition. Without it, a
    learned model cleans first and inspects after, and the reference refuses the plan: clean
    deletes the dirt inspect needs.
    """

    CHORES = Domain(
        "chores",
        {"dirty": 1, "washed": 1, "done": 1},
        {
            "clean": Action(
                "clean",
                ("?x",),
                add=frozenset({("washed", "?x")}),
                delete=frozenset({("dirty", "?x")}),
            ),
            "inspect": Action(
                "inspect",
                ("?x",),
                pre_pos=frozenset({("dirty", "?x")}),
                add=frozenset({("done", "?x")}),
            ),
        },
    )
    DEMOS = (
        (Transition(frozenset(), ("clean", "a"), frozenset({("washed", "a")})),),
        (
            Transition(
                frozenset({("dirty", "c")}),
                ("inspect", "c"),
                frozenset({("dirty", "c"), ("done", "c")}),
            ),
        ),
    )
    PROBLEM = Problem(
        "p", ("b",), frozenset({("dirty", "b")}), frozenset({("washed", "b"), ("done", "b")})
    )

    def test_a_learned_model_without_them_plans_what_the_reference_refuses(self):
        model = learn(Signature.of(self.CHORES), self.DEMOS)
        stripped = Domain(
            model.name,
            model.predicates,
            {
                n: Action(n, a.params, a.pre_pos, frozenset(), a.add, a.delete)
                for n, a in model.actions.items()
            },
        )
        plan = bfs(stripped, self.PROBLEM).plan
        assert plan == (("clean", "b"), ("inspect", "b"))
        assert not validate(self.CHORES, self.PROBLEM, plan).valid

    def test_the_learned_model_keeps_them_and_plans_nothing_invalid(self):
        model = learn(Signature.of(self.CHORES), self.DEMOS)
        assert ("dirty", "?x") in model.actions["clean"].pre_neg
        assert bfs(model, self.PROBLEM).status == "unsolvable"


def test_the_optimistic_model_finds_plans_the_reference_refuses(blocks):

    model = optimistic.learn(Signature.of(blocks.domain), _traces(blocks, 0, 5))
    rng = random.Random("optimistic")
    invalid = 0
    for problem in draw(blocks, rng, 10, blocks.test_sizes):
        plan = bfs(model, problem, max_nodes=20_000).plan
        if plan is not None and not validate(blocks.domain, problem, plan).valid:
            invalid += 1
    assert invalid > 0
