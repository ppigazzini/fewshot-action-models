import random

import pytest

from amlearn.domains import draw
from amlearn.learn import Signature, optimistic
from amlearn.learn.sam import learn, unexplained
from amlearn.plan.bfs import bfs
from amlearn.strips import Transition
from amlearn.traces import demonstrate, execute, random_walk
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

    def test_a_transition_that_repeats_an_object_is_skipped(self, blocks):
        step = Transition(frozenset(), ("stack", "a", "a"), frozenset({("on", "a", "a")}))
        model = learn(Signature.of(blocks.domain), [(step,)])
        assert model.actions == {}


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


def test_the_optimistic_model_finds_plans_the_reference_refuses(blocks):

    model = optimistic.learn(Signature.of(blocks.domain), _traces(blocks, 0, 5))
    rng = random.Random("optimistic")
    invalid = 0
    for problem in draw(blocks, rng, 10, blocks.test_sizes):
        plan = bfs(model, problem, max_nodes=20_000).plan
        if plan is not None and not validate(blocks.domain, problem, plan).valid:
            invalid += 1
    assert invalid > 0
