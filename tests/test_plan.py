from amlearn.domains import draw
from amlearn.learn import Signature
from amlearn.learn.sam import learn
from amlearn.plan.asp import asp
from amlearn.plan.bfs import bfs
from amlearn.strips import Action, Domain, Problem
from amlearn.traces import demonstrate
from amlearn.validate import validate


class TestBreadthFirst:
    def test_every_plan_it_returns_is_valid_under_the_same_domain(self, bench, rng):
        for problem in draw(bench, rng, 10, bench.test_sizes):
            out = bfs(bench.domain, problem)
            assert out.status == "solved"
            assert out.plan is not None
            assert validate(bench.domain, problem, out.plan).valid

    def test_it_proves_unsolvability_by_exhaustion(self):
        d = Domain("d", {"p": 1}, {})
        problem = Problem("x", ("o",), frozenset(), frozenset({("p", "o")}))
        assert bfs(d, problem).status == "unsolvable"

    def test_it_reports_the_node_bound_as_a_limit_not_a_proof(self, blocks, rng):
        problem = blocks.generate(rng, 5)
        assert bfs(blocks.domain, problem, max_nodes=3).status == "limit"

    def test_a_satisfied_goal_needs_no_steps(self):
        d = Domain("d", {"p": 1}, {"a": Action("a", ("?x",))})
        problem = Problem("x", ("o",), frozenset({("p", "o")}), frozenset({("p", "o")}))
        assert bfs(d, problem).plan == ()

    def test_the_bound_counts_the_states_stored(self, chain, walk_to):
        assert bfs(chain, walk_to("o4"), max_nodes=3).status == "limit"
        proof = bfs(chain, walk_to("o4"), max_nodes=4)
        assert proof.status == "unsolvable"
        assert proof.expanded == 3

    def test_a_goal_stored_as_the_bound_is_reached_is_solved(self, chain, walk_to):
        out = bfs(chain, walk_to("o3"), max_nodes=3)
        assert out.plan == (("step", "o1", "o2"), ("step", "o2", "o3"))


class TestAnswerSet:
    def test_a_satisfied_goal_needs_no_steps(self, chain, walk_to):
        assert asp(chain, walk_to("o1")).plan == ()

    def test_a_plan_as_long_as_the_horizon_is_found(self, chain, walk_to):
        assert asp(chain, walk_to("o3"), max_horizon=2).plan == (
            ("step", "o1", "o2"),
            ("step", "o2", "o3"),
        )
        assert asp(chain, walk_to("o3"), max_horizon=1).status == "limit"

    def test_an_atom_one_step_deletes_and_adds_holds_after_it(self):
        flip = Action(
            "flip", ("?x",), add=frozenset({("p", "?x")}), delete=frozenset({("p", "?x")})
        )
        d = Domain("d", {"p": 1}, {"flip": flip})
        problem = Problem("x", ("o",), frozenset(), frozenset({("p", "o")}))
        assert asp(d, problem).plan == bfs(d, problem).plan == (("flip", "o"),)

    def test_a_negative_precondition_holds_a_step_back(self, guarded):
        domain, problem = guarded
        plan = asp(domain, problem).plan
        assert plan == bfs(domain, problem).plan == (("unblock", "o"), ("make", "o"))


class TestCrossCheck:
    def test_both_planners_find_plans_of_equal_length(self, bench, rng):
        for problem in draw(bench, rng, 6, bench.test_sizes):
            by_bfs = bfs(bench.domain, problem)
            by_asp = asp(bench.domain, problem)
            assert by_asp.status == "solved"
            assert by_bfs.plan is not None
            assert by_asp.plan is not None
            assert len(by_asp.plan) == len(by_bfs.plan)
            assert validate(bench.domain, problem, by_asp.plan).valid

    def test_asp_reports_a_limit_when_the_horizon_is_too_short(self, blocks, rng):
        problem = blocks.generate(rng, 4)
        assert asp(blocks.domain, problem, max_horizon=0).status == "limit"


def test_both_planners_agree_on_learned_models_at_test_sizes(bench, rng):

    traces = [demonstrate(bench.domain, p) for p in draw(bench, rng, 3, bench.train_sizes)]
    model = learn(Signature.of(bench.domain), traces)
    assert any(a.pre_neg for a in model.actions.values())
    for problem in draw(bench, rng, 4, bench.test_sizes):
        by_bfs, by_asp = bfs(model, problem), asp(model, problem)
        assert (by_bfs.plan is None) == (by_asp.plan is None)
        if by_bfs.plan is not None and by_asp.plan is not None:
            assert len(by_bfs.plan) == len(by_asp.plan)
            assert validate(model, problem, by_asp.plan).valid
