from amlearn.plan.bfs import bfs
from amlearn.validate import validate


class TestValidate:
    def test_it_names_the_step_and_the_false_precondition(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        verdict = validate(blocks.domain, problem, [("put-down", "b1")])
        assert not verdict.valid
        assert verdict.step == 0
        assert "holding(b1) false" in verdict.reason

    def test_it_refuses_an_action_the_reference_does_not_have(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        verdict = validate(blocks.domain, problem, [("teleport", "b1")])
        assert "no action" in verdict.reason

    def test_it_refuses_a_step_with_the_wrong_number_of_objects(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        verdict = validate(blocks.domain, problem, [("stack", "b1")])
        assert not verdict.valid
        assert verdict.step == 0
        assert "stack takes 2 objects, got 1" in verdict.reason

    def test_it_names_a_negative_precondition_that_holds(self, guarded):
        domain, problem = guarded
        verdict = validate(domain, problem, [("make", "o")])
        assert not verdict.valid
        assert verdict.reason == "make(o): blocked(o) true"

    def test_it_refuses_an_object_the_problem_does_not_name(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        verdict = validate(blocks.domain, problem, [("pick-up", "zz")])
        assert verdict.reason == "pick-up(zz): unknown ['zz']"

    def test_it_refuses_a_repeated_object(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        verdict = validate(blocks.domain, problem, [("stack", "b1", "b1")])
        assert "repeated object" in verdict.reason

    def test_an_executable_plan_that_misses_the_goal_fails_at_its_end(self, blocks, rng):
        problem = blocks.generate(rng, 4)
        plan = bfs(blocks.domain, problem).plan
        assert plan
        verdict = validate(blocks.domain, problem, plan[:-1])
        assert verdict.step == len(plan) - 1
        assert verdict.reason.startswith("goal unmet")
