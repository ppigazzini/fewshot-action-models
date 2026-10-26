import random

import pytest

from amlearn.strips import StripsError
from amlearn.traces import execute, random_walk


class TestExecute:
    def test_it_refuses_a_step_whose_precondition_fails(self, blocks, rng):
        problem = blocks.generate(rng, 3)
        with pytest.raises(StripsError, match="step 0 of the plan is not applicable"):
            execute(blocks.domain, problem, (("put-down", "b1"),))


class TestRandomWalk:
    def test_it_chooses_among_the_applicable_actions(self, blocks, rng):
        problem = blocks.generate(rng, 4)
        first = {
            random_walk(blocks.domain, problem, 1, random.Random(i))[0].action for i in range(20)
        }
        assert len(first) > 1

    def test_it_stops_where_no_action_applies(self, chain, walk_to):
        walk = random_walk(chain, walk_to("o4"), 10, random.Random(0))
        assert [step.action for step in walk] == [("step", "o1", "o2"), ("step", "o2", "o3")]
