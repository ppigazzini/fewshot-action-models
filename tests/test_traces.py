import itertools
import random

import pytest

from amlearn.strips import StripsError
from amlearn.traces import demonstrate, execute, observe, random_walk


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


class TestObserve:
    def _demo(self, bench, rng):
        problem = bench.generate(rng, bench.train_sizes[-1])
        return problem, demonstrate(bench.domain, problem)

    def test_an_observer_that_misses_and_hides_nothing_sees_the_trace(self, bench, rng):
        problem, trace = self._demo(bench, rng)
        before = rng.getstate()
        seen = observe(trace, rng, predicates=bench.domain.predicates, objects=problem.objects)
        assert tuple(o.step for o in seen) == trace
        assert all(not o.hidden and not o.next_hidden for o in seen)
        assert rng.getstate() == before

    def test_each_state_is_seen_once(self, bench, rng):
        problem, trace = self._demo(bench, rng)
        seen = observe(
            trace,
            rng,
            predicates=bench.domain.predicates,
            objects=problem.objects,
            drop=0.3,
            hide=0.3,
        )
        for before, after in itertools.pairwise(seen):
            assert before.step.next_state == after.step.state
            assert before.next_hidden == after.hidden

    def test_a_hidden_atom_is_not_seen_true(self, bench, rng):
        problem, trace = self._demo(bench, rng)
        seen = observe(
            trace, rng, predicates=bench.domain.predicates, objects=problem.objects, hide=0.5
        )
        assert all(o.hidden.isdisjoint(o.step.state) for o in seen)
        assert any(o.hidden for o in seen)
