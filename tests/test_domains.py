import random

from amlearn.domains import draw


class TestGenerators:
    def test_no_drawn_goal_holds_in_its_initial_state(self, bench):
        rng = random.Random(f"{bench.name}:goals")
        sizes = bench.train_sizes + bench.test_sizes
        assert all(not p.goal <= p.init for p in draw(bench, rng, 200, sizes))

    def test_every_drawn_problem_names_only_its_objects(self, bench):
        rng = random.Random(f"{bench.name}:objects")
        for problem in draw(bench, rng, 50, bench.test_sizes):
            named = {o for atom in problem.init | problem.goal for o in atom[1:]}
            assert named <= set(problem.objects)
