import random

from amlearn.domains import BENCHMARKS, draw, logistics_problem


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


class TestLocalLogistics:
    @staticmethod
    def _away(problems):
        city = {"ap1": "c1", "po1": "c1", "ap2": "c2", "po2": "c2"}
        moved = []
        for problem in problems:
            start = {a[1]: a[2] for a in problem.init if a[0] == "at"}
            moved += [city[a[2]] != city[start[a[1]]] for a in problem.goal]
        return sum(moved) / len(moved)

    def test_a_training_goal_rarely_changes_city(self):
        rng = random.Random("local")
        share = self._away(draw(BENCHMARKS["logistics-local"], rng, 400, (2,)))
        assert 0.07 < share < 0.15

    def test_its_test_problems_come_from_the_logistics_generator(self):
        bench = BENCHMARKS["logistics-local"]
        assert bench.generate_test is logistics_problem
        share = self._away(draw(bench, random.Random("local"), 400, (2,), test=True))
        assert 0.45 < share < 0.6
