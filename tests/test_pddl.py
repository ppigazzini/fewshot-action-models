import pytest

from amlearn.pddl import PddlError, read_domain, read_problem, write_domain, write_problem


class TestReader:
    def test_it_reads_every_reference_domain(self, bench):
        assert bench.domain.actions

    def test_it_refuses_types(self):
        text = "(define (domain d) (:types block) (:predicates (p ?x)))"
        with pytest.raises(PddlError, match="types"):
            read_domain(text)

    def test_it_refuses_a_conditional_effect(self):
        text = """(define (domain d) (:predicates (p ?x) (q ?x))
          (:action a :parameters (?x) :precondition (p ?x)
           :effect (when (p ?x) (q ?x))))"""
        with pytest.raises(PddlError, match="outside the supported subset"):
            read_domain(text)

    def test_it_refuses_a_constant_in_a_schema(self):
        text = """(define (domain d) (:predicates (p ?x))
          (:action a :parameters (?x) :precondition (p c) :effect (p ?x)))"""
        with pytest.raises(PddlError, match="not parameters"):
            read_domain(text)

    def test_it_ignores_comments_and_case(self):
        d = read_domain("; note\n(DEFINE (DOMAIN D) (:PREDICATES (P ?X)))")
        assert d.predicates == {"p": 1}


class TestRoundTrip:
    def test_a_written_domain_reads_back_equal(self, bench):
        again = read_domain(write_domain(bench.domain))
        assert again.predicates == bench.domain.predicates
        assert again.actions == bench.domain.actions

    def test_a_written_problem_reads_back_equal(self, bench, rng):
        problem = bench.generate(rng, bench.train_sizes[0])
        again = read_problem(write_problem(problem, bench.domain.name))
        assert again.init == problem.init
        assert again.goal == problem.goal
        assert sorted(again.objects) == sorted(problem.objects)


def test_an_action_field_without_a_value_is_a_pddl_error():
    text = "(define (domain d) (:predicates (p ?x)) (:action a :parameters (?x) :effect))"
    with pytest.raises(PddlError, match="without a value"):
        read_domain(text)
