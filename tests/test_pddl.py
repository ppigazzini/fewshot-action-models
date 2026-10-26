import pytest

from amlearn.domains import draw
from amlearn.learn import Signature
from amlearn.learn.sam import learn
from amlearn.pddl import PddlError, read_domain, read_problem, write_domain, write_problem
from amlearn.traces import demonstrate


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

    def test_it_drops_an_inequality_of_two_parameters(self):
        text = """(define (domain d) (:predicates (p ?x))
          (:action a :parameters (?x ?y)
           :precondition (and (p ?x) (not (= ?x ?y))) :effect (p ?y)))"""
        assert read_domain(text).actions["a"].pre_pos == {("p", "?x")}

    def test_it_ignores_comments_and_case(self):
        d = read_domain("; note\n(DEFINE (DOMAIN D) (:PREDICATES (P ?X)))")
        assert d.predicates == {"p": 1}


class TestRoundTrip:
    def test_a_written_domain_reads_back_equal(self, bench):
        again = read_domain(write_domain(bench.domain))
        assert again.predicates == bench.domain.predicates
        assert again.actions == bench.domain.actions

    def test_a_learned_model_reads_back_equal(self, bench, rng):
        traces = [demonstrate(bench.domain, p) for p in draw(bench, rng, 3, bench.train_sizes)]
        model = learn(Signature.of(bench.domain), traces)
        assert any(a.pre_neg for a in model.actions.values())
        assert read_domain(write_domain(model)).actions == model.actions

    def test_the_writer_states_injective_grounding_as_inequalities(self, blocks):
        text = write_domain(blocks.domain)
        assert ":equality" in text
        assert "(not (= ?x ?y))" in text

    def test_a_written_problem_reads_back_equal(self, bench, rng):
        problem = bench.generate(rng, bench.train_sizes[0])
        again = read_problem(write_problem(problem, bench.domain.name))
        assert again.init == problem.init
        assert again.goal == problem.goal
        assert sorted(again.objects) == sorted(problem.objects)


DOMAIN = "(define (domain d) (:predicates (p ?x)) {})"
PROBLEM = "(define (problem x) (:domain d) (:objects {}) (:init) {})"


class TestMalformedText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (DOMAIN.format("(:action a :parameters (?x) :effect)"), "without a value"),
            (DOMAIN.format("(:action)"), "without a name"),
            (DOMAIN.format("(:predicates (q ?x))"), ":predicates appears twice"),
            (
                DOMAIN.format("(:action a :parameters (?x)) (:action a :parameters (?x))"),
                "action a defined twice",
            ),
            (DOMAIN.format("(:action a :parameters (?x) :effect (q ?x))"), "undeclared"),
            (
                DOMAIN.format("(:action a :parameters (?x) :precondition (not (= ?x ?x)))"),
                "not an inequality of two parameters",
            ),
            (
                DOMAIN.format("(:action a :parameters (?x) :precondition (not (= ?x c)))"),
                "not an inequality of two parameters",
            ),
            (
                DOMAIN.format("(:action a :parameters (?x ?y) :precondition (= ?x ?y))"),
                "'=' is outside the supported subset",
            ),
            (
                DOMAIN.format(
                    "(:action a :parameters (?x ?y) :precondition (and (not (= ?x ?y) (p ?x))))"
                ),
                "malformed negation",
            ),
        ],
    )
    def test_a_malformed_domain_is_a_pddl_error_naming_the_fault(self, text, expected):
        with pytest.raises(PddlError, match=expected):
            read_domain(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (PROBLEM.format("o", "(:goal)"), "holds 0 expressions"),
            (PROBLEM.format("o", "(:goal (p o) (q o))"), "holds 2 expressions"),
            (PROBLEM.format("o o", "(:goal (p o))"), "more than once"),
        ],
    )
    def test_a_malformed_problem_is_a_pddl_error_naming_the_fault(self, text, expected):
        with pytest.raises(PddlError, match=expected):
            read_problem(text)
