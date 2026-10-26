import pytest

from amlearn.strips import Action, Domain, Problem, StripsError, lifted_atoms, render


class TestAction:
    def test_it_refuses_an_atom_over_a_name_that_is_not_a_parameter(self):
        with pytest.raises(StripsError, match="not parameters"):
            Action("a", ("?x",), pre_pos=frozenset({("p", "?y")}))

    def test_it_refuses_a_repeated_parameter(self):
        with pytest.raises(StripsError, match="repeated parameter"):
            Action("a", ("?x", "?x"))

    def test_grounding_binds_parameters_in_order(self):
        act = Action("move", ("?f", "?t"), frozenset({("at", "?f")}), add=frozenset({("at", "?t")}))
        g = act.ground(("r1", "r2"))
        assert g.label == ("move", "r1", "r2")
        assert g.pre_pos == {("at", "r1")}
        assert g.add == {("at", "r2")}


class TestGroundedAction:
    def test_delete_comes_before_add(self):
        act = Action("a", ("?x",), add=frozenset({("p", "?x")}), delete=frozenset({("p", "?x")}))
        assert act.ground(("o",)).apply(frozenset()) == {("p", "o")}

    def test_a_negative_precondition_blocks_where_its_atom_holds(self):
        act = Action("a", ("?x",), pre_neg=frozenset({("p", "?x")}))
        assert not act.ground(("o",)).applicable(frozenset({("p", "o")}))
        assert act.ground(("o",)).applicable(frozenset())


class TestDomain:
    def test_it_refuses_an_undeclared_predicate(self):
        act = Action("a", ("?x",), pre_pos=frozenset({("q", "?x")}))
        with pytest.raises(StripsError, match="undeclared predicate"):
            Domain("d", {"p": 1}, {"a": act})

    def test_it_refuses_a_wrong_arity(self):
        act = Action("a", ("?x",), pre_pos=frozenset({("p", "?x")}))
        with pytest.raises(StripsError, match="takes 2"):
            Domain("d", {"p": 2}, {"a": act})

    def test_grounding_is_injective(self):
        d = Domain("d", {"p": 2}, {"a": Action("a", ("?x", "?y"))})
        labels = [g.label for g in d.grounded(("o1", "o2"))]
        assert labels == [("a", "o1", "o2"), ("a", "o2", "o1")]

    def test_relevant_drops_groundings_a_static_atom_rules_out(self):
        act = Action(
            "a", ("?x",), pre_pos=frozenset({("kind", "?x")}), add=frozenset({("p", "?x")})
        )
        d = Domain("d", {"kind": 1, "p": 1}, {"a": act})
        problem = Problem("p", ("o1", "o2"), frozenset({("kind", "o1")}), frozenset())
        assert [g.label for g in d.relevant(problem)] == [("a", "o1")]

    def test_relevant_keeps_the_order_of_grounded_whatever_the_order_of_objects(self, bench, rng):
        drawn = bench.generate(rng, bench.test_sizes[0])
        reverse = tuple(reversed(drawn.objects))
        problem = Problem(drawn.name, reverse, drawn.init, drawn.goal)
        kept = [g.label for g in bench.domain.relevant(problem)]
        everything = [g.label for g in bench.domain.grounded(problem.objects)]
        assert kept == [label for label in everything if label in set(kept)]


class TestProblem:
    def test_it_refuses_an_object_named_twice(self):
        with pytest.raises(StripsError, match=r"more than once: \['o'\]"):
            Problem("p", ("o", "o"), frozenset(), frozenset())


def test_lifted_atoms_include_repeated_variables():
    assert list(lifted_atoms({"on": 2}, ("?x", "?y"))) == [
        ("on", "?x", "?x"),
        ("on", "?x", "?y"),
        ("on", "?y", "?x"),
        ("on", "?y", "?y"),
    ]


def test_render_writes_a_nullary_atom_bare():
    assert render(("handempty",)) == "handempty"
    assert render(("on", "a", "b")) == "on(a,b)"
