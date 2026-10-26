import random

from amlearn.domains import draw
from amlearn.learn import Signature, optimistic
from amlearn.learn.sam import learn
from amlearn.metrics import Counts, predictive, probes, syntactic
from amlearn.strips import Action, Domain, Problem
from amlearn.traces import demonstrate


def _setup(bench, demos, seed=0):
    rng = random.Random(f"{bench.name}:metrics:{seed}")
    traces = [demonstrate(bench.domain, p) for p in draw(bench, rng, demos, bench.train_sizes)]
    cases = probes(bench.domain, draw(bench, rng, 6, bench.test_sizes), 8, rng)
    return traces, cases


def _brute(model, cases):
    """The counts over every injective grounding, without pruning."""
    tp = fp = fn = ctp = cfp = cfn = 0
    for case in cases:
        got = {g.label: g for g in model.grounded(case.problem.objects) if g.applicable(case.state)}
        tp += len(got.keys() & case.truth.keys())
        fp += len(got.keys() - case.truth.keys())
        fn += len(case.truth.keys() - got.keys())
        for label, flips in case.truth.items():
            if label in got:
                mine = got[label].apply(case.state) ^ case.state
                ctp, cfp, cfn = (
                    ctp + len(mine & flips),
                    cfp + len(mine - flips),
                    cfn + len(flips - mine),
                )
    return Counts(tp, fp, fn), Counts(ctp, cfp, cfn)


class TestSyntactic:
    def test_the_reference_matches_itself(self, bench):
        pre, eff = syntactic(bench.domain, bench.domain)
        assert pre.fp == pre.fn == eff.fp == eff.fn == 0
        assert pre.tp == sum(len(a.pre_pos) + len(a.pre_neg) for a in bench.domain.actions.values())

    def test_an_action_the_model_lacks_is_all_false_negatives(self, blocks):
        model = learn(Signature.of(blocks.domain), [])
        pre, eff = syntactic(model, blocks.domain)
        assert pre.tp == pre.fp == eff.tp == eff.fp == 0
        assert eff.fn == sum(len(a.add) + len(a.delete) for a in blocks.domain.actions.values())


class TestPredictive:
    def test_the_reference_predicts_itself(self, bench):
        _, cases = _setup(bench, 1)
        app, change = predictive(bench.domain, bench.domain, cases)
        assert app.fp == app.fn == change.fp == change.fn == 0
        assert app.tp == sum(len(c.truth) for c in cases)

    def test_a_safe_model_applies_nothing_the_reference_refuses(self, bench):
        for demos in (1, 3):
            traces, cases = _setup(bench, demos)
            app, change = predictive(learn(Signature.of(bench.domain), traces), bench.domain, cases)
            assert app.fp == change.fp == change.fn == 0

    def test_pruning_by_reference_statics_counts_what_every_grounding_counts(self, bench):
        if bench.name == "logistics":
            # 11,880 bindings of drive-truck per problem: minutes, not seconds.
            return
        traces, cases = _setup(bench, 1, seed=2)
        for model in (learn(Signature.of(bench.domain), traces), bench.domain):
            assert predictive(model, bench.domain, cases) == _brute(model, cases)

    def test_an_atom_only_an_unobserved_action_changes_is_not_pruned_as_static(self):
        # The model saw check alone, so wet looks static to it; paint makes wet true along a
        # reference walk, and check then applies in both models.
        check = Action("check", ("?x",), pre_pos=frozenset({("wet", "?x")}))
        paint = Action("paint", ("?x",), add=frozenset({("wet", "?x")}))
        reference = Domain("paints", {"wet": 1}, {"check": check, "paint": paint})
        model = Domain("paints-sam", {"wet": 1}, {"check": check})
        problem = Problem("p", ("o",), frozenset(), frozenset({("wet", "o")}))
        cases = probes(reference, [problem], 3, random.Random(1))
        checks = sum(("check", "o") in c.truth for c in cases)
        app, _ = predictive(model, reference, cases)
        assert app == _brute(model, cases)[0]
        assert app.tp == checks > 0

    def test_the_optimistic_model_applies_what_the_reference_refuses(self, blocks):
        traces, cases = _setup(blocks, 3)
        model = optimistic.learn(Signature.of(blocks.domain), traces)
        app, _ = predictive(model, blocks.domain, cases)
        assert app.fp > 0
        assert predictive(model, blocks.domain, cases) == _brute(model, cases)
