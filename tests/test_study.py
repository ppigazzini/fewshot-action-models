import json

import pytest
from valgebra import Validator

from amlearn.study import FORMAT, Row, Study, StudyError, differences, dumps, loads

ROW = Validator(Row)


def _row(**changes: int) -> Row:
    defaults = {
        "demos": 5,
        "seed": 0,
        "transitions": 20,
        "actions_learned": 4,
        "problems": 10,
        "solved": 8,
        "valid": 8,
        "limit": 0,
        "reference_solved": 10,
    }
    return ROW.ensure(defaults | changes)


def _study(*rows: Row) -> Study:
    return Study(
        format=FORMAT,
        benchmark="blocksworld",
        learner="sam",
        planner="bfs",
        max_nodes=1000,
        test_problems=10,
        rows=list(rows) or [_row()],
    )


class TestShape:
    def test_a_written_study_reads_back_equal(self):
        study = _study()
        assert loads(dumps(study)) == study

    def test_a_boolean_count_is_refused(self):
        payload = json.loads(dumps(_study()))
        payload["rows"][0]["solved"] = True
        with pytest.raises(StudyError, match=r"rows\[0\]\.solved"):
            loads(json.dumps(payload))

    def test_a_key_the_format_does_not_name_is_refused(self):
        payload = json.loads(dumps(_study()))
        payload["rows"][0]["sovled"] = 3
        with pytest.raises(StudyError, match="sovled"):
            loads(json.dumps(payload))

    def test_a_negative_count_is_refused(self):
        payload = json.loads(dumps(_study()))
        payload["rows"][0]["limit"] = -1
        with pytest.raises(StudyError, match=r"rows\[0\]\.limit"):
            loads(json.dumps(payload))

    def test_every_failure_is_reported_at_once(self):
        payload = json.loads(dumps(_study()))
        payload["format"] = 2
        payload["rows"][0]["seed"] = "0"
        with pytest.raises(StudyError) as caught:
            loads(json.dumps(payload))
        assert "format" in str(caught.value)
        assert "rows[0].seed" in str(caught.value)


class TestRowRules:
    def test_more_valid_than_solved_is_refused(self):
        with pytest.raises(StudyError, match="valid <= solved"):
            dumps(_study(_row(valid=9)))

    def test_a_row_over_a_different_test_set_is_refused(self):
        with pytest.raises(StudyError, match="problems 11"):
            dumps(_study(_row(problems=11, reference_solved=10)))

    def test_more_limit_hits_than_unsolved_problems_is_refused(self):
        with pytest.raises(StudyError, match="limit hits"):
            dumps(_study(_row(limit=3)))


def test_differences_name_the_row_and_the_count():
    assert differences(_study(), _study(_row(solved=9, valid=9))) == [
        "rows[0].solved: 8 != 9",
        "rows[0].valid: 8 != 9",
    ]


def test_more_valid_plans_than_the_reference_solves_is_refused():
    with pytest.raises(StudyError, match="the reference solves"):
        dumps(_study(_row(valid=8, solved=8, reference_solved=7)))
