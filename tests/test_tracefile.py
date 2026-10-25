import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from amlearn.domains import draw
from amlearn.learn import Signature
from amlearn.learn.sam import learn
from amlearn.strips import Transition
from amlearn.tracefile import MAX_REPORTED, TraceFileError, TraceSchema
from amlearn.traces import demonstrate

ROOT = Path(__file__).parent.parent


def _demos(bench, n=4):
    rng = random.Random(f"{bench.name}:tracefile")
    return [demonstrate(bench.domain, p) for p in draw(bench, rng, n, bench.train_sizes)]


def _line(text: str, number: int = 0) -> dict:
    return json.loads(text.splitlines()[number])


class TestRoundTrip:
    def test_written_traces_read_back_equal_and_learn_the_same_model(self, bench):
        schema = TraceSchema.of(Signature.of(bench.domain))
        traces = _demos(bench)
        again = schema.loads(schema.dumps(traces))
        assert again == traces
        signature = Signature.of(bench.domain)
        assert learn(signature, again).actions == learn(signature, traces).actions

    def test_the_schema_admits_every_state_the_reference_produces(self, bench):
        schema = TraceSchema.of(Signature.of(bench.domain))
        for trace in _demos(bench):
            record = {"steps": [_step(t) for t in trace]}
            assert record in schema.line


def _step(t: Transition) -> dict:
    return {
        "state": [list(a) for a in sorted(t.state)],
        "action": list(t.action),
        "next_state": [list(a) for a in sorted(t.next_state)],
    }


class TestRefusals:
    @pytest.fixture
    def schema_and_text(self, blocks):
        schema = TraceSchema.of(Signature.of(blocks.domain))
        return schema, schema.dumps(_demos(blocks, 1))

    def _broken(self, text: str, edit) -> str:
        record = _line(text)
        edit(record["steps"])
        return json.dumps(record) + "\n"

    @pytest.mark.parametrize(
        ("edit", "expected"),
        [
            (lambda s: s[0].__setitem__("action", ["stack", "b1"]), "stack takes 2 objects, got 1"),
            (lambda s: s[0]["state"].__setitem__(0, ["on", "b1"]), "on takes 2 objects, got 1"),
            (lambda s: s[0].__setitem__("action", ["teleport", "b1"]), "got 'teleport'"),
            (lambda s: s[0]["state"].__setitem__(0, ["clear", ""]), "expected length >= 1"),
            (lambda s: s[0]["state"].__setitem__(0, ["clear", 7]), "steps[0].state[0][1]"),
            (lambda s: s[0].__setitem__("extra", 1), "steps[0].extra"),
            (lambda s: s.append(s[0]), ".state differs from steps["),
        ],
    )
    def test_a_malformed_step_is_named_with_its_path(self, schema_and_text, edit, expected):
        schema, text = schema_and_text
        with pytest.raises(TraceFileError, match=expected.replace("[", r"\[").replace("]", r"\]")):
            schema.loads(self._broken(text, edit))

    def test_every_failure_on_every_line_is_reported_with_its_line(self, schema_and_text):
        schema, text = schema_and_text
        bad = self._broken(text, lambda s: s[0].__setitem__("extra", 1))
        with pytest.raises(TraceFileError) as caught:
            schema.loads(bad + "{not json\n" + bad)
        message = str(caught.value)
        assert "line 1:" in message
        assert "line 2: expected valid JSON" in message
        assert "line 3:" in message

    def test_a_long_list_of_failures_is_summarised(self, schema_and_text):
        schema, text = schema_and_text
        bad = self._broken(text, lambda s: s[0].__setitem__("extra", 1))
        with pytest.raises(TraceFileError, match=r"\(\+5 more\)"):
            schema.loads(bad * (MAX_REPORTED + 5))

    def test_writing_a_trace_the_signature_does_not_admit_is_refused(self, blocks):
        schema = TraceSchema.of(Signature.of(blocks.domain))
        step = Transition(frozenset({("levitating", "b1")}), ("pick-up", "b1"), frozenset())
        with pytest.raises(TraceFileError):
            schema.dumps([(step,)])


class TestStripsConsistency:
    def test_a_change_no_action_without_constants_makes_is_refused_at_its_line_and_step(
        self, blocks
    ):
        schema = TraceSchema.of(Signature.of(blocks.domain))
        lines = schema.dumps(_demos(blocks, 2)).splitlines()
        record = json.loads(lines[1])
        record["steps"][-1]["next_state"].append(["clear", "zz"])
        text = "\n".join([lines[0], json.dumps(record)])
        last = len(record["steps"]) - 1
        with pytest.raises(TraceFileError) as caught:
            schema.loads(text)
        assert str(caught.value).startswith(f"line 2: steps[{last}]: clear(zz) changes, but ")


def test_the_tool_learns_from_a_trace_file_and_refuses_a_bad_one(tmp_path):
    traces, bad = tmp_path / "demos.jsonl", tmp_path / "bad.jsonl"
    tool = [sys.executable, str(ROOT / "tools" / "learn.py"), "blocksworld", "--demos", "4"]
    written = subprocess.run(
        [*tool, "--write-traces", str(traces)], capture_output=True, text=True, check=True
    )
    read = subprocess.run(
        [*tool, "--traces", str(traces)], capture_output=True, text=True, check=True
    )
    assert written.stdout == read.stdout
    bad.write_text('{"steps": [{"state": [], "action": ["stack", "b1"], "next_state": []}]}\n')
    refused = subprocess.run(
        [*tool, "--traces", str(bad)], capture_output=True, text=True, check=False
    )
    assert refused.returncode == 1
    assert "stack takes 2 objects, got 1" in refused.stderr
