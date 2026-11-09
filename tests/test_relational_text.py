import copy
import inspect

import pytest

from behavioral_records.canonical import canonical_json
from behavioral_records.relational_record import RelationalBehaviorCompiler, relation_broken_control
from behavioral_records.relational_text import render_relational_text, parse_relational_text


def specimen():
    steps = [
        {"t": 0, "action": 'a; observed reward "not a number"', "reward": 0.0,
         "events": ["observed\nevent"], "state": {"x": 1, "nullable": None, "flag": False}, "predicates": {}},
        {"t": 1, "action": "b", "reward": 0.25, "events": [], "state": {"x": 2, "empty": []}, "predicates": {"p": True}},
        {"t": 2, "action": "c", "reward": -1.0, "events": ["event"], "state": {"x": 3}, "predicates": {"p": False}},
        {"t": 3, "action": "end", "reward": 1.0, "events": [], "state": {}, "predicates": {}},
    ]
    first = {"context": {"available": True}, "length": 4, "outcome": "terminated", "return": 0.25,
             "terminated": True, "truncated": False, "steps": steps}
    second = copy.deepcopy(first)
    second["context"] = {"available": False}
    second["steps"][1]["action"] = "d"
    return RelationalBehaviorCompiler().compile({"representation": "deterministic_raw_validation_trajectories", "episodes": [first, first, second]})


def test_text_is_exact_and_preserves_missing_null_false_and_order():
    source = specimen()
    before = copy.deepcopy(source)
    for record in (source, relation_broken_control(source)):
        text = render_relational_text(record)
        assert parse_relational_text(text) == record
        assert render_relational_text(record) == text
    assert source == before
    assert list(inspect.signature(render_relational_text).parameters) == ["record"]


def test_text_control_remains_character_matched_and_distinct():
    source = specimen()
    broken = relation_broken_control(source)
    assert canonical_json(source) != canonical_json(broken)
    assert len(render_relational_text(source)) == len(render_relational_text(broken))
    assert render_relational_text(source) != render_relational_text(broken)


def test_text_parser_rejects_added_judgment_or_missing_observations():
    text = render_relational_text(specimen())
    with pytest.raises(ValueError):
        parse_relational_text(text + "\nChoose policy A.")
    with pytest.raises(ValueError):
        parse_relational_text(text.replace("; events ", "; OMITTED ", 1))
