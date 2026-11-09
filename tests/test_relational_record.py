from __future__ import annotations

import inspect

from behavioral_records.canonical import canonical_json
from behavioral_records.relational_record import (
    RelationalBehaviorCompiler,
    audit_relation_break,
    relation_broken_control,
    validate_relational_record_contract,
)


def _episode(context: str, middle_action: str, middle_event: str) -> dict:
    return {
        "context": {"condition": context},
        "length": 3,
        "outcome": "success",
        "return": 1.0,
        "terminated": True,
        "truncated": False,
        "steps": [
            {
                "t": 0,
                "action": "begin",
                "reward": 0.0,
                "events": [],
                "state": {"progress": 0, "safe": True},
                "predicates": {"advanced": False},
            },
            {
                "t": 1,
                "action": middle_action,
                "reward": 0.0,
                "events": [middle_event],
                "state": {"progress": 1, "safe": context == "safe"},
                "predicates": {"advanced": True},
            },
            {
                "t": 2,
                "action": "finish",
                "reward": 1.0,
                "events": ["goal"],
                "state": {"progress": 2, "safe": context == "safe"},
                "predicates": {"advanced": True},
            },
        ],
    }


def _fulltrace() -> dict:
    safe = _episode("safe", "advance", "checkpoint")
    hazard = _episode("hazard", "detour", "hazard_observed")
    return {
        "representation": "deterministic_raw_validation_trajectories",
        "episodes": [safe for _ in range(10)] + [hazard for _ in range(10)],
    }


def test_relational_compiler_is_deterministic_compact_and_requirement_blind() -> None:
    compiler = RelationalBehaviorCompiler()
    first = compiler.compile(_fulltrace())
    second = compiler.compile(_fulltrace())
    assert first == second
    assert first["coverage"] == {
        "observed_episode_count": 20,
        "observed_step_count": 60,
        "unique_episode_pattern_count": 2,
    }
    assert sorted(row["multiplicity"] for row in first["patterns"]) == [10, 10]
    assert len(canonical_json(first)) < len(canonical_json(_fulltrace()))
    assert validate_relational_record_contract(first) == []
    parameters = inspect.signature(RelationalBehaviorCompiler.compile).parameters
    assert not {
        "objective",
        "requirement",
        "preference",
        "oracle",
        "judge_response",
    }.intersection(parameters)


def test_relation_break_is_deterministic_fact_and_character_matched() -> None:
    linked = RelationalBehaviorCompiler().compile(_fulltrace())
    first = relation_broken_control(linked)
    second = relation_broken_control(linked)
    assert first == second
    assert first != linked
    audit = audit_relation_break(linked, first)
    assert validate_relational_record_contract(first) == []
    ignored = {
        "changed_context_behavior_assignments",
        "changed_action_observation_assignments",
        "eligible_action_observation_assignments",
        "linked_characters",
        "broken_characters",
    }
    assert all(value for key, value in audit.items() if key not in ignored)
    assert audit["linked_characters"] == audit["broken_characters"]
