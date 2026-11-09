from __future__ import annotations

from copy import deepcopy

from behavioral_records.demo import make_demo_trajectories
from behavioral_records.experiment import build_records, group_by_policy
from behavioral_records.integrity import audit_record, extractor_signature_is_objective_free
from behavioral_records.io import read_json


def _protocol() -> dict:
    return read_json("configs/protocol_v2.json")


def _validation_groups():
    rows = [item for item in make_demo_trajectories() if item.context["split"] == "validation"]
    return group_by_policy(rows)


def test_extractor_is_deterministic_under_input_reordering() -> None:
    groups = _validation_groups()
    protocol = _protocol()
    forward = build_records(groups, protocol)
    reverse = build_records({key: list(reversed(value)) for key, value in groups.items()}, protocol)
    assert forward == reverse


def test_extractor_signature_and_audit_are_objective_free() -> None:
    groups = _validation_groups()
    protocol = _protocol()
    record = build_records(groups, protocol)["efficient_clean"]
    assert extractor_signature_is_objective_free()
    audit = audit_record(record, protocol)
    assert audit["passed"] is True
    assert audit["forbidden_key_paths"] == []


def test_information_budget_reports_omitted_mass() -> None:
    groups = _validation_groups()
    protocol = deepcopy(_protocol())
    protocol["aggregation"]["max_trace_patterns"] = 1
    protocol["aggregation"]["max_transition_rows"] = 1
    record = build_records(groups, protocol)["efficient_clean"]
    assert record["coverage"]["trace_episode_fraction"] < 1.0
    assert record["coverage"]["transition_occurrence_fraction"] < 1.0
    assert record["coverage"]["omitted_trace_episodes"] > 0
