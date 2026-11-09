from __future__ import annotations

from behavioral_records.demo import make_demo_trajectories
from behavioral_records.experiment import build_records, group_by_policy
from behavioral_records.integrity import (
    audit_prompt_view_pair,
    prompt_view_signatures_are_objective_free,
)
from behavioral_records.io import read_json
from behavioral_records.views import (
    build_compact_behavior_view,
    build_matched_raw_trajectory_view,
    public_evidence,
    public_evidence_chars,
)


def _materials():
    protocol = read_json("configs/protocol_v2.json")
    validation = group_by_policy(
        item for item in make_demo_trajectories() if item.context["split"] == "validation"
    )
    records = build_records(validation, protocol)
    return protocol, validation, records


def test_compact_view_is_deterministic_objective_free_and_bounded() -> None:
    protocol, validation, records = _materials()
    policy_id = "efficient_clean"
    first = build_compact_behavior_view(records[policy_id], protocol["prompt_view"])
    second = build_compact_behavior_view(records[policy_id], protocol["prompt_view"])
    assert first == second
    assert prompt_view_signatures_are_objective_free()
    assert public_evidence_chars(first) <= protocol["prompt_view"]["per_policy_char_budget"]
    assert first["context_groups"]
    assert first["transitions"]
    assert len(first["event_skeletons"]) == len(validation[policy_id])
    assert first["view_coverage"]["event_skeleton_episode_fraction"] == 1.0
    assert first["view_coverage"]["context_episode_fraction"] == 1.0
    assert policy_id not in str(public_evidence(first))
    assert validation[policy_id]


def test_raw_control_matches_compact_actual_size_without_identity_fields() -> None:
    protocol, validation, records = _materials()
    policy_id = "efficient_clean"
    compact = build_compact_behavior_view(records[policy_id], protocol["prompt_view"])
    target = public_evidence_chars(compact)
    raw = build_matched_raw_trajectory_view(
        policy_id,
        validation[policy_id],
        protocol,
        protocol["prompt_view"],
        target,
    )
    reversed_raw = build_matched_raw_trajectory_view(
        policy_id,
        reversed(validation[policy_id]),
        protocol,
        protocol["prompt_view"],
        target,
    )
    assert raw == reversed_raw
    assert public_evidence_chars(raw) <= target
    assert target - public_evidence_chars(raw) <= protocol["prompt_view"]["raw_control"][
        "max_match_slack_chars"
    ]
    public_text = str(public_evidence(raw)).lower()
    assert policy_id not in public_text
    assert "episode_id" not in public_text
    assert "metadata" not in public_text
    assert audit_prompt_view_pair(compact, raw, protocol)["passed"] is True
