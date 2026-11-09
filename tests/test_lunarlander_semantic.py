from __future__ import annotations

from behavioral_records.io import read_json
from behavioral_records.lunarlander_semantic import (
    build_hidden_policy_profile,
    build_metric_pair_candidates,
    compare_hidden_profiles,
    hidden_episode_facts,
    validate_semantic_gate_config,
)


def _state(**overrides):
    row = {
        "x": 0.0,
        "y": 0.1,
        "vx": 0.0,
        "vy": -0.05,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "left_leg_contact": False,
        "right_leg_contact": False,
    }
    row.update(overrides)
    return row


def _hidden_episode(candidate_id: str = "candidate") -> dict:
    states = [
        _state(x=0.4, vx=-0.25),
        _state(x=0.2, vx=-0.1),
        _state(x=0.1),
        _state(x=0.1),
        _state(x=0.1),
        _state(x=0.1),
        _state(x=0.1),
        _state(x=0.1),
        _state(x=0.08, left_leg_contact=True),
        _state(x=0.05, left_leg_contact=True, right_leg_contact=True),
    ]
    actions = [2, 1, 0, 0, 0, 0, 0, 1, 3, 0]
    return {
        "candidate_id": candidate_id,
        "episode_id": f"episode_{candidate_id}",
        "context": {"context_id": "LL-C0"},
        "return": 150.0,
        "length": len(states),
        "terminal_cause": "settled_termination",
        "steps": [
            {
                "t": index,
                "action_id": action,
                "action": str(action),
                "reward": 0.0,
                "state": state,
                "body_contact_game_over": False,
                "lander_awake": index < len(states) - 1,
            }
            for index, (state, action) in enumerate(zip(states, actions))
        ],
    }


def test_semantic_gate_config_matches_completed_training_run() -> None:
    config = read_json("configs/lunarlander_semantic_gate_v1.json")
    transfer = read_json("configs/lunarlander_transfer_v1.json")
    training = read_json("configs/lunarlander_policy_pool_training_v1.json")
    summary = read_json(
        "tests/fixtures/lunar_training_summary.json"
    )
    report = validate_semantic_gate_config(config, transfer, training, summary)
    assert report["passed"] is True


def test_hidden_episode_facts_detect_control_recovery_and_contact() -> None:
    config = read_json("configs/lunarlander_semantic_gate_v1.json")
    facts = hidden_episode_facts(
        _hidden_episode(), config["factual_operationalization"]
    )
    assert facts["settled_termination"] is True
    assert facts["controlled_first_contact"] is True
    assert facts["catastrophic_termination"] is False
    assert facts["post_recovery_alternations"] >= 1
    assert facts["first_contact_speed"] is not None


def test_hidden_profile_and_lexicographic_oracle_are_separate_from_objective_text() -> None:
    config = read_json("configs/lunarlander_semantic_gate_v1.json")
    rows_a = []
    rows_b = []
    for context_id in ("LL-C0", "LL-C1", "LL-C2"):
        for replicate in range(2):
            row_a = _hidden_episode("a")
            row_b = _hidden_episode("b")
            row_a["episode_id"] = f"a_{context_id}_{replicate}"
            row_b["episode_id"] = f"b_{context_id}_{replicate}"
            row_a["context"]["context_id"] = context_id
            row_b["context"]["context_id"] = context_id
            if replicate == 0:
                row_b["terminal_cause"] = "body_contact_termination"
            rows_a.append(row_a)
            rows_b.append(row_b)
    profile_a = build_hidden_policy_profile(
        "a", rows_a, config["factual_operationalization"]
    )
    profile_b = build_hidden_policy_profile(
        "b", rows_b, config["factual_operationalization"]
    )
    result = compare_hidden_profiles(
        profile_a, profile_b, config["hidden_oracle"]["LL-R1"]
    )
    assert result["status"] == "DETERMINATE"
    assert result["label"] == "A"
    assert "objective" not in str(profile_a).lower()


def test_metric_pair_eligibility_is_objective_independent() -> None:
    records = {
        candidate: {
            "aggregate": {"mean_return": value, "mean_length": 100.0 + value},
            "contexts": {},
        }
        for candidate, value in (("a", 0.0), ("b", 0.1), ("c", 4.0), ("d", 8.0))
    }
    result = build_metric_pair_candidates(
        records,
        ["mean_return", "mean_length"],
        nearest_fraction=0.5,
    )
    assert result["all_pair_count"] == 6
    assert result["eligible_pair_count"] == 3
    assert result["eligible_pairs"][0]["policy_a"] == "a"
    assert result["eligible_pairs"][0]["policy_b"] == "b"
    assert "oracle" not in str(result).lower()
