from __future__ import annotations

import pytest

from behavioral_records.io import read_json
from behavioral_records.lunarlander_transfer import (
    ACTION_LABELS,
    STATE_FIELDS,
    assert_lunarlander_objective_blind,
    build_candidate_blueprint,
    compile_lunarlander_step,
    lunarlander_state,
    probe_lunarlander_environment,
    validate_lunarlander_transfer_config,
)
from behavioral_records.lunarlander_training import (
    evaluation_seeds,
    rollout_lunarlander_policy,
    summarize_validation_trajectories,
    training_candidate_lookup,
    validate_lunarlander_training_config,
)


@pytest.fixture
def config() -> dict:
    return read_json("configs/lunarlander_transfer_v1.json")


def test_lunarlander_state_has_fixed_nonvisual_schema() -> None:
    state = lunarlander_state([0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 1.0, 0.0])
    assert tuple(state) == STATE_FIELDS
    assert state["left_leg_contact"] is True
    assert state["right_leg_contact"] is False


def test_lunarlander_step_is_factual_and_deterministic() -> None:
    kwargs = {
        "previous_observation": [0.1, 0.2, -0.3, -0.4, -0.2, 0.1, 0.0, 0.0],
        "observation": [0.1, 0.18, 0.2, -0.1, 0.1, 0.05, 1.0, 0.0],
        "action": 1,
        "reward": -0.25,
        "terminated": False,
        "truncated": False,
    }
    first = compile_lunarlander_step(**kwargs)
    second = compile_lunarlander_step(**kwargs)
    assert first == second
    assert first["action"] == "left_orientation_engine"
    assert set(ACTION_LABELS) == {
        "no_engine",
        "left_orientation_engine",
        "main_engine",
        "right_orientation_engine",
    }
    assert first["events"] == [
        "horizontal_velocity_sign_change",
        "angle_sign_change",
        "left_leg_contact_on",
    ]
    assert first["predicates"]["orientation_engine_commanded"] is True
    assert_lunarlander_objective_blind(first)


def test_lunarlander_public_guard_rejects_evaluative_label() -> None:
    with pytest.raises(ValueError, match="leaked"):
        assert_lunarlander_objective_blind({"label": "safe_landing"})


def test_transfer_config_and_candidate_blueprint_are_frozen(config: dict) -> None:
    validation = validate_lunarlander_transfer_config(config)
    assert validation["passed"] is True
    rows = build_candidate_blueprint(config)
    assert len(rows) == 45
    assert len({row["candidate_id"] for row in rows}) == 45
    assert all(row["training_reward"] == "unmodified_official_environment_reward_only" for row in rows)
    assert all(row["public_identity_allowed"] is False for row in rows)


def test_official_lunarlander_probe_when_box2d_is_available(config: dict) -> None:
    pytest.importorskip("Box2D")
    report = probe_lunarlander_environment(config, seeds=(6201,))
    assert report["status"] == "PASS_INTERFACE"
    assert report["seeded_determinism"] is True
    assert report["objective_blind_adapter"] == "PASS"


def test_training_protocol_matches_frozen_candidate_blueprint(config: dict) -> None:
    training_config = read_json("configs/lunarlander_policy_pool_training_v1.json")
    report = validate_lunarlander_training_config(training_config, config)
    assert report["passed"] is True
    assert report["rollout_size"] == 5000
    assert report["expected_training_chains"] == 9
    assert report["expected_candidates"] == 45
    assert report["validation_seed_hash"] != report["heldout_seed_hash"]
    lookup = training_candidate_lookup(config)
    assert len(lookup) == 45
    assert ("LL-C0", 5101, 50000) in lookup


def test_validation_seeds_are_frozen_and_smoke_only_reduces_count() -> None:
    training_config = read_json("configs/lunarlander_policy_pool_training_v1.json")
    formal = evaluation_seeds(training_config)
    smoke = evaluation_seeds(training_config, smoke=True)
    assert formal["LL-C0"] == list(range(710000, 710016))
    assert smoke["LL-C0"] == [710000, 710001]
    assert set(formal) == {"LL-C0", "LL-C1", "LL-C2"}


def test_objective_blind_lunarlander_rollout_and_summary(config: dict) -> None:
    pytest.importorskip("Box2D")

    class NoEnginePolicy:
        @staticmethod
        def predict(observation, deterministic=True):
            return 0, None

    trajectory = rollout_lunarlander_policy(
        model=NoEnginePolicy(),
        transfer_config=config,
        policy_id="opaque_candidate",
        context_id="LL-C0",
        seed=770001,
    )
    assert trajectory.steps
    assert trajectory.context["context_id"] == "LL-C0"
    assert trajectory.outcome in {"terminated", "truncated"}
    assert "training_profile" not in str(trajectory.to_dict()).lower()
    assert "oracle" not in str(trajectory.to_dict()).lower()
    summary = summarize_validation_trajectories([trajectory])
    assert summary["policy_id"] == "opaque_candidate"
    assert summary["episodes"] == 1
    assert len(summary["behavior_signature"]) == 20
