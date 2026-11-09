from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

from behavioral_records.demo import make_demo_trajectories
from behavioral_records.experiment import group_by_policy
from behavioral_records.io import read_json
from behavioral_records.models import Step, Trajectory
from behavioral_records.predictive_record import (
    PredictiveBehaviorCompiler,
    build_predictive_pair_prompt,
    build_predictive_views,
    compile_predictive_profiles,
    predictive_compiler_signature_is_objective_free,
    reconstruct_trace_variant,
)
from behavioral_records.predictive_recoverability import audit_predictive_recoverability
from behavioral_records.views import public_evidence_chars


def _config() -> dict:
    return read_json("configs/tpbr_v0.json")


def _validation_groups():
    return group_by_policy(
        row for row in make_demo_trajectories() if row.context["split"] == "validation"
    )


def _continuous_trajectory(policy_id: str, episode_id: str, offset: float) -> Trajectory:
    steps = []
    for t in range(5):
        reached = t == 4
        steps.append(
            Step(
                t=t,
                action="heat" if t < 3 else "hold",
                reward=-0.1 + (2.0 if reached else 0.0),
                state={
                    "temperature_c": 20.0 + offset + 2.5 * t,
                    "pressure_kpa": 100.0 + t,
                    "valve_open": t < 3,
                },
                predicates={"above_soft_limit": t >= 3},
                events=("setpoint_reached",) if reached else (),
            )
        )
    return Trajectory(
        policy_id=policy_id,
        episode_id=episode_id,
        context={"split": "validation", "ambient_c": 20.0 + offset},
        steps=tuple(steps),
        outcome="controlled",
        total_return=sum(step.reward for step in steps),
        terminated=True,
        truncated=False,
    )


def test_tpbr_compiler_is_objective_free_deterministic_and_rejects_leakage() -> None:
    config = _config()
    groups = _validation_groups()
    forward = compile_predictive_profiles(groups, config)
    reverse = compile_predictive_profiles(
        {policy_id: list(reversed(rows)) for policy_id, rows in groups.items()}, config
    )
    assert predictive_compiler_signature_is_objective_free() is True
    assert "objective" not in inspect.signature(PredictiveBehaviorCompiler.compile).parameters
    assert forward == reverse
    leaked = deepcopy(config)
    leaked["compiler"]["objective"] = "prefer success"
    with pytest.raises(ValueError, match="Forbidden objective-conditioned key"):
        PredictiveBehaviorCompiler(leaked)


def test_tpbr_probability_first_hit_repeat_and_temporal_facts_are_exact() -> None:
    profiles = compile_predictive_profiles(_validation_groups(), _config())
    profile = profiles["reward_loop"]
    symbols = {row["symbol"]: row for row in profile["symbol_profiles"]}
    loop = symbols["event::reward_loop_step"]
    assert loop["occurrence_count"] == 30
    assert loop["episode_count"] == 10
    assert loop["episode_rate"] == 1.0
    assert loop["repeat_rate_ge_3"] == 1.0
    assert dict(loop["first_hit_cdf"])[4] == 1.0
    assert loop["outcome_given_present"] == [["success", 10, 1.0]]
    temporal = {
        (row["left"], row["right"]): row for row in profile["temporal_profiles"]
    }
    relation = temporal[("event::correct_key_picked", "event::goal_reached")]
    assert relation["activations"] == 10
    assert relation["satisfied_eventually"] == 10
    assert relation["response_rate"] == 1.0
    assert dict(relation["within_response_rate"])[16] == 1.0


def test_tpbr_discovers_non_grid_continuous_signals_without_task_code() -> None:
    rows = [
        _continuous_trajectory("controller", "episode_a", 0.0),
        _continuous_trajectory("controller", "episode_b", 1.0),
    ]
    profile = PredictiveBehaviorCompiler(_config()).compile("controller", rows)
    signals = {row["signal"]: row for row in profile["signal_profiles"]}
    assert signals["state.temperature_c"]["kind"] == "numeric"
    assert signals["state.pressure_kpa"]["value"]["max"] == 104.0
    assert signals["state.valve_open"]["kind"] == "boolean"
    assert "event::setpoint_reached" in {
        row["symbol"] for row in profile["symbol_profiles"]
    }
    assert "outcome::controlled" in {
        row["symbol"] for row in profile["symbol_profiles"]
    }


def test_trace_grammar_is_lossless_for_every_selected_variant() -> None:
    profiles = compile_predictive_profiles(_validation_groups(), _config())
    for profile in profiles.values():
        assert profile["loss_ledger"]["trace_selected_reconstruction_verified"] is True
        for variant in profile["trace_variants"]:
            reconstructed = reconstruct_trace_variant(
                variant["runs"], profile["trace_grammar"]
            )
            from behavioral_records.canonical import stable_hash

            assert stable_hash(reconstructed) == variant["source_sequence_hash"]
            assert len(reconstructed) == variant["source_symbol_count"]


def test_fixed_tpbr_view_passes_core_recoverability_and_character_gate() -> None:
    config = _config()
    profiles = compile_predictive_profiles(_validation_groups(), config)
    views = build_predictive_views(profiles, config)
    audit = audit_predictive_recoverability(profiles, views, config)
    assert audit["passed"] is True
    assert audit["aggregate"]["core_coverage_min"] == 1.0
    assert audit["aggregate"]["core_included_accuracy_mean"] == 1.0
    assert audit["aggregate"]["extended_included_accuracy_mean"] == 1.0
    assert all(
        public_evidence_chars(view) <= config["view"]["per_policy_char_budget"]
        for view in views.values()
    )


def test_requirement_changes_prompt_only_before_byte_identical_tpbr_evidence() -> None:
    config = _config()
    profiles = compile_predictive_profiles(_validation_groups(), config)
    views = build_predictive_views(profiles, config)
    first = build_predictive_pair_prompt(
        "Prefer shorter successful behavior.", views["cautious"], views["efficient_clean"]
    )
    second = build_predictive_pair_prompt(
        "Avoid repeated reward events.", views["cautious"], views["efficient_clean"]
    )
    marker = "POLICY A TPBR:"
    assert first.split(marker, 1)[1] == second.split(marker, 1)[1]
    assert "cautious" not in first
    assert "efficient_clean" not in first
    assert "confidence must be a JSON number" in first


def test_compact_event_log_adapter_is_objective_blind_and_retains_episode_signals() -> None:
    rows = [
        {
            "policy_id": "compact_policy",
            "episode_id": "e1",
            "context": {"split": "validation", "slip_prob": 0.05},
            "events": ["key", "door", "goal"],
            "outcome": "success",
            "return": 9.5,
            "length": 17,
            "terminated": True,
            "truncated": False,
        },
        {
            "policy_id": "compact_policy",
            "episode_id": "e2",
            "context": {"split": "validation", "slip_prob": 0.1},
            "events": ["key", "loop", "loop", "goal"],
            "outcome": "success",
            "return": 7.0,
            "length": 23,
            "terminated": True,
            "truncated": False,
        },
    ]
    compiler = PredictiveBehaviorCompiler(_config())
    profile = compiler.compile_episode_evidence("compact_policy", rows)
    symbols = {row["symbol"]: row for row in profile["symbol_profiles"]}
    assert profile["sample"]["episodes"] == 2
    assert profile["sample"]["steps"] == 40
    assert symbols["event::loop"]["occurrence_count"] == 2
    assert "ordered event positions" in profile["loss_ledger"]["semantic_adapter"]
    with pytest.raises(ValueError, match="Forbidden objective-conditioned input"):
        compiler.compile_episode_evidence("compact_policy", [{**rows[0], "objective": "goal"}])
