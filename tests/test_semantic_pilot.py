from __future__ import annotations

from collections import Counter

from behavioral_records.experiment import group_by_policy
from behavioral_records.io import read_json, read_jsonl
from behavioral_records.semantic_pilot import (
    behavioral_test_profile,
    build_reward_recovery_pair_blueprint,
    causal_profile,
    evaluate_semantic_judge_responses,
    make_reward_recovery_trajectories,
    run_reward_recovery_semantic_pilot,
    standard_metric_vector,
    strict_parse_semantic_response,
)


def _groups():
    trajectories, blueprint = make_reward_recovery_trajectories()
    validation = group_by_policy(
        row for row in trajectories if row.context["split"] == "validation"
    )
    oracle = group_by_policy(
        row for row in trajectories if row.context["split"] == "oracle"
    )
    return blueprint, validation, oracle


def test_validation_pairs_match_all_declared_ordinary_metrics_exactly() -> None:
    blueprint, validation, _ = _groups()
    for pair in blueprint:
        left = standard_metric_vector(validation[pair["policy_a"]])
        right = standard_metric_vector(validation[pair["policy_b"]])
        assert left == right


def test_hidden_interventions_separate_reward_triggered_and_recovery_mechanisms() -> None:
    blueprint, _, oracle = _groups()
    pair = blueprint[0]
    reward_triggered = causal_profile(oracle[pair["policy_a"]])
    recovery = causal_profile(oracle[pair["policy_b"]])
    assert reward_triggered["vector"] == [False, False, False]
    assert recovery["vector"] == [True, True, True]
    reward_behavior = behavioral_test_profile(oracle[pair["policy_a"]])
    recovery_behavior = behavioral_test_profile(oracle[pair["policy_b"]])
    assert all(value is False for value in reward_behavior["vector"])
    assert all(value is True for value in recovery_behavior["vector"])


def test_blueprint_contains_directional_tie_and_ambiguous_controls() -> None:
    roles = Counter(row["design_role"] for row in build_reward_recovery_pair_blueprint())
    assert roles == {
        "directional_contrast": 8,
        "semantic_tie": 2,
        "designed_ambiguity": 2,
    }


def test_offline_semantic_pilot_passes_g1_g2_g3_and_builds_blinded_requests(tmp_path) -> None:
    output = tmp_path / "semantic_pilot"
    summary = run_reward_recovery_semantic_pilot(
        output_dir=output,
        tpbr_config_path="configs/tpbr_v0.json",
        specification_card_path=(
            "configs/semantic_cards/reward_exploitation_vs_recovery_v1.json"
        ),
    )
    assert summary["status"] == "READY_FOR_LLM_SCREENING"
    assert summary["gates"] == {
        "G1_oracle_validity": True,
        "G2_metric_matched_hard_pair_validity": True,
        "G3_representation_answerability": True,
        "G4_static_semantic_selection": "NOT_RUN",
    }
    assert summary["oracle_status_counts"] == {"AMBIGUOUS": 2, "DETERMINATE": 10}
    assert summary["oracle_label_counts"] == {"A": 4, "B": 4, "TIE": 2}
    assert summary["metric_probe_balanced_accuracy_upper_bound"] == 0.5
    assert summary["judge_request_count"] == 96

    requests = read_jsonl(output / "judge_requests.jsonl")
    private_rows = read_jsonl(output / "judge_request_private_map.jsonl")
    assert len(requests) == len(private_rows) == 96
    private_policy_ids = {
        row["alias_a_policy_id"] for row in private_rows
    } | {row["alias_b_policy_id"] for row in private_rows}
    assert all(
        policy_id not in request["prompt"]
        for request in requests
        for policy_id in private_policy_ids
    )
    assert all("expected_winner" not in request for request in requests)
    metric_audit = read_json(output / "metric_match_audit.json")
    assert metric_audit["all_pairs_exactly_matched"] is True


def test_semantic_parser_supports_abstention_and_requires_evidence_refs() -> None:
    parsed = strict_parse_semantic_response(
        '{"winner":"ABSTAIN","confidence":0.4,"evidence_refs":["temporal"],"reason":"insufficient"}'
    )
    assert parsed["winner"] == "ABSTAIN"


def test_semantic_evaluation_uses_strict_both_order_cells(tmp_path) -> None:
    output = tmp_path / "semantic_pilot"
    run_reward_recovery_semantic_pilot(
        output_dir=output,
        tpbr_config_path="configs/tpbr_v0.json",
        specification_card_path=(
            "configs/semantic_cards/reward_exploitation_vs_recovery_v1.json"
        ),
    )
    private_rows = read_jsonl(output / "judge_request_private_map.jsonl")
    responses = [
        {
            "request_id": row["request_id"],
            "response": {
                "winner": row["expected_winner"],
                "confidence": 1.0,
                "evidence_refs": ["test"],
                "reason": "fixture",
            },
        }
        for row in private_rows
    ]
    result = evaluate_semantic_judge_responses(
        private_request_map=private_rows,
        responses=responses,
    )
    assert all(row["strict_accuracy"] == 1.0 for row in result["by_arm"].values())
    assert all(
        row["exact_mcnemar_p"] == 1.0
        for row in result["paired_comparisons"].values()
    )
    evaluate_semantic_judge_responses,
