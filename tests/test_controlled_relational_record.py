from __future__ import annotations

from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(not Path("outputs/controlled_confirmatory_20260822_v1/C1/public_evidence.jsonl").exists(), reason="Extract the evidence release to run frozen-input integration tests")

from behavioral_records.io import read_json, read_jsonl
from behavioral_records.relational_record import (
    RELATIONAL_ARMS,
    build_controlled_relational_record,
    evaluate_controlled_relational_record,
)


def test_controlled_relational_bundle_passes_all_offline_gates(tmp_path) -> None:
    result = build_controlled_relational_record(
        output_dir=tmp_path,
        plan_path="configs/controlled_relational_record_v1.json",
        repository_root=".",
    )
    assert result["status"] == "READY_FOR_RELATIONAL_JUDGE"
    assert result["pair_count"] == 96
    assert result["request_count"] == 3072
    assert set(result["arms"]) == set(RELATIONAL_ARMS)
    audit = read_json(tmp_path / "relational_integrity_audit.json")
    assert audit["passed"]
    assert all(audit["gates"].values())
    assert len(read_jsonl(tmp_path / "judge_requests.jsonl")) == 3072
    assert build_controlled_relational_record(
        output_dir=tmp_path,
        plan_path="configs/controlled_relational_record_v1.json",
        repository_root=".",
    ) == result


def test_controlled_relational_evaluation_uses_pair_level_primary_contrast(
    tmp_path,
) -> None:
    build_controlled_relational_record(
        output_dir=tmp_path,
        plan_path="configs/controlled_relational_record_v1.json",
        repository_root=".",
    )
    private = read_jsonl(tmp_path / "judge_private_map.jsonl")
    responses = []
    for row in private:
        winner = (
            row["expected_winner"]
            if row["arm"] in {"Relational-Linked", "FullTrace-Control"}
            else "ABSTAIN"
        )
        responses.append(
            {
                "request_id": row["request_id"],
                "response": {
                    "winner": winner,
                    "confidence": 0.8,
                    "evidence_refs": ["test"],
                    "reason": "synthetic test response",
                },
            }
        )
    result = evaluate_controlled_relational_record(
        private_map=private,
        responses=responses,
        plan=read_json("configs/controlled_relational_record_v1.json"),
    )
    assert result["status"] == "RELATIONAL_EXPERIMENT_COMPLETE"
    assert all(result["quality_gates"].values())
    assert result["contrasts"]["primary_relational_link_contribution"][
        "mean_difference"
    ] > 0
    assert result["support"]["relational_link_contribution_supported"]

    linked_only = [
        response
        for response, private_row in zip(responses, private)
        if private_row["arm"] == "Relational-Linked"
    ]
    failed_quality = evaluate_controlled_relational_record(
        private_map=private,
        responses=linked_only,
        plan=read_json("configs/controlled_relational_record_v1.json"),
    )
    assert failed_quality["support"]["primary_statistical_condition_met"]
    assert not failed_quality["quality_gates"][
        "minimum_scenario_arm_valid_rate_met"
    ]
    assert not failed_quality["support"][
        "relational_link_contribution_supported"
    ]
