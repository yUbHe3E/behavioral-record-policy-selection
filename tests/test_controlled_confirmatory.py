from __future__ import annotations

from behavioral_records.controlled_confirmatory import (
    PRIMARY_ARMS,
    build_controlled_confirmatory,
    evaluate_controlled_static_responses,
)
from behavioral_records.io import read_json, read_jsonl


def test_controlled_confirmatory_passes_offline_gates(tmp_path) -> None:
    result = build_controlled_confirmatory(
        output_dir=tmp_path,
        config_path="configs/controlled_scenarios_confirmatory_v1.json",
    )
    assert result["status"] == "READY_FOR_STATIC_JUDGE"
    assert result["total_request_count"] == 2304
    for scenario in ("C1", "C2"):
        summary = read_json(tmp_path / scenario / "summary.json")
        assert summary["pair_count"] == 48
        assert summary["oracle_status_counts"] == {"AMBIGUOUS": 7, "DETERMINATE": 41}
        assert summary["oracle_label_counts"] == {"A": 17, "B": 17, "TIE": 7}
        assert summary["unique_pair_fingerprint_count"] == 48
        assert all(summary["gates"].values())
        assert read_json(tmp_path / scenario / "metric_match_audit.json")[
            "all_pairs_exactly_matched"
        ]
        requests = read_jsonl(tmp_path / scenario / "judge_requests.jsonl")
        assert len(requests) == 1152
        assert {row["arm"] for row in requests} == set(PRIMARY_ARMS)


def test_controlled_static_evaluation_uses_pair_as_unit(tmp_path) -> None:
    build_controlled_confirmatory(
        output_dir=tmp_path,
        config_path="configs/controlled_scenarios_confirmatory_v1.json",
    )
    private = read_jsonl(tmp_path / "judge_private_map.jsonl")
    responses = []
    for row in private:
        winner = row["expected_winner"] if row["arm"] == "B2-FullTrace" else "ABSTAIN"
        responses.append(
            {
                "request_id": row["request_id"],
                "response": {
                    "winner": winner,
                    "confidence": 0.8,
                    "evidence_refs": ["e1"],
                    "reason": "test",
                },
            }
        )
    result = evaluate_controlled_static_responses(
        private_map=private,
        responses=responses,
    )
    assert result["expected_request_count"] == 2304
    assert result["combined_B2_minus_B1"]["mean_difference"] > 0.0
    for scenario in ("C1", "C2"):
        assert result["by_scenario"][scenario]["by_arm"]["B2-FullTrace"][
            "determinate_pair_count"
        ] == 41
