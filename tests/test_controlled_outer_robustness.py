from __future__ import annotations

from behavioral_records.controlled_outer_robustness import (
    apply_exact_score_tie_break,
    difference_in_differences,
    rerank_frozen_outer_cell,
    shared_seed_hash_tie_key,
)


def test_shared_seed_hash_tie_break_is_deterministic_and_not_arm_conditioned() -> None:
    first = shared_seed_hash_tie_key(
        scenario="C1", outer_seed=5, generation=0, policy_id="slot_00"
    )
    assert first == shared_seed_hash_tie_key(
        scenario="C1", outer_seed=5, generation=0, policy_id="slot_00"
    )
    assert first != shared_seed_hash_tie_key(
        scenario="C1", outer_seed=6, generation=0, policy_id="slot_00"
    )


def test_exact_score_tie_break_preserves_unequal_score_order() -> None:
    rows = [
        {"policy_id": "z", "score": 1.0, "response_graph_connected": True},
        {"policy_id": "b", "score": 0.0, "response_graph_connected": True},
        {"policy_id": "a", "score": 0.0, "response_graph_connected": True},
    ]
    legacy = apply_exact_score_tie_break(
        rows, mode="policy_id", scenario="C1", outer_seed=5, generation=0
    )
    hashed = apply_exact_score_tie_break(
        rows, mode="shared_seed_hash", scenario="C1", outer_seed=5, generation=0
    )
    assert legacy[0]["policy_id"] == hashed[0]["policy_id"] == "z"
    assert [row["policy_id"] for row in legacy[1:]] == ["a", "b"]
    assert {row["policy_id"] for row in hashed[1:]} == {"a", "b"}


def test_frozen_cell_reranking_uses_heldout_rows_only_after_ranking() -> None:
    private = []
    responses = []
    for edge_index, (left, right) in enumerate((("p0", "p1"), ("p1", "p2"))):
        for order in ("AB", "BA"):
            alias_a, alias_b = (left, right) if order == "AB" else (right, left)
            request_id = f"r{edge_index}_{order}"
            private.append(
                {
                    "request_id": request_id,
                    "scenario_id": "C1",
                    "outer_seed": 5,
                    "generation": 0,
                    "selector_arm": "B2-FullTrace-LLM-PBT",
                    "edge_index": edge_index,
                    "left_policy_id": left,
                    "right_policy_id": right,
                    "alias_a_policy_id": alias_a,
                    "alias_b_policy_id": alias_b,
                }
            )
            responses.append(
                {"request_id": request_id, "response": {"winner": "A" if order == "AB" else "B"}}
            )
    oracle = [
        {
            "policy_id": policy_id,
            "semantic_violation_rate": violation,
            "heldout_success_rate": 1.0,
        }
        for policy_id, violation in (("p0", 0.0), ("p1", 0.5), ("p2", 1.0))
    ]
    result = rerank_frozen_outer_cell(
        private_rows=private,
        responses=responses,
        oracle_rows=oracle,
        parent_count=1,
        regularization=0.1,
        probability_clip=0.05,
        tie_break_mode="shared_seed_hash",
    )
    assert result["selected_policy_ids"] == ["p0"]
    assert result["selected_parent_mean_semantic_violation_rate"] == 0.0
    assert result["valid_request_rate"] == 1.0


def test_difference_in_differences_is_seed_paired_and_same_for_pass_or_violation() -> None:
    rows = []
    for seed in (5, 6):
        for scenario in ("C1", "C2"):
            for generation in (0, 2):
                for arm in ("B1-LLM-PBT", "B2-FullTrace-LLM-PBT"):
                    violation = 0.5
                    if generation == 2 and arm == "B2-FullTrace-LLM-PBT":
                        violation = 0.1
                    rows.append(
                        {
                            "scenario_id": scenario,
                            "outer_seed": seed,
                            "generation": generation,
                            "selector_arm": arm,
                            "selected_parent_mean_semantic_violation_rate": violation,
                        }
                    )
    result = difference_in_differences(
        rows, scenarios=["C1", "C2"], outer_seeds=[5, 6]
    )
    assert result["mean_effect"] == 0.4
    assert result["outer_seed_cluster_effects"] == {"5": 0.4, "6": 0.4}
    assert result["positive_value_favors"] == "B2-FullTrace-LLM-PBT"
