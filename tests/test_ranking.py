from __future__ import annotations

from behavioral_records.ranking import aggregate_pairwise_judgments, rank_pairwise_consensus


def test_ab_ba_consensus_produces_a_fixed_final_ranking() -> None:
    requests = []
    responses = []
    winners = {("alpha", "beta"): "alpha", ("alpha", "gamma"): "alpha", ("beta", "gamma"): "beta"}
    for pair_index, ((left, right), preferred) in enumerate(winners.items()):
        for order in ("AB", "BA"):
            policy_a, policy_b = (left, right) if order == "AB" else (right, left)
            request_id = f"request_{pair_index}_{order}"
            requests.append(
                {
                    "request_id": request_id,
                    "pair_id": f"pair_{pair_index}",
                    "objective_id": "complex_requirement",
                    "evidence_arm": "P_compact_behavioral_record",
                    "order": order,
                    "policy_a_id": policy_a,
                    "policy_b_id": policy_b,
                }
            )
            responses.append(
                {
                    "request_id": request_id,
                    "winner": "A" if policy_a == preferred else "B",
                    "confidence": 0.9,
                    "reason": "synthetic test response",
                }
            )

    consensus = aggregate_pairwise_judgments(requests, responses)
    assert len(consensus) == 3
    assert all(row["order_consistent"] for row in consensus)
    ranking = rank_pairwise_consensus(consensus)
    assert [row["policy_id"] for row in ranking] == ["alpha", "beta", "gamma"]
    assert ranking[0]["selected"] is True
    assert ranking[0]["response_graph_connected"] is True
