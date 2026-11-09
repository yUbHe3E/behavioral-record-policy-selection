from __future__ import annotations

from behavioral_records.lunarlander_reanalysis import (
    build_candidate_chain_map,
    summarize_pair_coverage,
    summarize_selected_pair_reuse,
    training_chain_group_disjoint_cv,
)


def _candidate_rows() -> list[dict]:
    rows = []
    for seed in (1, 2, 3):
        for profile in ("P0", "P1"):
            for checkpoint in (100, 200):
                rows.append(
                    {
                        "candidate_id": f"c_{profile}_{seed}_{checkpoint}",
                        "training_profile_private": profile,
                        "training_seed_private": seed,
                        "checkpoint_timestep_private": checkpoint,
                    }
                )
    return rows


def _pair_rows(requirement_id: str = "R1") -> list[dict]:
    candidates = [row["candidate_id"] for row in _candidate_rows()]
    rows = []
    pair_index = 0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            label = "A" if pair_index % 2 == 0 else "B"
            rows.append(
                {
                    "requirement_id": requirement_id,
                    "pair_id": f"pair_{pair_index}",
                    "policy_a": left,
                    "policy_b": right,
                    "oracle_status": "DETERMINATE",
                    "oracle_label": label,
                    "oracle_detail_private": {"decisive_field": "temporal_relation"},
                    "oriented_metric_difference": [
                        1.0 if label == "A" else -1.0,
                        float(pair_index % 3),
                    ],
                }
            )
            pair_index += 1
    return rows


def test_candidate_chain_map_keeps_checkpoints_in_profile_seed_chain() -> None:
    chain_map = build_candidate_chain_map(_candidate_rows())
    assert len(chain_map) == 12
    assert len({row["training_chain_id"] for row in chain_map.values()}) == 6
    assert (
        chain_map["c_P0_1_100"]["training_chain_id"]
        == chain_map["c_P0_1_200"]["training_chain_id"]
    )
    assert (
        chain_map["c_P0_1_100"]["training_chain_id"]
        != chain_map["c_P1_1_100"]["training_chain_id"]
    )


def test_group_disjoint_probe_reports_exact_test_coverage_without_inference() -> None:
    rows = _pair_rows()
    chain_map = build_candidate_chain_map(_candidate_rows())
    result = training_chain_group_disjoint_cv(
        rows,
        chain_map,
        {"iterations": 200, "learning_rate": 0.05, "l2": 1.0},
    )
    assert result["status"] == "EVALUATED_DESCRIPTIVE_ONLY"
    assert result["inference_authorized"] is False
    assert result["eligible_determinate_pairs"] == 66
    assert result["unique_test_pair_ids"] == 18
    assert result["test_pair_coverage_of_determinate"] == 18 / 66
    assert result["unique_test_policies"] == 12
    assert result["test_policy_coverage_of_determinate"] == 1.0
    assert result["evaluated_folds"] == 3
    for fold in result["folds"]:
        assert fold["test_pairs"] == 6
        assert fold["train_test_policy_overlap"] == []
        assert fold["train_test_training_chain_overlap"] == []


def test_coverage_and_reuse_have_explicit_pair_and_policy_denominators() -> None:
    all_r1 = _pair_rows("R1")
    all_r2 = _pair_rows("R2")
    chain_map = build_candidate_chain_map(_candidate_rows())
    coverage = summarize_pair_coverage(all_r1, chain_map)
    assert coverage["eligible_pair_rows"] == 66
    assert coverage["determinate_pair_rows"] == 66
    assert coverage["determinate_unique_policies"] == 12
    assert coverage["determinate_training_chains"] == 6

    selected_r1 = all_r1[:5]
    selected_r2 = all_r2[:5]
    reuse = summarize_selected_pair_reuse(
        {"R1": selected_r1, "R2": selected_r2},
        chain_map,
        {"R1": all_r1, "R2": all_r2},
    )
    assert reuse["by_requirement"]["R1"]["selected_requirement_pair_units"] == 5
    assert reuse["by_requirement"]["R1"]["maximum_pair_appearances_for_one_policy"] == 5
    assert reuse["combined_across_requirements"][
        "maximum_requirement_pair_appearances_for_one_policy"
    ] == 10
    assert reuse["combined_across_requirements"]["unique_physical_pair_ids"] == 5
    assert reuse["combined_across_requirements"][
        "physical_pair_ids_reused_across_requirements"
    ] == 5
