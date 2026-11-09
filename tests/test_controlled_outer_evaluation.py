from __future__ import annotations

from behavioral_records.controlled_outer_evaluation import (
    evaluate_controlled_outer_confirmatory,
    exact_two_sided_sign_flip_p,
)


def test_exact_sign_flip_is_two_sided_and_exact() -> None:
    assert exact_two_sided_sign_flip_p([1.0] * 8) == 2 / 256
    assert exact_two_sided_sign_flip_p([0.0] * 8) == 1.0


def test_confirmatory_evaluation_aggregates_scenarios_within_seed() -> None:
    seeds = list(range(5, 13))
    scenarios = ["C1", "C2"]
    arms = [
        "B1-LLM-PBT",
        "B2-FullTrace-LLM-PBT",
        "Metric-PBT",
        "Random-PBT",
        "Oracle-PBT",
    ]
    rows = []
    for generation in range(3):
        for scenario in scenarios:
            for seed in seeds:
                for arm in arms:
                    violation = 0.5
                    if arm == "B1-LLM-PBT":
                        violation = 0.6
                    elif arm == "B2-FullTrace-LLM-PBT":
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
    plan = {
        "design": {
            "population": 16,
            "parents": 4,
            "generations": 3,
            "outer_seeds": seeds,
            "scenarios": scenarios,
            "selector_arms": arms,
        },
        "judge_freeze": {"response_format_type": "json_object", "max_concurrency": 1},
        "primary_analysis": {
            "endpoint_generation": 2,
            "contrast": "B1 minus B2",
            "positive_value_favors": "B2",
            "bootstrap_seed": 20260822,
            "alpha": 0.05,
        },
        "claim_boundary": "frozen primary only",
    }
    controls = {
        "population": 16,
        "parents": 4,
        "generations": 3,
        "outer_seeds": seeds,
        "arms": arms,
        "response_format_type": "json_object",
        "max_concurrency": 1,
    }
    result = evaluate_controlled_outer_confirmatory(
        selection_rows=rows,
        stage_summary={"gates": {"complete": True}},
        frozen_controls=controls,
        plan=plan,
    )
    assert result["status"] == "FROZEN_PRIMARY_CLAIM_SUPPORTED"
    assert result["primary"]["mean_effect"] == 0.5
    assert result["primary"]["exact_two_sided_sign_flip_p"] == 2 / 256
    assert result["data_integrity"]["previous_pilot_in_primary_inference"] is False
