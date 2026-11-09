from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from statistics import fmean
from typing import Any, Mapping, Sequence


def exact_two_sided_sign_flip_p(values: Sequence[float]) -> float:
    """Exact paired randomization p-value, dropping only numerical zeros."""
    nonzero = [float(value) for value in values if not math.isclose(float(value), 0.0)]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, nonzero)))
        extreme += statistic + 1e-12 >= observed
        total += 1
    return extreme / total


def seed_cluster_bootstrap_ci(
    values: Sequence[float], *, draws: int = 10_000, seed: int = 20260822
) -> list[float]:
    if not values:
        raise ValueError("Cluster bootstrap requires at least one seed effect")
    rng = random.Random(seed)
    estimates = sorted(
        fmean(float(values[rng.randrange(len(values))]) for _ in values)
        for _ in range(draws)
    )
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]]


def evaluate_controlled_outer_confirmatory(
    *,
    selection_rows: Sequence[Mapping[str, Any]],
    stage_summary: Mapping[str, Any],
    frozen_controls: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    design = plan["design"]
    analysis = plan["primary_analysis"]
    expected_seeds = [int(value) for value in design["outer_seeds"]]
    expected_scenarios = [str(value) for value in design["scenarios"]]
    expected_arms = [str(value) for value in design["selector_arms"]]
    expected_generations = int(design["generations"])
    endpoint_generation = int(analysis["endpoint_generation"])

    design_matches = (
        int(frozen_controls.get("population", -1)) == int(design["population"])
        and int(frozen_controls.get("parents", -1)) == int(design["parents"])
        and int(frozen_controls.get("generations", -1)) == expected_generations
        and [int(value) for value in frozen_controls.get("outer_seeds", [])]
        == expected_seeds
        and list(frozen_controls.get("arms", [])) == expected_arms
        and frozen_controls.get("response_format_type")
        == plan["judge_freeze"]["response_format_type"]
        and int(frozen_controls.get("max_concurrency", -1))
        == int(plan["judge_freeze"]["max_concurrency"])
    )

    index: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    duplicate_cells: list[list[Any]] = []
    for row in selection_rows:
        key = (
            str(row["scenario_id"]),
            int(row["outer_seed"]),
            int(row["generation"]),
            str(row["selector_arm"]),
        )
        if key in index:
            duplicate_cells.append(list(key))
        index[key] = row
    expected_keys = {
        (scenario, seed, generation, arm)
        for scenario in expected_scenarios
        for seed in expected_seeds
        for generation in range(expected_generations)
        for arm in expected_arms
    }
    observed_keys = set(index)
    complete_cells = not duplicate_cells and observed_keys == expected_keys

    stage_gates = dict(stage_summary.get("gates", {}))
    all_stage_gates = bool(stage_gates) and all(bool(value) for value in stage_gates.values())
    execution_valid = design_matches and complete_cells and all_stage_gates

    primary_seed_effects: dict[str, float] = {}
    scenario_effects: dict[str, list[float]] = defaultdict(list)
    if complete_cells:
        for seed in expected_seeds:
            within_seed = []
            for scenario in expected_scenarios:
                b1 = float(
                    index[(scenario, seed, endpoint_generation, "B1-LLM-PBT")][
                        "selected_parent_mean_semantic_violation_rate"
                    ]
                )
                b2 = float(
                    index[(scenario, seed, endpoint_generation, "B2-FullTrace-LLM-PBT")][
                        "selected_parent_mean_semantic_violation_rate"
                    ]
                )
                effect = b1 - b2
                scenario_effects[scenario].append(effect)
                within_seed.append(effect)
            primary_seed_effects[str(seed)] = fmean(within_seed)

    values = [primary_seed_effects[str(seed)] for seed in expected_seeds] if complete_cells else []
    primary = {
        "endpoint_generation": endpoint_generation,
        "effect_definition": analysis["contrast"],
        "positive_value_favors": analysis["positive_value_favors"],
        "outer_seed_cluster_count": len(values),
        "outer_seed_cluster_effects": primary_seed_effects,
        "mean_effect": fmean(values) if values else None,
        "bootstrap_95_ci": (
            seed_cluster_bootstrap_ci(values, seed=int(analysis["bootstrap_seed"]))
            if values
            else None
        ),
        "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(values) if values else None,
        "alpha": float(analysis["alpha"]),
    }

    endpoint_by_arm: dict[str, Any] = {}
    if complete_cells:
        for arm in expected_arms:
            rows = [
                index[(scenario, seed, endpoint_generation, arm)]
                for scenario in expected_scenarios
                for seed in expected_seeds
            ]
            violations = [
                float(row["selected_parent_mean_semantic_violation_rate"])
                for row in rows
            ]
            endpoint_by_arm[arm] = {
                "mean_semantic_violation_rate": fmean(violations),
                "mean_semantic_pass_rate": 1.0 - fmean(violations),
            }

    scenario_summary = {
        scenario: {
            "mean_B1_minus_B2_semantic_violation_rate": fmean(values_for_scenario),
            "outer_seed_effects": values_for_scenario,
            "inference_status": "SECONDARY_DESCRIPTIVE",
        }
        for scenario, values_for_scenario in scenario_effects.items()
    }
    support_checks = {
        "execution_and_quality_gates_pass": execution_valid,
        "primary_mean_effect_strictly_positive": bool(values)
        and float(primary["mean_effect"]) > 0.0,
        "primary_exact_p_below_alpha": bool(values)
        and float(primary["exact_two_sided_sign_flip_p"]) < float(primary["alpha"]),
        "C1_descriptive_effect_strictly_positive": bool(scenario_effects.get("C1"))
        and fmean(scenario_effects["C1"]) > 0.0,
        "C2_descriptive_effect_strictly_positive": bool(scenario_effects.get("C2"))
        and fmean(scenario_effects["C2"]) > 0.0,
    }
    if not execution_valid:
        conclusion = "INVALID_CONFIRMATORY_RUN"
    elif all(support_checks.values()):
        conclusion = "FROZEN_PRIMARY_CLAIM_SUPPORTED"
    else:
        conclusion = "FROZEN_PRIMARY_CLAIM_NOT_SUPPORTED"

    return {
        "schema_version": "controlled-outer-confirmatory-evaluation-v3",
        "status": conclusion,
        "primary": primary,
        "support_checks": support_checks,
        "scenario_specific_secondary": scenario_summary,
        "endpoint_by_arm_secondary": endpoint_by_arm,
        "data_integrity": {
            "design_matches_frozen_plan": design_matches,
            "complete_unique_design_cells": complete_cells,
            "expected_cell_count": len(expected_keys),
            "observed_unique_cell_count": len(observed_keys),
            "missing_cells": [list(key) for key in sorted(expected_keys - observed_keys)],
            "unexpected_cells": [list(key) for key in sorted(observed_keys - expected_keys)],
            "duplicate_cells": duplicate_cells,
            "all_stage_gates_pass": all_stage_gates,
            "stage_gates": stage_gates,
            "previous_pilot_in_primary_inference": False,
        },
        "claim_boundary": plan["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "confirmatory_evaluation",
            "verification_status": "VERIFIED_AGAINST_FROZEN_PLAN" if execution_valid else "INVALID",
            "version_label": "controlled_outer_confirmatory_v3",
        },
    }
