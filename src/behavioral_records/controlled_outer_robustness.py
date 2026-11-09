from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import stable_hash
from .controlled_outer import _winner_probability_for_left
from .controlled_outer_evaluation import (
    exact_two_sided_sign_flip_p,
    seed_cluster_bootstrap_ci,
)
from .ranking import rank_pairwise_consensus


LLM_SELECTOR_ARMS = ("B1-LLM-PBT", "B2-FullTrace-LLM-PBT")
TIE_BREAK_MODES = ("policy_id", "shared_seed_hash")


def shared_seed_hash_tie_key(
    *,
    scenario: str,
    outer_seed: int,
    generation: int,
    policy_id: str,
) -> str:
    """Return an arm-independent deterministic priority for an exact score tie.

    The selector arm is intentionally absent.  Policies occupying the same named
    slot therefore receive the same tie priority in B1 and B2, avoiding a
    representation-specific secondary selection rule.
    """

    return stable_hash(
        [
            "controlled-outer-shared-exact-tie-v1",
            str(scenario),
            int(outer_seed),
            int(generation),
            str(policy_id),
        ]
    )


def apply_exact_score_tie_break(
    ranking: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    scenario: str,
    outer_seed: int,
    generation: int,
) -> list[dict[str, Any]]:
    """Reorder only policies with exactly equal fitted ranking scores."""

    if mode not in TIE_BREAK_MODES:
        raise ValueError(f"Unsupported exact-score tie-break mode: {mode}")

    def secondary(row: Mapping[str, Any]) -> str:
        policy_id = str(row["policy_id"])
        if mode == "policy_id":
            return policy_id
        return shared_seed_hash_tie_key(
            scenario=scenario,
            outer_seed=outer_seed,
            generation=generation,
            policy_id=policy_id,
        )

    ordered = sorted(
        (dict(row) for row in ranking),
        key=lambda row: (-float(row["score"]), secondary(row)),
    )
    for position, row in enumerate(ordered, start=1):
        row["rank"] = position
        row["selected"] = position == 1 and bool(row.get("response_graph_connected", True))
        row["exact_score_tie_break"] = mode
    return ordered


def build_frozen_outer_consensus(
    private_rows: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the original AB/BA-neutralized pairwise consensus offline."""

    response_by_id = {str(row["request_id"]): row for row in responses}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    validity: list[bool] = []
    for raw in private_rows:
        row = dict(raw)
        probability, valid, physical = _winner_probability_for_left(
            row, response_by_id.get(str(row["request_id"]))
        )
        validity.append(valid)
        groups[int(row["edge_index"])].append(
            {**row, "left_probability": probability, "valid": valid, "physical": physical}
        )
    if not groups:
        raise ValueError("At least one frozen private-map row is required")

    consensus: list[dict[str, Any]] = []
    order_agreements: list[bool] = []
    for edge_index, rows in sorted(groups.items()):
        left_ids = {str(row["left_policy_id"]) for row in rows}
        right_ids = {str(row["right_policy_id"]) for row in rows}
        selector_arms = {str(row["selector_arm"]) for row in rows}
        if len(left_ids) != 1 or len(right_ids) != 1 or len(selector_arms) != 1:
            raise ValueError(f"Inconsistent frozen mapping for edge {edge_index}")
        physical = [row["physical"] for row in rows if row["physical"] is not None]
        order_consistent = len(physical) == 2 and len(set(physical)) == 1
        order_agreements.append(order_consistent)
        consensus.append(
            {
                "objective_id": "controlled_outer_requirement",
                "evidence_arm": next(iter(selector_arms)),
                "pair_id": f"edge_{edge_index:02d}",
                "left_policy_id": next(iter(left_ids)),
                "right_policy_id": next(iter(right_ids)),
                "left_win_probability": fmean(row["left_probability"] for row in rows),
                "response_count": len(rows),
                "order_consistent": order_consistent,
            }
        )
    return consensus, {
        "valid_request_rate": fmean(validity) if validity else 0.0,
        "order_consistent_edge_rate": (
            fmean(order_agreements) if order_agreements else 0.0
        ),
        "edge_count": len(consensus),
    }


def rerank_frozen_outer_cell(
    *,
    private_rows: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    parent_count: int,
    regularization: float,
    probability_clip: float,
    tie_break_mode: str,
) -> dict[str, Any]:
    """Rerank one observed candidate cell without changing its frozen population."""

    if not private_rows:
        raise ValueError("private_rows must not be empty")
    first = private_rows[0]
    scenario = str(first["scenario_id"])
    outer_seed = int(first["outer_seed"])
    generation = int(first["generation"])
    selector_arm = str(first["selector_arm"])
    expected_cell = (scenario, outer_seed, generation, selector_arm)
    if any(
        (
            str(row["scenario_id"]),
            int(row["outer_seed"]),
            int(row["generation"]),
            str(row["selector_arm"]),
        )
        != expected_cell
        for row in private_rows
    ):
        raise ValueError("private_rows span more than one selection cell")

    consensus, response_diagnostics = build_frozen_outer_consensus(private_rows, responses)
    base_ranking = rank_pairwise_consensus(
        consensus,
        regularization=regularization,
        probability_clip=probability_clip,
    )
    ranking = apply_exact_score_tie_break(
        base_ranking,
        mode=tie_break_mode,
        scenario=scenario,
        outer_seed=outer_seed,
        generation=generation,
    )
    selected_ids = [str(row["policy_id"]) for row in ranking[:parent_count]]
    oracle_by_id = {str(row["policy_id"]): row for row in oracle_rows}
    missing = sorted(set(selected_ids) - set(oracle_by_id))
    if missing:
        raise ValueError(f"Held-out oracle rows missing selected policies: {missing}")
    violations = [
        float(oracle_by_id[policy_id]["semantic_violation_rate"])
        for policy_id in selected_ids
    ]
    successes = [
        float(oracle_by_id[policy_id]["heldout_success_rate"])
        for policy_id in selected_ids
    ]

    score_counts = Counter(float(row["score"]) for row in ranking)
    tie_sizes = sorted((count for count in score_counts.values() if count > 1), reverse=True)
    cutoff_tie_size = 0
    if parent_count < len(ranking):
        cutoff_score = float(ranking[parent_count - 1]["score"])
        if cutoff_score == float(ranking[parent_count]["score"]):
            cutoff_tie_size = score_counts[cutoff_score]

    return {
        "scenario_id": scenario,
        "outer_seed": outer_seed,
        "generation": generation,
        "selector_arm": selector_arm,
        "regularization": regularization,
        "probability_clip": probability_clip,
        "tie_break_mode": tie_break_mode,
        "selected_policy_ids": selected_ids,
        "selected_policy_set_hash": stable_hash(sorted(selected_ids)),
        "selected_parent_mean_semantic_violation_rate": fmean(violations),
        "selected_parent_mean_semantic_pass_rate": 1.0 - fmean(violations),
        "selected_parent_mean_heldout_success_rate": fmean(successes),
        "exact_score_tie_group_count": len(tie_sizes),
        "exact_score_tied_policy_count": sum(tie_sizes),
        "largest_exact_score_tie_size": max(tie_sizes, default=1),
        "selection_cutoff_exact_tie_size": cutoff_tie_size,
        "ranking": ranking,
        **response_diagnostics,
    }


def _index_selection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], Mapping[str, Any]]:
    index: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["scenario_id"]),
            int(row["outer_seed"]),
            int(row["generation"]),
            str(row["selector_arm"]),
        )
        if key in index:
            raise ValueError(f"Duplicate selection cell: {key}")
        index[key] = row
    return index


def difference_in_differences(
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[str],
    outer_seeds: Sequence[int],
    baseline_generation: int = 0,
    endpoint_generation: int = 2,
    b1_arm: str = "B1-LLM-PBT",
    b2_arm: str = "B2-FullTrace-LLM-PBT",
    bootstrap_seed: int = 20260901,
) -> dict[str, Any]:
    """Compute a seed-paired generation-2 minus generation-0 DID.

    The same numeric effect is obtained from pass rates as
    ``(B2-B1)_endpoint - (B2-B1)_baseline`` and from violation rates as
    ``(B1-B2)_endpoint - (B1-B2)_baseline``.  Positive values favor B2.
    """

    index = _index_selection_rows(selection_rows)
    by_scenario: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[str, float] = {}
    cell_details: list[dict[str, Any]] = []
    for seed in outer_seeds:
        within_seed: list[float] = []
        for scenario in scenarios:
            required = [
                (scenario, int(seed), generation, arm)
                for generation in (baseline_generation, endpoint_generation)
                for arm in (b1_arm, b2_arm)
            ]
            missing = [key for key in required if key not in index]
            if missing:
                raise ValueError(f"Missing cells for DID: {missing}")
            b1_g0 = float(
                index[(scenario, int(seed), baseline_generation, b1_arm)][
                    "selected_parent_mean_semantic_violation_rate"
                ]
            )
            b2_g0 = float(
                index[(scenario, int(seed), baseline_generation, b2_arm)][
                    "selected_parent_mean_semantic_violation_rate"
                ]
            )
            b1_g2 = float(
                index[(scenario, int(seed), endpoint_generation, b1_arm)][
                    "selected_parent_mean_semantic_violation_rate"
                ]
            )
            b2_g2 = float(
                index[(scenario, int(seed), endpoint_generation, b2_arm)][
                    "selected_parent_mean_semantic_violation_rate"
                ]
            )
            contrast_g0 = b1_g0 - b2_g0
            contrast_g2 = b1_g2 - b2_g2
            did = contrast_g2 - contrast_g0
            by_scenario[scenario].append(did)
            within_seed.append(did)
            cell_details.append(
                {
                    "scenario_id": scenario,
                    "outer_seed": int(seed),
                    "B1_minus_B2_violation_baseline": contrast_g0,
                    "B1_minus_B2_violation_endpoint": contrast_g2,
                    "difference_in_differences": did,
                }
            )
        by_seed[str(seed)] = fmean(within_seed)
    values = [by_seed[str(seed)] for seed in outer_seeds]
    return {
        "baseline_generation": baseline_generation,
        "endpoint_generation": endpoint_generation,
        "violation_effect_definition": (
            "(B1_minus_B2_violation_at_endpoint) minus "
            "(B1_minus_B2_violation_at_baseline)"
        ),
        "pass_effect_definition": (
            "(B2_minus_B1_pass_at_endpoint) minus "
            "(B2_minus_B1_pass_at_baseline)"
        ),
        "positive_value_favors": b2_arm,
        "outer_seed_cluster_count": len(values),
        "outer_seed_cluster_effects": by_seed,
        "mean_effect": fmean(values),
        "bootstrap_95_ci": seed_cluster_bootstrap_ci(values, seed=bootstrap_seed),
        "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(values),
        "scenario_specific_descriptive": {
            scenario: {
                "mean_effect": fmean(scenario_values),
                "outer_seed_effects": scenario_values,
            }
            for scenario, scenario_values in by_scenario.items()
        },
        "cell_details": cell_details,
        "inference_status": "POST_HOC_DESCRIPTIVE",
    }


def summarize_sensitivity_grid(
    cell_rows: Sequence[Mapping[str, Any]],
    *,
    frozen_selection_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    outer_seeds: Sequence[int],
    endpoint_generation: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Summarize each finite ranking/tie-break specification on observed candidates."""

    frozen_index = _index_selection_rows(frozen_selection_rows)
    groups: dict[tuple[float, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        groups[
            (
                float(row["regularization"]),
                float(row["probability_clip"]),
                str(row["tie_break_mode"]),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (regularization, probability_clip, tie_break_mode), rows in sorted(groups.items()):
        row_index = _index_selection_rows(rows)
        set_matches: list[bool] = []
        jaccards: list[float] = []
        for row in rows:
            key = (
                str(row["scenario_id"]),
                int(row["outer_seed"]),
                int(row["generation"]),
                str(row["selector_arm"]),
            )
            observed = set(map(str, row["selected_policy_ids"]))
            frozen = set(map(str, frozen_index[key]["selected_policy_ids"]))
            set_matches.append(observed == frozen)
            jaccards.append(len(observed & frozen) / len(observed | frozen))

        endpoint_seed_effects: dict[str, float] = {}
        generation_effects: dict[str, float] = {}
        for generation in sorted({int(row["generation"]) for row in rows}):
            seed_effects: list[float] = []
            for seed in outer_seeds:
                scenario_effects = []
                for scenario in scenarios:
                    b1 = float(
                        row_index[(scenario, int(seed), generation, "B1-LLM-PBT")][
                            "selected_parent_mean_semantic_violation_rate"
                        ]
                    )
                    b2 = float(
                        row_index[
                            (scenario, int(seed), generation, "B2-FullTrace-LLM-PBT")
                        ]["selected_parent_mean_semantic_violation_rate"]
                    )
                    scenario_effects.append(b1 - b2)
                seed_effects.append(fmean(scenario_effects))
                if generation == endpoint_generation:
                    endpoint_seed_effects[str(seed)] = seed_effects[-1]
            generation_effects[str(generation)] = fmean(seed_effects)

        endpoint_values = [endpoint_seed_effects[str(seed)] for seed in outer_seeds]
        summaries.append(
            {
                "config_id": stable_hash(
                    [regularization, probability_clip, tie_break_mode], length=16
                ),
                "regularization": regularization,
                "probability_clip": probability_clip,
                "tie_break_mode": tie_break_mode,
                "observed_candidate_cell_count": len(rows),
                "selected_set_exact_match_to_frozen_fraction": fmean(set_matches),
                "selected_set_mean_jaccard_to_frozen": fmean(jaccards),
                "cells_with_any_exact_score_tie": sum(
                    int(row["exact_score_tie_group_count"]) > 0 for row in rows
                ),
                "cells_with_selection_cutoff_exact_tie": sum(
                    int(row["selection_cutoff_exact_tie_size"]) > 0 for row in rows
                ),
                "generation_mean_B1_minus_B2_violation": generation_effects,
                "endpoint_observed_candidate_effect": {
                    "generation": endpoint_generation,
                    "outer_seed_cluster_effects": endpoint_seed_effects,
                    "mean_effect": fmean(endpoint_values),
                    "bootstrap_95_ci": seed_cluster_bootstrap_ci(
                        endpoint_values, seed=bootstrap_seed
                    ),
                    "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(
                        endpoint_values
                    ),
                    "positive_value_favors": "B2-FullTrace-LLM-PBT",
                    "inference_status": "POST_HOC_OBSERVED_CANDIDATE_SENSITIVITY",
                },
            }
        )
    return summaries
