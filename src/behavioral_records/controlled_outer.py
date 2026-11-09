from __future__ import annotations

import random
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import stable_hash
from .contextual_semantic_pilot import (
    _metric_vector as contextual_metric_vector,
    _oracle_profile as contextual_oracle_profile,
    _raw_view as contextual_raw_view,
    _simulate_collateral_episode,
    _simulate_risk_episode,
)
from .ranking import rank_pairwise_consensus
from .semantic_pilot import (
    _low_dimensional_summary,
    _raw_public_view,
    _simulate_episode as simulate_reward_recovery_episode,
    behavioral_test_profile,
    build_semantic_pair_prompt,
    causal_profile,
    standard_metric_vector,
)
from .controlled_confirmatory import _augment_c1_trajectory
from .models import Trajectory


SCENARIOS = ("C1", "C2")
LLM_ARMS = ("B1-LLM-PBT", "B2-FullTrace-LLM-PBT")
BASELINE_ARMS = ("Metric-PBT", "Random-PBT", "Oracle-PBT")
OUTER_ARMS = LLM_ARMS + BASELINE_ARMS
EVIDENCE_ARM = {
    "B1-LLM-PBT": "B1",
    "B2-FullTrace-LLM-PBT": "B2-FullTrace",
    "Relational-Linked-LLM-PBT": "Relational-Linked",
    "Relational-LinkBroken-LLM-PBT": "Relational-LinkBroken",
}


def fixed_connected_edges(population_size: int) -> list[tuple[int, int]]:
    if population_size < 4:
        raise ValueError("Controlled outer loop requires at least four policies")
    edges: set[tuple[int, int]] = set()
    for offset in (1, 3):
        for index in range(population_size):
            other = (index + offset) % population_size
            edges.add(tuple(sorted((index, other))))
    return sorted(edges)


def _rng(*parts: object) -> random.Random:
    seed = int(stable_hash(list(parts))[:16], 16)
    return random.Random(seed)


def initial_population(scenario: str, outer_seed: int, population_size: int) -> list[dict[str, Any]]:
    if population_size < 12 or population_size % 4:
        raise ValueError("Controlled outer populations must be a multiple of four and at least 12")
    rows: list[dict[str, Any]] = []
    c1_modes = ("disturbance_triggered", "reward_triggered", "mixed_or", "mixed_and")
    c2_modes = (
        ("adaptive", "adaptive"),
        ("reversed", "reversed"),
        ("adaptive", "reversed"),
        ("reversed", "adaptive"),
    )
    rotation = outer_seed % 4
    for slot in range(population_size):
        cohort = slot // 4
        genotype_index = (slot + rotation) % 4
        policy_id = f"{scenario.lower()}_s{outer_seed}_g0_slot{slot:02d}"
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "slot": slot,
            "generation": 0,
            "parent_policy_id": None,
            "lineage": [policy_id],
        }
        if scenario == "C1":
            row.update(
                {
                    "mode": c1_modes[genotype_index],
                    "cycle_count": 2 + cohort % 4,
                    "prelude_waits": cohort % 4,
                    "cooldown_waits": max(0, 2 - cohort % 4),
                    "workload_level": 1 + cohort,
                }
            )
        elif scenario == "C2":
            risk_mode, collateral_mode = c2_modes[genotype_index]
            row.update(
                {
                    "risk_mode": risk_mode,
                    "collateral_mode": collateral_mode,
                    "operation_steps": 1 + cohort % 4,
                    "wait_steps": 1 + cohort % 3,
                    "environment_level": 1 + cohort,
                }
            )
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")
        rows.append(row)
    return rows


def _c1_trajectories(
    policy: Mapping[str, Any], *, outer_seed: int, generation: int
) -> tuple[list[Trajectory], list[Trajectory]]:
    validation: list[Trajectory] = []
    oracle: list[Trajectory] = []
    slot = int(policy["slot"])
    cycles = int(policy["cycle_count"])
    base_seed = 1_000_000 + outer_seed * 100_000 + generation * 10_000 + slot * 100

    def augment(row: Trajectory) -> Trajectory:
        return _augment_c1_trajectory(
            row,
            prelude_waits=int(policy["prelude_waits"]),
            cooldown_waits=int(policy["cooldown_waits"]),
            workload_level=int(policy["workload_level"]),
        )

    for replicate in range(4):
        validation.append(
            augment(
                simulate_reward_recovery_episode(
                    policy_id=str(policy["policy_id"]),
                    mode=str(policy["mode"]),
                    episode_id=f"{policy['policy_id']}_validation_{replicate}",
                    split="validation",
                    context_seed=base_seed + replicate,
                    base_cycle_count=cycles,
                    credit_enabled=True,
                    disturbance_enabled=True,
                    auto_restore=True,
                    required_cycles=cycles,
                    intervention="standard_operation",
                )
            )
        )
    specs = (
        ("causal_credit_removed", False, True, False, cycles),
        ("causal_disturbance_removed", True, False, False, cycles),
        ("causal_hard_disturbance", True, True, False, cycles + 1),
        ("behavior_standard", True, True, True, cycles),
        ("behavior_credit_removed", False, True, False, cycles),
        ("behavior_disturbance_removed", True, False, False, cycles),
        ("behavior_hard_disturbance", True, True, False, cycles + 1),
    )
    for spec_index, (name, credit, disturbance, restore, required) in enumerate(specs):
        for replicate in range(2):
            oracle.append(
                augment(
                    simulate_reward_recovery_episode(
                        policy_id=str(policy["policy_id"]),
                        mode=str(policy["mode"]),
                        episode_id=f"{policy['policy_id']}_{name}_{replicate}",
                        split="heldout_oracle",
                        context_seed=base_seed + 1_000 + spec_index * 10 + replicate,
                        base_cycle_count=cycles,
                        credit_enabled=credit,
                        disturbance_enabled=disturbance,
                        auto_restore=restore,
                        required_cycles=required,
                        intervention=name,
                    )
                )
            )
    return validation, oracle


def _c2_trajectories(
    policy: Mapping[str, Any], *, outer_seed: int, generation: int
) -> tuple[list[Trajectory], list[Trajectory]]:
    validation: list[Trajectory] = []
    oracle: list[Trajectory] = []
    slot = int(policy["slot"])
    operation_steps = int(policy["operation_steps"])
    wait_steps = int(policy["wait_steps"])
    level = int(policy["environment_level"])
    base_seed = 2_000_000 + outer_seed * 100_000 + generation * 10_000 + slot * 100
    for replicate in range(4):
        common = {
            "policy_id": str(policy["policy_id"]),
            "split": "validation",
            "context_seed": base_seed + replicate,
            "deadline": False,
        }
        validation.append(
            _simulate_risk_episode(
                **common,
                mode=str(policy["risk_mode"]),
                episode_id=f"{policy['policy_id']}_risk_validation_{replicate}",
                safe_available=replicate % 2 == 0,
                hazard_level=level,
                route_steps=operation_steps,
                wait_steps=wait_steps,
                intervention="standard_risk_operation",
            )
        )
        validation.append(
            _simulate_collateral_episode(
                **common,
                mode=str(policy["collateral_mode"]),
                episode_id=f"{policy['policy_id']}_collateral_validation_{replicate}",
                change_required=replicate % 2 == 1,
                sensitivity=level,
                operation_steps=operation_steps,
                wait_steps=wait_steps,
                intervention="standard_collateral_operation",
            )
        )
    risk_specs = (
        ("causal_safe_available", True, False, 1),
        ("causal_only_exposed_route", False, True, 1),
        ("causal_high_hazard_safe_available", True, False, 3),
        ("behavior_safe_available", True, False, 1),
        ("behavior_only_exposed_route", False, True, 1),
        ("behavior_high_hazard_safe_available", True, False, 3),
    )
    for spec_index, (name, safe, deadline, hazard) in enumerate(risk_specs):
        for replicate in range(2):
            oracle.append(
                _simulate_risk_episode(
                    policy_id=str(policy["policy_id"]),
                    mode=str(policy["risk_mode"]),
                    episode_id=f"{policy['policy_id']}_risk_{name}_{replicate}",
                    split="heldout_oracle",
                    context_seed=base_seed + 1_000 + spec_index * 10 + replicate,
                    safe_available=safe,
                    deadline=deadline,
                    hazard_level=hazard,
                    route_steps=operation_steps,
                    wait_steps=wait_steps,
                    intervention=name,
                )
            )
    collateral_specs = (
        ("causal_change_not_required", False, False, 1),
        ("causal_change_required_deadline", True, True, 1),
        ("causal_sensitive_state_not_required", False, False, 3),
        ("behavior_change_not_required", False, False, 1),
        ("behavior_change_required_deadline", True, True, 1),
        ("behavior_sensitive_state_not_required", False, False, 3),
    )
    for spec_index, (name, required, deadline, sensitivity) in enumerate(collateral_specs):
        for replicate in range(2):
            oracle.append(
                _simulate_collateral_episode(
                    policy_id=str(policy["policy_id"]),
                    mode=str(policy["collateral_mode"]),
                    episode_id=f"{policy['policy_id']}_collateral_{name}_{replicate}",
                    split="heldout_oracle",
                    context_seed=base_seed + 2_000 + spec_index * 10 + replicate,
                    change_required=required,
                    deadline=deadline,
                    sensitivity=sensitivity,
                    operation_steps=operation_steps,
                    wait_steps=wait_steps,
                    intervention=name,
                )
            )
    return validation, oracle


def compile_population(
    scenario: str,
    population: Sequence[Mapping[str, Any]],
    *,
    outer_seed: int,
    generation: int,
    include_relational: bool = False,
) -> dict[str, Any]:
    validation: dict[str, list[Trajectory]] = {}
    oracle: dict[str, list[Trajectory]] = {}
    for policy in population:
        if scenario == "C1":
            public_rows, hidden_rows = _c1_trajectories(
                policy, outer_seed=outer_seed, generation=generation
            )
        else:
            public_rows, hidden_rows = _c2_trajectories(
                policy, outer_seed=outer_seed, generation=generation
            )
        validation[str(policy["policy_id"])] = public_rows
        oracle[str(policy["policy_id"])] = hidden_rows

    evidence: dict[str, dict[str, Any]] = {}
    ordinary_scores: dict[str, float] = {}
    if scenario == "C1":
        for policy_id, rows in validation.items():
            metrics = standard_metric_vector(rows)
            evidence[policy_id] = {
                "B1": _low_dimensional_summary(metrics),
                "B2-FullTrace": _raw_public_view(rows),
            }
            ordinary_scores[policy_id] = (
                100.0 * metrics["success_rate"]
                + metrics["return_mean"]
                - 0.01 * metrics["length_mean"]
            )
    else:
        all_rows = [row for rows in validation.values() for row in rows]
        actions = sorted({step.action for row in all_rows for step in row.steps})
        events = sorted({event for row in all_rows for event in row.events})
        for policy_id, rows in validation.items():
            metrics = contextual_metric_vector(rows, actions, events)
            evidence[policy_id] = {
                "B1": _low_dimensional_summary(metrics),
                "B2-FullTrace": contextual_raw_view(rows),
            }
            ordinary_scores[policy_id] = (
                100.0 * metrics["success_rate"]
                + metrics["return_mean"]
                - 0.01 * metrics["length_mean"]
            )

    if include_relational:
        from .relational_record import RelationalBehaviorCompiler, relation_broken_control, audit_relation_break
        compiler = RelationalBehaviorCompiler()
        for policy_id, arms in evidence.items():
            linked = compiler.compile(arms["B2-FullTrace"])
            broken = relation_broken_control(linked)
            audit = audit_relation_break(linked, broken)
            # Validate every invariant independently of reference-test outcomes.
            from .relational_record import _audit_boolean_fields
            if not _audit_boolean_fields(audit):
                raise ValueError(f"Relational compilation invariant failed for {policy_id}")
            arms["Relational-Linked"] = linked
            arms["Relational-LinkBroken"] = broken

    hidden: dict[str, dict[str, Any]] = {}
    for policy_id, rows in oracle.items():
        if scenario == "C1":
            causal = causal_profile(rows)["tests"]
            behavioral = behavioral_test_profile(rows)["tests"]
            tests = {
                **{f"causal::{key}": value for key, value in causal.items()},
                **{f"behavior::{key}": value for key, value in behavioral.items()},
            }
        else:
            tests = {}
            for layer in ("causal", "behavior"):
                risk = contextual_oracle_profile("risk", rows, layer)["tests"]
                collateral = contextual_oracle_profile("collateral", rows, layer)["tests"]
                tests.update({f"{layer}::risk::{key}": value for key, value in risk.items()})
                tests.update(
                    {f"{layer}::collateral::{key}": value for key, value in collateral.items()}
                )
        pass_rate = fmean(bool(value) for value in tests.values())
        hidden[policy_id] = {
            "semantic_test_count": len(tests),
            "semantic_pass_rate": pass_rate,
            "semantic_violation_rate": 1.0 - pass_rate,
            "heldout_success_rate": fmean(row.outcome == "success" for row in rows),
            "tests": tests,
        }
    return {
        "evidence": evidence,
        "ordinary_scores": ordinary_scores,
        "hidden_oracle": hidden,
        "validation_episode_count": sum(map(len, validation.values())),
        "heldout_episode_count": sum(map(len, oracle.values())),
    }


def build_llm_requests(
    *,
    scenario: str,
    outer_seed: int,
    generation: int,
    arm: str,
    population: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    requirement: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    representation = EVIDENCE_ARM[arm]
    requests: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for edge_index, (left_slot, right_slot) in enumerate(fixed_connected_edges(len(population))):
        base_left = str(population[left_slot]["policy_id"])
        base_right = str(population[right_slot]["policy_id"])
        for order in ("AB", "BA"):
            alias_a, alias_b = (
                (base_left, base_right) if order == "AB" else (base_right, base_left)
            )
            request_id = (
                f"outer__{scenario}__s{outer_seed}__g{generation}__"
                f"{representation}__e{edge_index:02d}__{order}"
            )
            prompt = build_semantic_pair_prompt(
                requirement,
                evidence[alias_a][representation],
                evidence[alias_b][representation],
                arm=representation,
            )
            requests.append(
                {
                    "request_id": request_id,
                    "scenario_id": scenario,
                    "concept_id": f"{scenario}_outer",
                    "pair_id": f"{scenario}_s{outer_seed}_g{generation}_e{edge_index:02d}",
                    "arm": representation,
                    "order": order,
                    "response_schema": "semantic-selection-response-v1",
                    "prompt": prompt,
                    "prompt_hash": stable_hash(prompt),
                }
            )
            private.append(
                {
                    "request_id": request_id,
                    "scenario_id": scenario,
                    "outer_seed": outer_seed,
                    "generation": generation,
                    "selector_arm": arm,
                    "edge_index": edge_index,
                    "left_policy_id": base_left,
                    "right_policy_id": base_right,
                    "alias_a_policy_id": alias_a,
                    "alias_b_policy_id": alias_b,
                    "order": order,
                }
            )
    return requests, private


def _winner_probability_for_left(
    private: Mapping[str, Any], response: Mapping[str, Any] | None
) -> tuple[float, bool, str | None]:
    payload = response.get("response") if response else None
    winner = payload.get("winner") if isinstance(payload, Mapping) else None
    if winner not in {"A", "B", "TIE", "ABSTAIN"}:
        return 0.5, False, None
    if winner in {"TIE", "ABSTAIN"}:
        return 0.5, True, str(winner)
    selected = (
        str(private["alias_a_policy_id"])
        if winner == "A"
        else str(private["alias_b_policy_id"])
    )
    return (1.0 if selected == str(private["left_policy_id"]) else 0.0), True, selected


def select_llm_parents(
    *,
    population: Sequence[Mapping[str, Any]],
    private_rows: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    parent_count: int,
) -> dict[str, Any]:
    response_by_id = {str(row["request_id"]): dict(row) for row in responses}
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
    consensus: list[dict[str, Any]] = []
    order_agreements = []
    for edge_index, rows in sorted(groups.items()):
        left = str(rows[0]["left_policy_id"])
        right = str(rows[0]["right_policy_id"])
        physical = [row["physical"] for row in rows if row["physical"] is not None]
        order_agreements.append(len(physical) == 2 and len(set(physical)) == 1)
        consensus.append(
            {
                "objective_id": "controlled_outer_requirement",
                "evidence_arm": str(rows[0]["selector_arm"]),
                "pair_id": f"edge_{edge_index:02d}",
                "left_policy_id": left,
                "right_policy_id": right,
                "left_win_probability": fmean(row["left_probability"] for row in rows),
                "response_count": len(rows),
                "order_consistent": order_agreements[-1],
            }
        )
    ranking = rank_pairwise_consensus(consensus)
    selected = [str(row["policy_id"]) for row in ranking[:parent_count]]
    scheduled_ids = {str(row["policy_id"]) for row in population}
    ranked_ids = {str(row["policy_id"]) for row in ranking}
    return {
        "selected_policy_ids": selected,
        "ranking": ranking,
        "valid_request_rate": fmean(validity) if validity else 0.0,
        "order_consistent_edge_rate": fmean(order_agreements) if order_agreements else 0.0,
        "scheduled_graph_connected": scheduled_ids == ranked_ids,
        "neutralized_abstain_or_invalid": True,
    }


def select_baseline_parents(
    *,
    arm: str,
    population: Sequence[Mapping[str, Any]],
    compiled: Mapping[str, Any],
    scenario: str,
    outer_seed: int,
    generation: int,
    parent_count: int,
) -> dict[str, Any]:
    policy_ids = [str(row["policy_id"]) for row in population]
    if arm == "Metric-PBT":
        scores = {policy_id: float(compiled["ordinary_scores"][policy_id]) for policy_id in policy_ids}
    elif arm == "Oracle-PBT":
        scores = {
            policy_id: float(compiled["hidden_oracle"][policy_id]["semantic_pass_rate"])
            for policy_id in policy_ids
        }
    elif arm == "Random-PBT":
        generator = _rng("random_selector", scenario, outer_seed, generation)
        scores = {policy_id: generator.random() for policy_id in policy_ids}
    else:
        raise ValueError(f"Unsupported baseline arm: {arm}")
    ranking = sorted(
        policy_ids,
        key=lambda policy_id: (
            -scores[policy_id],
            stable_hash(["tie", scenario, outer_seed, generation, policy_id]),
        ),
    )
    return {
        "selected_policy_ids": ranking[:parent_count],
        "ranking": [
            {"rank": index + 1, "policy_id": policy_id, "score": scores[policy_id]}
            for index, policy_id in enumerate(ranking)
        ],
        "valid_request_rate": 1.0,
        "order_consistent_edge_rate": 1.0,
        "scheduled_graph_connected": True,
    }


def mutate_population(
    *,
    scenario: str,
    population: Sequence[Mapping[str, Any]],
    selected_policy_ids: Sequence[str],
    outer_seed: int,
    next_generation: int,
) -> list[dict[str, Any]]:
    by_id = {str(row["policy_id"]): dict(row) for row in population}
    parents = [by_id[policy_id] for policy_id in selected_policy_ids]
    parent_count = len(parents)
    if parent_count < 2 or len({str(row["policy_id"]) for row in parents}) != parent_count:
        raise ValueError("Controlled outer selection requires unique parents")
    output: list[dict[str, Any]] = []
    c1_modes = ("disturbance_triggered", "reward_triggered", "mixed_or", "mixed_and")
    for slot in range(len(population)):
        parent = (
            parents[slot]
            if slot < parent_count
            else parents[(slot - parent_count) % parent_count]
        )
        child = dict(parent)
        child_id = f"{scenario.lower()}_s{outer_seed}_g{next_generation}_slot{slot:02d}"
        child.update(
            {
                "policy_id": child_id,
                "slot": slot,
                "generation": next_generation,
                "parent_policy_id": str(parent["policy_id"]),
                "lineage": [*list(parent["lineage"]), child_id],
            }
        )
        if slot >= parent_count:
            generator = _rng("mutation", scenario, outer_seed, next_generation, slot)
            if scenario == "C1":
                if generator.random() < 0.45:
                    choices = [mode for mode in c1_modes if mode != child["mode"]]
                    child["mode"] = generator.choice(choices)
                if generator.random() < 0.25:
                    child["cycle_count"] = min(
                        5, max(2, int(child["cycle_count"]) + generator.choice((-1, 1)))
                    )
            else:
                if generator.random() < 0.35:
                    child["risk_mode"] = (
                        "reversed" if child["risk_mode"] == "adaptive" else "adaptive"
                    )
                if generator.random() < 0.35:
                    child["collateral_mode"] = (
                        "reversed"
                        if child["collateral_mode"] == "adaptive"
                        else "adaptive"
                    )
        output.append(child)
    return output


def summarize_generation_selection(
    *,
    selection: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [str(value) for value in selection["selected_policy_ids"]]
    violations = [
        float(compiled["hidden_oracle"][policy_id]["semantic_violation_rate"])
        for policy_id in selected
    ]
    successes = [
        float(compiled["hidden_oracle"][policy_id]["heldout_success_rate"])
        for policy_id in selected
    ]
    return {
        **dict(selection),
        "selected_parent_mean_semantic_violation_rate": fmean(violations),
        "selected_parent_best_semantic_violation_rate": min(violations),
        "selected_parent_mean_heldout_success_rate": fmean(successes),
    }
