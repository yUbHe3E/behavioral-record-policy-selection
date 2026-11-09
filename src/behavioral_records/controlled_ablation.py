from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, stable_hash
from .confirmatory import holm_adjust
from .controlled_confirmatory import _bootstrap_mean_difference, _paired_sign_flip_p
from .io import read_json, read_jsonl, write_json, write_jsonl
from .semantic_pilot import build_semantic_pair_prompt


ABLATION_ARMS = (
    "B2-FullTrace-Control",
    "B2-TemporalShuffle",
    "B2-StateMasked",
)
SCENARIOS = ("C1", "C2")
MASK_TOKENS = {
    "boolean": "<BOOL_MASKED>",
    "number": "<NUMBER_MASKED>",
    "text": "<TEXT_MASKED>",
    "null": "<NULL_MASKED>",
    "array": "<ARRAY_MASKED>",
    "object": "<OBJECT_MASKED>",
}
COMPACT_MASK_TOKENS = {
    "boolean": "<B>",
    "number": "<N>",
    "text": "<S>",
    "null": "<0>",
    "array": "<A>",
    "object": "<O>",
}
MASK_TOKEN_PROFILES = {
    "verbose_type_tags_v1": MASK_TOKENS,
    "compact_type_tags_v1": COMPACT_MASK_TOKENS,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_without_t(step: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in step.items() if key != "t"}


def temporal_shuffle_view(public_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Remove displayed temporal order while preserving every non-t step value."""
    result = copy.deepcopy(dict(public_evidence))
    if result.get("representation") != "deterministic_raw_validation_trajectories":
        raise ValueError("Temporal shuffle requires the frozen FullTrace representation")
    # Equal-length label keeps the canonical serialized character count exact.
    result["representation"] = "deterministic_step_multiset_no_time_order"
    for episode_index, episode in enumerate(result.get("episodes", [])):
        original = list(episode.get("steps", []))
        order = list(range(len(original)))
        if len(order) > 1:
            seed = int(
                stable_hash(
                    [
                        "controlled-temporal-shuffle-v1",
                        episode.get("evidence_id", ""),
                        episode_index,
                        len(order),
                    ],
                    16,
                ),
                16,
            )
            random.Random(seed).shuffle(order)
            if order == list(range(len(original))):
                order = order[1:] + order[:1]
        shuffled = []
        for displayed_t, source_index in enumerate(order):
            step = copy.deepcopy(original[source_index])
            step["t"] = displayed_t
            shuffled.append(step)
        episode["steps"] = shuffled
    return result


def _mask_value(value: Any, mask_tokens: Mapping[str, str] = MASK_TOKENS) -> str:
    if value is None:
        return mask_tokens["null"]
    if isinstance(value, bool):
        return mask_tokens["boolean"]
    if isinstance(value, (int, float)):
        return mask_tokens["number"]
    if isinstance(value, str):
        return mask_tokens["text"]
    if isinstance(value, list):
        return mask_tokens["array"]
    return mask_tokens["object"]


def state_masked_view(
    public_evidence: Mapping[str, Any],
    *,
    mask_tokens: Mapping[str, str] = MASK_TOKENS,
) -> dict[str, Any]:
    """Mask step-state/predicate values without changing other trajectory facts."""
    result = copy.deepcopy(dict(public_evidence))
    if result.get("representation") != "deterministic_raw_validation_trajectories":
        raise ValueError("State mask requires the frozen FullTrace representation")
    result["representation"] = "deterministic_steps_state_values_masked"
    for episode in result.get("episodes", []):
        for step in episode.get("steps", []):
            for field in ("state", "predicates"):
                values = step.get(field, {})
                if not isinstance(values, Mapping):
                    raise ValueError(f"FullTrace step {field} must be an object")
                step[field] = {
                    str(key): _mask_value(value, mask_tokens)
                    for key, value in values.items()
                }
    return result


def audit_temporal_shuffle(
    source: Mapping[str, Any], transformed: Mapping[str, Any]
) -> dict[str, Any]:
    source_episodes = list(source.get("episodes", []))
    transformed_episodes = list(transformed.get("episodes", []))
    episode_metadata_exact = len(source_episodes) == len(transformed_episodes)
    step_multiset_exact = episode_metadata_exact
    displayed_time_axis_exact = episode_metadata_exact
    all_informative_orders_changed = True
    for original, shuffled in zip(source_episodes, transformed_episodes):
        episode_metadata_exact = episode_metadata_exact and (
            {key: value for key, value in original.items() if key != "steps"}
            == {key: value for key, value in shuffled.items() if key != "steps"}
        )
        original_steps = list(original.get("steps", []))
        shuffled_steps = list(shuffled.get("steps", []))
        original_payloads = [canonical_json(_step_without_t(step)) for step in original_steps]
        shuffled_payloads = [canonical_json(_step_without_t(step)) for step in shuffled_steps]
        step_multiset_exact = step_multiset_exact and Counter(original_payloads) == Counter(
            shuffled_payloads
        )
        displayed_time_axis_exact = displayed_time_axis_exact and [
            step.get("t") for step in shuffled_steps
        ] == list(range(len(shuffled_steps)))
        if len(set(original_payloads)) > 1:
            all_informative_orders_changed = all_informative_orders_changed and (
                original_payloads != shuffled_payloads
            )
    return {
        "episode_metadata_exact": episode_metadata_exact,
        "non_t_step_multiset_exact": step_multiset_exact,
        "displayed_time_axis_is_0_to_length_minus_1": displayed_time_axis_exact,
        "all_informative_episode_orders_changed": all_informative_orders_changed,
        "serialized_character_count_exact": len(canonical_json(source))
        == len(canonical_json(transformed)),
        "source_characters": len(canonical_json(source)),
        "transformed_characters": len(canonical_json(transformed)),
    }


def audit_state_mask(
    source: Mapping[str, Any],
    transformed: Mapping[str, Any],
    *,
    mask_tokens: Mapping[str, str] = MASK_TOKENS,
) -> dict[str, Any]:
    source_episodes = list(source.get("episodes", []))
    transformed_episodes = list(transformed.get("episodes", []))
    episode_metadata_exact = len(source_episodes) == len(transformed_episodes)
    non_state_step_fields_exact = episode_metadata_exact
    field_names_exact = episode_metadata_exact
    all_values_masked = episode_metadata_exact
    allowed_masks = set(mask_tokens.values())
    for original, masked in zip(source_episodes, transformed_episodes):
        episode_metadata_exact = episode_metadata_exact and (
            {key: value for key, value in original.items() if key != "steps"}
            == {key: value for key, value in masked.items() if key != "steps"}
        )
        original_steps = list(original.get("steps", []))
        masked_steps = list(masked.get("steps", []))
        non_state_step_fields_exact = non_state_step_fields_exact and len(
            original_steps
        ) == len(masked_steps)
        for original_step, masked_step in zip(original_steps, masked_steps):
            non_state_step_fields_exact = non_state_step_fields_exact and (
                {
                    key: value
                    for key, value in original_step.items()
                    if key not in {"state", "predicates"}
                }
                == {
                    key: value
                    for key, value in masked_step.items()
                    if key not in {"state", "predicates"}
                }
            )
            for field in ("state", "predicates"):
                original_values = dict(original_step.get(field, {}))
                masked_values = dict(masked_step.get(field, {}))
                field_names_exact = field_names_exact and set(original_values) == set(
                    masked_values
                )
                all_values_masked = all_values_masked and all(
                    value in allowed_masks for value in masked_values.values()
                )
    return {
        "episode_metadata_exact": episode_metadata_exact,
        "non_state_step_fields_exact": non_state_step_fields_exact,
        "state_and_predicate_field_names_exact": field_names_exact,
        "all_state_and_predicate_values_masked": all_values_masked,
        "source_characters": len(canonical_json(source)),
        "transformed_characters": len(canonical_json(transformed)),
    }


def _expected_winner(oracle: Mapping[str, Any], order: str) -> str:
    if oracle["status"] == "AMBIGUOUS":
        return "ABSTAIN"
    label = str(oracle["label"])
    if label == "TIE":
        return "TIE"
    if order == "AB":
        return label
    return "B" if label == "A" else "A"


def build_controlled_representation_ablation(
    *,
    output_dir: str | Path,
    plan_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository_root)
    plan = read_json(plan_path)
    request_namespace = str(plan.get("request_namespace", "ablation_v1"))
    mask_token_profile = str(
        plan.get("state_mask_token_profile", "verbose_type_tags_v1")
    )
    try:
        mask_tokens = MASK_TOKEN_PROFILES[mask_token_profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown state_mask_token_profile: {mask_token_profile}"
        ) from exc
    source = root / str(plan["source_bundle"])
    scenario_config_path = root / str(plan["scenario_config"])
    scenario_config = read_json(scenario_config_path)

    source_hash_checks = {
        relative: _file_sha256(source / relative) == str(expected).lower()
        for relative, expected in plan["source_hashes_sha256"].items()
    }
    scenario_config_hash_matches = _file_sha256(scenario_config_path) == str(
        plan["scenario_config_file_sha256"]
    ).lower()

    all_requests: list[dict[str, Any]] = []
    all_private: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    policy_ids: set[str] = set()
    pair_count = 0
    temporal_audits = []
    state_audits = []

    for scenario in SCENARIOS:
        scenario_source = source / scenario
        source_evidence_rows = read_jsonl(scenario_source / "public_evidence.jsonl")
        full_by_policy = {
            str(row["policy_id"]): copy.deepcopy(row["public_evidence"])
            for row in source_evidence_rows
            if row["arm"] == "B2-FullTrace"
        }
        blueprint = read_json(scenario_source / "pair_manifest_private.json")
        oracle = read_json(scenario_source / "oracle_adjudication.json")
        requirements = scenario_config["scenarios"][scenario]["requirement_paraphrases"]
        pair_count += len(blueprint)
        policy_ids.update(full_by_policy)
        evidence_by_policy: dict[str, dict[str, Any]] = {}
        for policy_id, full in sorted(full_by_policy.items()):
            temporal = temporal_shuffle_view(full)
            state_masked = state_masked_view(full, mask_tokens=mask_tokens)
            temporal_audit = audit_temporal_shuffle(full, temporal)
            state_audit = audit_state_mask(
                full, state_masked, mask_tokens=mask_tokens
            )
            temporal_audits.append(temporal_audit)
            state_audits.append(state_audit)
            evidence_by_policy[policy_id] = {
                "B2-FullTrace-Control": full,
                "B2-TemporalShuffle": temporal,
                "B2-StateMasked": state_masked,
            }
            for arm, evidence in evidence_by_policy[policy_id].items():
                all_evidence.append(
                    {
                        "scenario_id": scenario,
                        "policy_id": policy_id,
                        "arm": arm,
                        "public_evidence": evidence,
                        "characters": len(canonical_json(evidence)),
                        "evidence_hash": stable_hash(evidence),
                    }
                )

        for pair in blueprint:
            pair_id = str(pair["pair_id"])
            base_left = str(pair["policy_a"])
            base_right = str(pair["policy_b"])
            pair_oracle = oracle[pair_id]
            for arm in ABLATION_ARMS:
                for paraphrase_index, requirement in enumerate(requirements):
                    paraphrase = f"P{paraphrase_index}"
                    for order in ("AB", "BA"):
                        left, right = (
                            (base_left, base_right)
                            if order == "AB"
                            else (base_right, base_left)
                        )
                        prompt = build_semantic_pair_prompt(
                            requirement,
                            evidence_by_policy[left][arm],
                            evidence_by_policy[right][arm],
                            arm=arm,
                        )
                        for repeat in range(2):
                            request_id = (
                                f"{request_namespace}__{scenario}__{pair_id}__{arm}__"
                                f"{paraphrase}__{order}__R{repeat}"
                            )
                            all_requests.append(
                                {
                                    "request_id": request_id,
                                    "scenario_id": scenario,
                                    "concept_id": scenario,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase,
                                    "order": order,
                                    "repeat": repeat,
                                    "response_schema": "semantic-selection-response-v1",
                                    "prompt": prompt,
                                    "prompt_hash": stable_hash(prompt),
                                }
                            )
                            all_private.append(
                                {
                                    "request_id": request_id,
                                    "scenario_id": scenario,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase,
                                    "order": order,
                                    "repeat": repeat,
                                    "alias_a_policy_id": left,
                                    "alias_b_policy_id": right,
                                    "oracle_status": pair_oracle["status"],
                                    "expected_winner": _expected_winner(pair_oracle, order),
                                }
                            )

    prompts = [str(row["prompt"]) for row in all_requests]
    forbidden_literals = {
        "oracle_status",
        "expected_winner",
        "semantic_violation_rate",
        "preferred policy",
        "correct_strategy",
    }
    no_private_leakage = not any(
        literal in prompt.lower() for prompt in prompts for literal in forbidden_literals
    ) and not any(policy_id in prompt for prompt in prompts for policy_id in policy_ids)
    temporal_passed = all(all(row.values()) for row in (
        {
            key: value
            for key, value in audit.items()
            if key not in {"source_characters", "transformed_characters"}
        }
        for audit in temporal_audits
    ))
    state_passed = all(all(row.values()) for row in (
        {
            key: value
            for key, value in audit.items()
            if key not in {"source_characters", "transformed_characters"}
        }
        for audit in state_audits
    ))
    expected_requests = int(plan["design"]["expected_total_requests"])
    gates = {
        "source_hashes_match": all(source_hash_checks.values())
        and scenario_config_hash_matches,
        "all_96_source_pairs_retained": pair_count == 96,
        "all_three_arms_present": set(row["arm"] for row in all_requests)
        == set(ABLATION_ARMS),
        "expected_2304_requests": len(all_requests) == expected_requests,
        "temporal_shuffle_audit_passed": temporal_passed,
        "state_mask_audit_passed": state_passed,
        "maximum_prompt_characters_at_most_70000": max(map(len, prompts), default=0)
        <= int(plan["offline_gates"]["maximum_prompt_characters"]),
        "no_hidden_oracle_or_policy_identity_leakage": no_private_leakage,
    }
    write_jsonl(output / "public_evidence.jsonl", all_evidence)
    write_jsonl(output / "judge_requests.jsonl", all_requests)
    write_jsonl(output / "judge_private_map.jsonl", all_private)
    write_json(
        output / "ablation_integrity_audit.json",
        {
            "schema_version": "controlled-representation-ablation-audit-v1",
            "source_hash_checks": source_hash_checks,
            "scenario_config_hash_matches": scenario_config_hash_matches,
            "state_mask_token_profile": mask_token_profile,
            "temporal_policy_audits": temporal_audits,
            "state_policy_audits": state_audits,
            "maximum_prompt_characters": max(map(len, prompts), default=0),
            "minimum_prompt_characters": min(map(len, prompts), default=0),
            "gates": gates,
            "passed": all(gates.values()),
        },
    )
    summary = {
        "schema_version": "controlled-representation-ablation-bundle-v1",
        "status": "READY_FOR_ABLATION_JUDGE" if all(gates.values()) else "OFFLINE_GATE_FAILED",
        "pair_count": pair_count,
        "policy_count": len(policy_ids),
        "arms": list(ABLATION_ARMS),
        "request_count": len(all_requests),
        "request_hash": stable_hash(all_requests),
        "plan_hash": stable_hash(plan),
        "gates": gates,
        "claim_boundary": plan["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "prepare",
            "origin_date": "2026-08-23",
            "verification_status": "VERIFIED_OFFLINE" if all(gates.values()) else "FAILED",
            "version_label": str(
                plan.get("version_label", "controlled_representation_ablation_v1")
            ),
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def evaluate_controlled_representation_ablation(
    *,
    private_map: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    private_rows = [dict(row) for row in private_map]
    response_by_id = {str(row["request_id"]): dict(row) for row in responses}
    calls: list[dict[str, Any]] = []
    for row in private_rows:
        response = response_by_id.get(str(row["request_id"]))
        parsed = response.get("response") if response else None
        observed = parsed.get("winner") if isinstance(parsed, Mapping) else None
        valid = observed in {"A", "B", "TIE", "ABSTAIN"}
        calls.append(
            {
                **row,
                "observed_winner": observed,
                "valid": valid,
                "correct": bool(valid and observed == row["expected_winner"]),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        grouped[(str(row["scenario_id"]), str(row["pair_id"]), str(row["arm"]))].append(row)
    pair_rows: list[dict[str, Any]] = []
    for (scenario, pair_id, arm), rows in sorted(grouped.items()):
        pair_rows.append(
            {
                "scenario_id": scenario,
                "pair_id": pair_id,
                "arm": arm,
                "oracle_status": rows[0]["oracle_status"],
                "determinate": rows[0]["oracle_status"] == "DETERMINATE",
                "request_count": len(rows),
                "valid_request_count": sum(row["valid"] for row in rows),
                "oracle_label_mass": fmean(row["correct"] for row in rows),
                "strict_all_context_correct": len(rows) == 8 and all(
                    row["correct"] for row in rows
                ),
            }
        )

    by_scenario: dict[str, Any] = {}
    cell_valid_rates = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in pair_rows if row["scenario_id"] == scenario]
        determinate = [row for row in scenario_rows if row["determinate"]]
        by_arm = {}
        for arm in ABLATION_ARMS:
            arm_rows = [row for row in determinate if row["arm"] == arm]
            all_arm_rows = [row for row in scenario_rows if row["arm"] == arm]
            valid_rate = sum(row["valid_request_count"] for row in all_arm_rows) / sum(
                row["request_count"] for row in all_arm_rows
            )
            cell_valid_rates[f"{scenario}::{arm}"] = valid_rate
            by_arm[arm] = {
                "determinate_pair_count": len(arm_rows),
                "context_marginalized_oracle_label_mass": fmean(
                    row["oracle_label_mass"] for row in arm_rows
                ),
                "strict_all_context_correct_pairs": sum(
                    row["strict_all_context_correct"] for row in arm_rows
                ),
                "valid_request_rate": valid_rate,
            }
        by_scenario[scenario] = {"by_arm": by_arm}

    reference = "B2-FullTrace-Control"
    ablations = ("B2-TemporalShuffle", "B2-StateMasked")
    raw_p: dict[str, float] = {}
    contrasts: dict[str, Any] = {}
    for ablation in ablations:
        pair_effects = []
        by_scenario_effects: dict[str, list[float]] = defaultdict(list)
        for scenario in SCENARIOS:
            rows = [
                row
                for row in pair_rows
                if row["scenario_id"] == scenario and row["determinate"]
            ]
            index = {(row["pair_id"], row["arm"]): row for row in rows}
            for pair_id in sorted({row["pair_id"] for row in rows}):
                effect = (
                    index[(pair_id, reference)]["oracle_label_mass"]
                    - index[(pair_id, ablation)]["oracle_label_mass"]
                )
                pair_effects.append(effect)
                by_scenario_effects[scenario].append(effect)
        key = f"{reference}_minus_{ablation}"
        p_value = _paired_sign_flip_p(pair_effects, seed=20260830 + len(contrasts))
        raw_p[key] = p_value
        contrasts[key] = {
            "effect_definition": f"{reference} minus {ablation}; positive favors FullTrace",
            "determinate_pair_count": len(pair_effects),
            "mean_difference": fmean(pair_effects),
            "pair_bootstrap_95_ci": _bootstrap_mean_difference(
                pair_effects, seed=20260828 + len(contrasts)
            ),
            "raw_paired_sign_flip_p": p_value,
            "fulltrace_better_pairs": sum(value > 0 for value in pair_effects),
            "equal_pairs": sum(value == 0 for value in pair_effects),
            "ablation_better_pairs": sum(value < 0 for value in pair_effects),
            "scenario_mean_differences_secondary": {
                scenario: fmean(values)
                for scenario, values in by_scenario_effects.items()
            },
        }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        contrasts[key]["holm_adjusted_p"] = value

    call_index = {
        (
            str(row["scenario_id"]),
            str(row["pair_id"]),
            str(row["arm"]),
            str(row["paraphrase"]),
            str(row["order"]),
            int(row["repeat"]),
        ): row
        for row in calls
    }
    determinate_pair_keys = sorted(
        {
            (str(row["scenario_id"]), str(row["pair_id"]))
            for row in calls
            if row["oracle_status"] == "DETERMINATE"
        }
    )
    posthoc_missingness_sensitivity: dict[str, Any] = {}
    context_keys = [
        (paraphrase, order, repeat)
        for paraphrase in ("P0", "P1")
        for order in ("AB", "BA")
        for repeat in range(2)
    ]
    for contrast_index, ablation in enumerate(ablations):
        matched_pair_effects = []
        normalized_pair_effects = []
        valid_rate_pair_effects = []
        matched_context_counts = []
        for scenario, pair_id in determinate_pair_keys:
            reference_rows = []
            ablation_rows = []
            matched_differences = []
            for paraphrase, order, repeat in context_keys:
                reference_row = call_index[
                    (
                        scenario,
                        pair_id,
                        reference,
                        paraphrase,
                        order,
                        repeat,
                    )
                ]
                ablation_row = call_index[
                    (
                        scenario,
                        pair_id,
                        ablation,
                        paraphrase,
                        order,
                        repeat,
                    )
                ]
                reference_rows.append(reference_row)
                ablation_rows.append(ablation_row)
                if reference_row["valid"] and ablation_row["valid"]:
                    matched_differences.append(
                        float(reference_row["correct"])
                        - float(ablation_row["correct"])
                    )
            matched_context_counts.append(len(matched_differences))
            if matched_differences:
                matched_pair_effects.append(fmean(matched_differences))
            valid_reference = [
                float(row["correct"]) for row in reference_rows if row["valid"]
            ]
            valid_ablation = [
                float(row["correct"]) for row in ablation_rows if row["valid"]
            ]
            if valid_reference and valid_ablation:
                normalized_pair_effects.append(
                    fmean(valid_reference) - fmean(valid_ablation)
                )
            valid_rate_pair_effects.append(
                fmean(float(row["valid"]) for row in reference_rows)
                - fmean(float(row["valid"]) for row in ablation_rows)
            )

        def sensitivity_summary(
            values: Sequence[float], *, seed: int
        ) -> dict[str, Any]:
            return {
                "pair_count": len(values),
                "mean_difference": fmean(values) if values else None,
                "pair_bootstrap_95_ci": _bootstrap_mean_difference(values, seed=seed)
                if values
                else None,
                "paired_sign_flip_p": _paired_sign_flip_p(values, seed=seed + 1)
                if values
                else None,
            }

        posthoc_missingness_sensitivity[
            f"{reference}_minus_{ablation}"
        ] = {
            "status": "POSTHOC_NOT_A_REPLACEMENT_FOR_THE_FROZEN_PRIMARY_ANALYSIS",
            "matched_context_complete_case": {
                **sensitivity_summary(
                    matched_pair_effects, seed=20260910 + contrast_index * 10
                ),
                "contexts_per_pair_min": min(matched_context_counts, default=0),
                "contexts_per_pair_median": median(matched_context_counts)
                if matched_context_counts
                else 0,
                "contexts_per_pair_max": max(matched_context_counts, default=0),
                "pairs_with_no_jointly_valid_context": sum(
                    count == 0 for count in matched_context_counts
                ),
            },
            "armwise_valid_normalized": sensitivity_summary(
                normalized_pair_effects, seed=20260912 + contrast_index * 10
            ),
            "reference_minus_ablation_valid_rate": sensitivity_summary(
                valid_rate_pair_effects, seed=20260914 + contrast_index * 10
            ),
        }

    alpha = float(plan["primary_analysis"]["alpha"])
    temporal_key = f"{reference}_minus_B2-TemporalShuffle"
    state_key = f"{reference}_minus_B2-StateMasked"
    support = {
        "temporal_contribution_supported": contrasts[temporal_key]["mean_difference"] > 0
        and contrasts[temporal_key]["holm_adjusted_p"] < alpha,
        "state_contribution_supported": contrasts[state_key]["mean_difference"] > 0
        and contrasts[state_key]["holm_adjusted_p"] < alpha,
    }
    quality_gates = {
        "expected_2304_request_cells": len(private_rows)
        == int(plan["design"]["expected_total_requests"]),
        "all_288_pair_arm_cells_present": len(pair_rows) == 288,
        "minimum_scenario_arm_valid_rate_at_least_0_8": min(
            cell_valid_rates.values(), default=0.0
        )
        >= 0.8,
    }
    return {
        "schema_version": "controlled-representation-ablation-evaluation-v1",
        "status": "MECHANISTIC_FOLLOWUP_COMPLETE"
        if all(quality_gates.values())
        else "ABLATION_QUALITY_GATE_FAILED",
        "response_count": len(response_by_id),
        "expected_request_count": len(private_rows),
        "by_scenario": by_scenario,
        "primary_contrasts": contrasts,
        "support": support,
        "quality_gates": quality_gates,
        "cell_valid_rates": cell_valid_rates,
        "posthoc_missingness_sensitivity": posthoc_missingness_sensitivity,
        "pair_rows": pair_rows,
        "claim_boundary": plan["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-08-23",
            "verification_status": "ANALYZED",
            "version_label": "controlled_representation_ablation_v1",
        },
    }
