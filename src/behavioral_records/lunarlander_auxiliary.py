from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, stable_hash
from .io import read_json, read_jsonl, write_json, write_jsonl


ARMS = ("B1", "B2-DenseTrace")
ORDERS = ("AB", "BA")
TRACE_COLUMNS = (
    "start_fraction",
    "end_fraction",
    "mean_x",
    "mean_y",
    "mean_vx",
    "mean_vy",
    "mean_angle",
    "mean_angular_velocity",
    "speed_p90",
    "no_engine_fraction",
    "left_engine_fraction",
    "main_engine_fraction",
    "right_engine_fraction",
    "action_switches",
    "left_contact_fraction",
    "right_contact_fraction",
    "reward_sum",
)
STATE_COLUMNS = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)
ACTION_INDEX = {
    "no_engine": 0,
    "left_orientation_engine": 1,
    "main_engine": 2,
    "right_orientation_engine": 3,
}


def _rounded(value: float, decimals: int) -> float:
    result = round(float(value), decimals)
    return 0.0 if result == -0.0 else result


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def anonymize_low_dimensional_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove policy identity without changing any factual metric value."""
    return {
        "record_type": "anonymous_policy_behavior_evidence",
        "aggregate": dict(record["aggregate"]),
        "contexts": {
            str(key): dict(value)
            for key, value in sorted(dict(record["contexts"]).items())
        },
        "coverage": {
            "episodes": int(record["aggregate"]["episodes"]),
            "temporal_step_order_available": False,
        },
    }


def _state_array(state: Mapping[str, Any], decimals: int) -> list[Any]:
    return [
        _rounded(state["x"], decimals),
        _rounded(state["y"], decimals),
        _rounded(state["vx"], decimals),
        _rounded(state["vy"], decimals),
        _rounded(state["angle"], decimals),
        _rounded(state["angular_velocity"], decimals),
        bool(state["left_leg_contact"]),
        bool(state["right_leg_contact"]),
    ]


def _first_contact(steps: Sequence[Mapping[str, Any]], decimals: int) -> list[Any] | None:
    for step in steps:
        state = dict(step["state"])
        if bool(state["left_leg_contact"]) or bool(state["right_leg_contact"]):
            return [int(step["t"]), *_state_array(state, decimals)]
    return None


def _time_bin(
    steps: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    decimals: int,
) -> list[Any]:
    selected = list(steps[start:end])
    if not selected:
        raise ValueError("A normalized time bin cannot be empty")
    states = [dict(step["state"]) for step in selected]
    actions = [str(step["action"]) for step in selected]
    unknown = sorted(set(actions) - set(ACTION_INDEX))
    if unknown:
        raise ValueError(f"Unknown LunarLander actions in trace: {unknown}")
    count = len(selected)
    action_counts = [0, 0, 0, 0]
    for action in actions:
        action_counts[ACTION_INDEX[action]] += 1
    speeds = [
        math.hypot(float(state["vx"]), float(state["vy"])) for state in states
    ]
    length = len(steps)
    return [
        _rounded(start / length, decimals),
        _rounded(end / length, decimals),
        *[
            _rounded(fmean(float(state[field]) for state in states), decimals)
            for field in ("x", "y", "vx", "vy", "angle", "angular_velocity")
        ],
        _rounded(_quantile(speeds, 0.9), decimals),
        *[_rounded(value / count, decimals) for value in action_counts],
        sum(before != after for before, after in zip(actions, actions[1:])),
        _rounded(fmean(bool(state["left_leg_contact"]) for state in states), decimals),
        _rounded(fmean(bool(state["right_leg_contact"]) for state in states), decimals),
        _rounded(sum(float(step["reward"]) for step in selected), decimals),
    ]


def compile_dense_trace_record(
    low_dimensional_record: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
    *,
    normalized_time_bins: int,
    decimals: int,
) -> dict[str, Any]:
    """Compile a fixed, objective-blind, all-episode temporal evidence record."""
    if normalized_time_bins <= 0:
        raise ValueError("normalized_time_bins must be positive")
    expected_policy_ids = {
        str(row.get("policy_id", "")) for row in trajectories if row.get("policy_id")
    }
    if len(expected_policy_ids) > 1:
        raise ValueError("Dense trace compilation received multiple policies")
    episodes = []
    total_steps = 0
    accounted_steps = 0
    for row in sorted(
        trajectories,
        key=lambda item: (
            str(dict(item["context"])["context_id"]),
            str(item["episode_id"]),
        ),
    ):
        steps = list(row["steps"])
        if not steps:
            raise ValueError("A validation trajectory contains no steps")
        length = len(steps)
        boundaries = [math.floor(index * length / normalized_time_bins) for index in range(normalized_time_bins + 1)]
        boundaries[-1] = length
        bins = []
        for index in range(normalized_time_bins):
            start, end = boundaries[index], boundaries[index + 1]
            if end <= start:
                end = min(length, start + 1)
            bins.append(_time_bin(steps, start, end, decimals))
            accounted_steps += end - start
        actions = [str(step["action"]) for step in steps]
        public_fingerprint = {
            "context": dict(row["context"]),
            "return": row["return"],
            "length": row["length"],
            "outcome": row["outcome"],
            "steps": steps,
        }
        context = dict(row["context"])
        context.pop("split", None)
        context.pop("seed", None)
        context.pop("context_seed", None)
        episodes.append(
            {
                "evidence_id": stable_hash(public_fingerprint)[:16],
                "context": context,
                "outcome": str(row["outcome"]),
                "return": _rounded(row["return"], decimals),
                "length": length,
                "total_action_switches": sum(
                    before != after for before, after in zip(actions, actions[1:])
                ),
                "first_contact": _first_contact(steps, decimals),
                "final_state": _state_array(dict(steps[-1]["state"]), decimals),
                "time_bins": bins,
            }
        )
        total_steps += length
    if accounted_steps != total_steps:
        raise AssertionError(
            f"Dense trace step accounting mismatch: {accounted_steps} != {total_steps}"
        )
    nested_b1 = anonymize_low_dimensional_record(low_dimensional_record)
    return {
        **nested_b1,
        "coverage": {
            "episodes": len(episodes),
            "steps": total_steps,
            "temporal_step_order_available": "between_bins_only",
        },
        "trace_schema": {
            "state_columns": list(STATE_COLUMNS),
            "first_contact_columns": ["t", *STATE_COLUMNS],
            "time_bin_columns": list(TRACE_COLUMNS),
            "normalized_bins_per_episode": normalized_time_bins,
        },
        "episodes": episodes,
        "loss_ledger": {
            "episode_coverage_fraction": 1.0,
            "step_accounting_fraction": 1.0,
            "within_bin_order": "not_represented",
            "within_bin_extreme_timing": "not_represented",
            "numeric_rounding_decimals": decimals,
            "candidate_identity": "removed",
        },
    }


def build_auxiliary_prompt(
    requirement: str,
    evidence_a: Mapping[str, Any],
    evidence_b: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "Choose the policy whose observed behavior best satisfies the requirement.",
            "The records were generated by a deterministic program without access to the requirement.",
            "Apply the priorities and contextual exceptions expressed in the requirement; do not replace them with return maximization.",
            "Use only the supplied factual validation evidence. Do not infer training method or policy identity.",
            "Use TIE only for equivalent satisfaction and ABSTAIN only when the evidence cannot resolve the comparison.",
            f"REQUIREMENT: {requirement}",
            f"POLICY A EVIDENCE: {canonical_json(evidence_a)}",
            f"POLICY B EVIDENCE: {canonical_json(evidence_b)}",
            "Return exactly one JSON object with keys winner, confidence, evidence_refs, and reason.",
            'winner must be "A", "B", "TIE", or "ABSTAIN".',
            "confidence must be a JSON number from 0 to 1.",
            "evidence_refs must be a JSON list of short factual field names from the records.",
        ]
    )


def _expected_winner(oracle_label: str, order: str) -> str:
    if oracle_label not in {"A", "B", "TIE"}:
        raise ValueError(f"Unsupported oracle label: {oracle_label}")
    if order == "AB" or oracle_label == "TIE":
        return oracle_label
    return "B" if oracle_label == "A" else "A"


def _contains_forbidden_prompt_text(
    prompt: str,
    *,
    candidate_ids: Sequence[str],
) -> list[str]:
    lowered = prompt.lower()
    forbidden_literals = [
        "oracle",
        "heldout",
        "held-out",
        "expected_winner",
        "training_seed",
        "checkpoint",
    ]
    hits = [literal for literal in forbidden_literals if literal in lowered]
    hits.extend(candidate_id for candidate_id in candidate_ids if candidate_id in prompt)
    return sorted(set(hits))


def build_lunarlander_auxiliary_bundle(
    *,
    config: Mapping[str, Any],
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root_path = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = dict(config["source"])
    training_dir = root_path / source["training_run_dir"]
    offline_dir = root_path / source["offline_gate_run_dir"]
    offline_summary = read_json(offline_dir / "summary.json")
    training_summary = read_json(training_dir / "summary.json")
    low_rows = read_jsonl(offline_dir / "low_dimensional_validation_records.jsonl")
    pair_rows = read_jsonl(offline_dir / config["pair_freeze"]["source_file"])
    low_by_id = {str(row["candidate_id"]): row for row in low_rows}
    candidate_rows = {
        str(row["candidate_id"]): row for row in training_summary["candidate_rows"]
    }
    candidate_ids = sorted(candidate_rows)

    source_gates = {
        "offline_summary_hash_matches": stable_hash(offline_summary)
        == source["expected_offline_gate_summary_hash"],
        "original_failure_status_preserved": offline_summary.get("status")
        == source["required_original_status"],
        "original_llm_calls_were_zero": int(offline_summary.get("api_calls", -1)) == 0,
        "candidate_sets_match": set(low_by_id) == set(candidate_rows),
    }
    trace_cfg = dict(config["dense_trace_compiler"])
    evidence: dict[str, dict[str, Any]] = {}
    record_audit_rows = []
    for candidate_id in candidate_ids:
        trajectory_path = training_dir / candidate_rows[candidate_id][
            "validation_trajectory_path"
        ]
        trajectories = read_jsonl(trajectory_path)
        b1 = anonymize_low_dimensional_record(low_by_id[candidate_id])
        b2 = compile_dense_trace_record(
            low_by_id[candidate_id],
            trajectories,
            normalized_time_bins=int(trace_cfg["normalized_time_bins_per_episode"]),
            decimals=int(trace_cfg["floating_point_decimals"]),
        )
        nested_b1 = {key: b2[key] for key in ("record_type", "aggregate", "contexts")}
        b1_core = {key: b1[key] for key in ("record_type", "aggregate", "contexts")}
        record_audit_rows.append(
            {
                "candidate_id_private": candidate_id,
                "B1_hash": stable_hash(b1),
                "B2_hash": stable_hash(b2),
                "B2_nests_B1_factual_panel": nested_b1 == b1_core,
                "episode_count": int(b2["coverage"]["episodes"]),
                "step_count": int(b2["coverage"]["steps"]),
                "B1_characters": len(canonical_json(b1)),
                "B2_characters": len(canonical_json(b2)),
            }
        )
        evidence[candidate_id] = {"B1": b1, "B2-DenseTrace": b2}

    requirements = dict(config["requirements"])
    frozen_pairs: dict[str, list[dict[str, Any]]] = {}
    pair_gates: dict[str, Any] = {}
    expected_pairs = int(config["pair_freeze"]["pairs_per_requirement"])
    for requirement_id in sorted(requirements):
        rows = [
            dict(row)
            for row in pair_rows
            if str(row["requirement_id"]) == requirement_id
        ]
        frozen_pairs[requirement_id] = rows
        pair_gates[requirement_id] = {
            "pair_count": len(rows),
            "pair_count_passed": len(rows) == expected_pairs,
            "all_determinate": all(row["oracle_status"] == "DETERMINATE" for row in rows),
            "labels": {
                label: sum(row["oracle_label"] == label for row in rows)
                for label in ("A", "B", "TIE")
            },
        }

    requests = []
    private_map = []
    prompt_lengths = []
    prompt_forbidden_hits: dict[str, list[str]] = {}
    repeats = int(config["judge_design"]["repeats_per_requirement_paraphrase_and_order"])
    for requirement_id in sorted(requirements):
        for pair in frozen_pairs[requirement_id]:
            pair_id = str(pair["pair_id"])
            base_left = str(pair["policy_a"])
            base_right = str(pair["policy_b"])
            for arm in ARMS:
                for paraphrase_index, requirement in enumerate(requirements[requirement_id]):
                    paraphrase_id = f"P{paraphrase_index}"
                    for order in ORDERS:
                        left, right = (
                            (base_left, base_right)
                            if order == "AB"
                            else (base_right, base_left)
                        )
                        prompt = build_auxiliary_prompt(
                            str(requirement), evidence[left][arm], evidence[right][arm]
                        )
                        prompt_lengths.append(len(prompt))
                        hits = _contains_forbidden_prompt_text(
                            prompt, candidate_ids=candidate_ids
                        )
                        for repeat in range(repeats):
                            request_id = (
                                f"{requirement_id}__{pair_id}__{arm}__"
                                f"{paraphrase_id}__{order}__R{repeat}"
                            )
                            if hits:
                                prompt_forbidden_hits[request_id] = hits
                            requests.append(
                                {
                                    "request_id": request_id,
                                    "objective_id": requirement_id,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase_id,
                                    "order": order,
                                    "repeat": repeat,
                                    "response_schema": "semantic-selection-response-v1",
                                    "prompt": prompt,
                                    "prompt_hash": stable_hash(prompt),
                                }
                            )
                            private_map.append(
                                {
                                    "request_id": request_id,
                                    "requirement_id": requirement_id,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase_id,
                                    "order": order,
                                    "repeat": repeat,
                                    "alias_a_policy_id": left,
                                    "alias_b_policy_id": right,
                                    "oracle_status": pair["oracle_status"],
                                    "expected_winner": _expected_winner(
                                        str(pair["oracle_label"]), order
                                    ),
                                }
                            )

    maximum_prompt_chars = int(config["judge_design"]["maximum_prompt_characters"])
    expected_requests = int(config["judge_design"]["expected_total_requests"])
    compiler_parameters = inspect.signature(compile_dense_trace_record).parameters
    bundle_gates = {
        **source_gates,
        "pair_freeze_passed": all(
            row["pair_count_passed"] and row["all_determinate"]
            for row in pair_gates.values()
        ),
        "B2_nests_B1_factual_panel": all(
            row["B2_nests_B1_factual_panel"] for row in record_audit_rows
        ),
        "all_records_cover_48_episodes": all(
            row["episode_count"] == 48 for row in record_audit_rows
        ),
        "compiler_signature_is_objective_blind": not any(
            key in compiler_parameters
            for key in ("objective", "requirement", "oracle", "judge_response")
        ),
        "prompts_have_no_forbidden_identity_or_oracle_text": not prompt_forbidden_hits,
        "prompt_budget_passed": max(prompt_lengths, default=0) <= maximum_prompt_chars,
        "request_count_passed": len(requests) == expected_requests,
        "request_ids_unique": len({row["request_id"] for row in requests})
        == len(requests),
        "api_calls": 0,
    }
    summary = {
        "schema_version": "lunarlander-post-failure-auxiliary-bundle-summary-v1",
        "status": "READY_FOR_AUXILIARY_JUDGE"
        if all(value is True or key == "api_calls" and value == 0 for key, value in bundle_gates.items())
        else "OFFLINE_BUNDLE_GATE_FAILED",
        "study_role": config["study_role"],
        "config_hash": stable_hash(config),
        "source_offline_summary_hash": stable_hash(offline_summary),
        "source_original_status": offline_summary.get("status"),
        "gates": bundle_gates,
        "pair_audits": pair_gates,
        "candidate_count": len(candidate_ids),
        "record_character_range": {
            "B1_min": min(row["B1_characters"] for row in record_audit_rows),
            "B1_max": max(row["B1_characters"] for row in record_audit_rows),
            "B2_min": min(row["B2_characters"] for row in record_audit_rows),
            "B2_max": max(row["B2_characters"] for row in record_audit_rows),
        },
        "prompt_character_range": {
            "min": min(prompt_lengths, default=0),
            "max": max(prompt_lengths, default=0),
            "mean": fmean(prompt_lengths) if prompt_lengths else 0.0,
        },
        "request_count": len(requests),
        "request_hash": stable_hash(requests),
        "private_map_hash": stable_hash(private_map),
        "prompt_forbidden_hits": prompt_forbidden_hits,
        "api_calls": 0,
        "human_construct_audit_complete": False,
        "manuscript_claim_authorized": False,
        "claim_boundary": config["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-08-23",
            "verification_status": "OFFLINE_BUNDLE_VERIFIED",
            "version_label": "lunarlander_auxiliary_v1",
        },
    }
    write_jsonl(output / "judge_requests.jsonl", requests)
    write_jsonl(output / "private_request_map.jsonl", private_map)
    write_jsonl(output / "record_audit_private.jsonl", record_audit_rows)
    write_json(output / "summary.json", summary)
    return summary


def _bootstrap_mean_difference(
    values: Sequence[float], *, seed: int = 20260823
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = [
        fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(10_000)
    ]
    means.sort()
    return [means[249], means[9749]]


def _paired_sign_flip_p(values: Sequence[float], *, seed: int = 20260824) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(fmean(nonzero))
    if len(nonzero) <= 20:
        scores = [
            abs(
                fmean(
                    value if mask & (1 << index) else -value
                    for index, value in enumerate(nonzero)
                )
            )
            for mask in range(2 ** len(nonzero))
        ]
        return sum(score >= observed - 1e-12 for score in scores) / len(scores)
    rng = random.Random(seed)
    draws = 100_000
    exceed = 0
    for _ in range(draws):
        score = fmean(value if rng.random() < 0.5 else -value for value in nonzero)
        exceed += abs(score) >= observed - 1e-12
    return (exceed + 1) / (draws + 1)


def _physical_choice(row: Mapping[str, Any]) -> str:
    observed = row.get("observed_winner")
    if observed == "A":
        return str(row["alias_a_policy_id"])
    if observed == "B":
        return str(row["alias_b_policy_id"])
    return str(observed)


def evaluate_lunarlander_auxiliary(
    *, private_map: Iterable[Mapping[str, Any]], responses: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    private_rows = [dict(row) for row in private_map]
    response_by_id = {str(row["request_id"]): dict(row) for row in responses}
    calls = []
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
        grouped[(row["requirement_id"], row["pair_id"], row["arm"])].append(row)
    pair_rows = []
    for (requirement_id, pair_id, arm), rows in sorted(grouped.items()):
        order_contexts: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            order_contexts[(row["paraphrase"], int(row["repeat"]))].append(row)
        both_order = [
            len(context_rows) == 2 and all(row["correct"] for row in context_rows)
            for context_rows in order_contexts.values()
        ]
        order_agreement = [
            len(context_rows) == 2
            and len({_physical_choice(row) for row in context_rows}) == 1
            for context_rows in order_contexts.values()
        ]
        pair_rows.append(
            {
                "requirement_id": requirement_id,
                "pair_id": pair_id,
                "arm": arm,
                "request_count": len(rows),
                "valid_response_rate": fmean(row["valid"] for row in rows),
                "oracle_label_mass": fmean(row["correct"] for row in rows),
                "strict_both_order_rate": fmean(both_order),
                "AB_BA_physical_choice_agreement": fmean(order_agreement),
            }
        )

    by_requirement = {}
    for requirement_id in sorted({row["requirement_id"] for row in pair_rows}):
        selected = [row for row in pair_rows if row["requirement_id"] == requirement_id]
        indexed = {(row["pair_id"], row["arm"]): row for row in selected}
        pair_ids = sorted({row["pair_id"] for row in selected})
        complete_pair_ids = [
            pair_id
            for pair_id in pair_ids
            if all((pair_id, arm) in indexed for arm in ARMS)
        ]
        differences = [
            indexed[(pair_id, "B2-DenseTrace")]["oracle_label_mass"]
            - indexed[(pair_id, "B1")]["oracle_label_mass"]
            for pair_id in complete_pair_ids
        ]
        arm_summary = {}
        for arm in ARMS:
            rows = [row for row in selected if row["arm"] == arm]
            arm_summary[arm] = {
                "pair_count": len(rows),
                "context_marginalized_oracle_label_mass": fmean(
                    row["oracle_label_mass"] for row in rows
                )
                if rows
                else 0.0,
                "strict_both_order_rate": fmean(
                    row["strict_both_order_rate"] for row in rows
                )
                if rows
                else 0.0,
                "AB_BA_physical_choice_agreement": fmean(
                    row["AB_BA_physical_choice_agreement"] for row in rows
                )
                if rows
                else 0.0,
                "valid_response_rate": fmean(row["valid_response_rate"] for row in rows)
                if rows
                else 0.0,
            }
        by_requirement[requirement_id] = {
            "by_arm": arm_summary,
            "B2_minus_B1": {
                "complete_pair_count": len(complete_pair_ids),
                "mean_difference": fmean(differences) if differences else 0.0,
                "paired_pair_bootstrap_95_ci": _bootstrap_mean_difference(differences),
                "paired_sign_flip_p": _paired_sign_flip_p(differences),
                "B2_better_pairs": sum(value > 0 for value in differences),
                "equal_pairs": sum(value == 0 for value in differences),
                "B1_better_pairs": sum(value < 0 for value in differences),
            },
        }
    return {
        "schema_version": "lunarlander-post-failure-auxiliary-evaluation-v1",
        "response_count": len(response_by_id),
        "expected_request_count": len(private_rows),
        "by_requirement": by_requirement,
        "pair_rows": pair_rows,
        "calls": calls,
        "manuscript_claim_authorized": False,
        "claim_boundary": (
            "Post-failure auxiliary standard-task stress test only. The policy pair is "
            "the inference unit; repeated calls are probability probes, not independent samples."
        ),
    }
