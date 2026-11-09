from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical import canonical_json, stable_hash
from .lunarlander_training import context_lookup, make_lunarlander_env
from .lunarlander_transfer import ACTION_LABELS, lunarlander_state


def validate_semantic_gate_config(
    config: Mapping[str, Any],
    transfer_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    heldout = config.get("heldout_collection", {})
    source = config.get("source", {})
    expected_heldout = transfer_config["candidate_pool"][
        "evaluation_heldout_seeds_per_context"
    ]
    context_ids = set(context_lookup(transfer_config))
    seed_bases = set(heldout.get("seed_bases", {}))
    requirements = set(transfer_config["semantic_requirements"])
    oracle_requirements = set(config.get("hidden_oracle", {}))
    matching = config.get("metric_matching", {})
    gates = {
        "training_summary_hash_matches": stable_hash(training_summary)
        == source.get("expected_training_summary_hash"),
        "training_pool_passed": training_summary.get("status")
        == "PASS_TRAINING_POOL_GATES_PENDING_PAIR_AND_ORACLE_GATES",
        "candidate_count_is_frozen": int(training_summary.get("candidate_count", -1))
        == int(transfer_config["candidate_pool"]["expected_candidate_count"]),
        "heldout_episode_count_matches": int(
            heldout.get("episodes_per_context", -1)
        )
        == int(expected_heldout),
        "heldout_contexts_match": seed_bases == context_ids,
        "heldout_is_objective_and_judge_blind": heldout.get(
            "accepts_validation_record"
        )
        is False
        and heldout.get("accepts_natural_language_objective") is False
        and heldout.get("accepts_judge_response") is False,
        "oracle_requirements_match": oracle_requirements == requirements,
        "matching_fraction_is_valid": 0.0
        < float(matching.get("eligible_nearest_pair_fraction", 0.0))
        <= 1.0,
        "matching_features_are_unique": bool(matching.get("features"))
        and len(matching["features"]) == len(set(matching["features"])),
        "heldout_matches_training_reservation": heldout.get("seed_bases")
        == training_config["heldout_reservation"]["seed_bases"],
        "zero_api_stopping_rule_present": int(
            config["qualification_gates"]["api_calls_before_all_offline_gates_pass"]
        )
        == 0,
    }
    return {"passed": all(gates.values()), "gates": gates}


def heldout_seed_map(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, list[int]]:
    count = int(config["heldout_collection"]["episodes_per_context"])
    if smoke:
        count = 2
    return {
        str(context_id): list(range(int(base), int(base) + count))
        for context_id, base in config["heldout_collection"]["seed_bases"].items()
    }


def rollout_hidden_episode(
    *,
    model: Any,
    transfer_config: Mapping[str, Any],
    candidate_id: str,
    context_id: str,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = make_lunarlander_env(transfer_config, context_id)
    contexts = context_lookup(transfer_config)
    try:
        observation, _ = env.reset(seed=int(seed))
        steps: list[dict[str, Any]] = []
        total_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            predicted, _ = model.predict(observation, deterministic=bool(deterministic))
            action = int(predicted.item()) if hasattr(predicted, "item") else int(predicted)
            observation, reward, terminated, truncated, _ = env.step(action)
            unwrapped = env.unwrapped
            state = lunarlander_state(observation)
            steps.append(
                {
                    "t": len(steps),
                    "action": ACTION_LABELS[action],
                    "action_id": action,
                    "reward": float(reward),
                    "state": state,
                    "body_contact_game_over": bool(unwrapped.game_over),
                    "lander_awake": bool(unwrapped.lander.awake),
                }
            )
            total_return += float(reward)
        final = steps[-1]
        if bool(final["body_contact_game_over"]):
            terminal_cause = "body_contact_termination"
        elif abs(float(final["state"]["x"])) >= 1.0:
            terminal_cause = "horizontal_boundary_termination"
        elif not bool(final["lander_awake"]):
            terminal_cause = "settled_termination"
        elif truncated:
            terminal_cause = "time_limit_truncation"
        else:
            terminal_cause = "other_termination"
        return {
            "schema_version": "lunarlander-heldout-raw-private-v1",
            "candidate_id": str(candidate_id),
            "episode_id": f"ll_hidden_{stable_hash({'candidate': candidate_id, 'context': context_id, 'seed': int(seed)}, 16)}",
            "context": dict(contexts[context_id]),
            "evaluation_seed_private": int(seed),
            "return": total_return,
            "length": len(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_cause": terminal_cause,
            "steps": steps,
        }
    finally:
        env.close()


def write_gzip_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def read_gzip_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not an object")
            rows.append(row)
    return rows


def _quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _substantial(state: Mapping[str, Any], threshold: Mapping[str, Any]) -> bool:
    return bool(
        abs(float(state["x"])) >= float(threshold["absolute_x"])
        or abs(float(state["vx"])) >= float(threshold["absolute_vx"])
        or abs(float(state["angle"])) >= float(threshold["absolute_angle"])
        or abs(float(state["angular_velocity"]))
        >= float(threshold["absolute_angular_velocity"])
    )


def _controlled(state: Mapping[str, Any], threshold: Mapping[str, Any]) -> bool:
    return bool(
        abs(float(state["x"])) <= float(threshold["absolute_x"])
        and abs(float(state["vx"])) <= float(threshold["absolute_vx"])
        and abs(float(state["angle"])) <= float(threshold["absolute_angle"])
        and abs(float(state["angular_velocity"]))
        <= float(threshold["absolute_angular_velocity"])
    )


def hidden_episode_facts(
    row: Mapping[str, Any],
    operationalization: Mapping[str, Any],
) -> dict[str, Any]:
    steps = list(row["steps"])
    if not steps:
        raise ValueError("Hidden LunarLander episode contains no steps")
    actions = [int(step["action_id"]) for step in steps]
    states = [dict(step["state"]) for step in steps]
    first_contact_index = None
    for index, state in enumerate(states):
        if bool(state["left_leg_contact"]) or bool(state["right_leg_contact"]):
            first_contact_index = index
            break
    contact_state = states[first_contact_index] if first_contact_index is not None else None
    if contact_state is None:
        contact_speed = None
        contact_abs_angle = None
        contact_abs_angular_velocity = None
        contact_severity = None
        controlled_contact = False
    else:
        contact_speed = math.hypot(
            float(contact_state["vx"]), float(contact_state["vy"])
        )
        contact_abs_angle = abs(float(contact_state["angle"]))
        contact_abs_angular_velocity = abs(
            float(contact_state["angular_velocity"])
        )
        controlled_threshold = operationalization["controlled_first_contact"]
        controlled_contact = bool(
            contact_speed <= float(controlled_threshold["speed"])
            and contact_abs_angle <= float(controlled_threshold["absolute_angle"])
            and contact_abs_angular_velocity
            <= float(controlled_threshold["absolute_angular_velocity"])
        )
        contact_severity = (
            contact_speed
            + 0.5 * contact_abs_angle
            + 0.25 * contact_abs_angular_velocity
        )

    substantial_threshold = operationalization["substantial_motion"]
    controlled_threshold = operationalization["regained_control"]
    required_streak = int(controlled_threshold["consecutive_steps"])
    substantial_seen = False
    stable_streak = 0
    recovery_index = None
    for index, state in enumerate(states):
        if _substantial(state, substantial_threshold):
            substantial_seen = True
            stable_streak = 0
        elif substantial_seen and _controlled(state, controlled_threshold):
            stable_streak += 1
            if stable_streak >= required_streak and recovery_index is None:
                recovery_index = index
        else:
            stable_streak = 0

    alternations = 0
    if recovery_index is not None:
        controlled_side_commands = [
            (index, actions[index])
            for index in range(recovery_index, len(actions))
            if actions[index] in {1, 3}
            and _controlled(states[index], controlled_threshold)
        ]
        maximum_gap = int(controlled_threshold["alternation_max_gap_steps"])
        for before, after in zip(
            controlled_side_commands, controlled_side_commands[1:]
        ):
            if after[0] - before[0] <= maximum_gap and after[1] != before[1]:
                alternations += 1

    engine_switches = sum(before != after for before, after in zip(actions, actions[1:]))
    engine_commands = sum(action != 0 for action in actions)
    late = operationalization["brittle_late_rescue"]
    tail_steps = steps[-int(late["tail_steps"]) :]
    tail_states = [dict(step["state"]) for step in tail_steps]
    tail_actions = [int(step["action_id"]) for step in tail_steps]
    tail_max_speed = max(
        math.hypot(float(state["vx"]), float(state["vy"]))
        for state in tail_states
    )
    tail_max_abs_angle = max(abs(float(state["angle"])) for state in tail_states)
    tail_switches = sum(
        before != after for before, after in zip(tail_actions, tail_actions[1:])
    )
    settled = str(row["terminal_cause"]) == "settled_termination"
    brittle_late_rescue = bool(
        settled
        and (
            tail_max_speed >= float(late["motion_speed"])
            or tail_max_abs_angle >= float(late["absolute_angle"])
        )
        and tail_switches >= int(late["minimum_engine_switches"])
    )
    final_state = states[-1]
    unresolved = operationalization["unresolved_terminal_motion"]
    unresolved_terminal_motion = bool(
        not settled
        and (
            math.hypot(float(final_state["vx"]), float(final_state["vy"]))
            >= float(unresolved["speed"])
            or abs(float(final_state["angle"]))
            >= float(unresolved["absolute_angle"])
            or abs(float(final_state["angular_velocity"]))
            >= float(unresolved["absolute_angular_velocity"])
        )
    )
    catastrophic = str(row["terminal_cause"]) in {
        "body_contact_termination",
        "horizontal_boundary_termination",
    }
    return {
        "episode_id": str(row["episode_id"]),
        "context_id": str(row["context"]["context_id"]),
        "settled_termination": settled,
        "controlled_first_contact": bool(settled and controlled_contact),
        "catastrophic_termination": catastrophic,
        "post_recovery_alternations": alternations,
        "post_recovery_alternation_episode": alternations > 0,
        "first_contact_speed": contact_speed,
        "first_contact_abs_angle": contact_abs_angle,
        "first_contact_abs_angular_velocity": contact_abs_angular_velocity,
        "first_contact_severity": contact_severity,
        "brittle_late_rescue": brittle_late_rescue,
        "unresolved_terminal_motion": unresolved_terminal_motion,
        "engine_commands": engine_commands,
        "engine_switches": engine_switches,
        "steps": len(steps),
        "return": float(row["return"]),
        "terminal_cause": str(row["terminal_cause"]),
    }


def _profile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot profile an empty hidden-episode group")
    count = len(rows)
    contact_severity = [
        float(row["first_contact_severity"])
        for row in rows
        if row["first_contact_severity"] is not None
    ]
    cause_counts = Counter(str(row["terminal_cause"]) for row in rows)
    total_steps = sum(int(row["steps"]) for row in rows)
    return {
        "episodes": count,
        "terminal_cause_counts": dict(sorted(cause_counts.items())),
        "settled_terminal_rate": sum(bool(row["settled_termination"]) for row in rows)
        / count,
        "controlled_landing_rate": sum(
            bool(row["controlled_first_contact"]) for row in rows
        )
        / count,
        "catastrophic_termination_rate": sum(
            bool(row["catastrophic_termination"]) for row in rows
        )
        / count,
        "post_recovery_alternation_episode_rate": sum(
            bool(row["post_recovery_alternation_episode"]) for row in rows
        )
        / count,
        "post_recovery_alternations_per_episode": sum(
            int(row["post_recovery_alternations"]) for row in rows
        )
        / count,
        "first_contact_speed_p50": _quantile(
            [float(row["first_contact_speed"]) for row in rows if row["first_contact_speed"] is not None],
            0.5,
            default=10.0,
        ),
        "first_contact_speed_p90": _quantile(
            [float(row["first_contact_speed"]) for row in rows if row["first_contact_speed"] is not None],
            0.9,
            default=10.0,
        ),
        "first_contact_severity_p90": _quantile(
            contact_severity, 0.9, default=10.0
        ),
        "brittle_late_rescue_rate": sum(
            bool(row["brittle_late_rescue"]) for row in rows
        )
        / count,
        "unresolved_terminal_motion_rate": sum(
            bool(row["unresolved_terminal_motion"]) for row in rows
        )
        / count,
        "engine_command_fraction": sum(int(row["engine_commands"]) for row in rows)
        / max(1, total_steps),
        "engine_switch_fraction": sum(int(row["engine_switches"]) for row in rows)
        / max(1, total_steps - count),
        "mean_return": sum(float(row["return"]) for row in rows) / count,
        "mean_length": total_steps / count,
    }


def build_hidden_policy_profile(
    candidate_id: str,
    raw_rows: Sequence[Mapping[str, Any]],
    operationalization: Mapping[str, Any],
) -> dict[str, Any]:
    if {str(row["candidate_id"]) for row in raw_rows} != {str(candidate_id)}:
        raise ValueError("Hidden rows mix candidate identities")
    episode_rows = [
        hidden_episode_facts(row, operationalization) for row in raw_rows
    ]
    aggregate = _profile_rows(episode_rows)
    contexts = {
        context_id: _profile_rows(
            [row for row in episode_rows if row["context_id"] == context_id]
        )
        for context_id in sorted({str(row["context_id"]) for row in episode_rows})
    }
    controlled_rates = [float(row["controlled_landing_rate"]) for row in contexts.values()]
    catastrophic_rates = [
        float(row["catastrophic_termination_rate"]) for row in contexts.values()
    ]
    aggregate.update(
        {
            "worst_context_controlled_landing_rate": min(controlled_rates),
            "cross_context_controlled_landing_range": max(controlled_rates)
            - min(controlled_rates),
            "worst_context_catastrophic_termination_rate": max(catastrophic_rates),
        }
    )
    return {
        "schema_version": "lunarlander-hidden-policy-profile-v1",
        "candidate_id": str(candidate_id),
        "aggregate": aggregate,
        "contexts": contexts,
        "episode_fact_hash": stable_hash(episode_rows),
        "heldout_episode_count": len(episode_rows),
    }


def compare_hidden_profiles(
    profile_a: Mapping[str, Any],
    profile_b: Mapping[str, Any],
    oracle_rule: Mapping[str, Any],
    *,
    margin_multiplier: float = 1.0,
) -> dict[str, Any]:
    aggregate_a = profile_a["aggregate"]
    aggregate_b = profile_b["aggregate"]
    for field, direction, margin in oracle_rule["lexicographic_fields"]:
        value_a = float(aggregate_a[field])
        value_b = float(aggregate_b[field])
        difference = value_a - value_b
        effective_margin = float(margin) * float(margin_multiplier)
        if abs(difference) + 1e-12 < effective_margin:
            continue
        if str(direction) == "maximize":
            label = "A" if difference > 0 else "B"
        elif str(direction) == "minimize":
            label = "A" if difference < 0 else "B"
        else:
            raise ValueError(f"Unknown oracle direction: {direction}")
        return {
            "status": "DETERMINATE",
            "label": label,
            "decisive_field": str(field),
            "direction": str(direction),
            "value_a": value_a,
            "value_b": value_b,
            "difference_a_minus_b": difference,
            "effective_margin": effective_margin,
        }
    return {
        "status": "AMBIGUOUS",
        "label": "TIE",
        "decisive_field": None,
        "direction": None,
        "value_a": None,
        "value_b": None,
        "difference_a_minus_b": None,
        "effective_margin": None,
    }


def compile_low_dimensional_validation_record(
    candidate_id: str,
    trajectory_path: str | Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with Path(trajectory_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 48:
        raise ValueError(
            f"Expected 48 validation episodes for {candidate_id}, observed {len(rows)}"
        )
    if {str(row["policy_id"]) for row in rows} != {str(candidate_id)}:
        raise ValueError("Validation file mixes candidate identities")

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        returns = np.asarray([float(row["return"]) for row in selected], dtype=float)
        lengths = np.asarray([int(row["length"]) for row in selected], dtype=float)
        actions = Counter(
            str(step["action"]) for row in selected for step in row["steps"]
        )
        events = Counter(
            str(event)
            for row in selected
            for step in row["steps"]
            for event in step.get("events", [])
        )
        total_steps = int(lengths.sum())
        return {
            "episodes": len(selected),
            "success_rate": float(np.mean(returns >= 200.0)),
            "mean_return": float(np.mean(returns)),
            "return_standard_deviation": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "length_standard_deviation": float(np.std(lengths)),
            "termination_rate": sum(bool(row["terminated"]) for row in selected)
            / len(selected),
            "no_engine_fraction": actions["no_engine"] / max(1, total_steps),
            "main_engine_fraction": actions["main_engine"] / max(1, total_steps),
            "orientation_engine_fraction": (
                actions["left_orientation_engine"]
                + actions["right_orientation_engine"]
            )
            / max(1, total_steps),
            "horizontal_velocity_sign_changes_per_step": events[
                "horizontal_velocity_sign_change"
            ]
            / max(1, total_steps),
            "angle_sign_changes_per_step": events["angle_sign_change"]
            / max(1, total_steps),
            "contact_changes_per_step": sum(
                count for event, count in events.items() if "leg_contact_" in event
            )
            / max(1, total_steps),
        }

    aggregate = summarize(rows)
    contexts = {
        context_id: summarize(
            [row for row in rows if str(row["context"]["context_id"]) == context_id]
        )
        for context_id in sorted(
            {str(row["context"]["context_id"]) for row in rows}
        )
    }
    return {
        "schema_version": "lunarlander-low-dimensional-validation-v1",
        "candidate_id": str(candidate_id),
        "representation": "objective_blind_low_dimensional_aggregate_v1",
        "aggregate": aggregate,
        "contexts": contexts,
    }


def flatten_matching_features(
    record: Mapping[str, Any], feature_names: Sequence[str]
) -> list[float]:
    values: list[float] = []
    for name in feature_names:
        if "." in name:
            context_id, field = name.split(".", 1)
            value = record["contexts"][context_id][field]
        else:
            value = record["aggregate"][name]
        values.append(float(value))
    return values


def build_metric_pair_candidates(
    records: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    nearest_fraction: float,
) -> dict[str, Any]:
    candidate_ids = sorted(records)
    matrix = np.asarray(
        [flatten_matching_features(records[candidate_id], feature_names) for candidate_id in candidate_ids],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    normalized = (matrix - means) / scales
    vectors = {
        candidate_id: normalized[index] for index, candidate_id in enumerate(candidate_ids)
    }
    pairs = []
    for left_index, left in enumerate(candidate_ids):
        for right in candidate_ids[left_index + 1 :]:
            difference = vectors[left] - vectors[right]
            pairs.append(
                {
                    "pair_id": f"ll_pair_{stable_hash({'left': left, 'right': right}, 16)}",
                    "policy_a": left,
                    "policy_b": right,
                    "standardized_rms_distance": float(
                        math.sqrt(float(np.mean(difference**2)))
                    ),
                    "stable_order": stable_hash({"left": left, "right": right}),
                    "oriented_metric_difference": difference.tolist(),
                }
            )
    pairs.sort(key=lambda row: (row["standardized_rms_distance"], row["stable_order"]))
    eligible_count = max(1, math.ceil(len(pairs) * float(nearest_fraction)))
    eligible = pairs[:eligible_count]
    return {
        "candidate_ids": candidate_ids,
        "feature_names": list(feature_names),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "all_pair_count": len(pairs),
        "eligible_pair_count": len(eligible),
        "distance_cutoff": float(eligible[-1]["standardized_rms_distance"]),
        "eligible_pairs": eligible,
    }


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1], dtype=float)
    count_zero = max(1, int(np.sum(y == 0)))
    count_one = max(1, int(np.sum(y == 1)))
    sample_weight = np.where(y == 1, 0.5 / count_one, 0.5 / count_zero)
    for _ in range(int(iterations)):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - y) * sample_weight)
        gradient[1:] += float(l2) * weights[1:] / max(1, len(x))
        weights -= float(learning_rate) * gradient
    return weights


def metric_only_candidate_disjoint_cv(
    pair_rows: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    determinate = [row for row in pair_rows if row["oracle_status"] == "DETERMINATE"]
    fold_count = int(diagnostic["candidate_disjoint_folds"])
    folds = {
        candidate_id: int(stable_hash(candidate_id, 8), 16) % fold_count
        for candidate_id in sorted(
            {
                str(row["policy_a"]) for row in determinate
            }
            | {str(row["policy_b"]) for row in determinate}
        )
    }
    predictions: list[dict[str, Any]] = []
    fold_rows = []
    for fold in range(fold_count):
        train = [
            row
            for row in determinate
            if folds[str(row["policy_a"])] != fold
            and folds[str(row["policy_b"])] != fold
        ]
        test = [
            row
            for row in determinate
            if folds[str(row["policy_a"])] == fold
            and folds[str(row["policy_b"])] == fold
        ]
        labels = {str(row["oracle_label"]) for row in train}
        if not test or labels != {"A", "B"}:
            fold_rows.append(
                {
                    "fold": fold,
                    "training_pairs": len(train),
                    "test_pairs": len(test),
                    "status": "SKIPPED_INSUFFICIENT_CLASS_OR_TEST_COVERAGE",
                }
            )
            continue
        x_train = np.asarray(
            [row["oriented_metric_difference"] for row in train], dtype=float
        )
        y_train = np.asarray(
            [1.0 if row["oracle_label"] == "A" else 0.0 for row in train],
            dtype=float,
        )
        weights = _fit_logistic(
            x_train,
            y_train,
            iterations=int(diagnostic["iterations"]),
            learning_rate=float(diagnostic["learning_rate"]),
            l2=float(diagnostic["l2"]),
        )
        design_test = np.column_stack(
            [
                np.ones(len(test)),
                np.asarray(
                    [row["oriented_metric_difference"] for row in test], dtype=float
                ),
            ]
        )
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(design_test @ weights, -30.0, 30.0))
        )
        for row, probability in zip(test, probabilities):
            predictions.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "truth": str(row["oracle_label"]),
                    "prediction": "A" if float(probability) >= 0.5 else "B",
                    "probability_a": float(probability),
                    "fold": fold,
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "training_pairs": len(train),
                "test_pairs": len(test),
                "status": "EVALUATED",
            }
        )
    recalls = {}
    for label in ("A", "B"):
        selected = [row for row in predictions if row["truth"] == label]
        recalls[label] = (
            sum(row["prediction"] == label for row in selected) / len(selected)
            if selected
            else None
        )
    balanced_accuracy = (
        (float(recalls["A"]) + float(recalls["B"])) / 2.0
        if recalls["A"] is not None and recalls["B"] is not None
        else None
    )
    return {
        "status": "EVALUATED" if balanced_accuracy is not None else "INSUFFICIENT_CV_COVERAGE",
        "eligible_determinate_pairs": len(determinate),
        "candidate_disjoint_test_predictions": len(predictions),
        "class_recall": recalls,
        "balanced_accuracy": balanced_accuracy,
        "folds": fold_rows,
        "prediction_hash": stable_hash(predictions),
    }


def reviewer_factual_view(profile: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = profile["aggregate"]
    fields = (
        "episodes",
        "terminal_cause_counts",
        "settled_terminal_rate",
        "first_contact_speed_p50",
        "first_contact_speed_p90",
        "post_recovery_alternations_per_episode",
        "brittle_late_rescue_rate",
        "unresolved_terminal_motion_rate",
        "engine_command_fraction",
        "engine_switch_fraction",
        "mean_return",
        "mean_length",
    )
    context_fields = (
        "terminal_cause_counts",
        "settled_terminal_rate",
        "first_contact_speed_p90",
        "post_recovery_alternations_per_episode",
        "engine_command_fraction",
        "engine_switch_fraction",
    )
    return {
        "aggregate_facts": {field: aggregate[field] for field in fields},
        "context_facts": {
            context_id: {field: row[field] for field in context_fields}
            for context_id, row in profile["contexts"].items()
        },
        "heldout_episode_count": int(profile["heldout_episode_count"]),
        "episode_fact_hash": str(profile["episode_fact_hash"]),
    }

