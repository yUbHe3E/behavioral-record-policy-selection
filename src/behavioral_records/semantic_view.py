from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, json_ready, stable_hash
from .models import Trajectory
from .views import public_evidence_chars


FORBIDDEN_CONFIG_KEYS = {
    "objective",
    "objective_id",
    "requirement",
    "preference",
    "judge",
    "oracle",
    "gold_label",
}


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(value[key], child))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.update(_flatten(item, f"{prefix}[{index}]"))
        return rows
    rows[prefix] = value
    return rows


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float], quantiles: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": fmean(rows) if rows else None,
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
        "quantiles": [[value, _quantile(rows, value)] for value in quantiles],
    }


def _episode_symbols(trajectory: Trajectory) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = defaultdict(list)
    for step in trajectory.steps:
        rows[f"action::{step.action}"].append(step.t)
        for event in step.events:
            rows[f"event::{event}"].append(step.t)
    rows[f"outcome::{trajectory.outcome}"].append(trajectory.steps[-1].t + 1)
    return dict(rows)


def _profile_counts(
    trajectories: Sequence[Trajectory],
    *,
    category: str,
    vocabulary: Sequence[str],
    quantiles: Sequence[float],
) -> list[dict[str, Any]]:
    profiles = []
    for name in vocabulary:
        if category == "action":
            counts = [sum(step.action == name for step in row.steps) for row in trajectories]
        else:
            counts = [row.events.count(name) for row in trajectories]
        episode_count = sum(value > 0 for value in counts)
        profiles.append(
            {
                "name": name,
                "occurrence_count": sum(counts),
                "episode_count": episode_count,
                "episode_rate": episode_count / len(trajectories),
                "count_per_episode": _distribution(counts, quantiles),
            }
        )
    profiles.sort(key=lambda row: (-row["episode_count"], -row["occurrence_count"], row["name"]))
    return profiles


def _temporal_relations(trajectories: Sequence[Trajectory]) -> list[dict[str, Any]]:
    typed = [_episode_symbols(row) for row in trajectories]
    vocabulary = sorted({symbol for row in typed for symbol in row})
    relations = []
    for left_index, left in enumerate(vocabulary):
        for right in vocabulary[left_index + 1 :]:
            left_before = 0
            right_before = 0
            same_step = 0
            both = 0
            for symbols in typed:
                if left not in symbols or right not in symbols:
                    continue
                both += 1
                left_first = min(symbols[left])
                right_first = min(symbols[right])
                if left_first < right_first:
                    left_before += 1
                elif right_first < left_first:
                    right_before += 1
                else:
                    same_step += 1
            relations.append(
                {
                    "left": left,
                    "right": right,
                    "both_episode_count": both,
                    "both_episode_rate": both / len(trajectories),
                    "first_occurrence_order": {
                        "left_before_right": left_before,
                        "right_before_left": right_before,
                        "same_step": same_step,
                    },
                    "not_both_episode_count": len(trajectories) - both,
                }
            )
    relations.sort(
        key=lambda row: (
            -row["both_episode_count"],
            row["left"],
            row["right"],
        )
    )
    return relations


def _action_state_effects(
    trajectories: Sequence[Trajectory], quantiles: Sequence[float]
) -> list[dict[str, Any]]:
    numeric: dict[tuple[str, str], list[float]] = defaultdict(list)
    missing_prior: Counter[tuple[str, str]] = Counter()
    for trajectory in trajectories:
        prior: dict[str, Any] | None = None
        for step in trajectory.steps:
            current = {
                **_flatten(step.state, "state"),
                **_flatten(step.predicates, "predicate"),
            }
            action = str(step.action)
            for signal, value in current.items():
                key = (action, signal)
                if prior is None or signal not in prior:
                    missing_prior[key] += 1
                    continue
                prior_value = prior[signal]
                if isinstance(value, bool) and isinstance(prior_value, bool):
                    numeric[key].append(float(value) - float(prior_value))
                elif (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isinstance(prior_value, (int, float))
                    and not isinstance(prior_value, bool)
                    and math.isfinite(float(value))
                    and math.isfinite(float(prior_value))
                ):
                    numeric[key].append(float(value) - float(prior_value))
            prior = current
    rows = []
    for (action, signal), deltas in sorted(numeric.items()):
        support = len(deltas)
        rows.append(
            {
                "action": action,
                "signal": signal,
                "transition_support": support,
                "missing_prior_count": missing_prior[(action, signal)],
                "delta": _distribution(deltas, quantiles),
                "nonzero_rate": sum(value != 0.0 for value in deltas) / support,
                "positive_rate": sum(value > 0.0 for value in deltas) / support,
                "negative_rate": sum(value < 0.0 for value in deltas) / support,
            }
        )
    rows.sort(key=lambda row: (-row["transition_support"], row["action"], row["signal"]))
    return rows


def _trace_variants(trajectories: Sequence[Trajectory]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    sequences: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        sequence = [
            {
                "action": str(step.action),
                "events": list(step.events),
            }
            for step in trajectory.steps
        ]
        key = canonical_json(sequence)
        grouped[key].append(trajectory)
        sequences[key] = sequence
    rows = []
    for key, episodes in grouped.items():
        outcomes = Counter(row.outcome for row in episodes)
        rows.append(
            {
                "episode_count": len(episodes),
                "episode_rate": len(episodes) / len(trajectories),
                "steps": sequences[key],
                "outcomes": dict(sorted(outcomes.items())),
                "return": _distribution([row.total_return for row in episodes], (0.1, 0.5, 0.9)),
                "evidence_id": stable_hash(key)[:16],
            }
        )
    rows.sort(key=lambda row: (-row["episode_count"], row["evidence_id"]))
    return rows


def _context_profiles(trajectories: Sequence[Trajectory]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        for field, value in sorted(trajectory.context.items()):
            if field in {"split", "context_seed"}:
                continue
            groups[(str(field), canonical_json(json_ready(value)))].append(trajectory)
    rows = []
    for (field, value), episodes in groups.items():
        rows.append(
            {
                "field": field,
                "value": json_ready(value),
                "episode_count": len(episodes),
                "outcomes": dict(sorted(Counter(row.outcome for row in episodes).items())),
            }
        )
    rows.sort(key=lambda row: (-row["episode_count"], row["field"], row["value"]))
    return rows


class SemanticBehaviorCompiler:
    """Objective-blind high-density behavior evidence with explicit relations."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = json_ready(dict(config))
        self.config_hash = stable_hash(self.config)
        self._validate()

    def _validate(self) -> None:
        if set(self.config) != {"schema_version", "compiler", "view", "audit"}:
            raise ValueError("Semantic-view config requires schema_version, compiler, view, audit")

        def visit(value: Any, path: str = "config") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key).lower() in FORBIDDEN_CONFIG_KEYS:
                        raise ValueError(f"Forbidden objective-conditioned config key at {path}.{key}")
                    visit(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(self.config)

    def compile(self, policy_id: str, trajectories: Iterable[Trajectory]) -> dict[str, Any]:
        episodes = sorted(list(trajectories), key=lambda row: row.episode_id)
        if not episodes or any(row.policy_id != policy_id for row in episodes):
            raise ValueError("Semantic compiler requires one non-empty policy group")
        quantiles = [float(value) for value in self.config["compiler"]["quantiles"]]
        actions = sorted({str(step.action) for row in episodes for step in row.steps})
        events = sorted({event for row in episodes for event in row.events})
        profile = {
            "representation": "objective_blind_semantic_behavior_evidence",
            "schema_version": self.config["schema_version"],
            "policy_id": policy_id,
            "config_hash": self.config_hash,
            "sample": {
                "episodes": len(episodes),
                "steps": sum(row.length for row in episodes),
                "contexts": len({canonical_json(row.context) for row in episodes}),
            },
            "outcomes": dict(sorted(Counter(row.outcome for row in episodes).items())),
            "return": _distribution([row.total_return for row in episodes], quantiles),
            "length": _distribution([row.length for row in episodes], quantiles),
            "action_profiles": _profile_counts(
                episodes, category="action", vocabulary=actions, quantiles=quantiles
            ),
            "event_profiles": _profile_counts(
                episodes, category="event", vocabulary=events, quantiles=quantiles
            ),
            "temporal_relations": _temporal_relations(episodes),
            "action_state_effects": _action_state_effects(episodes, quantiles),
            "trace_variants": _trace_variants(episodes),
            "context_profiles": _context_profiles(episodes),
            "loss_ledger": {
                "semantic_adapter": "all observed actions, events, numeric/boolean state deltas, contexts, outcomes, and full step-order variants; no objective input",
                "raw_episode_count": len(episodes),
                "raw_step_count": sum(row.length for row in episodes),
            },
        }
        profile["profile_hash"] = stable_hash(profile)
        return profile


def compile_semantic_profiles(
    trajectories_by_policy: Mapping[str, Iterable[Trajectory]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    compiler = SemanticBehaviorCompiler(config)
    return {
        policy_id: compiler.compile(policy_id, rows)
        for policy_id, rows in sorted(trajectories_by_policy.items())
    }


def build_semantic_view(profile: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["view"]
    source_keys = {
        "action_profiles": "max_action_profiles",
        "event_profiles": "max_event_profiles",
        "temporal_relations": "max_temporal_relations",
        "action_state_effects": "max_action_state_effects",
        "trace_variants": "max_trace_variants",
        "context_profiles": "max_context_rows",
    }
    limits = {
        key: min(len(profile[key]), int(cfg[max_key]))
        for key, max_key in source_keys.items()
    }
    floors = {
        "action_profiles": limits["action_profiles"],
        "event_profiles": limits["event_profiles"],
        "temporal_relations": min(limits["temporal_relations"], int(cfg["minimum_temporal_relations"])),
        "action_state_effects": min(limits["action_state_effects"], int(cfg["minimum_action_state_effects"])),
        "trace_variants": min(limits["trace_variants"], int(cfg["minimum_trace_variants"])),
        "context_profiles": 0,
    }

    def assemble() -> dict[str, Any]:
        view = {
            "representation": "objective_blind_semantic_behavior_evidence",
            "schema_version": config["schema_version"],
            "policy_id": profile["policy_id"],
            "config_hash": profile["config_hash"],
            "sample": profile["sample"],
            "outcomes": profile["outcomes"],
            "return": profile["return"],
            "length": profile["length"],
            "notation": {
                "profile_row": ["name", "occurrences", "episodes", "episode_rate", "mean_count_per_episode"],
                "temporal_relation_row": ["left", "right", "both_episodes", "both_rate", "left_before_right", "right_before_left", "same_step", "not_both_episodes"],
                "action_state_effect_row": ["action", "signal", "support", "missing_prior", "mean_delta", "q10", "q50", "q90", "nonzero_rate", "positive_rate", "negative_rate"],
                "trace_variant_row": ["episodes", "episode_rate", "steps_as_action_events", "outcomes", "evidence_id"],
                "context_row": ["field", "value", "episodes", "outcomes"],
                "temporal_semantics": "first occurrences among episodes where both symbols occur; direction counts are explicit",
                "effect_semantics": "state_after_action_step minus state_at_previous_step",
                "missingness": "omitted rows are counted in loss_ledger and are not zero",
            },
        }
        view["action_profiles"] = [
            [
                row["name"], row["occurrence_count"], row["episode_count"],
                row["episode_rate"], row["count_per_episode"]["mean"],
            ]
            for row in list(profile["action_profiles"])[: limits["action_profiles"]]
        ]
        view["event_profiles"] = [
            [
                row["name"], row["occurrence_count"], row["episode_count"],
                row["episode_rate"], row["count_per_episode"]["mean"],
            ]
            for row in list(profile["event_profiles"])[: limits["event_profiles"]]
        ]
        view["temporal_relations"] = [
            [
                row["left"], row["right"], row["both_episode_count"],
                row["both_episode_rate"],
                row["first_occurrence_order"]["left_before_right"],
                row["first_occurrence_order"]["right_before_left"],
                row["first_occurrence_order"]["same_step"],
                row["not_both_episode_count"],
            ]
            for row in list(profile["temporal_relations"])[: limits["temporal_relations"]]
        ]
        view["action_state_effects"] = [
            [
                row["action"], row["signal"], row["transition_support"],
                row["missing_prior_count"], row["delta"]["mean"],
                dict(row["delta"]["quantiles"]).get(0.1),
                dict(row["delta"]["quantiles"]).get(0.5),
                dict(row["delta"]["quantiles"]).get(0.9),
                row["nonzero_rate"], row["positive_rate"], row["negative_rate"],
            ]
            for row in list(profile["action_state_effects"])[: limits["action_state_effects"]]
        ]
        view["trace_variants"] = [
            [
                row["episode_count"], row["episode_rate"],
                [[step["action"], step["events"]] for step in row["steps"]],
                row["outcomes"], row["evidence_id"],
            ]
            for row in list(profile["trace_variants"])[: limits["trace_variants"]]
        ]
        view["context_profiles"] = [
            [row["field"], row["value"], row["episode_count"], row["outcomes"]]
            for row in list(profile["context_profiles"])[: limits["context_profiles"]]
        ]
        view["loss_ledger"] = {
            **profile["loss_ledger"],
            "judge_view": {
                f"included_{key}": limits[key] for key in source_keys
            },
        }
        view["loss_ledger"]["judge_view"].update(
            {f"source_{key}": len(profile[key]) for key in source_keys}
        )
        return view

    shrink_order = (
        "context_profiles",
        "action_state_effects",
        "temporal_relations",
        "trace_variants",
    )
    cap = int(cfg["per_policy_char_budget"])
    while True:
        view = assemble()
        if public_evidence_chars(view) <= cap:
            break
        changed = False
        for key in shrink_order:
            if limits[key] > floors[key]:
                limits[key] -= 1
                changed = True
                break
        if not changed:
            raise ValueError(
                f"Semantic-view structural minimum exceeds {cap} characters"
            )
    view["loss_ledger"]["judge_view"]["public_characters"] = public_evidence_chars(view)
    view["view_hash"] = stable_hash(view)
    return view


def build_semantic_views(
    profiles: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        policy_id: build_semantic_view(profile, config)
        for policy_id, profile in sorted(profiles.items())
    }
