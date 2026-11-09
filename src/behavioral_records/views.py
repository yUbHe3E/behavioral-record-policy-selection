from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, json_ready, stable_hash
from .models import Step, Trajectory


PRIVATE_EVIDENCE_KEYS = {
    "policy_id",
    "record_hash",
    "source_record_hash",
    "view_hash",
}


def public_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove storage-only identity and integrity fields before prompting."""
    return {key: value for key, value in record.items() if key not in PRIVATE_EVIDENCE_KEYS}


def public_evidence_chars(record: Mapping[str, Any]) -> int:
    return len(canonical_json(public_evidence(record)))


def _compact_token_definition(token: str, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": token}
    if "action" in descriptor:
        row["a"] = descriptor["action"]
    if "reward_bin" in descriptor:
        row["rb"] = descriptor["reward_bin"]
    if descriptor.get("events"):
        row["e"] = descriptor["events"]
    if descriptor.get("state"):
        row["s"] = descriptor["state"]
    if descriptor.get("predicates"):
        row["p"] = descriptor["predicates"]
    return row


def _compact_context_counts(counts: Mapping[str, int]) -> list[list[Any]]:
    return [[json_ready(json.loads(context)), count] for context, count in sorted(counts.items())]


def _compact_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "runs": [[item["token"], item["repeat"]] for item in row["runs"]],
        "episodes": row["episode_count"],
        "outcomes": row["outcome_counts"],
        "return": row["return"],
        "length": row["length"],
        "contexts": _compact_context_counts(row["context_counts"]),
    }


def _assemble_compact_view(
    record: Mapping[str, Any],
    prompt_view_config: Mapping[str, Any],
    limits: Mapping[str, int],
) -> dict[str, Any]:
    token_rows = list(record["token_counts"])[: limits["token_rows"]]
    transition_rows = list(record["transition_counts"])[: limits["transition_rows"]]
    trace_rows = list(record["trace_patterns"])[: limits["trace_patterns"]]

    all_nodes = list(record["prefix_automaton"]["nodes"])
    selected_nodes = all_nodes[: limits["automaton_nodes"]]
    selected_node_ids = {row["id"] for row in selected_nodes}
    selected_edges = [
        row
        for row in record["prefix_automaton"]["edges"]
        if row["source"] in selected_node_ids and row["target"] in selected_node_ids
    ]

    referenced_tokens = {row["token"] for row in token_rows}
    for row in transition_rows:
        referenced_tokens.update((row["source"], row["target"]))
    for row in trace_rows:
        referenced_tokens.update(item["token"] for item in row["runs"])
    referenced_tokens.update(row["token"] for row in selected_edges)
    token_definitions = record["token_definitions"]

    represented_token_occurrences = sum(row["count"] for row in token_rows)
    source_token_occurrences = sum(row["count"] for row in record["token_counts"])
    represented_transition_occurrences = sum(row["count"] for row in transition_rows)
    source_transition_occurrences = sum(row["count"] for row in record["transition_counts"])
    represented_trace_episodes = sum(row["episode_count"] for row in trace_rows)
    represented_context_episodes = sum(row["episode_count"] for row in record["context_groups"])

    return {
        "representation": "compact_behavioral_record",
        "schema_version": prompt_view_config["schema_version"],
        "protocol_hash": record["protocol_hash"],
        "policy_id": record["policy_id"],
        "source_record_hash": record["record_hash"],
        "char_cap": int(prompt_view_config["per_policy_char_budget"]),
        "notation": {
            "token_definition": {"a": "action", "rb": "reward_bin", "e": "events", "s": "state", "p": "predicates"},
            "token_count_row": ["token", "occurrences", "episodes"],
            "transition_row": ["source", "target", "occurrences", "episodes"],
            "automaton_node_row": ["id", "visits", "terminal_episodes", "outcome_counts"],
            "automaton_edge_row": ["source", "target", "token", "repeat", "visits"],
            "event_run_row": ["event", "repeat"],
            "false_or_empty_fields": "retained in s/p; empty event lists are omitted only in token definitions",
        },
        "sample": record["sample"],
        "episode_distributions": record["episode_distributions"],
        "context_groups": record["context_groups"],
        "event_skeletons": record.get("event_skeletons", []),
        "token_definitions": [
            _compact_token_definition(token, token_definitions[token]) for token in sorted(referenced_tokens)
        ],
        "token_counts": [
            [row["token"], row["count"], row["episode_count"]] for row in token_rows
        ],
        "transitions": [
            [row["source"], row["target"], row["count"], row["episode_count"]]
            for row in transition_rows
        ],
        "trace_patterns": [_compact_trace(row) for row in trace_rows],
        "prefix_automaton": {
            "nodes": [
                [row["id"], row["visits"], row["terminal_count"], row["outcome_counts"]]
                for row in selected_nodes
            ],
            "edges": [
                [row["source"], row["target"], row["token"], row["repeat"], row["visits"]]
                for row in selected_edges
            ],
        },
        "source_extractor_coverage": record["coverage"],
        "view_coverage": {
            "included_token_rows": len(token_rows),
            "source_token_rows": len(record["token_counts"]),
            "token_occurrence_fraction": represented_token_occurrences / max(1, source_token_occurrences),
            "included_transition_rows": len(transition_rows),
            "source_transition_rows": len(record["transition_counts"]),
            "transition_occurrence_fraction_within_source_record": (
                represented_transition_occurrences / max(1, source_transition_occurrences)
            ),
            "included_trace_patterns": len(trace_rows),
            "source_trace_patterns": len(record["trace_patterns"]),
            "trace_episode_fraction": represented_trace_episodes / max(1, record["sample"]["episodes"]),
            "included_automaton_nodes": len(selected_nodes),
            "source_automaton_nodes": len(all_nodes),
            "automaton_node_fraction_within_source_record": len(selected_nodes) / max(1, len(all_nodes)),
            "included_context_groups": len(record["context_groups"]),
            "context_episode_fraction": represented_context_episodes / max(1, record["sample"]["episodes"]),
            "included_event_skeletons": len(record.get("event_skeletons", [])),
            "event_skeleton_episode_fraction": (
                len(record.get("event_skeletons", [])) / max(1, record["sample"]["episodes"])
            ),
        },
    }


def build_compact_behavior_view(
    record: Mapping[str, Any],
    prompt_view_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic, objective-free structured view under a hard character cap."""
    compact_config = prompt_view_config["compact"]
    limits = {
        "token_rows": min(len(record["token_counts"]), int(compact_config["max_token_rows"])),
        "transition_rows": min(len(record["transition_counts"]), int(compact_config["max_transition_rows"])),
        "trace_patterns": min(len(record["trace_patterns"]), int(compact_config["max_trace_patterns"])),
        "automaton_nodes": min(
            len(record["prefix_automaton"]["nodes"]), int(compact_config["max_automaton_nodes"])
        ),
    }
    floors_config = compact_config["minimum_rows"]
    floors = {
        key: min(limits[key], int(floors_config[key]))
        for key in limits
    }
    decrement = {
        "token_rows": 4,
        "transition_rows": 8,
        "trace_patterns": 1,
        "automaton_nodes": 8,
    }
    shrink_order = ("automaton_nodes", "trace_patterns", "transition_rows", "token_rows")
    char_cap = int(prompt_view_config["per_policy_char_budget"])

    while True:
        view = _assemble_compact_view(record, prompt_view_config, limits)
        if public_evidence_chars(view) <= char_cap:
            break
        changed = False
        for key in shrink_order:
            if limits[key] > floors[key]:
                limits[key] = max(floors[key], limits[key] - decrement[key])
                changed = True
        if not changed:
            raise ValueError(
                f"Compact view cannot fit char cap {char_cap} for policy {record['policy_id']}; "
                f"minimum view requires {public_evidence_chars(view)} characters"
            )

    view["view_hash"] = stable_hash(view)
    return view


def _selected_context(trajectory: Trajectory, context_fields: Iterable[str]) -> dict[str, Any]:
    return json_ready({field: trajectory.context.get(field) for field in context_fields})


def _raw_step(step: Step) -> list[Any]:
    return [
        step.t,
        step.action,
        step.reward,
        json_ready(dict(step.state)),
        list(step.events),
        json_ready(dict(step.predicates)),
    ]


def _raw_episode(
    trajectory: Trajectory,
    context_fields: Iterable[str],
    observed_steps: int,
) -> dict[str, Any]:
    complete = observed_steps == trajectory.length
    row: dict[str, Any] = {
        "context": _selected_context(trajectory, context_fields),
        "complete": complete,
        "observed_length": observed_steps,
        "source_length": trajectory.length,
        "steps": [_raw_step(step) for step in trajectory.steps[:observed_steps]],
    }
    if complete:
        row.update(
            {
                "outcome": trajectory.outcome,
                "return": trajectory.total_return,
                "terminated": trajectory.terminated,
                "truncated": trajectory.truncated,
            }
        )
    return row


def _round_robin_context_order(
    trajectories: Iterable[Trajectory],
    context_fields: Iterable[str],
) -> list[Trajectory]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        context_key = canonical_json(_selected_context(trajectory, context_fields))
        groups[context_key].append(trajectory)
    for context_key in groups:
        groups[context_key].sort(
            key=lambda row: (stable_hash(["raw-view-v1", row.episode_id]), row.episode_id)
        )
    ordered: list[Trajectory] = []
    context_keys = sorted(groups)
    index = 0
    while True:
        added = False
        for context_key in context_keys:
            if index < len(groups[context_key]):
                ordered.append(groups[context_key][index])
                added = True
        if not added:
            return ordered
        index += 1


def _assemble_raw_view(
    policy_id: str,
    protocol_hash: str,
    prompt_view_config: Mapping[str, Any],
    episodes: list[dict[str, Any]],
    source_trajectories: list[Trajectory],
    context_fields: Iterable[str],
    matched_public_chars: int,
) -> dict[str, Any]:
    observed_steps = sum(row["observed_length"] for row in episodes)
    complete_episodes = sum(bool(row["complete"]) for row in episodes)
    represented_contexts = {canonical_json(row["context"]) for row in episodes}
    source_contexts = {
        canonical_json(_selected_context(trajectory, context_fields)) for trajectory in source_trajectories
    }
    total_steps = sum(trajectory.length for trajectory in source_trajectories)
    return {
        "representation": "raw_trajectories_matched_chars",
        "schema_version": prompt_view_config["schema_version"],
        "protocol_hash": protocol_hash,
        "policy_id": policy_id,
        "char_cap": int(prompt_view_config["per_policy_char_budget"]),
        "matched_public_chars": matched_public_chars,
        "selection": prompt_view_config["raw_control"]["selection"],
        "step_columns": ["t", "action", "reward", "state", "events", "predicates"],
        "episodes": episodes,
        "view_coverage": {
            "complete_episodes": complete_episodes,
            "partial_episodes": len(episodes) - complete_episodes,
            "source_episodes": len(source_trajectories),
            "complete_episode_fraction": complete_episodes / max(1, len(source_trajectories)),
            "observed_steps": observed_steps,
            "source_steps": total_steps,
            "step_fraction": observed_steps / max(1, total_steps),
            "represented_contexts": len(represented_contexts),
            "source_contexts": len(source_contexts),
            "context_fraction": len(represented_contexts) / max(1, len(source_contexts)),
        },
    }


def build_matched_raw_trajectory_view(
    policy_id: str,
    trajectories: Iterable[Trajectory],
    protocol: Mapping[str, Any],
    prompt_view_config: Mapping[str, Any],
    matched_public_chars: int,
) -> dict[str, Any]:
    """Select raw validation evidence without seeing an objective, up to a matched size."""
    source = sorted(list(trajectories), key=lambda row: row.episode_id)
    if not source:
        raise ValueError("At least one raw trajectory is required")
    if any(row.policy_id != policy_id for row in source):
        raise ValueError("Raw-view trajectories contain another policy ID")
    hard_cap = int(prompt_view_config["per_policy_char_budget"])
    if matched_public_chars > hard_cap:
        raise ValueError("Matched public size exceeds the frozen per-policy character cap")
    context_fields = list(protocol["context_fields"])
    ordered = _round_robin_context_order(source, context_fields)
    selected: list[dict[str, Any]] = []

    def candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return _assemble_raw_view(
            policy_id,
            stable_hash(protocol),
            prompt_view_config,
            rows,
            source,
            context_fields,
            matched_public_chars,
        )

    if public_evidence_chars(candidate([])) > matched_public_chars:
        raise ValueError("Matched character budget is too small for raw-view metadata")

    for trajectory in ordered:
        full_row = _raw_episode(trajectory, context_fields, trajectory.length)
        if public_evidence_chars(candidate(selected + [full_row])) <= matched_public_chars:
            selected.append(full_row)
            continue

        low, high = 0, trajectory.length - 1
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if middle == 0:
                fits = True
            else:
                prefix_row = _raw_episode(trajectory, context_fields, middle)
                fits = public_evidence_chars(candidate(selected + [prefix_row])) <= matched_public_chars
            if fits:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best > 0:
            selected.append(_raw_episode(trajectory, context_fields, best))
        break

    view = candidate(selected)
    view["view_hash"] = stable_hash(view)
    return view
