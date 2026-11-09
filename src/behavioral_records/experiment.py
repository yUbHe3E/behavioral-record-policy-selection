from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

from .baselines import scalar_summary, unordered_event_summary
from .canonical import stable_hash
from .extractor import BehaviorRecordExtractor
from .integrity import audit_prompt_view_pair, audit_record, build_protocol_manifest
from .io import read_json, read_trajectories, write_json, write_jsonl
from .legacy_baseline import legacy_handcrafted_summary
from .models import Trajectory
from .oracle import compare_results, rank_policies
from .prompts import make_pair_requests
from .views import (
    build_compact_behavior_view,
    build_matched_raw_trajectory_view,
    public_evidence_chars,
)


def group_by_policy(trajectories: Iterable[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.policy_id].append(trajectory)
    return {policy_id: sorted(rows, key=lambda row: row.episode_id) for policy_id, rows in sorted(groups.items())}


def build_records(
    trajectories_by_policy: Mapping[str, Iterable[Trajectory]],
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    extractor = BehaviorRecordExtractor(protocol)
    return {
        policy_id: extractor.extract(policy_id, trajectories)
        for policy_id, trajectories in sorted(trajectories_by_policy.items())
    }


def fixed_pair_graph(
    policy_ids: Iterable[str],
    *,
    max_pairs: int | None = None,
    seed: str = "pair-graph-v1",
) -> list[tuple[str, str]]:
    """Build a deterministic objective-independent connected pair graph."""
    policies = sorted(set(policy_ids))
    if len(policies) < 2:
        return []
    all_pairs = list(combinations(policies, 2))
    all_pairs.sort(key=lambda pair: (stable_hash([seed, *pair]), pair))
    if max_pairs is None or max_pairs >= len(all_pairs):
        return all_pairs

    ring_order = sorted(policies, key=lambda policy_id: (stable_hash([seed, policy_id]), policy_id))
    required_edges: list[tuple[str, str]] = []
    for index, policy_id in enumerate(ring_order):
        neighbor = ring_order[(index + 1) % len(ring_order)]
        edge = tuple(sorted((policy_id, neighbor)))
        if edge not in required_edges:
            required_edges.append(edge)
    if max_pairs < len(required_edges):
        raise ValueError(
            f"max_pairs={max_pairs} is too small to keep {len(policies)} policies connected; "
            f"use at least {len(required_edges)}"
        )
    selected = list(required_edges)
    selected_set = set(selected)
    for pair in all_pairs:
        if len(selected) >= max_pairs:
            break
        if pair not in selected_set:
            selected.append(pair)
            selected_set.add(pair)
    selected.sort(key=lambda pair: (stable_hash([seed, *pair]), pair))
    return selected


def _number_summary(values: Iterable[int | float]) -> dict[str, int | float | None]:
    rows = list(values)
    if not rows:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(rows),
        "min": min(rows),
        "median": median(rows),
        "mean": fmean(rows),
        "max": max(rows),
    }


def run_offline_experiment(
    trajectory_path: str | Path,
    protocol_path: str | Path,
    objectives_path: str | Path,
    output_dir: str | Path,
    *,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectories = read_trajectories(trajectory_path)
    validation_trajectories = [item for item in trajectories if item.context.get("split") == "validation"]
    heldout_trajectories = [item for item in trajectories if item.context.get("split") == "heldout"]
    if not validation_trajectories or not heldout_trajectories:
        raise ValueError("Offline experiments require both validation and heldout trajectory splits")
    validation_groups = group_by_policy(validation_trajectories)
    heldout_groups = group_by_policy(heldout_trajectories)
    if set(validation_groups) != set(heldout_groups):
        raise ValueError("Validation and heldout splits must contain the same policy IDs")
    protocol = read_json(protocol_path)
    objectives = read_json(objectives_path)
    records = build_records(validation_groups, protocol)
    if "prompt_view" not in protocol:
        raise ValueError("Protocol must freeze a prompt_view configuration")
    compact_records = {
        policy_id: build_compact_behavior_view(record, protocol["prompt_view"])
        for policy_id, record in records.items()
    }
    raw_records = {
        policy_id: build_matched_raw_trajectory_view(
            policy_id,
            validation_groups[policy_id],
            protocol,
            protocol["prompt_view"],
            public_evidence_chars(compact_records[policy_id]),
        )
        for policy_id in sorted(validation_groups)
    }

    scalar_records = {
        policy_id: scalar_summary(policy_id, rows) for policy_id, rows in validation_groups.items()
    }
    unordered_records = {
        policy_id: unordered_event_summary(policy_id, rows) for policy_id, rows in validation_groups.items()
    }
    legacy_records = {
        policy_id: legacy_handcrafted_summary(policy_id, rows)
        for policy_id, rows in validation_groups.items()
    }
    evidence_arms = {
        "B0_scalar": scalar_records,
        "B1_legacy_handcrafted": legacy_records,
        "B2_unordered_events": unordered_records,
        "P_compact_behavioral_record": compact_records,
        "B3_raw_trajectories_matched_chars": raw_records,
    }
    pairs = fixed_pair_graph(validation_groups, max_pairs=max_pairs)
    write_jsonl(
        output / "pair_graph.jsonl",
        (
            {
                "pair_id": stable_hash(["pair-graph-v1", left_id, right_id], 16),
                "left_policy_id": left_id,
                "right_policy_id": right_id,
            }
            for left_id, right_id in pairs
        ),
    )
    write_jsonl(output / "behavioral_records.jsonl", records.values())
    write_jsonl(output / "compact_behavior_views.jsonl", compact_records.values())
    write_jsonl(output / "raw_trajectory_views.jsonl", raw_records.values())
    write_jsonl(output / "baseline_scalar.jsonl", scalar_records.values())
    write_jsonl(output / "baseline_legacy_handcrafted.jsonl", legacy_records.values())
    write_jsonl(output / "baseline_unordered_events.jsonl", unordered_records.values())

    audits = [audit_record(record, protocol) | {"policy_id": policy_id} for policy_id, record in records.items()]
    write_json(output / "integrity_audit.json", {"passed": all(row["passed"] for row in audits), "records": audits})
    view_audits = [
        audit_prompt_view_pair(compact_records[policy_id], raw_records[policy_id], protocol)
        for policy_id in sorted(records)
    ]
    write_json(
        output / "prompt_view_audit.json",
        {
            "passed": all(row["passed"] for row in view_audits),
            "hard_cap_chars_per_policy": protocol["prompt_view"]["per_policy_char_budget"],
            "compact_public_chars": _number_summary(row["compact_public_chars"] for row in view_audits),
            "raw_public_chars": _number_summary(row["raw_public_chars"] for row in view_audits),
            "match_slack_chars": _number_summary(row["match_slack_chars"] for row in view_audits),
            "records": view_audits,
        },
    )

    oracle_rows: list[dict[str, Any]] = []
    pair_labels: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for objective in objectives:
        ranked = rank_policies(objective, heldout_groups)
        rankings[str(objective["id"])] = [result.policy_id for result in ranked]
        oracle_rows.extend(result.to_dict() for result in ranked)
        by_policy = {result.policy_id: result for result in ranked}
        for left_id, right_id in pairs:
            verdict = compare_results(by_policy[left_id], by_policy[right_id])
            pair_labels.append(
                {
                    "objective_id": objective["id"],
                    "left_policy_id": left_id,
                    "right_policy_id": right_id,
                    "oracle_preference": verdict,
                }
            )
        for evidence_arm, arm_records in evidence_arms.items():
            prompt_rows.extend(
                make_pair_requests(objective, arm_records, evidence_arm=evidence_arm, pairs=pairs)
            )

    write_jsonl(output / "oracle_policy_results.jsonl", oracle_rows)
    write_jsonl(output / "oracle_pair_labels.jsonl", pair_labels)
    write_jsonl(output / "judge_requests.jsonl", prompt_rows)
    manifest = build_protocol_manifest(protocol_path)
    prompt_chars_by_arm = {
        evidence_arm: _number_summary(
            len(row["prompt"]) for row in prompt_rows if row["evidence_arm"] == evidence_arm
        )
        for evidence_arm in evidence_arms
    }
    summary = {
        "status": "COMPLETE_OFFLINE_ONLY",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_file": str(Path(trajectory_path).resolve()),
        "protocol_file": str(Path(protocol_path).resolve()),
        "objectives_file": str(Path(objectives_path).resolve()),
        "protocol_hash": manifest["protocol_hash"],
        "policy_count": len(validation_groups),
        "validation_episode_count": len(validation_trajectories),
        "heldout_episode_count": len(heldout_trajectories),
        "objective_count": len(objectives),
        "evidence_arms": list(evidence_arms),
        "pair_count": len(pairs),
        "pair_graph_design": "objective-independent stable-hash connected graph",
        "judge_request_count": len(prompt_rows),
        "integrity_passed": all(row["passed"] for row in audits) and all(
            row["passed"] for row in view_audits
        ),
        "prompt_view_audit_passed": all(row["passed"] for row in view_audits),
        "per_policy_char_cap": protocol["prompt_view"]["per_policy_char_budget"],
        "compact_public_chars": _number_summary(
            public_evidence_chars(row) for row in compact_records.values()
        ),
        "raw_public_chars": _number_summary(public_evidence_chars(row) for row in raw_records.values()),
        "judge_prompt_chars_by_arm": prompt_chars_by_arm,
        "oracle_rankings": rankings,
        "claims": "No LLM/model-judge result was produced by this offline run.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "protocol_manifest.json", manifest)
    return summary
