from __future__ import annotations

from pathlib import Path

from behavioral_records.demo import make_demo_trajectories
from behavioral_records.experiment import fixed_pair_graph, run_offline_experiment
from behavioral_records.io import read_json, read_jsonl, write_trajectories


def test_offline_pipeline_separates_validation_and_heldout(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    output_dir = tmp_path / "run"
    write_trajectories(trajectory_path, make_demo_trajectories())
    summary = run_offline_experiment(
        trajectory_path,
        "configs/protocol_v2.json",
        "configs/objectives_demo.json",
        output_dir,
    )
    assert summary["status"] == "COMPLETE_OFFLINE_ONLY"
    assert summary["validation_episode_count"] == 40
    assert summary["heldout_episode_count"] == 40
    assert summary["integrity_passed"] is True
    assert len(read_jsonl(output_dir / "behavioral_records.jsonl")) == 4
    assert len(read_jsonl(output_dir / "compact_behavior_views.jsonl")) == 4
    assert len(read_jsonl(output_dir / "raw_trajectory_views.jsonl")) == 4
    assert len(read_jsonl(output_dir / "baseline_legacy_handcrafted.jsonl")) == 4
    assert len(read_jsonl(output_dir / "judge_requests.jsonl")) == 180
    assert read_json(output_dir / "integrity_audit.json")["passed"] is True
    assert read_json(output_dir / "prompt_view_audit.json")["passed"] is True


def test_fixed_pair_graph_is_connected_and_objective_independent() -> None:
    policies = [f"policy_{index}" for index in range(10)]
    pairs = fixed_pair_graph(policies, max_pairs=12)
    assert len(pairs) == 12
    adjacency = {policy_id: set() for policy_id in policies}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = {policies[0]}
    frontier = [policies[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    assert visited == set(policies)
