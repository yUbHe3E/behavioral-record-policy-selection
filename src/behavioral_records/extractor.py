from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from statistics import fmean
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, json_ready, stable_hash
from .models import Step, Trajectory


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": fmean(rows) if rows else None,
        "min": min(rows) if rows else None,
        "p10": _quantile(rows, 0.10),
        "p50": _quantile(rows, 0.50),
        "p90": _quantile(rows, 0.90),
        "max": max(rows) if rows else None,
    }


def _run_length(tokens: Iterable[str]) -> tuple[tuple[str, int], ...]:
    runs: list[tuple[str, int]] = []
    for token in tokens:
        if runs and runs[-1][0] == token:
            previous, count = runs[-1]
            runs[-1] = (previous, count + 1)
        else:
            runs.append((token, 1))
    return tuple(runs)


class BehaviorRecordExtractor:
    """Frozen objective-agnostic mapping from raw trajectories to evidence.

    The public extraction API intentionally has no objective, preference, or
    judge argument. A protocol hash is attached to every output record.
    """

    def __init__(self, protocol: Mapping[str, Any]):
        self.protocol = json_ready(dict(protocol))
        self.protocol_hash = stable_hash(self.protocol)
        self._validate_protocol()

    def _validate_protocol(self) -> None:
        required = {"schema_version", "step_fields", "context_fields", "episode_fields", "aggregation", "ordering"}
        missing = sorted(required - set(self.protocol))
        if missing:
            raise ValueError(f"Protocol is missing required keys: {missing}")
        forbidden = {"objective", "objective_id", "preference", "alignment_score", "oracle_score", "heldout_score"}

        def visit(value: Any, path: str = "protocol") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in forbidden:
                        raise ValueError(f"Objective-conditioned key is forbidden at {path}.{key}")
                    visit(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(self.protocol)

    def extract(self, policy_id: str, trajectories: Iterable[Trajectory]) -> dict[str, Any]:
        episodes = sorted(list(trajectories), key=lambda item: item.episode_id)
        if not episodes:
            raise ValueError("At least one trajectory is required")
        mismatches = sorted({item.policy_id for item in episodes if item.policy_id != policy_id})
        if mismatches:
            raise ValueError(f"Trajectories contain other policy IDs: {mismatches}")

        token_definitions: dict[str, dict[str, Any]] = {}
        episode_tokens: list[tuple[Trajectory, tuple[str, ...]]] = []
        token_counts: Counter[str] = Counter()
        token_episode_counts: Counter[str] = Counter()
        transition_counts: Counter[tuple[str, str]] = Counter()
        transition_episode_counts: Counter[tuple[str, str]] = Counter()

        for episode in episodes:
            tokens: list[str] = []
            for step in episode.steps:
                descriptor = self._step_descriptor(step)
                token_id = f"t_{stable_hash(descriptor, 12)}"
                prior = token_definitions.get(token_id)
                if prior is not None and canonical_json(prior) != canonical_json(descriptor):
                    raise RuntimeError(f"Token hash collision for {token_id}")
                token_definitions[token_id] = descriptor
                tokens.append(token_id)
                token_counts[token_id] += 1
            token_episode_counts.update(set(tokens))
            transitions = list(zip(tokens, tokens[1:]))
            transition_counts.update(transitions)
            transition_episode_counts.update(set(transitions))
            episode_tokens.append((episode, tuple(tokens)))

        aggregation = self.protocol["aggregation"]
        episode_distributions: dict[str, Any] = {}
        episode_fields = set(self.protocol["episode_fields"])
        if "outcome" in episode_fields:
            episode_distributions["outcome_counts"] = dict(sorted(Counter(item.outcome for item in episodes).items()))
        if "return" in episode_fields:
            episode_distributions["return"] = _distribution(item.total_return for item in episodes)
        if "length" in episode_fields:
            episode_distributions["length"] = _distribution(item.length for item in episodes)
        if "terminated" in episode_fields:
            episode_distributions["terminated_count"] = sum(item.terminated for item in episodes)
        if "truncated" in episode_fields:
            episode_distributions["truncated_count"] = sum(item.truncated for item in episodes)

        record = {
            "schema_version": self.protocol["schema_version"],
            "policy_id": policy_id,
            "protocol_hash": self.protocol_hash,
            "sample": {
                "episodes": len(episodes),
                "steps": sum(item.length for item in episodes),
                "contexts": len({self._context_key(item) for item in episodes}),
            },
            "episode_distributions": episode_distributions,
            "token_definitions": {key: token_definitions[key] for key in sorted(token_definitions)},
            "token_counts": self._token_rows(token_counts, token_episode_counts, len(episodes)),
            "transition_counts": self._transition_rows(
                transition_counts,
                transition_episode_counts,
                len(episodes),
                int(aggregation["max_transition_rows"]),
            ),
            "trace_patterns": self._trace_patterns(episode_tokens, int(aggregation["max_trace_patterns"])),
            "prefix_automaton": self._prefix_automaton(episode_tokens, int(aggregation["max_automaton_nodes"])),
            "context_groups": self._context_groups(episodes, int(aggregation["max_context_groups"])),
        }
        if aggregation.get("include_event_skeletons", False):
            record["event_skeletons"] = self._event_skeletons(episodes)
        record["coverage"] = self._coverage(record, episodes, transition_counts)
        record["record_hash"] = stable_hash(record)
        return record

    def _step_descriptor(self, step: Step) -> dict[str, Any]:
        cfg = self.protocol["step_fields"]
        descriptor: dict[str, Any] = {}
        selected_state = {key: step.state[key] for key in cfg["state"] if key in step.state}
        selected_predicates = {key: step.predicates[key] for key in cfg["predicates"] if key in step.predicates}
        if selected_state:
            descriptor["state"] = selected_state
        if selected_predicates:
            descriptor["predicates"] = selected_predicates
        if cfg.get("include_action", False):
            descriptor["action"] = step.action
        if cfg.get("include_events", False):
            descriptor["events"] = list(step.events)
        if cfg.get("include_reward_bin", False):
            descriptor["reward_bin"] = self._reward_bin(step.reward, [float(value) for value in cfg["reward_bins"]])
        return json_ready(descriptor)

    @staticmethod
    def _reward_bin(reward: float, boundaries: list[float]) -> int:
        if boundaries != sorted(boundaries):
            raise ValueError("reward_bins must be sorted")
        for index, boundary in enumerate(boundaries):
            if reward <= boundary:
                return index
        return len(boundaries)

    def _context_key(self, episode: Trajectory) -> str:
        fields = self.protocol["context_fields"]
        return canonical_json({field: episode.context.get(field) for field in fields})

    @staticmethod
    def _token_rows(counts: Counter[str], episode_counts: Counter[str], n_episodes: int) -> list[dict[str, Any]]:
        keys = sorted(counts, key=lambda key: (-counts[key], key))
        return [
            {
                "token": key,
                "count": counts[key],
                "episode_count": episode_counts[key],
                "episode_rate": episode_counts[key] / n_episodes,
            }
            for key in keys
        ]

    @staticmethod
    def _transition_rows(
        counts: Counter[tuple[str, str]],
        episode_counts: Counter[tuple[str, str]],
        n_episodes: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        keys = sorted(counts, key=lambda key: (-counts[key], key))[:limit]
        return [
            {
                "source": source,
                "target": target,
                "count": counts[(source, target)],
                "episode_count": episode_counts[(source, target)],
                "episode_rate": episode_counts[(source, target)] / n_episodes,
            }
            for source, target in keys
        ]

    def _trace_patterns(
        self,
        episode_tokens: list[tuple[Trajectory, tuple[str, ...]]],
        limit: int,
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[tuple[str, int], ...], list[Trajectory]] = defaultdict(list)
        for episode, tokens in episode_tokens:
            groups[_run_length(tokens)].append(episode)
        ordered = sorted(groups, key=lambda key: (-len(groups[key]), canonical_json(key)))[:limit]
        rows: list[dict[str, Any]] = []
        for runs in ordered:
            episodes = groups[runs]
            rows.append(
                {
                    "runs": [{"token": token, "repeat": repeat} for token, repeat in runs],
                    "episode_count": len(episodes),
                    "episode_rate": len(episodes) / len(episode_tokens),
                    "outcome_counts": dict(sorted(Counter(item.outcome for item in episodes).items())),
                    "return": _distribution(item.total_return for item in episodes),
                    "length": _distribution(item.length for item in episodes),
                    "context_counts": dict(sorted(Counter(self._context_key(item) for item in episodes).items())),
                }
            )
        return rows

    @staticmethod
    def _prefix_automaton(
        episode_tokens: list[tuple[Trajectory, tuple[str, ...]]],
        limit: int,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = [{"visits": len(episode_tokens), "terminal_count": 0, "outcomes": Counter()}]
        edges: dict[tuple[int, str, int], int] = {}
        child_lookup: dict[tuple[int, str, int], int] = {}
        edge_visits: Counter[tuple[int, str, int]] = Counter()
        for episode, tokens in episode_tokens:
            parent = 0
            for token, repeat in _run_length(tokens):
                key = (parent, token, repeat)
                if key not in child_lookup:
                    child_lookup[key] = len(nodes)
                    nodes.append({"visits": 0, "terminal_count": 0, "outcomes": Counter()})
                child = child_lookup[key]
                edges[key] = child
                edge_visits[key] += 1
                nodes[child]["visits"] += 1
                parent = child
            nodes[parent]["terminal_count"] += 1
            nodes[parent]["outcomes"][episode.outcome] += 1

        included = {0}
        queue: deque[int] = deque([0])
        selected_edges: list[tuple[tuple[int, str, int], int]] = []
        while queue and len(included) < limit:
            parent = queue.popleft()
            children = [(key, child) for key, child in edges.items() if key[0] == parent]
            children.sort(key=lambda item: (-edge_visits[item[0]], item[0][1], item[0][2]))
            for key, child in children:
                if len(included) >= limit:
                    break
                included.add(child)
                selected_edges.append((key, child))
                queue.append(child)

        id_map = {old: new for new, old in enumerate(sorted(included))}
        node_rows = [
            {
                "id": id_map[old],
                "visits": nodes[old]["visits"],
                "terminal_count": nodes[old]["terminal_count"],
                "outcome_counts": dict(sorted(nodes[old]["outcomes"].items())),
            }
            for old in sorted(included)
        ]
        edge_rows = [
            {
                "source": id_map[key[0]],
                "target": id_map[child],
                "token": key[1],
                "repeat": key[2],
                "visits": edge_visits[key],
            }
            for key, child in selected_edges
            if key[0] in included and child in included
        ]
        return {
            "nodes": node_rows,
            "edges": edge_rows,
            "total_nodes_before_budget": len(nodes),
            "omitted_nodes": len(nodes) - len(included),
        }

    def _context_groups(self, episodes: list[Trajectory], limit: int) -> list[dict[str, Any]]:
        groups: dict[str, list[Trajectory]] = defaultdict(list)
        for episode in episodes:
            groups[self._context_key(episode)].append(episode)
        keys = sorted(groups, key=lambda key: (-len(groups[key]), key))[:limit]
        rows = []
        for key in keys:
            values = groups[key]
            rows.append(
                {
                    "context": json_ready(json.loads(key)),
                    "episode_count": len(values),
                    "outcome_counts": dict(sorted(Counter(item.outcome for item in values).items())),
                    "return": _distribution(item.total_return for item in values),
                    "length": _distribution(item.length for item in values),
                    "event_counts": dict(sorted(Counter(event for item in values for event in item.events).items())),
                }
            )
        return rows

    def _event_skeletons(self, episodes: list[Trajectory]) -> list[dict[str, Any]]:
        """Retain every episode's observed event order without step-level state repetition."""
        rows: list[dict[str, Any]] = []
        for episode in episodes:
            runs = _run_length(episode.events)
            rows.append(
                {
                    "context": json_ready(json.loads(self._context_key(episode))),
                    "outcome": episode.outcome,
                    "return": episode.total_return,
                    "length": episode.length,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                    "event_runs": [[event, repeat] for event, repeat in runs],
                }
            )
        return rows

    @staticmethod
    def _coverage(
        record: Mapping[str, Any],
        episodes: list[Trajectory],
        transition_counts: Counter[tuple[str, str]],
    ) -> dict[str, Any]:
        represented_trace_episodes = sum(row["episode_count"] for row in record["trace_patterns"])
        represented_transition_occurrences = sum(row["count"] for row in record["transition_counts"])
        total_transition_occurrences = sum(transition_counts.values())
        represented_context_episodes = sum(row["episode_count"] for row in record["context_groups"])
        coverage = {
            "all_steps_tokenized": True,
            "trace_episode_fraction": represented_trace_episodes / len(episodes),
            "transition_occurrence_fraction": (
                represented_transition_occurrences / total_transition_occurrences
                if total_transition_occurrences
                else 1.0
            ),
            "context_episode_fraction": represented_context_episodes / len(episodes),
            "automaton_node_fraction": (
                len(record["prefix_automaton"]["nodes"])
                / max(1, record["prefix_automaton"]["total_nodes_before_budget"])
            ),
            "omitted_trace_episodes": len(episodes) - represented_trace_episodes,
            "omitted_transition_occurrences": total_transition_occurrences - represented_transition_occurrences,
            "omitted_context_episodes": len(episodes) - represented_context_episodes,
        }
        if "event_skeletons" in record:
            coverage["event_skeleton_episode_fraction"] = len(record["event_skeletons"]) / len(episodes)
            coverage["omitted_event_skeleton_episodes"] = len(episodes) - len(record["event_skeletons"])
        return coverage
