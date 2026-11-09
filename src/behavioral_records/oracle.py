from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from .models import Trajectory


SUPPORTED_OPERATORS = {"<=", "<", ">=", ">", "==", "!="}


def _compare(left: Any, op: str, right: Any) -> bool:
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(f"Unsupported operator: {op}")
    if op == "<=":
        return left <= right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == ">":
        return left > right
    if op == "==":
        return left == right
    return left != right


def _contains_sequence(events: Iterable[str], sequence: Iterable[str]) -> bool:
    targets = list(sequence)
    if not targets:
        return True
    index = 0
    for event in events:
        if event == targets[index]:
            index += 1
            if index == len(targets):
                return True
    return False


def _episode_field(episode: Trajectory, field: str) -> float:
    if field == "return":
        return episode.total_return
    if field == "length":
        return float(episode.length)
    if field == "terminated":
        return float(episode.terminated)
    if field == "truncated":
        return float(episode.truncated)
    if field.startswith("context."):
        value = episode.context.get(field.split(".", 1)[1])
        return float(value)
    raise ValueError(f"Unsupported numeric episode field: {field}")


def _matches_context(episode: Trajectory, condition: Mapping[str, Any]) -> bool:
    field = str(condition["field"])
    value = episode.context.get(field)
    return _compare(value, str(condition["op"]), condition["value"])


class ObjectiveOracle:
    """Executable hidden oracle evaluated on raw held-out trajectories."""

    def __init__(self, objective: Mapping[str, Any]):
        self.objective = dict(objective)
        if "id" not in self.objective or "natural_language" not in self.objective:
            raise ValueError("Objective requires id and natural_language")

    def metric(self, trajectories: Iterable[Trajectory], spec: Mapping[str, Any]) -> float:
        episodes = list(trajectories)
        if not episodes:
            return math.nan
        kind = str(spec["kind"])
        if kind == "mean":
            return fmean(_episode_field(item, str(spec["field"])) for item in episodes)
        if kind == "event_rate":
            event = str(spec["event"])
            return fmean(float(event in item.events) for item in episodes)
        if kind == "outcome_rate":
            outcome = str(spec["outcome"])
            return fmean(float(item.outcome == outcome) for item in episodes)
        if kind == "event_count_mean":
            event = str(spec["event"])
            return fmean(item.events.count(event) for item in episodes)
        if kind == "sequence_rate":
            sequence = [str(item) for item in spec["sequence"]]
            return fmean(float(_contains_sequence(item.events, sequence)) for item in episodes)
        if kind in {"conditional_event_rate", "conditional_outcome_rate"}:
            selected = [item for item in episodes if _matches_context(item, spec["where"])]
            if not selected:
                return math.nan
            if kind == "conditional_event_rate":
                event = str(spec["event"])
                return fmean(float(event in item.events) for item in selected)
            outcome = str(spec["outcome"])
            return fmean(float(item.outcome == outcome) for item in selected)
        if kind in {"worst_context_event_rate", "worst_context_outcome_rate"}:
            field = str(spec["context_field"])
            groups: dict[str, list[Trajectory]] = {}
            for item in episodes:
                groups.setdefault(str(item.context.get(field)), []).append(item)
            if kind == "worst_context_event_rate":
                event = str(spec["event"])
                return min(fmean(float(event in item.events) for item in group) for group in groups.values())
            outcome = str(spec["outcome"])
            return min(fmean(float(item.outcome == outcome) for item in group) for group in groups.values())
        raise ValueError(f"Unsupported metric kind: {kind}")

    def evaluate(self, policy_id: str, trajectories: Iterable[Trajectory]) -> "OracleResult":
        episodes = list(trajectories)
        constraints: list[dict[str, Any]] = []
        for index, constraint in enumerate(self.objective.get("constraints", [])):
            value = self.metric(episodes, constraint["metric"])
            threshold = float(constraint["value"])
            op = str(constraint["op"])
            passed = bool(math.isfinite(value) and _compare(value, op, threshold))
            margin = self._normalized_margin(value, op, threshold)
            constraints.append(
                {
                    "index": index,
                    "metric": dict(constraint["metric"]),
                    "op": op,
                    "threshold": threshold,
                    "value": value,
                    "passed": passed,
                    "normalized_margin": margin,
                }
            )
        preferences: list[dict[str, Any]] = []
        ordered_preferences = sorted(self.objective.get("preferences", []), key=lambda row: int(row.get("priority", 0)))
        for preference in ordered_preferences:
            value = self.metric(episodes, preference["metric"])
            direction = str(preference["direction"])
            if direction not in {"min", "max"}:
                raise ValueError(f"Unsupported preference direction: {direction}")
            utility = value if direction == "max" else -value
            preferences.append(
                {
                    "metric": dict(preference["metric"]),
                    "direction": direction,
                    "priority": int(preference.get("priority", 0)),
                    "value": value,
                    "utility": utility,
                }
            )
        feasible = all(row["passed"] for row in constraints)
        minimum_margin = min((row["normalized_margin"] for row in constraints), default=0.0)
        preference_vector = tuple(float(row["utility"]) for row in preferences)
        comparison_key = (1.0, 0.0, *preference_vector) if feasible else (0.0, minimum_margin, *preference_vector)
        return OracleResult(
            objective_id=str(self.objective["id"]),
            policy_id=policy_id,
            feasible=feasible,
            minimum_constraint_margin=minimum_margin,
            preference_vector=preference_vector,
            comparison_key=comparison_key,
            constraints=tuple(constraints),
            preferences=tuple(preferences),
            episode_count=len(episodes),
        )

    @staticmethod
    def _normalized_margin(value: float, op: str, threshold: float) -> float:
        if not math.isfinite(value):
            return -math.inf
        scale = max(1.0, abs(threshold))
        if op in {">=", ">"}:
            return (value - threshold) / scale
        if op in {"<=", "<"}:
            return (threshold - value) / scale
        return 0.0 if _compare(value, op, threshold) else -1.0


@dataclass(frozen=True)
class OracleResult:
    objective_id: str
    policy_id: str
    feasible: bool
    minimum_constraint_margin: float
    preference_vector: tuple[float, ...]
    comparison_key: tuple[float, ...]
    constraints: tuple[dict[str, Any], ...]
    preferences: tuple[dict[str, Any], ...]
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "feasible": self.feasible,
            "minimum_constraint_margin": self.minimum_constraint_margin,
            "preference_vector": list(self.preference_vector),
            "comparison_key": list(self.comparison_key),
            "constraints": list(self.constraints),
            "preferences": list(self.preferences),
            "episode_count": self.episode_count,
        }


def rank_policies(
    objective: Mapping[str, Any],
    trajectories_by_policy: Mapping[str, Iterable[Trajectory]],
) -> list[OracleResult]:
    oracle = ObjectiveOracle(objective)
    results = [oracle.evaluate(policy_id, trajectories) for policy_id, trajectories in trajectories_by_policy.items()]
    results.sort(key=lambda result: result.policy_id)
    results.sort(key=lambda result: result.comparison_key, reverse=True)
    return results


def compare_results(left: OracleResult, right: OracleResult, tolerance: float = 1e-9) -> str:
    if left.objective_id != right.objective_id:
        raise ValueError("Oracle results must share an objective")
    if len(left.comparison_key) != len(right.comparison_key):
        raise ValueError("Oracle comparison keys have different lengths")
    for left_value, right_value in zip(left.comparison_key, right.comparison_key):
        if left_value == right_value:
            continue
        if abs(left_value - right_value) <= tolerance:
            continue
        return "left" if left_value > right_value else "right"
    return "tie"
