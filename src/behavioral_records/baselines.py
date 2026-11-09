from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any, Iterable

from .models import Trajectory


def scalar_summary(policy_id: str, trajectories: Iterable[Trajectory]) -> dict[str, Any]:
    episodes = list(trajectories)
    if not episodes:
        raise ValueError("At least one trajectory is required")
    success = [float(item.outcome == "success" or "goal_reached" in item.events) for item in episodes]
    return {
        "representation": "scalar_summary",
        "policy_id": policy_id,
        "episode_count": len(episodes),
        "mean_return": fmean(item.total_return for item in episodes),
        "mean_length": fmean(item.length for item in episodes),
        "success_rate": fmean(success),
    }


def unordered_event_summary(policy_id: str, trajectories: Iterable[Trajectory]) -> dict[str, Any]:
    episodes = list(trajectories)
    if not episodes:
        raise ValueError("At least one trajectory is required")
    event_counts = Counter(event for item in episodes for event in item.events)
    event_episode_counts = Counter(event for item in episodes for event in set(item.events))
    action_counts = Counter(str(step.action) for item in episodes for step in item.steps)
    outcome_counts = Counter(item.outcome for item in episodes)
    return {
        "representation": "unordered_event_summary",
        "policy_id": policy_id,
        "episode_count": len(episodes),
        "event_counts": dict(sorted(event_counts.items())),
        "event_episode_rates": {
            event: event_episode_counts[event] / len(episodes) for event in sorted(event_episode_counts)
        },
        "action_counts": dict(sorted(action_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "mean_return": fmean(item.total_return for item in episodes),
        "mean_length": fmean(item.length for item in episodes),
    }

