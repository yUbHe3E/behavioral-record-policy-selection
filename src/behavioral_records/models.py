from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Step:
    t: int
    action: str | int
    reward: float
    state: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    predicates: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Step":
        events = row.get("events", ())
        if isinstance(events, str):
            events = (events,) if events else ()
        return cls(
            t=int(row["t"]),
            action=row.get("action", "unknown"),
            reward=float(row.get("reward", 0.0)),
            state=dict(row.get("state", {})),
            events=tuple(str(item) for item in events),
            predicates=dict(row.get("predicates", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "action": self.action,
            "reward": self.reward,
            "state": dict(self.state),
            "events": list(self.events),
            "predicates": dict(self.predicates),
        }


@dataclass(frozen=True)
class Trajectory:
    policy_id: str
    episode_id: str
    context: Mapping[str, Any]
    steps: tuple[Step, ...]
    outcome: str
    total_return: float
    terminated: bool
    truncated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(event for step in self.steps for event in step.events)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Trajectory":
        raw_steps = row.get("steps", ())
        steps = tuple(Step.from_dict(item) for item in raw_steps)
        if not steps:
            raise ValueError("A trajectory must contain at least one step")
        times = [step.t for step in steps]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Step times must be unique and monotonically increasing")
        total_return = row.get("return", sum(step.reward for step in steps))
        return cls(
            policy_id=str(row["policy_id"]),
            episode_id=str(row["episode_id"]),
            context=dict(row.get("context", {})),
            steps=steps,
            outcome=str(row.get("outcome", "unknown")),
            total_return=float(total_return),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "episode_id": self.episode_id,
            "context": dict(self.context),
            "outcome": self.outcome,
            "return": self.total_return,
            "length": self.length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }

