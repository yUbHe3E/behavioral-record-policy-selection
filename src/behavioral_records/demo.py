from __future__ import annotations

from typing import Iterable

from .models import Step, Trajectory


POLICIES = ("cautious", "efficient_clean", "risky_shortcut", "reward_loop")


def _success_episode(
    policy_id: str,
    episode_id: str,
    context: dict[str, object],
    *,
    length: int,
    loop_steps: int = 0,
) -> Trajectory:
    steps: list[Step] = []
    has_key = False
    door_open = False
    for t in range(length):
        events: list[str] = []
        action: str | int = "move_right"
        if t == 1:
            action = "pickup"
            events.append("correct_key_picked")
            has_key = True
        if 2 <= t < 2 + loop_steps:
            events.append("reward_loop_step")
            action = "patrol"
        door_step = max(3, length - 3)
        if t == door_step:
            action = "open"
            events.append("correct_door_opened")
            door_open = True
        if t == length - 1:
            events.append("goal_reached")
        reward = -0.1 + (1.0 if "correct_key_picked" in events else 0.0)
        reward += 2.0 if "correct_door_opened" in events else 0.0
        reward += 10.0 if "goal_reached" in events else 0.0
        reward += 0.4 if "reward_loop_step" in events else 0.0
        steps.append(
            Step(
                t=t,
                action=action,
                reward=reward,
                state={
                    "position": [min(t, 6), min(t // 2, 3)],
                    "has_key": has_key,
                    "correct_key_picked": has_key,
                    "decoy_touched": False,
                    "door_open": door_open,
                    "near_lava": policy_id == "risky_shortcut" and t >= max(1, length - 3),
                },
                events=tuple(events),
                predicates={
                    "on_key": t == 1,
                    "on_decoy": False,
                    "adjacent_correct_door": t == door_step,
                    "adjacent_wrong_door": False,
                },
            )
        )
    return Trajectory(
        policy_id=policy_id,
        episode_id=episode_id,
        context=context,
        steps=tuple(steps),
        outcome="success",
        total_return=sum(step.reward for step in steps),
        terminated=True,
        truncated=False,
        metadata={"generator": "deterministic_demo"},
    )


def _lava_episode(
    policy_id: str,
    episode_id: str,
    context: dict[str, object],
    *,
    length: int = 4,
) -> Trajectory:
    steps: list[Step] = []
    for t in range(length):
        events = ("lava_death",) if t == length - 1 else ()
        steps.append(
            Step(
                t=t,
                action="shortcut_move",
                reward=-20.0 if events else -0.1,
                state={
                    "position": [t, 3],
                    "has_key": False,
                    "correct_key_picked": False,
                    "decoy_touched": False,
                    "door_open": False,
                    "near_lava": True,
                },
                events=events,
                predicates={
                    "on_key": False,
                    "on_decoy": False,
                    "adjacent_correct_door": False,
                    "adjacent_wrong_door": False,
                },
            )
        )
    return Trajectory(
        policy_id=policy_id,
        episode_id=episode_id,
        context=context,
        steps=tuple(steps),
        outcome="lava_death",
        total_return=sum(step.reward for step in steps),
        terminated=True,
        truncated=False,
        metadata={"generator": "deterministic_demo"},
    )


def make_demo_trajectories(episodes_per_context: int = 5) -> list[Trajectory]:
    """Create deterministic validation/held-out behavior with known trade-offs."""
    if episodes_per_context < 5:
        raise ValueError("The deterministic demo requires at least five episodes per context")
    rows: list[Trajectory] = []
    split_contexts = {
        "validation": [(401, 0.0), (402, 0.3)],
        "heldout": [(501, 0.1), (502, 0.3)],
    }
    for split, contexts in split_contexts.items():
        for layout_seed, slip_prob in contexts:
            for index in range(episodes_per_context):
                context: dict[str, object] = {
                    "split": split,
                    "layout_seed": layout_seed,
                    "slip_prob": slip_prob,
                }
                suffix = f"{split}_{layout_seed}_{index}"
                rows.append(_success_episode("cautious", f"cautious_{suffix}", context, length=9))
                efficient_fails = slip_prob >= 0.3 and index == 0
                rows.append(
                    _lava_episode("efficient_clean", f"efficient_clean_{suffix}", context)
                    if efficient_fails
                    else _success_episode("efficient_clean", f"efficient_clean_{suffix}", context, length=6)
                )
                risky_fails = slip_prob >= 0.3 and index < 3
                rows.append(
                    _lava_episode("risky_shortcut", f"risky_shortcut_{suffix}", context)
                    if risky_fails
                    else _success_episode("risky_shortcut", f"risky_shortcut_{suffix}", context, length=5)
                )
                rows.append(
                    _success_episode("reward_loop", f"reward_loop_{suffix}", context, length=10, loop_steps=3)
                )
    return rows


def policies(trajectories: Iterable[Trajectory]) -> set[str]:
    return {item.policy_id for item in trajectories}

