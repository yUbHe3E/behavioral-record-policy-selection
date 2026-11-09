from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .canonical import stable_hash
from .lunarlander_transfer import (
    _environment_kwargs,
    assert_lunarlander_objective_blind,
    build_candidate_blueprint,
    compile_lunarlander_step,
)
from .models import Step, Trajectory


def context_lookup(transfer_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contexts = {
        str(row["context_id"]): dict(row)
        for row in transfer_config["environment"]["contexts"]
    }
    if len(contexts) != len(transfer_config["environment"]["contexts"]):
        raise ValueError("LunarLander context identifiers must be unique")
    return contexts


def validate_lunarlander_training_config(
    training_config: Mapping[str, Any],
    transfer_config: Mapping[str, Any],
) -> dict[str, Any]:
    training = dict(training_config.get("training", {}))
    validation = dict(training_config.get("validation", {}))
    heldout = dict(training_config.get("heldout_reservation", {}))
    source = dict(training_config.get("source_transfer_config", {}))
    transfer_pool = dict(transfer_config.get("candidate_pool", {}))
    contexts = context_lookup(transfer_config)
    vector_environments = int(training.get("vector_environments", 0))
    n_steps = int(training.get("ppo", {}).get("n_steps", 0))
    rollout_size = vector_environments * n_steps
    checkpoints = [int(value) for value in training.get("checkpoint_timesteps", [])]
    validation_seeds = {
        context_id: set(
            range(
                int(validation.get("seed_bases", {}).get(context_id, -1)),
                int(validation.get("seed_bases", {}).get(context_id, -1))
                + int(validation.get("episodes_per_context", 0)),
            )
        )
        for context_id in contexts
    }
    heldout_seeds = {
        context_id: set(
            range(
                int(heldout.get("seed_bases", {}).get(context_id, -1)),
                int(heldout.get("seed_bases", {}).get(context_id, -1))
                + int(heldout.get("episodes_per_context", 0)),
            )
        )
        for context_id in contexts
    }
    gates = {
        "source_transfer_hash_matches": source.get("expected_config_hash")
        == stable_hash(transfer_config),
        "official_reward_only": training.get("official_environment_reward_only")
        is True
        and training.get("semantic_reward_shaping") is False,
        "profiles_match_frozen_candidate_blueprint": list(training.get("profiles", []))
        == list(transfer_pool.get("training_profiles", [])),
        "seeds_match_frozen_candidate_blueprint": list(training.get("seeds", []))
        == list(transfer_pool.get("training_seeds", [])),
        "checkpoints_match_frozen_candidate_blueprint": checkpoints
        == list(transfer_pool.get("checkpoint_timesteps", [])),
        "profiles_exist_as_contexts": set(training.get("profiles", []))
        == set(contexts),
        "rollout_size_is_positive": rollout_size > 0,
        "checkpoints_are_strictly_increasing": bool(checkpoints)
        and checkpoints == sorted(set(checkpoints)),
        "checkpoints_align_to_complete_ppo_rollouts": rollout_size > 0
        and all(value % rollout_size == 0 for value in checkpoints),
        "ppo_batch_divides_rollout": rollout_size > 0
        and int(training.get("ppo", {}).get("batch_size", 0)) > 0
        and rollout_size % int(training.get("ppo", {}).get("batch_size", 0)) == 0,
        "validation_count_matches_transfer_plan": int(
            validation.get("episodes_per_context", -1)
        )
        == int(transfer_pool.get("evaluation_validation_seeds_per_context", -2)),
        "heldout_count_matches_transfer_plan": int(
            heldout.get("episodes_per_context", -1)
        )
        == int(transfer_pool.get("evaluation_heldout_seeds_per_context", -2)),
        "validation_and_heldout_seeds_disjoint": all(
            validation_seeds[context_id].isdisjoint(heldout_seeds[context_id])
            for context_id in contexts
        ),
        "heldout_not_collected_during_training": heldout.get(
            "collect_during_policy_pool_training"
        )
        is False,
        "automatic_retry_disabled": training_config.get("monitoring", {}).get(
            "automatic_retry"
        )
        is False,
        "overwrite_disabled": training_config.get("monitoring", {}).get(
            "overwrite_existing_output"
        )
        is False,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "rollout_size": rollout_size,
        "expected_training_chains": len(training.get("profiles", []))
        * len(training.get("seeds", [])),
        "expected_candidates": len(training.get("profiles", []))
        * len(training.get("seeds", []))
        * len(checkpoints),
        "validation_seed_hash": stable_hash(
            {key: sorted(value) for key, value in validation_seeds.items()}
        ),
        "heldout_seed_hash": stable_hash(
            {key: sorted(value) for key, value in heldout_seeds.items()}
        ),
    }


def training_candidate_lookup(
    transfer_config: Mapping[str, Any],
) -> dict[tuple[str, int, int], str]:
    rows = build_candidate_blueprint(transfer_config)
    return {
        (
            str(row["environment_profile"]),
            int(row["training_seed"]),
            int(row["checkpoint_timestep"]),
        ): str(row["candidate_id"])
        for row in rows
    }


def evaluation_seeds(
    training_config: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, list[int]]:
    validation = training_config["validation"]
    count = int(validation["episodes_per_context"])
    if smoke:
        count = int(training_config["smoke_override"]["validation_episodes_per_context"])
    return {
        str(context_id): list(range(int(seed_base), int(seed_base) + count))
        for context_id, seed_base in validation["seed_bases"].items()
    }


def make_lunarlander_env(
    transfer_config: Mapping[str, Any],
    context_id: str,
):
    import gymnasium as gym

    contexts = context_lookup(transfer_config)
    if context_id not in contexts:
        raise KeyError(f"Unknown LunarLander context: {context_id}")
    return gym.make(
        str(transfer_config["environment"]["environment_id"]),
        **_environment_kwargs(contexts[context_id]),
    )


def make_lunarlander_vector_env(
    transfer_config: Mapping[str, Any],
    context_id: str,
    *,
    training_seed: int,
    n_envs: int,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def factory(rank: int):
        def _make():
            env = make_lunarlander_env(transfer_config, context_id)
            env.reset(seed=int(training_seed) + rank)
            return Monitor(env)

        return _make

    vector_env = DummyVecEnv([factory(rank) for rank in range(int(n_envs))])
    vector_env.seed(int(training_seed))
    return vector_env


def rollout_lunarlander_policy(
    *,
    model: Any,
    transfer_config: Mapping[str, Any],
    policy_id: str,
    context_id: str,
    seed: int,
    deterministic: bool = True,
    split: str = "validation",
) -> Trajectory:
    env = make_lunarlander_env(transfer_config, context_id)
    contexts = context_lookup(transfer_config)
    try:
        observation, _ = env.reset(seed=int(seed))
        previous_observation = observation
        steps: list[Step] = []
        total_return = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            predicted, _ = model.predict(observation, deterministic=bool(deterministic))
            action = int(predicted.item()) if hasattr(predicted, "item") else int(predicted)
            observation, reward, terminated, truncated, _ = env.step(action)
            compiled = compile_lunarlander_step(
                previous_observation=previous_observation,
                observation=observation,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            steps.append(
                Step(
                    t=len(steps),
                    action=str(compiled["action"]),
                    reward=float(compiled["reward"]),
                    state=dict(compiled["state"]),
                    events=tuple(str(value) for value in compiled["events"]),
                    predicates=dict(compiled["predicates"]),
                )
            )
            total_return += float(reward)
            previous_observation = observation
        context = dict(contexts[context_id])
        trajectory = Trajectory(
            policy_id=str(policy_id),
            episode_id=f"ll_episode_{stable_hash({'policy_id': policy_id, 'context_id': context_id, 'seed': int(seed), 'split': split}, 16)}",
            context=context,
            steps=tuple(steps),
            outcome="terminated" if terminated else "truncated",
            total_return=total_return,
            terminated=bool(terminated),
            truncated=bool(truncated),
            metadata={
                "adapter_schema_version": transfer_config["public_adapter"][
                    "schema_version"
                ],
                "evaluation_split": str(split),
            },
        )
        assert_lunarlander_objective_blind(trajectory.to_dict())
        return trajectory
    finally:
        env.close()


def _rounded_summary(trajectory: Trajectory) -> dict[str, Any]:
    states = [dict(step.state) for step in trajectory.steps]
    action_counts = Counter(str(step.action) for step in trajectory.steps)
    event_counts = Counter(trajectory.events)
    numeric_fields = (
        "x",
        "y",
        "vx",
        "vy",
        "angle",
        "angular_velocity",
    )
    extrema = {
        field: {
            "min": round(min(float(row[field]) for row in states), 3),
            "max": round(max(float(row[field]) for row in states), 3),
            "terminal": round(float(states[-1][field]), 3),
        }
        for field in numeric_fields
    }
    sequence_hash = stable_hash(
        [
            {
                "action": step.action,
                "events": list(step.events),
                "state": {
                    field: round(float(step.state[field]), 3)
                    for field in numeric_fields
                },
            }
            for step in trajectory.steps
        ],
        20,
    )
    return {
        "context_id": str(trajectory.context["context_id"]),
        "outcome": trajectory.outcome,
        "return": round(float(trajectory.total_return), 2),
        "length": trajectory.length,
        "action_counts": dict(sorted(action_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "extrema": extrema,
        "sequence_hash": sequence_hash,
    }


def summarize_validation_trajectories(
    trajectories: Sequence[Trajectory],
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("At least one validation trajectory is required")
    policy_ids = {trajectory.policy_id for trajectory in trajectories}
    if len(policy_ids) != 1:
        raise ValueError("A candidate summary cannot mix policy identifiers")
    returns = [float(trajectory.total_return) for trajectory in trajectories]
    lengths = [trajectory.length for trajectory in trajectories]
    episode_summaries = [_rounded_summary(trajectory) for trajectory in trajectories]
    by_context: dict[str, dict[str, Any]] = {}
    for context_id in sorted({row["context_id"] for row in episode_summaries}):
        selected = [row for row in episode_summaries if row["context_id"] == context_id]
        by_context[context_id] = {
            "episodes": len(selected),
            "mean_return": sum(float(row["return"]) for row in selected)
            / len(selected),
            "success_rate": sum(float(row["return"]) >= 200.0 for row in selected)
            / len(selected),
            "mean_length": sum(int(row["length"]) for row in selected) / len(selected),
        }
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return {
        "policy_id": next(iter(policy_ids)),
        "episodes": len(trajectories),
        "mean_return": mean_return,
        "return_standard_deviation": math.sqrt(variance),
        "minimum_return": min(returns),
        "maximum_return": max(returns),
        "mean_length": sum(lengths) / len(lengths),
        "official_return_200_success_rate": sum(value >= 200.0 for value in returns)
        / len(returns),
        "context_metrics": by_context,
        "behavior_signature": stable_hash(episode_summaries, 20),
    }
