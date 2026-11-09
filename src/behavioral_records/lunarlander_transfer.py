from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from typing import Any, Mapping, Sequence

from .canonical import stable_hash


STATE_FIELDS = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)
ACTION_LABELS = (
    "no_engine",
    "left_orientation_engine",
    "main_engine",
    "right_orientation_engine",
)
FORBIDDEN_PUBLIC_TERMS = {
    "aligned",
    "brittle",
    "correct_strategy",
    "good_policy",
    "oracle",
    "preferred",
    "recovery_quality",
    "safe_landing",
    "semantic_score",
    "stable_landing",
    "unsafe",
    "wasteful",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _first_package_version(*names: str) -> str | None:
    for name in names:
        version = _package_version(name)
        if version is not None:
            return version
    return None


def lunarlander_runtime_inventory() -> dict[str, Any]:
    packages = {}
    for module_name, package_names in (
        ("gymnasium", ("gymnasium",)),
        ("Box2D", ("Box2D", "box2d-py")),
        ("stable_baselines3", ("stable-baselines3",)),
        ("numpy", ("numpy",)),
        ("torch", ("torch",)),
    ):
        packages[module_name] = {
            "available": _module_available(module_name),
            "version": _first_package_version(*package_names),
        }
    cuda = False
    cuda_device = None
    if packages["torch"]["available"]:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            cuda_device = str(torch.cuda.get_device_name(0))
    return {
        "packages": packages,
        "cuda_available": cuda,
        "cuda_device": cuda_device,
    }


def assert_lunarlander_objective_blind(payload: Mapping[str, Any]) -> None:
    lowered = str(payload).lower()
    found = sorted(term for term in FORBIDDEN_PUBLIC_TERMS if term in lowered)
    if found:
        raise ValueError(
            f"Evaluative labels leaked into LunarLander public facts: {found}"
        )


def lunarlander_state(observation: Sequence[float]) -> dict[str, Any]:
    if len(observation) != len(STATE_FIELDS):
        raise ValueError(
            f"LunarLander observation must contain 8 values, received {len(observation)}"
        )
    numeric = [float(value) for value in observation]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("LunarLander observation contains a non-finite value")
    return {
        "x": numeric[0],
        "y": numeric[1],
        "vx": numeric[2],
        "vy": numeric[3],
        "angle": numeric[4],
        "angular_velocity": numeric[5],
        "left_leg_contact": numeric[6] >= 0.5,
        "right_leg_contact": numeric[7] >= 0.5,
    }


def _changed_sign(before: float, after: float) -> bool:
    return before != 0.0 and after != 0.0 and (before < 0.0) != (after < 0.0)


def compile_lunarlander_step(
    *,
    previous_observation: Sequence[float],
    observation: Sequence[float],
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    if not 0 <= int(action) < len(ACTION_LABELS):
        raise ValueError(f"LunarLander discrete action must be in [0, 3], received {action}")
    before = lunarlander_state(previous_observation)
    after = lunarlander_state(observation)
    events: list[str] = []
    for field, event in (
        ("vx", "horizontal_velocity_sign_change"),
        ("vy", "vertical_velocity_sign_change"),
        ("angle", "angle_sign_change"),
    ):
        if _changed_sign(float(before[field]), float(after[field])):
            events.append(event)
    for field, prefix in (
        ("left_leg_contact", "left_leg_contact"),
        ("right_leg_contact", "right_leg_contact"),
    ):
        if bool(before[field]) != bool(after[field]):
            events.append(f"{prefix}_{'on' if after[field] else 'off'}")
    if terminated:
        events.append("episode_terminated")
    if truncated:
        events.append("episode_truncated")
    action_label = ACTION_LABELS[int(action)]
    payload = {
        "action": action_label,
        "reward": float(reward),
        "state": after,
        "events": events,
        "predicates": {
            "no_engine_commanded": int(action) == 0,
            "main_engine_commanded": int(action) == 2,
            "orientation_engine_commanded": int(action) in {1, 3},
            "left_leg_contact": bool(after["left_leg_contact"]),
            "right_leg_contact": bool(after["right_leg_contact"]),
        },
    }
    assert_lunarlander_objective_blind(payload)
    return payload


def validate_lunarlander_transfer_config(config: Mapping[str, Any]) -> dict[str, Any]:
    environment = dict(config.get("environment", {}))
    adapter = dict(config.get("public_adapter", {}))
    pool = dict(config.get("candidate_pool", {}))
    requirements = dict(config.get("semantic_requirements", {}))
    contexts = list(environment.get("contexts", []))
    expected_candidates = (
        len(pool.get("training_profiles", []))
        * len(pool.get("training_seeds", []))
        * len(pool.get("checkpoint_timesteps", []))
    )
    gates = {
        "environment_is_frozen_lunarlander_v3_discrete": environment.get(
            "environment_id"
        )
        == "LunarLander-v3"
        and environment.get("continuous") is False,
        "three_unique_contexts": len(contexts) == 3
        and len({str(row.get("context_id")) for row in contexts}) == 3,
        "public_adapter_has_exact_state_schema": tuple(adapter.get("state_fields", []))
        == STATE_FIELDS,
        "public_adapter_has_exact_action_schema": tuple(
            adapter.get("action_labels", [])
        )
        == ACTION_LABELS,
        "compiler_accepts_no_objective_or_oracle": adapter.get(
            "accepts_natural_language_objective"
        )
        is False
        and adapter.get("accepts_oracle_label") is False
        and adapter.get("accepts_judge_output") is False,
        "two_requirements_each_with_two_phrasings": len(requirements) == 2
        and all(
            bool(row.get("public_text")) and bool(row.get("independent_paraphrase"))
            for row in requirements.values()
        ),
        "official_reward_only_without_semantic_shaping": pool.get(
            "training_reward"
        )
        == "unmodified_official_environment_reward_only"
        and pool.get("semantic_reward_shaping") is False,
        "candidate_blueprint_count_matches": expected_candidates
        == int(pool.get("expected_candidate_count", -1)),
        "hidden_oracle_is_heldout_only": config.get("hidden_oracle", {}).get(
            "input_split"
        )
        == "heldout_raw_trajectories_only"
        and config.get("hidden_oracle", {}).get("validation_record_access") is False
        and config.get("hidden_oracle", {}).get("judge_response_access") is False,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "expected_candidate_count": expected_candidates,
    }


def build_candidate_blueprint(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validation = validate_lunarlander_transfer_config(config)
    if not validation["passed"]:
        raise ValueError("LunarLander transfer config failed structural validation")
    pool = config["candidate_pool"]
    rows = []
    for profile in pool["training_profiles"]:
        for seed in pool["training_seeds"]:
            for timestep in pool["checkpoint_timesteps"]:
                private_spec = {
                    "environment_profile": str(profile),
                    "training_seed": int(seed),
                    "checkpoint_timestep": int(timestep),
                    "algorithm": str(pool["algorithm"]),
                    "training_reward": str(pool["training_reward"]),
                }
                rows.append(
                    {
                        "candidate_id": f"ll_candidate_{stable_hash(private_spec, 12)}",
                        **private_spec,
                        "public_identity_allowed": False,
                    }
                )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise RuntimeError("Candidate blueprint contains duplicate identifiers")
    return rows


def _environment_kwargs(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "continuous": False,
        "gravity": float(context["gravity"]),
        "enable_wind": bool(context["enable_wind"]),
        "wind_power": float(context["wind_power"]),
        "turbulence_power": float(context["turbulence_power"]),
    }


def probe_lunarlander_environment(
    config: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (6201, 6202),
) -> dict[str, Any]:
    inventory = lunarlander_runtime_inventory()
    if not inventory["packages"]["Box2D"]["available"]:
        return {
            "status": "SKIPPED_DEPENDENCY_MISSING",
            "missing": ["Box2D"],
            "seeded_determinism": None,
            "objective_blind_adapter": "VERIFIED_SYNTHETIC_ONLY",
        }

    import gymnasium as gym

    action_sequence = (0, 2, 0, 1, 0, 3, 2, 0) * 4
    context_rows = []
    all_deterministic = True
    for context in config["environment"]["contexts"]:
        seed_rows = []
        for seed in seeds:
            repeat_hashes = []
            for _ in range(2):
                env = gym.make(
                    str(config["environment"]["environment_id"]),
                    **_environment_kwargs(context),
                )
                try:
                    observation, _ = env.reset(seed=int(seed))
                    initial_hash = stable_hash(lunarlander_state(observation))
                    transitions = []
                    previous = observation
                    for action in action_sequence:
                        observation, reward, terminated, truncated, _ = env.step(action)
                        transitions.append(
                            compile_lunarlander_step(
                                previous_observation=previous,
                                observation=observation,
                                action=action,
                                reward=reward,
                                terminated=terminated,
                                truncated=truncated,
                            )
                        )
                        previous = observation
                        if terminated or truncated:
                            break
                    repeat_hashes.append(
                        stable_hash(
                            {
                                "initial_state_hash": initial_hash,
                                "transitions": transitions,
                            }
                        )
                    )
                finally:
                    env.close()
            deterministic = len(set(repeat_hashes)) == 1
            all_deterministic = all_deterministic and deterministic
            seed_rows.append(
                {
                    "seed": int(seed),
                    "repeat_hashes": repeat_hashes,
                    "deterministic": deterministic,
                }
            )
        context_rows.append(
            {
                "context_id": str(context["context_id"]),
                "seed_rows": seed_rows,
            }
        )
    return {
        "status": "PASS_INTERFACE" if all_deterministic else "FAIL_NONDETERMINISTIC",
        "seeded_determinism": all_deterministic,
        "objective_blind_adapter": "PASS",
        "contexts": context_rows,
        "scope": "environment_interface_and_adapter_only_not_policy_pool_qualification",
    }


def build_lunarlander_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_lunarlander_transfer_config(config)
    blueprint = build_candidate_blueprint(config) if validation["passed"] else []
    inventory = lunarlander_runtime_inventory()
    probe = probe_lunarlander_environment(config) if validation["passed"] else {
        "status": "SKIPPED_INVALID_CONFIG"
    }
    runtime_ready = bool(
        inventory["packages"]["gymnasium"]["available"]
        and inventory["packages"]["Box2D"]["available"]
        and inventory["packages"]["stable_baselines3"]["available"]
    )
    status = (
        "READY_FOR_POLICY_POOL_TRAINING"
        if validation["passed"]
        and runtime_ready
        and probe.get("status") == "PASS_INTERFACE"
        else "PREPARED_DEPENDENCY_MISSING"
        if validation["passed"] and not runtime_ready
        else "PREFLIGHT_FAILED"
    )
    return {
        "schema_version": "lunarlander-standard-transfer-preflight-v1",
        "status": status,
        "config_hash": stable_hash(config),
        "config_validation": validation,
        "runtime_inventory": inventory,
        "environment_probe": probe,
        "candidate_blueprint_count": len(blueprint),
        "candidate_blueprint_hash": stable_hash(blueprint),
        "api_calls": 0,
        "claim": "Offline preparation and interface evidence only; no policy-selection result.",
    }
