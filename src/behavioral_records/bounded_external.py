"""Offline facts and qualification for the finite, independently trained cohort."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from statistics import fmean
from typing import Any

from .canonical import stable_hash
from .io import read_json
from .lunarlander_semantic import build_hidden_policy_profile, compare_hidden_profiles
from .lunarlander_transfer import compile_lunarlander_step, lunarlander_state


def validate_seed_partitions(plan: dict) -> None:
    observed = set()
    for environment_index, environment in enumerate(("lunar", "highway")):
        for split in ("development", "validation", "reference"):
            spec = plan["splits"][split]
            for context_index, _ in enumerate(plan[environment]["contexts"]):
                start = (spec["base_seed"] + context_index * plan["splits"]["context_seed_offset"]
                         + environment_index * plan["splits"]["environment_seed_offset"])
                seeds = set(range(start, start + spec["episodes_per_context"]))
                if observed & seeds:
                    raise ValueError("Evaluation seed partitions overlap")
                observed.update(seeds)


def training_schedule(plan: dict) -> list[dict]:
    validate_seed_partitions(plan)
    jobs = []
    for index in range(6):
        for algorithm in ("ppo", "dqn"):
            spec = plan["lunar"]
            jobs.append({"environment": "lunar", "algorithm": algorithm,
                         "seed": spec[f"{algorithm}_seeds"][index],
                         "context": spec["training_context"],
                         "checkpoints": spec[f"{algorithm}_checkpoints"]})
    for index, seed in enumerate(plan["highway"]["seeds"]):
        jobs.append({"environment": "highway", "algorithm": "dqn", "seed": seed,
                     "context": "light" if index < 6 else "dense",
                     "checkpoints": plan["highway"]["checkpoints"]})
    for job in jobs:
        job["chain_id"] = f"{job['environment']}_{job['algorithm']}_{job['seed']}"
    if len(jobs) != 24 or len({j['seed'] for j in jobs}) != 24:
        raise ValueError("Expected 24 independently seeded chains")
    return jobs


def make_environment(plan: dict, environment: str, context_id: str):
    import gymnasium as gym

    spec = plan[environment]
    context = next(c for c in spec["contexts"] if c["context_id"] == context_id)
    kwargs = {k: v for k, v in context.items() if k != "context_id"}
    if environment == "highway":
        import highway_env  # noqa: F401 -- registers the official environments
        return gym.make(spec["environment_id"], config=kwargs)
    return gym.make(spec["environment_id"], **kwargs)


def highway_state(env) -> dict[str, Any]:
    """Physical ego and lane-neighbour facts; no preference or reference input."""
    base = env.unwrapped
    ego = base.vehicle
    lane_index = ego.lane_index
    state = {"position_x": float(ego.position[0]), "position_y": float(ego.position[1]),
             "speed": float(ego.speed), "heading": float(ego.heading),
             "lane": int(lane_index[2]), "target_lane": int(ego.target_lane_index[2]),
             "crashed": bool(ego.crashed), "on_road": bool(ego.on_road)}
    for name, offset in (("current", 0), ("left", -1), ("right", 1)):
        lane = int(lane_index[2]) + offset
        exists = 0 <= lane < base.config["lanes_count"]
        state[f"{name}_lane_exists"] = exists
        front, rear = (None, None)
        if exists:
            front, rear = base.road.neighbour_vehicles(ego, (lane_index[0], lane_index[1], lane))
        for position, vehicle in (("front", front), ("rear", rear)):
            distance = None if vehicle is None else abs(float(ego.lane_distance_to(vehicle)))
            state[f"{name}_{position}_distance"] = distance
            state[f"{name}_{position}_speed"] = None if vehicle is None else float(vehicle.speed)
    return state


def rollout(model, plan: dict, environment: str, context_id: str, seed: int,
            candidate_id: str, split: str, progress=None, heuristic: bool = False) -> dict:
    env = make_environment(plan, environment, context_id)
    context = next(c for c in plan[environment]["contexts"] if c["context_id"] == context_id)
    try:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        total = 0.0
        steps = []
        while not (terminated or truncated):
            if progress is not None and len(steps) % 100 == 0:
                progress()
            previous = observation
            before = highway_state(env) if environment == "highway" else lunarlander_state(previous)
            if heuristic:
                from gymnasium.envs.box2d.lunar_lander import heuristic as official_heuristic
                action = official_heuristic(env.unwrapped, observation)
            else:
                predicted, _ = model.predict(observation, deterministic=True)
                action = int(predicted.item()) if hasattr(predicted, "item") else int(predicted)
            observation, reward, terminated, truncated, _ = env.step(action)
            if environment == "lunar":
                factual = compile_lunarlander_step(previous_observation=previous, observation=observation,
                    action=action, reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
                step = {**factual, "action_id": action,
                        "body_contact_game_over": bool(env.unwrapped.game_over),
                        "lander_awake": bool(env.unwrapped.lander.awake)}
            else:
                state = highway_state(env)
                events = []
                if before["lane"] != state["lane"]:
                    events.append("lane_changed")
                if state["crashed"]:
                    events.append("collision")
                step = {"action": env.unwrapped.action_type.actions[action], "action_id": action,
                        "reward": float(reward), "state": state, "events": events, "predicates": {},
                        "before_state_private": before, "seconds": float(env.unwrapped.time)}
            steps.append({"t": len(steps), **step})
            total += float(reward)
        final = steps[-1]
        if environment == "lunar":
            cause = "other_termination"
            if final["body_contact_game_over"]:
                cause = "body_contact_termination"
            elif abs(final["state"]["x"]) >= 1:
                cause = "horizontal_boundary_termination"
            elif not final["lander_awake"]:
                cause = "settled_termination"
            elif truncated:
                cause = "time_limit_truncation"
        else:
            cause = "collision" if final["state"]["crashed"] else "completed_horizon"
        return {"candidate_id": candidate_id, "split": split,
                "episode_id": "episode_" + stable_hash([candidate_id, context_id, seed, split], 20),
                "context": dict(context), "evaluation_seed_private": seed, "steps": steps,
                "return": total, "length": len(steps), "outcome": cause, "terminal_cause": cause,
                "terminated": bool(terminated), "truncated": bool(truncated)}
    finally:
        env.close()


def highway_episode_facts(episode: dict, thresholds: dict) -> dict:
    unsafe = 0
    reversals = 0
    last_direction = 0
    last_change_time = -1000.0
    for step in episode["steps"]:
        before, after = step["before_state_private"], step["state"]
        direction = after["lane"] - before["lane"]
        if direction == 0:
            continue
        side = "right" if direction > 0 else "left"
        front = before[f"{side}_front_distance"]
        rear = before[f"{side}_rear_distance"]
        rear_speed = before[f"{side}_rear_speed"] or 0.0
        front_min = max(thresholds["minimum_front_gap_meters"], thresholds["front_gap_seconds"] * before["speed"])
        rear_min = max(thresholds["minimum_rear_gap_meters"], thresholds["rear_gap_seconds"] * rear_speed)
        unsafe += int((front is not None and front < front_min) or (rear is not None and rear < rear_min))
        lead_gap = before["current_front_distance"]
        lead_speed = before["current_front_speed"]
        obstructed = (lead_gap is not None and lead_gap < thresholds["lead_gap_seconds"] * before["speed"]
                      and lead_speed < before["speed"] - thresholds["lead_speed_difference_mps"])
        reversals += int(direction == -last_direction and not obstructed
                         and step["seconds"] - last_change_time <= thresholds["reversal_window_seconds"])
        last_direction, last_change_time = direction, step["seconds"]
    mean_speed = fmean(s["state"]["speed"] for s in episode["steps"])
    initial = episode["steps"][0]["before_state_private"]["position_x"]
    return {"completion_rate": float(episode["terminal_cause"] == "completed_horizon"),
            "mean_speed_mps": mean_speed,
            "low_progress_episode_rate": float(mean_speed < thresholds["low_progress_speed_mps"]),
            "unsafe_change_episode_rate": float(unsafe > 0),
            "unnecessary_reversal_episode_rate": float(reversals > 0),
            "mean_progress_meters": episode["steps"][-1]["state"]["position_x"] - initial}


def profile_for(episodes: list[dict], plan: dict, environment: str, repository) -> dict:
    if environment == "lunar":
        definitions = read_json(repository / plan["lunar"]["factual_definition_source"])
        return build_hidden_policy_profile(episodes[0]["candidate_id"], episodes,
                                          definitions["factual_operationalization"])
    facts = [highway_episode_facts(ep, plan["highway"]["operationalization"]) for ep in episodes]
    return {"aggregate": {key: fmean(r[key] for r in facts) for key in facts[0]}}


def qualifies(profile: dict, plan: dict, environment: str) -> bool:
    p, q = profile["aggregate"], plan["qualification"]
    if environment == "lunar":
        return p["controlled_landing_rate"] >= q["lunar_controlled_landing_rate"]
    return (p["completion_rate"] >= q["highway_collision_free_completion_rate"]
            and p["mean_speed_mps"] >= q["highway_mean_speed_mps"])


def qualify_pairs(candidates: list[dict], plan: dict, environment: str, repository) -> dict:
    spec = plan[environment]
    rules = spec.get("reference_rules")
    if environment == "lunar":
        rules = read_json(repository / spec["factual_definition_source"])["hidden_oracle"]["LL-R1"]
    completion_key = "controlled_landing_rate" if environment == "lunar" else "completion_rate"
    pairs = []
    for a, b in combinations(sorted(candidates, key=lambda r: r["chain_id"]), 2):
        reference = compare_hidden_profiles(a["profile"], b["profile"], rules)
        gap = abs(a["profile"]["aggregate"][completion_key] - b["profile"]["aggregate"][completion_key])
        relevant = (gap <= plan["qualification"]["maximum_completion_rate_gap"]
                    and reference["status"] == "DETERMINATE"
                    and reference["decisive_field"] in spec["relation_decisive_fields"])
        pairs.append({"pair_id": "bounded_pair_" + stable_hash([a["chain_id"], b["chain_id"]], 16),
                      "chain_a": a["chain_id"], "chain_b": b["chain_id"],
                      "development_reference": reference, "completion_gap": gap,
                      "relation_decisive": relevant})
    count = sum(p["relation_decisive"] for p in pairs)
    return {"qualified_chain_count": len(candidates), "pair_count": len(pairs),
            "relation_decisive_pair_count": count,
            "passed": len(candidates) >= plan["qualification"]["minimum_chains_per_environment"]
                      and count >= plan["qualification"]["minimum_relation_decisive_pairs"],
            "decisive_fields": dict(Counter(p["development_reference"]["decisive_field"] or "AMBIGUOUS" for p in pairs)),
            "pairs": sorted(pairs, key=lambda r: stable_hash(r["pair_id"]))}
