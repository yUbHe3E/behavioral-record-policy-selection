from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import shutil
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import stable_hash


FORBIDDEN_EVALUATIVE_TERMS = {
    "aligned",
    "correct_strategy",
    "preferred",
    "reckless",
    "reward_exploitation",
    "safe_strategy",
    "unnecessary_change",
}


@dataclass(frozen=True)
class StandardEnvironmentChoice:
    provider: str | None
    environment_id: str | None
    status: str
    reason: str


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_inventory() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "docker_executable": shutil.which("docker"),
        "packages": {
            name: {
                "available": _module_available(name),
                "version": _package_version(name),
            }
            for name in ("gymnasium", "minihack", "nle", "minigrid")
        },
    }


def choose_standard_environment(
    inventory: Mapping[str, Any],
) -> StandardEnvironmentChoice:
    packages = inventory.get("packages", {})
    minihack_available = bool(packages.get("minihack", {}).get("available"))
    nle_available = bool(packages.get("nle", {}).get("available"))
    minigrid_available = bool(packages.get("minigrid", {}).get("available"))
    docker_available = bool(inventory.get("docker_executable"))

    if minihack_available and nle_available:
        return StandardEnvironmentChoice(
            provider="minihack",
            environment_id="MiniHack-River-Lava-v0",
            status="ELIGIBLE",
            reason="MiniHack and NLE are importable.",
        )
    if minigrid_available:
        reason = (
            "MiniHack/NLE are unavailable; using the prospectively declared "
            "native MiniGrid fallback."
        )
        if not docker_available and platform.system().lower() == "windows":
            reason += " Docker is absent on this Windows host."
        return StandardEnvironmentChoice(
            provider="minigrid",
            environment_id="MiniGrid-ObstructedMaze-Full-v1",
            status="ELIGIBLE_FALLBACK",
            reason=reason,
        )
    return StandardEnvironmentChoice(
        provider=None,
        environment_id=None,
        status="SKIPPED_DEPENDENCY_MISSING",
        reason=(
            "Neither MiniHack/NLE nor the declared MiniGrid fallback is "
            "importable. Install the optional standard dependency before the "
            "environment probe."
        ),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _object_snapshot(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    result: dict[str, Any] = {
        "type": str(getattr(obj, "type", obj.__class__.__name__)),
        "color": getattr(obj, "color", None),
    }
    for field in ("is_open", "is_locked"):
        if hasattr(obj, field):
            result[field] = bool(getattr(obj, field))
    contains = getattr(obj, "contains", None)
    if contains is not None:
        result["contains"] = _object_snapshot(contains)
    return result


def minigrid_symbolic_snapshot(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    grid = getattr(base, "grid", None)
    objects: list[dict[str, Any]] = []
    if grid is not None:
        for y in range(int(grid.height)):
            for x in range(int(grid.width)):
                obj = grid.get(x, y)
                snapshot = _object_snapshot(obj)
                if snapshot is not None:
                    snapshot["x"] = x
                    snapshot["y"] = y
                    objects.append(snapshot)
    carrying = _object_snapshot(getattr(base, "carrying", None))
    agent_pos = getattr(base, "agent_pos", None)
    return {
        "agent": {
            "x": int(agent_pos[0]) if agent_pos is not None else None,
            "y": int(agent_pos[1]) if agent_pos is not None else None,
            "direction": int(getattr(base, "agent_dir", -1)),
            "carrying": carrying,
        },
        "grid": {
            "width": int(grid.width) if grid is not None else None,
            "height": int(grid.height) if grid is not None else None,
            "objects": objects,
        },
        "step_count": int(getattr(base, "step_count", 0)),
    }


def factual_state_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(path: tuple[str, ...], left: Any, right: Any) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                walk(path + (str(key),), left.get(key), right.get(key))
            return
        if left != right:
            rows.append({"field": ".".join(path), "before": _plain(left), "after": _plain(right)})

    walk((), before, after)
    return rows


def assert_objective_blind_facts(payload: Mapping[str, Any]) -> None:
    lowered = str(_plain(payload)).lower()
    found = sorted(term for term in FORBIDDEN_EVALUATIVE_TERMS if term in lowered)
    if found:
        raise ValueError(f"Evaluative labels leaked into standard-environment facts: {found}")


def _register_provider(provider: str) -> None:
    importlib.import_module(provider)


def probe_environment(
    provider: str,
    environment_id: str,
    *,
    seeds: Sequence[int] = (1301, 1302),
    actions: Sequence[int] = (0, 1, 2, 2, 5, 3, 2, 4),
) -> dict[str, Any]:
    import gymnasium as gym

    _register_provider(provider)
    runs: list[dict[str, Any]] = []
    deterministic = True
    for seed in seeds:
        seed_runs: list[dict[str, Any]] = []
        for repeat in range(2):
            env = gym.make(environment_id)
            try:
                observation, info = env.reset(seed=int(seed))
                initial = (
                    minigrid_symbolic_snapshot(env)
                    if provider == "minigrid"
                    else {"observation": _plain(observation), "info": _plain(info)}
                )
                transitions: list[dict[str, Any]] = []
                before = initial
                for requested_action in actions:
                    action = int(requested_action) % int(env.action_space.n)
                    observation, reward, terminated, truncated, info = env.step(action)
                    after = (
                        minigrid_symbolic_snapshot(env)
                        if provider == "minigrid"
                        else {"observation": _plain(observation), "info": _plain(info)}
                    )
                    delta = factual_state_delta(before, after)
                    transition = {
                        "action": action,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "state_delta": delta,
                    }
                    assert_objective_blind_facts(transition)
                    transitions.append(transition)
                    before = after
                    if terminated or truncated:
                        break
                seed_runs.append(
                    {
                        "repeat": repeat,
                        "initial_state_hash": stable_hash(initial),
                        "transition_hash": stable_hash(transitions),
                        "steps": len(transitions),
                        "nonempty_state_delta_steps": sum(bool(row["state_delta"]) for row in transitions),
                    }
                )
            finally:
                env.close()
        deterministic = deterministic and (
            seed_runs[0]["initial_state_hash"] == seed_runs[1]["initial_state_hash"]
            and seed_runs[0]["transition_hash"] == seed_runs[1]["transition_hash"]
        )
        runs.extend(seed_runs)
    return {
        "provider": provider,
        "environment_id": environment_id,
        "status": "PASS_INTERFACE" if deterministic else "FAIL_NONDETERMINISTIC",
        "seeded_determinism": deterministic,
        "objective_blind_adapter": "PASS",
        "runs": runs,
        "scope": "interface_and_seeded_transition_probe_only_not_semantic_diversity",
    }


def probe_minigrid_structure(
    environment_id: str,
    *,
    seeds: Iterable[int] = range(1400, 1432),
) -> dict[str, Any]:
    import gymnasium as gym

    _register_provider("minigrid")
    layouts: list[str] = []
    missions: set[str] = set()
    type_counts: dict[str, list[int]] = {}
    locked_door_counts: list[int] = []
    contained_object_counts: list[int] = []
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = gym.make(environment_id)
        try:
            observation, _ = env.reset(seed=int(seed))
            snapshot = minigrid_symbolic_snapshot(env)
            objects = list(snapshot["grid"]["objects"])
            counts: dict[str, int] = {}
            locked = 0
            contained = 0
            for obj in objects:
                object_type = str(obj.get("type"))
                counts[object_type] = counts.get(object_type, 0) + 1
                if object_type == "door" and bool(obj.get("is_locked")):
                    locked += 1
                contained_obj = obj.get("contains")
                if contained_obj is not None:
                    contained += 1
                    contained_type = f"contained::{contained_obj.get('type', 'unknown')}"
                    counts[contained_type] = counts.get(contained_type, 0) + 1
            for object_type in sorted(set(type_counts) | set(counts)):
                type_counts.setdefault(object_type, []).append(counts.get(object_type, 0))
            mission = (
                str(observation.get("mission", ""))
                if isinstance(observation, Mapping)
                else ""
            )
            missions.add(mission)
            layout_hash = stable_hash(snapshot["grid"])
            layouts.append(layout_hash)
            locked_door_counts.append(locked)
            contained_object_counts.append(contained)
            rows.append(
                {
                    "seed": int(seed),
                    "layout_hash": layout_hash,
                    "mission": mission,
                    "object_counts": counts,
                    "locked_doors": locked,
                    "objects_with_contents": contained,
                }
            )
        finally:
            env.close()
    sample_count = len(rows)
    distinct_nonstructural_types = sorted(
        object_type
        for object_type in type_counts
        if object_type not in {"empty", "wall"}
    )
    structural_pass = bool(
        sample_count
        and len(set(layouts)) >= max(2, int(0.80 * sample_count))
        and len(distinct_nonstructural_types) >= 4
        and min(locked_door_counts, default=0) >= 1
        and min(contained_object_counts, default=0) >= 1
    )
    return {
        "environment_id": environment_id,
        "status": "PASS_STRUCTURAL_COMPLEXITY" if structural_pass else "FAIL_STRUCTURAL_COMPLEXITY",
        "sampled_seeds": sample_count,
        "unique_layouts": len(set(layouts)),
        "unique_missions": len(missions),
        "nonstructural_object_types": distinct_nonstructural_types,
        "locked_doors_min_max": [min(locked_door_counts, default=0), max(locked_door_counts, default=0)],
        "objects_with_contents_min_max": [
            min(contained_object_counts, default=0),
            max(contained_object_counts, default=0),
        ],
        "structural_gate_pass": structural_pass,
        "scope": "environment_structure_only_not_learned_policy_or_semantic_pair_qualification",
        "rows": rows,
    }


def run_standard_preflight() -> dict[str, Any]:
    inventory = runtime_inventory()
    choice = choose_standard_environment(inventory)
    report: dict[str, Any] = {
        "schema_version": "standard-environment-preflight-v1",
        "inventory": inventory,
        "choice": {
            "provider": choice.provider,
            "environment_id": choice.environment_id,
            "status": choice.status,
            "reason": choice.reason,
        },
        "minihack_scientific_status": "NOT_TESTED_RUNTIME_INELIGIBLE",
    }
    if choice.provider is None or choice.environment_id is None:
        report["probe"] = {"status": "SKIPPED"}
        return report
    report["probe"] = probe_environment(choice.provider, choice.environment_id)
    if choice.provider == "minigrid":
        report["smoke_probe"] = probe_environment(
            "minigrid", "MiniGrid-KeyCorridorS6R3-v0", seeds=(1201, 1202)
        )
        report["structural_probe"] = probe_minigrid_structure(choice.environment_id)
    return report
