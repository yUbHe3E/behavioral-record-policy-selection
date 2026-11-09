"""Capability-first selection and requirement-blind complete Highway evidence."""
from __future__ import annotations

import copy
from .bounded_external import qualifies, training_schedule
from .canonical import canonical_json
from .external_validation import rounded
from .relational_record import (
    RelationalBehaviorCompiler, audit_relation_break, relation_broken_control,
    validate_relational_record_contract,
)


STATE_FIELDS = {
    "x": "position_x", "y": "position_y", "v": "speed", "h": "heading",
    "ln": "lane", "tln": "target_lane", "hit": "crashed", "road": "on_road",
}
for short, long in (("c", "current"), ("l", "left"), ("r", "right")):
    STATE_FIELDS[short + "_exists"] = long + "_lane_exists"
    for suffix, field in (("fd", "front_distance"), ("fv", "front_speed"),
                          ("rd", "rear_distance"), ("rv", "rear_speed")):
        STATE_FIELDS[short + "_" + suffix] = long + "_" + field


def highway_schedule(plan):
    jobs = [job for job in training_schedule(plan) if job["environment"] == "highway"]
    if len(jobs) != 12 or any(job["checkpoints"] != [20000, 35000, 50000] for job in jobs):
        raise ValueError("The approved twelve Highway chains and step caps must be preserved")
    return jobs


def select_checkpoint(evaluations, plan):
    eligible = [row for row in evaluations if qualifies(row["profile"], plan, "highway")]
    return max(eligible, key=lambda row: (row["mean_official_return"], -row["checkpoint"])) if eligible else None


def public_state(state):
    if set(state) != set(STATE_FIELDS.values()):
        raise ValueError("Unmapped physical state field")
    return rounded({short: state[long] for short, long in STATE_FIELDS.items()})


def public_highway_record(episodes):
    """No requirement, oracle, identity, policy source, or reference argument."""
    output = []
    for episode in episodes:
        if episode["split"] != "validation":
            raise ValueError("Only the independent display split is public")
        rows = episode["steps"]
        if not rows or len(rows) != episode["length"]:
            raise ValueError("Incomplete supplied episode")
        if len(rows) > 30:
            raise ValueError("Unexpected official Highway-fast horizon")
        # Every action has its pre-action context: initial_state for the first,
        # then the previous recorded post-action state for all remaining actions.
        for before, after in zip(rows, rows[1:]):
            if before["state"] != after["before_state_private"]:
                raise ValueError("Physical state continuity lost")
        output.append({
            "context": {
                "vehicles_density": episode["context"]["vehicles_density"],
                "policy_interval_seconds": 1, "horizon_seconds": 30,
                "state_fields": STATE_FIELDS,
                "units": "position and distance in meters, speed in meters/second, heading in radians; lanes are zero-indexed; c/l/r denote current/left/right lane; null neighbour means none observed",
                "initial_state": public_state(rows[0]["before_state_private"]),
                "recording": "Every policy step; state is post-action. Changes in physical lane can follow an earlier lane-change command. No steps omitted. Numbers rounded to four decimal places.",
            },
            "length": len(rows), "outcome": episode["outcome"],
            "return": round(episode["return"], 4),
            "terminated": episode["terminated"], "truncated": episode["truncated"],
            "steps": [{"t": row["t"], "action": row["action"],
                       "reward": round(row["reward"], 4), "events": copy.deepcopy(row["events"]),
                       "state": public_state(row["state"]), "predicates": {}} for row in rows],
        })
    return {"representation": "deterministic_raw_validation_trajectories", "episodes": output}


def audit_complete_evidence(full):
    linked = RelationalBehaviorCompiler().compile(full)
    broken = relation_broken_control(linked)
    if validate_relational_record_contract(linked) or validate_relational_record_contract(broken):
        raise ValueError("Compiler contract failed")
    reconstructed = []
    for pattern in linked["patterns"]:
        behavior = pattern["behavior"]
        states, predicates, steps = {}, {}, []
        for relation in behavior["relations"]:
            observation = relation["observation"]
            for name, target in (("state_relation", states), ("predicate_relation", predicates)):
                piece = observation[name]
                if piece["mode"] == "snapshot":
                    target.clear()
                    target.update(copy.deepcopy(piece["values"]))
                else:
                    for field, (old, new) in piece["values"].items():
                        if target.get(field, "<NOT_OBSERVED>") != old:
                            raise ValueError("State delta cannot be reconstructed")
                        if new == "<NOT_OBSERVED>":
                            target.pop(field, None)
                        else:
                            target[field] = copy.deepcopy(new)
            steps.append({"t": relation["t"], "action": relation["action"],
                          "reward": observation["reward"], "events": observation["events"],
                          "state": copy.deepcopy(states), "predicates": copy.deepcopy(predicates)})
        episode = {"context": pattern["context"], "steps": steps,
                   **{key: behavior[key] for key in ("length", "return", "outcome", "terminated", "truncated")}}
        reconstructed.extend([episode] * pattern["multiplicity"])
    exact = sorted(map(canonical_json, reconstructed)) == sorted(map(canonical_json, full["episodes"]))
    if not exact:
        raise ValueError("Round trip changed supplied factual evidence")
    binding = audit_relation_break(linked, broken)
    if not all(value for value in binding.values() if isinstance(value, bool)):
        raise ValueError("Matched relation perturbation failed")
    return {"round_trip_exact": exact, "complete_step_count": sum(e["length"] for e in full["episodes"]),
            "linked_bytes": len(canonical_json(linked).encode()), "binding_audit": binding}, linked, broken
