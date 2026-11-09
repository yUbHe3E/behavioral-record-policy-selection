import copy
from pathlib import Path
import pytest
from behavioral_records.io import read_json
from behavioral_records.highway_cohort_v2 import (
    STATE_FIELDS, audit_complete_evidence, highway_schedule, public_highway_record, select_checkpoint,
)

PLAN = read_json(Path(__file__).resolve().parents[1] / "configs/kbs_highway_external_v2_20261106.json")


def evaluation(step, reward, completion=.9, speed=25., relation=0.):
    return {"checkpoint": step, "mean_official_return": reward,
            "profile": {"aggregate": {"completion_rate": completion, "mean_speed_mps": speed,
                                        "unsafe_change_episode_rate": relation}}}


def test_capability_precedes_return_and_never_selects_ineligible_checkpoint():
    rows = [evaluation(20000, 50, .7), evaluation(35000, 20), evaluation(50000, 40, speed=10)]
    assert select_checkpoint(rows, PLAN)["checkpoint"] == 35000
    assert select_checkpoint([rows[0], rows[2]], PLAN) is None


def test_tie_prefers_earlier_step_without_semantic_score_selection():
    rows = [evaluation(35000, 20, relation=0), evaluation(20000, 20, relation=1)]
    assert select_checkpoint(rows, PLAN)["checkpoint"] == 20000


def test_highway_preserves_original_seeds_contexts_and_limits():
    jobs = highway_schedule(PLAN)
    assert len(jobs) == len({j["seed"] for j in jobs}) == 12
    assert {j["environment"] for j in jobs} == {"highway"}
    assert sum(j["context"] == "light" for j in jobs) == 6
    assert sum(j["checkpoints"][-1] for j in jobs) == 600000
    assert PLAN["limits"]["remaining_compute_seconds"] < 86400
    assert PLAN["limits"]["judge_requests"] == 1584


def episodes():
    result = []
    for density in (.8, 1.2):
        previous = {long: 0 for long in STATE_FIELDS.values()}
        rows = []
        for i, action in enumerate(("IDLE", "LANE_RIGHT", "LANE_LEFT")):
            current = dict(previous, position_x=(i+1)*25., speed=25.+i+density)
            rows.append({"t": i, "action": action, "reward": density+i,
                         "events": [] if i == 0 else ["lane_changed"],
                         "before_state_private": previous, "state": current})
            previous = current
        result.append({"candidate_id": "SECRET_LINEAGE", "evaluation_seed_private": 123456,
                       "split": "validation", "context": {"vehicles_density": density},
                       "length": 3, "outcome": "completed_horizon", "return": 4.,
                       "terminated": False, "truncated": True, "steps": rows})
    return result


def test_complete_evidence_round_trip_identity_boundary_and_initial_context():
    raw = episodes()
    full = public_highway_record(raw)
    audit, linked, broken = audit_complete_evidence(full)
    assert audit["round_trip_exact"] and audit["complete_step_count"] == 6
    assert "SECRET_LINEAGE" not in str(full) and "evaluation_seed_private" not in str(full)
    assert full["episodes"][0]["context"]["initial_state"]["x"] == 0
    assert len(full["episodes"][0]["steps"]) == 3
    assert linked != broken


def test_display_rejects_private_split_or_missing_physical_context():
    raw = episodes()
    raw[0]["split"] = "reference"
    with pytest.raises(ValueError, match="display"):
        public_highway_record(raw)
    raw = episodes()
    raw[0]["steps"][1]["before_state_private"] = copy.deepcopy(raw[0]["steps"][0]["state"])
    raw[0]["steps"][1]["before_state_private"]["position_x"] += 1
    with pytest.raises(ValueError, match="continuity"):
        public_highway_record(raw)
