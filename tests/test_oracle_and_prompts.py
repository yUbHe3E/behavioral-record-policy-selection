from __future__ import annotations

from behavioral_records.demo import make_demo_trajectories
from behavioral_records.experiment import build_records, group_by_policy
from behavioral_records.io import read_json
from behavioral_records.oracle import rank_policies
from behavioral_records.prompts import make_pair_requests


def test_hidden_objectives_rank_known_demo_policy_first() -> None:
    heldout = group_by_policy([item for item in make_demo_trajectories() if item.context["split"] == "heldout"])
    objectives = read_json("configs/objectives_demo.json")
    rankings = {objective["id"]: rank_policies(objective, heldout) for objective in objectives}
    assert rankings["heldout_robust_conditional"][0].policy_id == "efficient_clean"
    assert rankings["heldout_anti_loop"][0].policy_id == "efficient_clean"


def test_pair_prompts_hide_policy_identity_and_oracle_values() -> None:
    validation = group_by_policy([item for item in make_demo_trajectories() if item.context["split"] == "validation"])
    records = build_records(validation, read_json("configs/protocol_v2.json"))
    objective = read_json("configs/objectives_demo.json")[1]
    requests = make_pair_requests(objective, records)
    assert requests
    for row in requests:
        prompt = row["prompt"]
        assert '"policy_id":' not in prompt
        # The demo ID ``reward_loop`` intentionally overlaps with the factual
        # environment event ``reward_loop_step``; other opaque IDs must vanish.
        if row["policy_a_id"] != "reward_loop":
            assert row["policy_a_id"] not in prompt
        if row["policy_b_id"] != "reward_loop":
            assert row["policy_b_id"] not in prompt
        assert "oracle_score" not in prompt
        assert "model_path" not in prompt
        assert "curriculum" not in prompt
