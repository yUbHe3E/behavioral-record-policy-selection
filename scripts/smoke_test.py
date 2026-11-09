from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_records.demo import make_demo_trajectories  # noqa: E402
from behavioral_records.experiment import run_offline_experiment  # noqa: E402
from behavioral_records.io import write_trajectories  # noqa: E402
from behavioral_records.predictive_experiment import run_predictive_offline_audit  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="behavior_record_smoke_") as temp_dir:
        root = Path(temp_dir)
        trajectories = root / "trajectories.jsonl"
        write_trajectories(trajectories, make_demo_trajectories())
        summary = run_offline_experiment(
            trajectories,
            REPO_ROOT / "configs" / "protocol_v2.json",
            REPO_ROOT / "configs" / "objectives_demo.json",
            root / "result",
        )
        assert summary["status"] == "COMPLETE_OFFLINE_ONLY"
        assert summary["integrity_passed"] is True
        assert summary["policy_count"] == 4
        assert summary["oracle_rankings"]["heldout_robust_conditional"][0] == "efficient_clean"
        tpbr = run_predictive_offline_audit(
            trajectories,
            REPO_ROOT / "configs" / "tpbr_v0.json",
            root / "tpbr_v0",
        )
        assert tpbr["recoverability_passed"] is True
        assert tpbr["core_fact_coverage_min"] == 1.0
        print(
            "SMOKE_OK: offline records, TPBR, hidden oracle, prompts, and integrity checks completed"
        )


if __name__ == "__main__":
    main()
