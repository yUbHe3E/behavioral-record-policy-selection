from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_records.demo import make_demo_trajectories  # noqa: E402
from behavioral_records.experiment import run_offline_experiment  # noqa: E402
from behavioral_records.io import write_json, write_trajectories  # noqa: E402
from behavioral_records.predictive_experiment import run_predictive_offline_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline behavioral-record experiment")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.profile}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    episodes_per_context = 5 if args.profile == "smoke" else 25
    trajectory_path = run_dir / "demo_trajectories.jsonl"
    write_trajectories(trajectory_path, make_demo_trajectories(episodes_per_context))
    summary = run_offline_experiment(
        trajectory_path,
        REPO_ROOT / "configs" / "protocol_v2.json",
        REPO_ROOT / "configs" / "objectives_demo.json",
        run_dir,
    )
    tpbr_summary = run_predictive_offline_audit(
        trajectory_path,
        REPO_ROOT / "configs" / "tpbr_v0.json",
        run_dir / "tpbr_v0",
    )
    summary["tpbr_v0"] = tpbr_summary
    summary["integrity_passed"] = bool(
        summary["integrity_passed"] and tpbr_summary["recoverability_passed"]
    )
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary | {"run_dir": str(run_dir.resolve())}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
