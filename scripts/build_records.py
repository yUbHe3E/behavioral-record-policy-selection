from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_records.experiment import build_records, group_by_policy  # noqa: E402
from behavioral_records.io import read_json, read_trajectories, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build objective-agnostic records from validation trajectories")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--protocol", default=str(REPO_ROOT / "configs" / "protocol_v2.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trajectories = [
        item for item in read_trajectories(args.trajectories) if item.context.get("split") == "validation"
    ]
    records = build_records(group_by_policy(trajectories), read_json(args.protocol))
    write_jsonl(args.output, records.values())
    print(json.dumps({"status": "ok", "policy_count": len(records), "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
