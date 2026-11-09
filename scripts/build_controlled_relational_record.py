from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.relational_record import (  # noqa: E402
    build_controlled_relational_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen controlled relational-record experiment."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_relational_record_v1.json"),
    )
    args = parser.parse_args()
    result = build_controlled_relational_record(
        output_dir=args.output_dir,
        plan_path=args.plan,
        repository_root=ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "READY_FOR_RELATIONAL_JUDGE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
