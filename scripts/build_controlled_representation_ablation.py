from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.controlled_ablation import (  # noqa: E402
    build_controlled_representation_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen controlled representation ablation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_representation_ablation_v1.json"),
    )
    args = parser.parse_args()
    result = build_controlled_representation_ablation(
        output_dir=args.output_dir,
        plan_path=args.plan,
        repository_root=ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "READY_FOR_ABLATION_JUDGE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
