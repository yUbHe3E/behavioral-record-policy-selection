from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.controlled_ablation import (  # noqa: E402
    evaluate_controlled_representation_ablation,
)
from behavioral_records.io import read_json, read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen representation ablation.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--judge-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_representation_ablation_v1.json"),
    )
    args = parser.parse_args()
    result = evaluate_controlled_representation_ablation(
        private_map=read_jsonl(Path(args.bundle_dir) / "judge_private_map.jsonl"),
        responses=read_jsonl(Path(args.judge_dir) / "judge_responses.jsonl"),
        plan=read_json(args.plan),
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "response_count": result["response_count"],
                "primary_contrasts": result["primary_contrasts"],
                "support": result["support"],
                "quality_gates": result["quality_gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
