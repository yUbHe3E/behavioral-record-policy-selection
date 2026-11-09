from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.deepseek_relational_replication import (  # noqa: E402
    evaluate_deepseek_relational_replication,
    prepare_deepseek_relational_replication,
    validate_deepseek_relational_run_preflight,
)
from behavioral_records.io import read_json, read_jsonl, write_json  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen DeepSeek relational replication offline"
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "controlled_relational_deepseek_replication_v1.json"
        ),
    )
    parser.add_argument("--bundle-dir")
    parser.add_argument("--responses")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = read_json(args.config)
    bundle = _repo_path(args.bundle_dir or config["default_output_dir"])
    source_bundle = _repo_path(config["source_bundle"])
    source_plan = _repo_path(config["source_plan"])
    manifest = prepare_deepseek_relational_replication(
        source_bundle=source_bundle,
        source_plan_path=source_plan,
        output_dir=bundle,
        config=config,
    )
    preflight = read_json(bundle / "run" / "judge_preflight.json")
    validate_deepseek_relational_run_preflight(
        preflight=preflight,
        manifest=manifest,
        config=config,
    )
    responses_path = (
        _repo_path(args.responses)
        if args.responses
        else bundle / "run" / "judge_responses.jsonl"
    )
    output_path = (
        _repo_path(args.output)
        if args.output
        else bundle / "run" / "evaluation.json"
    )
    result = evaluate_deepseek_relational_replication(
        public_requests=read_jsonl(bundle / "bundle" / "judge_requests.jsonl"),
        private_map=read_jsonl(bundle / "private" / "evaluation_map.jsonl"),
        responses=read_jsonl(responses_path),
        config=config,
    )
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "response_count": result["response_count"],
                "primary": result["primary"],
                "support": result["support"],
                "quality_gates": result["quality_gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
