from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.io import read_json, read_jsonl, write_json  # noqa: E402
from behavioral_records.judge_runner import JudgeConfig, cache_key  # noqa: E402
from behavioral_records.relational_record import (  # noqa: E402
    evaluate_controlled_relational_record,
    validate_controlled_relational_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen relational-record experiment."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--judge-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_relational_record_v1.json"),
    )
    args = parser.parse_args()
    plan = read_json(args.plan)
    bundle = Path(args.bundle_dir)
    validate_controlled_relational_bundle(bundle, plan)
    requests = read_jsonl(bundle / "judge_requests.jsonl")
    private_map = read_jsonl(bundle / "judge_private_map.jsonl")
    responses = read_jsonl(Path(args.judge_dir) / "judge_responses.jsonl")
    response_ids = [str(row.get("request_id", "")) for row in responses]
    if len(response_ids) != len(set(response_ids)) or not all(response_ids):
        raise RuntimeError("Judge responses must have unique non-empty request IDs")
    request_by_id = {str(row["request_id"]): row for row in requests}
    if not set(response_ids) <= set(request_by_id):
        raise RuntimeError("Judge responses contain IDs outside the frozen request bundle")
    judge = plan["judge_freeze"]
    provenance_config = JudgeConfig(
        provider=str(judge["provider"]),
        base_url="https://provenance-check.invalid",
        model=str(judge["model"]),
        api_key="not-used-for-offline-provenance-check",
        max_requests=int(plan["design"]["expected_total_requests"]),
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        max_output_tokens=int(judge["max_output_tokens"]),
        max_prompt_chars=int(plan["offline_gates"]["maximum_prompt_characters"]),
        temperature=float(judge["temperature"]),
        max_concurrency=int(judge["max_concurrency"]),
        response_format_type=str(judge["response_format_type"]),
    )
    for row in responses:
        request_id = str(row["request_id"])
        if row.get("provider") != judge["provider"] or row.get("model") != judge["model"]:
            raise RuntimeError("Judge response provider/model differs from the frozen plan")
        if row.get("cache_key") != cache_key(request_by_id[request_id], provenance_config):
            raise RuntimeError("Judge response cache provenance does not match its request")
    result = evaluate_controlled_relational_record(
        private_map=private_map,
        responses=responses,
        plan=plan,
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "response_count": result["response_count"],
                "primary": result["contrasts"][
                    "primary_relational_link_contribution"
                ],
                "support": result["support"],
                "quality_gates": result["quality_gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
