from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.canonical import canonical_json, stable_hash  # noqa: E402
from behavioral_records.io import read_json, read_jsonl, write_json  # noqa: E402
from behavioral_records.judge_runner import JudgeConfig, run_judge_requests  # noqa: E402
from behavioral_records.relational_record import (  # noqa: E402
    validate_controlled_relational_bundle,
)
from behavioral_records.semantic_pilot import strict_parse_semantic_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the frozen MiniMax relational-record batch."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_relational_record_v1.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    bundle_summary = validate_controlled_relational_bundle(bundle, plan)
    requests = read_jsonl(bundle / "judge_requests.jsonl")
    expected = int(plan["design"]["expected_total_requests"])
    if len(requests) != expected or stable_hash(requests) != bundle_summary["request_hash"]:
        raise RuntimeError("Relational request count or hash does not match the bundle")
    attempts_path = output / "judge_attempts.jsonl"
    responses_path = output / "judge_responses.jsonl"
    if attempts_path.exists() or responses_path.exists():
        raise RuntimeError(
            "Refusing to resume or overwrite a frozen batch with existing attempts or responses"
        )

    frozen_base_url = str(plan["judge_freeze"]["base_url"])
    configured_base_url = os.environ.get("MINIMAX_BASE_URL", "")
    if configured_base_url and configured_base_url.rstrip("/") != frozen_base_url.rstrip("/"):
        raise RuntimeError("MINIMAX_BASE_URL differs from the authorized frozen endpoint")
    base_url = frozen_base_url
    model = os.environ.get("MINIMAX_MODEL", "")
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    preflight = {
        "status": "READY" if args.execute else "PREPARED_NO_NETWORK_CALL",
        "request_count": len(requests),
        "request_hash": stable_hash(requests),
        "plan_hash": stable_hash(plan),
        "provider": "minimax",
        "model_configured": bool(model),
        "authorized_base_url_frozen": True,
        "base_url_override_configured": bool(configured_base_url),
        "api_key_configured": bool(api_key),
        "api_key_recorded": False,
        "response_format_type": "json_object",
        "max_concurrency": int(plan["judge_freeze"]["max_concurrency"]),
        "automatic_retries": int(plan["judge_freeze"]["automatic_retries"]),
    }
    write_json(output / "judge_preflight.json", preflight)
    if not args.execute:
        print(canonical_json(preflight))
        return
    if not all((model, api_key)):
        result = {
            "status": "SKIPPED",
            "reason": "Missing MiniMax model or API key",
            "api_key_recorded": False,
        }
        write_json(output / "judge_run_summary.json", result)
        print(canonical_json(result))
        return
    if model != str(plan["judge_freeze"]["model"]):
        raise RuntimeError("Configured MiniMax model differs from the frozen plan")

    judge = plan["judge_freeze"]
    config = JudgeConfig(
        provider="minimax",
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_requests=expected,
        max_cost_usd=float(judge["maximum_cost_usd"]),
        input_cost_per_million=float(
            judge["input_accounting_rate_usd_per_million_tokens"]
        ),
        output_cost_per_million=float(
            judge["output_accounting_rate_usd_per_million_tokens"]
        ),
        max_output_tokens=int(judge["max_output_tokens"]),
        max_prompt_chars=int(plan["offline_gates"]["maximum_prompt_characters"]),
        timeout_seconds=float(judge["timeout_seconds"]),
        temperature=float(judge["temperature"]),
        max_concurrency=int(judge["max_concurrency"]),
        response_format_type=str(judge["response_format_type"]),
    )
    summary = run_judge_requests(
        requests,
        output_dir=output,
        config=config,
        response_parser=strict_parse_semantic_response,
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
