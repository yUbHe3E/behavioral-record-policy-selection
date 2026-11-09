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
from behavioral_records.semantic_pilot import strict_parse_semantic_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the frozen MiniMax representation ablation.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_representation_ablation_v1.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bundle = Path(args.bundle_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    requests = read_jsonl(bundle / "judge_requests.jsonl")
    bundle_summary = read_json(bundle / "summary.json")
    expected = int(plan["design"]["expected_total_requests"])
    if bundle_summary["status"] != "READY_FOR_ABLATION_JUDGE" or len(requests) != expected:
        raise RuntimeError("Ablation bundle did not pass its frozen offline gate")
    attempts_path = output / "judge_attempts.jsonl"
    if attempts_path.exists():
        raise RuntimeError(f"Refusing to resume or overwrite an ablation batch: {attempts_path}")

    base_url = os.environ.get("MINIMAX_BASE_URL", "")
    model = os.environ.get("MINIMAX_MODEL", "")
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    preflight = {
        "status": "READY" if args.execute else "PREPARED_NO_NETWORK_CALL",
        "request_count": len(requests),
        "request_hash": stable_hash(requests),
        "expected_request_hash": bundle_summary["request_hash"],
        "plan_hash": stable_hash(plan),
        "provider": "minimax",
        "model_configured": bool(model),
        "base_url_configured": bool(base_url),
        "api_key_configured": bool(api_key),
        "api_key_recorded": False,
        "response_format_type": "json_object",
        "max_concurrency": 1,
        "automatic_retries": 0,
    }
    write_json(output / "judge_preflight.json", preflight)
    if not args.execute:
        print(canonical_json(preflight))
        return
    if not all((base_url, model, api_key)):
        result = {
            "status": "SKIPPED",
            "reason": "Missing MiniMax base URL, model, or API key",
            "api_key_recorded": False,
        }
        write_json(output / "judge_run_summary.json", result)
        print(canonical_json(result))
        return
    judge = plan["judge_freeze"]
    config = JudgeConfig(
        provider="minimax",
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_requests=expected,
        max_cost_usd=float(judge["maximum_cost_usd"]),
        input_cost_per_million=0.3,
        output_cost_per_million=1.2,
        max_output_tokens=int(judge["max_output_tokens"]),
        max_prompt_chars=70_000,
        timeout_seconds=120.0,
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
