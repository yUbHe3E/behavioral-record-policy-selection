from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from behavioral_records.controlled_confirmatory import evaluate_controlled_static_responses  # noqa: E402
from behavioral_records.io import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controlled C1/C2 static judge responses.")
    parser.add_argument("--private-map", required=True)
    parser.add_argument("--responses", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-contexts-per-pair-arm", type=int, default=12)
    args = parser.parse_args()
    result = evaluate_controlled_static_responses(
        private_map=read_jsonl(args.private_map),
        responses=[row for path in args.responses for row in read_jsonl(path)],
        expected_contexts_per_pair_arm=args.expected_contexts_per_pair_arm,
    )
    write_json(args.output, result)
    print(json.dumps({
        "response_count": result["response_count"],
        "expected_request_count": result["expected_request_count"],
        "by_scenario": result["by_scenario"],
        "combined_B2_minus_B1": result["combined_B2_minus_B1"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
