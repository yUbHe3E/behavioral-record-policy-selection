from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.canonical import canonical_json  # noqa: E402
from behavioral_records.controlled_outer_evaluation import (  # noqa: E402
    evaluate_controlled_outer_confirmatory,
)
from behavioral_records.io import read_json, read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen controlled outer-loop v3 run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plan",
        default=str(ROOT / "configs" / "controlled_outer_confirmatory_v3.json"),
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    result = evaluate_controlled_outer_confirmatory(
        selection_rows=read_jsonl(output / "all_selection_results.jsonl"),
        stage_summary=read_json(output / "smoke_summary.json"),
        frozen_controls=read_json(output / "frozen_controls.json"),
        plan=read_json(args.plan),
    )
    write_json(output / "confirmatory_evaluation.json", result)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
