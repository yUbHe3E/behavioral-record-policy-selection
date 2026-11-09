from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from behavioral_records.controlled_confirmatory import build_controlled_confirmatory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen C1/C2 static confirmatory bundle.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "controlled_scenarios_confirmatory_v1.json"),
    )
    args = parser.parse_args()
    result = build_controlled_confirmatory(
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "READY_FOR_STATIC_JUDGE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
