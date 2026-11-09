from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact offline-run summary")
    parser.add_argument("--latest", action="store_true", help="Use the newest run under outputs")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = [path.parent for path in (REPO_ROOT / "outputs").glob("*/summary.json")]
        if not candidates:
            raise SystemExit("No completed run found under outputs/")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Run: {run_dir.resolve()}")
    print(f"Status: {summary['status']}")
    print(f"Policies: {summary['policy_count']}")
    print(f"Validation episodes: {summary['validation_episode_count']}")
    print(f"Held-out episodes: {summary['heldout_episode_count']}")
    print(f"Objectives: {summary['objective_count']}")
    print(f"Integrity passed: {summary['integrity_passed']}")
    if "per_policy_char_cap" in summary:
        print(f"Per-policy prompt-view cap: {summary['per_policy_char_cap']} characters")
        print(f"Compact view median: {summary['compact_public_chars']['median']} characters")
        print(f"Matched raw median: {summary['raw_public_chars']['median']} characters")
        for arm, sizes in summary["judge_prompt_chars_by_arm"].items():
            print(f"Prompt median {arm}: {sizes['median']} characters")
    if "tpbr_v0" in summary:
        tpbr = summary["tpbr_v0"]
        print(f"TPBR recoverability passed: {tpbr['recoverability_passed']}")
        print(f"TPBR core fact coverage min: {tpbr['core_fact_coverage_min']}")
        print(f"TPBR extended fact coverage mean: {tpbr['extended_fact_coverage_mean']}")
        print(f"TPBR public characters max: {tpbr['public_characters_max']}")
    for objective_id, ranking in summary["oracle_rankings"].items():
        print(f"Oracle {objective_id}: {' > '.join(ranking)}")
    print(summary["claims"])


if __name__ == "__main__":
    main()
