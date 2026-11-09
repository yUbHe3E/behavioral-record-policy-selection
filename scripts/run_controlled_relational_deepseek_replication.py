from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.canonical import canonical_json  # noqa: E402
from behavioral_records.deepseek_relational_replication import (  # noqa: E402
    execute_or_skip_deepseek_relational_replication,
    prepare_deepseek_relational_replication,
)
from behavioral_records.io import read_json  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or explicitly execute the frozen DeepSeek replication of "
            "the Relational-Linked-versus-Relational-LinkBroken primary contrast"
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "controlled_relational_deepseek_replication_v1.json"
        ),
    )
    parser.add_argument("--source-bundle")
    parser.add_argument("--source-plan")
    parser.add_argument("--output-dir")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    source_bundle = _repo_path(args.source_bundle or config["source_bundle"])
    source_plan = _repo_path(args.source_plan or config["source_plan"])
    output = _repo_path(args.output_dir or config["default_output_dir"])
    manifest = prepare_deepseek_relational_replication(
        source_bundle=source_bundle,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )

    judge = config["judge_freeze"]
    configured_base_url = os.environ.get("DEEPSEEK_BASE_URL", "")
    if configured_base_url and configured_base_url.rstrip("/") != str(
        judge["base_url"]
    ).rstrip("/"):
        raise RuntimeError("DEEPSEEK_BASE_URL differs from the frozen endpoint")
    configured_model = os.environ.get("DEEPSEEK_MODEL", "")
    if configured_model and configured_model != str(judge["model"]):
        raise RuntimeError("DEEPSEEK_MODEL differs from the frozen model")
    credential_name = str(judge["credential_environment_variable"])
    result = execute_or_skip_deepseek_relational_replication(
        output_dir=output,
        config=config,
        api_key=os.environ.get(credential_name, ""),
        execute=args.execute,
    )
    print(
        canonical_json(
            {
                "bundle_status": manifest["status"],
                "run_status": result["status"],
                "selected_request_count": manifest["selected_request_count"],
                "selected_request_hash": manifest["selected_request_hash"],
                "api_key_recorded": False,
            }
        )
    )


if __name__ == "__main__":
    main()
