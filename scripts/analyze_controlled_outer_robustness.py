from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.canonical import canonical_json  # noqa: E402
from behavioral_records.controlled_outer_robustness import (  # noqa: E402
    LLM_SELECTOR_ARMS,
    difference_in_differences,
    rerank_frozen_outer_cell,
    summarize_sensitivity_grid,
)
from behavioral_records.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_rows(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    outer_seed: int,
    generation: int,
    selector_arm: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["scenario_id"]) == scenario
        and int(row["outer_seed"]) == outer_seed
        and int(row["generation"]) == generation
        and str(row["selector_arm"]) == selector_arm
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline robustness analysis for the frozen controlled retention run."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "controlled_outer_robustness_v1.json"),
    )
    parser.add_argument("--source-run")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "controlled_outer_robustness_20260901_v1"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    source_run = Path(args.source_run or ROOT / config["source_run"]).resolve()
    frozen_source_run = (ROOT / config["source_run"]).resolve()
    if source_run != frozen_source_run:
        raise ValueError("--source-run must match the path frozen in the robustness plan")
    output_dir = Path(args.output_dir).resolve()
    if output_dir == source_run or source_run in output_dir.parents:
        raise ValueError("Robustness outputs must not be written into the frozen source run")

    source_hash_checks = {
        str(relative): (
            (ROOT / str(relative)).is_file()
            and _sha256(ROOT / str(relative)) == str(expected).lower()
        )
        for relative, expected in config["source_hashes_sha256"].items()
    }
    if not all(source_hash_checks.values()):
        failed = sorted(
            relative for relative, passed in source_hash_checks.items() if not passed
        )
        raise RuntimeError("Frozen robustness input hash mismatch: " + ", ".join(failed))

    source_plan_path = (ROOT / config["source_plan"]).resolve()
    source_plan = read_json(source_plan_path)
    frozen_controls_path = source_run / "frozen_controls.json"
    frozen_selections_path = source_run / "all_selection_results.jsonl"
    frozen_controls = read_json(frozen_controls_path)
    frozen_selections = read_jsonl(frozen_selections_path)
    design = source_plan["design"]
    scenarios = [str(value) for value in design["scenarios"]]
    outer_seeds = [int(value) for value in design["outer_seeds"]]
    generations = int(design["generations"])
    parent_count = int(design["parents"])
    grid = config["ranking_grid"]

    expected_llm_cells = len(scenarios) * len(outer_seeds) * generations * len(
        LLM_SELECTOR_ARMS
    )
    cell_outputs: list[dict[str, Any]] = []
    source_files = [config_path, source_plan_path, frozen_controls_path, frozen_selections_path]
    for generation in range(generations):
        generation_dir = source_run / f"generation_{generation}"
        private_path = generation_dir / "judge_private_map.jsonl"
        response_path = generation_dir / "judge" / "judge_responses.jsonl"
        oracle_path = generation_dir / "heldout_oracle_private.jsonl"
        source_files.extend([private_path, response_path, oracle_path])
        private_rows = read_jsonl(private_path)
        responses = read_jsonl(response_path)
        oracle_rows = read_jsonl(oracle_path)
        for scenario in scenarios:
            for outer_seed in outer_seeds:
                for selector_arm in LLM_SELECTOR_ARMS:
                    cell_private = _cell_rows(
                        private_rows,
                        scenario=scenario,
                        outer_seed=outer_seed,
                        generation=generation,
                        selector_arm=selector_arm,
                    )
                    cell_oracle = _cell_rows(
                        oracle_rows,
                        scenario=scenario,
                        outer_seed=outer_seed,
                        generation=generation,
                        selector_arm=selector_arm,
                    )
                    if not cell_private or not cell_oracle:
                        raise ValueError(
                            "Missing frozen cell artifacts for "
                            f"{scenario}, seed={outer_seed}, generation={generation}, "
                            f"arm={selector_arm}"
                        )
                    for regularization in map(float, grid["regularization"]):
                        for probability_clip in map(float, grid["probability_clip"]):
                            for tie_break_mode in map(
                                str, grid["exact_score_tie_break"]
                            ):
                                result = rerank_frozen_outer_cell(
                                    private_rows=cell_private,
                                    responses=responses,
                                    oracle_rows=cell_oracle,
                                    parent_count=parent_count,
                                    regularization=regularization,
                                    probability_clip=probability_clip,
                                    tie_break_mode=tie_break_mode,
                                )
                                result.pop("ranking")
                                cell_outputs.append(result)

    expected_grid_size = (
        len(grid["regularization"])
        * len(grid["probability_clip"])
        * len(grid["exact_score_tie_break"])
    )
    if len(cell_outputs) != expected_llm_cells * expected_grid_size:
        raise RuntimeError("Finite robustness grid is incomplete")

    summaries = summarize_sensitivity_grid(
        cell_outputs,
        frozen_selection_rows=frozen_selections,
        scenarios=scenarios,
        outer_seeds=outer_seeds,
        endpoint_generation=int(config["difference_in_differences"]["endpoint_generation"]),
        bootstrap_seed=int(config["difference_in_differences"]["bootstrap_seed"]),
    )
    reference_summary = next(
        row
        for row in summaries
        if float(row["regularization"]) == float(grid["reference_regularization"])
        and float(row["probability_clip"]) == float(
            grid["reference_probability_clip"]
        )
        and str(row["tie_break_mode"]) == str(grid["reference_tie_break"])
    )
    if reference_summary["selected_set_exact_match_to_frozen_fraction"] != 1.0:
        raise RuntimeError("Reference re-ranking did not reproduce all frozen LLM selections")

    did_config = config["difference_in_differences"]
    did = difference_in_differences(
        frozen_selections,
        scenarios=scenarios,
        outer_seeds=outer_seeds,
        baseline_generation=int(did_config["baseline_generation"]),
        endpoint_generation=int(did_config["endpoint_generation"]),
        bootstrap_seed=int(did_config["bootstrap_seed"]),
    )

    shared_hash_summaries = [
        row for row in summaries if row["tie_break_mode"] == "shared_seed_hash"
    ]
    shared_endpoint_means = [
        float(row["endpoint_observed_candidate_effect"]["mean_effect"])
        for row in shared_hash_summaries
    ]
    shared_all_seed_positive = [
        all(
            float(value) > 0.0
            for value in row["endpoint_observed_candidate_effect"][
                "outer_seed_cluster_effects"
            ].values()
        )
        for row in shared_hash_summaries
    ]
    legacy_reference = reference_summary
    shared_reference = next(
        row
        for row in summaries
        if float(row["regularization"]) == float(grid["reference_regularization"])
        and float(row["probability_clip"]) == float(
            grid["reference_probability_clip"]
        )
        and row["tie_break_mode"] == "shared_seed_hash"
    )
    summary = {
        "schema_version": "controlled-outer-offline-robustness-result-v1",
        "status": "COMPLETE_POST_HOC_OFFLINE_ANALYSIS",
        "source_run": str(source_run.relative_to(ROOT)),
        "analysis_scope": {
            "existing_frozen_responses_only": True,
            "external_api_calls": 0,
            "frozen_outputs_modified": False,
            "observed_candidate_reranking_only": True,
            "counterfactual_lineage_replay": False,
            "difference_in_differences_uses_executed_frozen_paths": True,
        },
        "data_integrity": {
            "all_frozen_input_hashes_match": all(source_hash_checks.values()),
            "expected_llm_selection_cells": expected_llm_cells,
            "observed_llm_selection_cells_per_grid_point": expected_llm_cells,
            "ranking_grid_point_count": expected_grid_size,
            "cell_result_count": len(cell_outputs),
            "reference_reproduces_all_frozen_llm_parent_sets": True,
            "frozen_design_controls_match_source_plan": (
                int(frozen_controls["generations"]) == generations
                and [int(value) for value in frozen_controls["outer_seeds"]]
                == outer_seeds
                and int(frozen_controls["parents"]) == parent_count
            ),
        },
        "tie_break_impact_at_reference_ranking": {
            "legacy_policy_id": legacy_reference,
            "shared_seed_hash": shared_reference,
            "endpoint_mean_effect_change": (
                float(shared_reference["endpoint_observed_candidate_effect"]["mean_effect"])
                - float(legacy_reference["endpoint_observed_candidate_effect"]["mean_effect"])
            ),
        },
        "ridge_and_clip_sensitivity_with_shared_seed_hash": {
            "grid_point_count": len(shared_hash_summaries),
            "endpoint_mean_effect_range": [
                min(shared_endpoint_means),
                max(shared_endpoint_means),
            ],
            "all_outer_seed_effects_positive_at_every_grid_point": all(
                shared_all_seed_positive
            ),
            "mean_selected_set_jaccard_to_frozen_across_grid": fmean(
                float(row["selected_set_mean_jaccard_to_frozen"])
                for row in shared_hash_summaries
            ),
        },
        "executed_path_generation_difference_in_differences": did,
        "grid_summaries": summaries,
        "claim_boundary": config["claim_boundary"],
    }
    manifest = {
        "schema_version": "controlled-outer-robustness-source-manifest-v1",
        "expected_source_hash_checks": source_hash_checks,
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in source_files
        ],
    }
    write_jsonl(output_dir / "observed_candidate_cell_sensitivity.jsonl", cell_outputs)
    write_json(output_dir / "robustness_summary.json", summary)
    write_json(output_dir / "source_manifest.json", manifest)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
