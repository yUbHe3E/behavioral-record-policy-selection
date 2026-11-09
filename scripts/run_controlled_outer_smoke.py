from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behavioral_records.canonical import canonical_json, stable_hash  # noqa: E402
from behavioral_records.controlled_outer import (  # noqa: E402
    BASELINE_ARMS,
    EVIDENCE_ARM,
    LLM_ARMS,
    OUTER_ARMS,
    SCENARIOS,
    build_llm_requests,
    compile_population,
    initial_population,
    mutate_population,
    select_baseline_parents,
    select_llm_parents,
    summarize_generation_selection,
)
from behavioral_records.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from behavioral_records.judge_runner import JudgeConfig, run_judge_requests  # noqa: E402
from behavioral_records.semantic_pilot import strict_parse_semantic_response  # noqa: E402


POPULATION = 12
PARENTS = 3
GENERATIONS = 3
OUTER_SEEDS = (0, 1)


def _key(scenario: str, outer_seed: int, arm: str) -> str:
    return f"{scenario}__s{outer_seed}__{arm}"


def _genotype_hash(policy: Mapping[str, Any]) -> str:
    excluded = {
        "policy_id",
        "slot",
        "generation",
        "parent_policy_id",
        "lineage",
    }
    return stable_hash({key: value for key, value in policy.items() if key not in excluded})


def _public_compilation_audit(
    requests: list[dict[str, Any]],
    compiled_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    forbidden = (
        "policy_id",
        "parent_policy_id",
        "lineage",
        '"mode"',
        '"risk_mode"',
        '"collateral_mode"',
        "semantic_violation_rate",
        "semantic_pass_rate",
        "hidden_oracle",
    )
    prompts = [str(row["prompt"]) for row in requests]
    objective_blind = all(
        term not in prompt.lower() for prompt in prompts for term in forbidden
    )
    evidence_hashes = []
    for compiled in compiled_by_key.values():
        for policy_evidence in compiled["evidence"].values():
            for arm in ("B1", "B2-FullTrace"):
                evidence_hashes.append(stable_hash(policy_evidence[arm]))
    return {
        "request_count": len(requests),
        "maximum_prompt_characters": max(map(len, prompts), default=0),
        "prompt_budget_passed": max(map(len, prompts), default=0) <= 70_000,
        "hidden_field_leakage_passed": objective_blind,
        "evidence_record_count": len(evidence_hashes),
        "unique_evidence_hash_count": len(set(evidence_hashes)),
        "passed": objective_blind and max(map(len, prompts), default=0) <= 70_000,
    }


def _prepare_generation(
    *,
    generation: int,
    populations: Mapping[str, list[dict[str, Any]]],
    requirements: Mapping[str, str],
    generation_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    requests: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    compiled_by_key: dict[str, dict[str, Any]] = {}
    population_rows: list[dict[str, Any]] = []
    policy_evaluation_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for outer_seed in OUTER_SEEDS:
            for arm in OUTER_ARMS:
                group_key = _key(scenario, outer_seed, arm)
                population = populations[group_key]
                compiled = compile_population(
                    scenario,
                    population,
                    outer_seed=outer_seed,
                    generation=generation,
                )
                compiled_by_key[group_key] = compiled
                for policy in population:
                    policy_id = str(policy["policy_id"])
                    population_rows.append(
                        {
                            "scenario_id": scenario,
                            "outer_seed": outer_seed,
                            "selector_arm": arm,
                            **dict(policy),
                            "genotype_hash": _genotype_hash(policy),
                        }
                    )
                    policy_evaluation_rows.append(
                        {
                            "scenario_id": scenario,
                            "outer_seed": outer_seed,
                            "selector_arm": arm,
                            "generation": generation,
                            "policy_id": policy_id,
                            "ordinary_score": compiled["ordinary_scores"][policy_id],
                            **dict(compiled["hidden_oracle"][policy_id]),
                        }
                    )
                if arm in LLM_ARMS:
                    group_requests, group_private = build_llm_requests(
                        scenario=scenario,
                        outer_seed=outer_seed,
                        generation=generation,
                        arm=arm,
                        population=population,
                        evidence=compiled["evidence"],
                        requirement=requirements[scenario],
                    )
                    requests.extend(group_requests)
                    private.extend(group_private)
    audit = _public_compilation_audit(requests, compiled_by_key)
    write_jsonl(generation_dir / "judge_requests.jsonl", requests)
    write_jsonl(generation_dir / "judge_private_map.jsonl", private)
    write_jsonl(generation_dir / "population_private.jsonl", population_rows)
    write_jsonl(generation_dir / "heldout_oracle_private.jsonl", policy_evaluation_rows)
    write_json(generation_dir / "public_compilation_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Generation {generation} public compilation audit failed")
    return requests, private, compiled_by_key


def _select_generation(
    *,
    generation: int,
    populations: Mapping[str, list[dict[str, Any]]],
    compiled_by_key: Mapping[str, Mapping[str, Any]],
    private_rows: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selection_rows: list[dict[str, Any]] = []
    next_populations: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        for outer_seed in OUTER_SEEDS:
            for arm in OUTER_ARMS:
                group_key = _key(scenario, outer_seed, arm)
                population = populations[group_key]
                compiled = compiled_by_key[group_key]
                if arm in LLM_ARMS:
                    group_private = [
                        row
                        for row in private_rows
                        if row["scenario_id"] == scenario
                        and int(row["outer_seed"]) == outer_seed
                        and row["selector_arm"] == arm
                    ]
                    selection = select_llm_parents(
                        population=population,
                        private_rows=group_private,
                        responses=responses,
                        parent_count=PARENTS,
                    )
                else:
                    selection = select_baseline_parents(
                        arm=arm,
                        population=population,
                        compiled=compiled,
                        scenario=scenario,
                        outer_seed=outer_seed,
                        generation=generation,
                        parent_count=PARENTS,
                    )
                summary = summarize_generation_selection(
                    selection=selection,
                    compiled=compiled,
                )
                selected_rows = {
                    str(row["policy_id"]): row for row in population
                }
                selected_genotype_hashes = [
                    _genotype_hash(selected_rows[policy_id])
                    for policy_id in summary["selected_policy_ids"]
                ]
                selection_rows.append(
                    {
                        "scenario_id": scenario,
                        "outer_seed": outer_seed,
                        "generation": generation,
                        "selector_arm": arm,
                        **summary,
                        "selected_genotype_hashes": selected_genotype_hashes,
                    }
                )
                if generation + 1 < GENERATIONS:
                    next_populations[group_key] = mutate_population(
                        scenario=scenario,
                        population=population,
                        selected_policy_ids=summary["selected_policy_ids"],
                        outer_seed=outer_seed,
                        next_generation=generation + 1,
                    )
    return selection_rows, next_populations


def _summarize_smoke(
    *,
    selection_rows: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    initial_non_degenerate: bool,
) -> dict[str, Any]:
    final = [row for row in selection_rows if int(row["generation"]) == GENERATIONS - 1]
    by_scenario_arm: dict[str, Any] = {}
    for scenario in SCENARIOS:
        for arm in OUTER_ARMS:
            rows = [
                row
                for row in final
                if row["scenario_id"] == scenario and row["selector_arm"] == arm
            ]
            by_scenario_arm[f"{scenario}::{arm}"] = {
                "outer_seed_count": len(rows),
                "mean_final_selected_semantic_violation_rate": fmean(
                    row["selected_parent_mean_semantic_violation_rate"] for row in rows
                ),
                "seed_values": [
                    row["selected_parent_mean_semantic_violation_rate"] for row in rows
                ],
                "mean_final_selected_heldout_success_rate": fmean(
                    row["selected_parent_mean_heldout_success_rate"] for row in rows
                ),
            }
    paired = {}
    for scenario in SCENARIOS:
        differences = []
        for outer_seed in OUTER_SEEDS:
            indexed = {
                row["selector_arm"]: row
                for row in final
                if row["scenario_id"] == scenario
                and int(row["outer_seed"]) == outer_seed
            }
            differences.append(
                indexed["B1-LLM-PBT"]["selected_parent_mean_semantic_violation_rate"]
                - indexed["B2-FullTrace-LLM-PBT"][
                    "selected_parent_mean_semantic_violation_rate"
                ]
            )
        paired[scenario] = {
            "B1_minus_B2_semantic_violation_improvement": fmean(differences),
            "outer_seed_differences": differences,
            "inference_status": "SMOKE_ONLY_TWO_SEEDS",
        }

    llm_rows = [row for row in selection_rows if row["selector_arm"] in LLM_ARMS]
    valid_parent_selection = all(
        len(set(row["selected_policy_ids"])) == PARENTS for row in selection_rows
    )
    connected = all(bool(row["scheduled_graph_connected"]) for row in llm_rows)
    minimum_valid = min((float(row["valid_request_rate"]) for row in llm_rows), default=0.0)
    no_provider_stop = all(row.get("stopped_reason") is None for row in run_summaries)
    complete_generations = len(run_summaries) == GENERATIONS
    final_index = {
        (row["scenario_id"], int(row["outer_seed"]), row["selector_arm"]): row
        for row in final
    }
    lineage_divergence = any(
        sorted(
            final_index[(scenario, outer_seed, "B1-LLM-PBT")][
                "selected_genotype_hashes"
            ]
        )
        != sorted(
            final_index[(scenario, outer_seed, "B2-FullTrace-LLM-PBT")][
                "selected_genotype_hashes"
            ]
        )
        for scenario in SCENARIOS
        for outer_seed in OUTER_SEEDS
    )
    gates = {
        "complete_requested_generations": complete_generations,
        "scheduled_selection_graph_connected": connected,
        "minimum_llm_valid_request_rate_at_least_0_8": minimum_valid >= 0.8,
        "valid_three_parent_selection": valid_parent_selection,
        "lineage_or_genotype_divergence": lineage_divergence,
        "non_degenerate_heldout_semantic_outcomes": initial_non_degenerate,
        "no_provider_budget_or_service_stop": no_provider_stop,
    }
    if GENERATIONS == 3 and len(OUTER_SEEDS) == 2:
        status = "READY_FOR_EIGHT_GENERATION_PILOT"
        stage = "smoke"
    elif GENERATIONS == 8 and len(OUTER_SEEDS) == 5:
        status = "READY_FOR_CONFIRMATORY_POWERING"
        stage = "pilot"
    elif GENERATIONS == 3 and len(OUTER_SEEDS) == 8:
        status = "READY_FOR_FROZEN_CONFIRMATORY_EVALUATION"
        stage = "confirmatory"
    else:
        status = "OUTER_STAGE_COMPLETE"
        stage = "custom"
    return {
        "schema_version": "controlled-outer-stage-summary-v1",
        "stage": stage,
        "status": status if all(gates.values()) else "OUTER_STAGE_GATE_FAILED",
        "population": POPULATION,
        "parents": PARENTS,
        "generations": GENERATIONS,
        "outer_seeds": list(OUTER_SEEDS),
        "by_scenario_arm": by_scenario_arm,
        "paired_primary_diagnostic": paired,
        "minimum_llm_valid_request_rate": minimum_valid,
        "estimated_total_cost_usd": sum(
            float(row.get("estimated_total_cost_usd", 0.0)) for row in run_summaries
        ),
        "gates": gates,
        "claim_boundary": (
            f"Operational controlled {stage} with {len(OUTER_SEEDS)} outer seeds. "
            "Only a separately frozen, adequately powered confirmatory stage supports a final claim."
        ),
    }


def main() -> None:
    global POPULATION, PARENTS, GENERATIONS, OUTER_SEEDS
    parser = argparse.ArgumentParser(description="Run the frozen controlled three-generation smoke.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "controlled_scenarios_confirmatory_v1.json"))
    parser.add_argument("--provider", choices=("minimax", "deepseek"), default="minimax")
    parser.add_argument("--max-cost-usd-per-generation", type=float, default=10.0)
    parser.add_argument("--input-cost-per-million", type=float, default=0.3)
    parser.add_argument("--output-cost-per-million", type=float, default=1.2)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--parents", type=int, default=PARENTS)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--outer-seeds", default=",".join(map(str, OUTER_SEEDS)))
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--response-format",
        choices=("none", "json_object"),
        default="none",
        help="Optional strict wire-level response format; parsing remains strict either way.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    POPULATION = args.population
    PARENTS = args.parents
    GENERATIONS = args.generations
    OUTER_SEEDS = tuple(int(value.strip()) for value in args.outer_seeds.split(",") if value.strip())
    if not OUTER_SEEDS:
        raise ValueError("At least one outer seed is required")
    if not 1 < PARENTS < POPULATION:
        raise ValueError("parents must be between one and population")
    if GENERATIONS <= 0:
        raise ValueError("generations must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    requirements = {
        scenario: str(config["scenarios"][scenario]["requirement_paraphrases"][0])
        for scenario in SCENARIOS
    }
    controls = {
        "schema_version": "controlled-outer-stage-controls-v1",
        "population": POPULATION,
        "parents": PARENTS,
        "generations": GENERATIONS,
        "outer_seeds": list(OUTER_SEEDS),
        "arms": list(OUTER_ARMS),
        "llm_orders": ["AB", "BA"],
        "edge_offsets": [1, 3],
        "abstain_or_invalid_pair_contribution": "neutral_probability_0.5",
        "compiler_accepts_objective": False,
        "heldout_oracle_enters_selection": False,
        "max_concurrency": args.max_concurrency,
        "response_format_type": (
            None if args.response_format == "none" else args.response_format
        ),
        "config_hash": stable_hash(config),
    }
    write_json(output / "frozen_controls.json", controls)

    populations: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        for outer_seed in OUTER_SEEDS:
            common = initial_population(scenario, outer_seed, POPULATION)
            for arm in OUTER_ARMS:
                populations[_key(scenario, outer_seed, arm)] = [dict(row) for row in common]

    prefix = args.provider.upper()
    base_url = os.environ.get(f"{prefix}_BASE_URL", "")
    model = os.environ.get(f"{prefix}_MODEL", "")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    preflight = {
        "status": "READY" if args.execute else "PREPARED_NO_NETWORK_CALL",
        "provider": args.provider,
        "base_url_configured": bool(base_url),
        "model_configured": bool(model),
        "api_key_configured": bool(api_key),
        "api_key_recorded": False,
        "controls_hash": stable_hash(controls),
    }
    write_json(output / "preflight.json", preflight)
    if args.execute and not all((base_url, model, api_key)):
        result = {
            "status": "SKIPPED",
            "reason": "Missing provider base URL, model, or API key",
            "api_key_recorded": False,
        }
        write_json(output / "smoke_summary.json", result)
        print(canonical_json(result))
        return

    all_selection_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    initial_non_degenerate = True
    for generation in range(GENERATIONS):
        generation_dir = output / f"generation_{generation}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        requests, private, compiled_by_key = _prepare_generation(
            generation=generation,
            populations=populations,
            requirements=requirements,
            generation_dir=generation_dir,
        )
        if generation == 0:
            initial_non_degenerate = all(
                len(
                    {
                        row["semantic_violation_rate"]
                        for row in compiled["hidden_oracle"].values()
                    }
                )
                > 1
                for compiled in compiled_by_key.values()
            )
        if not args.execute:
            result = {
                "status": "PREPARED_GENERATION_0_NO_NETWORK_CALL",
                "generation_0_request_count": len(requests),
                "expected_total_request_count": len(requests) * GENERATIONS,
                "public_audit": read_json(generation_dir / "public_compilation_audit.json"),
            }
            write_json(output / "smoke_summary.json", result)
            print(canonical_json(result))
            return
        attempts_path = generation_dir / "judge" / "judge_attempts.jsonl"
        if attempts_path.exists():
            raise RuntimeError(
                f"Refusing to resume an incomplete formal generation automatically: {attempts_path}"
            )
        judge_config = JudgeConfig(
            provider=args.provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_requests=len(requests),
            max_cost_usd=args.max_cost_usd_per_generation,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            max_output_tokens=args.max_output_tokens,
            max_prompt_chars=70_000,
            timeout_seconds=120.0,
            temperature=args.temperature,
            max_concurrency=args.max_concurrency,
            response_format_type=(
                None if args.response_format == "none" else args.response_format
            ),
        )
        run_summary = run_judge_requests(
            requests,
            output_dir=generation_dir / "judge",
            config=judge_config,
            response_parser=strict_parse_semantic_response,
        )
        run_summaries.append(dict(run_summary))
        responses = read_jsonl(generation_dir / "judge" / "judge_responses.jsonl")
        selection_rows, next_populations = _select_generation(
            generation=generation,
            populations=populations,
            compiled_by_key=compiled_by_key,
            private_rows=private,
            responses=responses,
        )
        all_selection_rows.extend(selection_rows)
        write_jsonl(generation_dir / "selection_results.jsonl", selection_rows)
        generation_gate = {
            "generation": generation,
            "run_summary": run_summary,
            "minimum_llm_valid_request_rate": min(
                row["valid_request_rate"]
                for row in selection_rows
                if row["selector_arm"] in LLM_ARMS
            ),
            "valid_parent_selection": all(
                len(set(row["selected_policy_ids"])) == PARENTS for row in selection_rows
            ),
            "scheduled_graph_connected": all(
                row["scheduled_graph_connected"]
                for row in selection_rows
                if row["selector_arm"] in LLM_ARMS
            ),
        }
        write_json(generation_dir / "generation_gate.json", generation_gate)
        if (
            run_summary.get("stopped_reason") is not None
            or generation_gate["minimum_llm_valid_request_rate"] < 0.8
            or not generation_gate["valid_parent_selection"]
            or not generation_gate["scheduled_graph_connected"]
        ):
            result = {
                "status": "GENERATION_GATE_FAILED",
                "failed_generation": generation,
                "generation_gate": generation_gate,
                "claim_boundary": "Outer smoke stopped before later generations.",
            }
            write_json(output / "smoke_summary.json", result)
            print(canonical_json(result))
            return
        populations = next_populations

    write_jsonl(output / "all_selection_results.jsonl", all_selection_rows)
    summary = _summarize_smoke(
        selection_rows=all_selection_rows,
        run_summaries=run_summaries,
        initial_non_degenerate=initial_non_degenerate,
    )
    write_json(output / "smoke_summary.json", summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
