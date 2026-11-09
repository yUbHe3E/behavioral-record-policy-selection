from __future__ import annotations

from pathlib import Path
import copy

import pytest

from behavioral_records.canonical import stable_hash
from behavioral_records.deepseek_relational_replication import (
    evaluate_deepseek_relational_replication,
    execute_or_skip_deepseek_relational_replication,
    file_sha256,
    prepare_deepseek_relational_replication,
    validate_deepseek_relational_run_preflight,
)
from behavioral_records.io import read_jsonl, write_json, write_jsonl
from behavioral_records.judge_runner import JudgeConfig, cache_key
import behavioral_records.judge_runner as judge_runner_module
import behavioral_records.canonical as canonical_module
import behavioral_records.controlled_confirmatory as controlled_module
import behavioral_records.deepseek_relational_replication as replication_module
import behavioral_records.io as io_module
import behavioral_records.semantic_pilot as semantic_module


def _make_source_and_config(
    tmp_path: Path, *, pair_count_per_scenario: int = 6
) -> tuple[Path, Path, dict]:
    source = tmp_path / "relational_source"
    source_plan_path = tmp_path / "relational_plan.json"
    source_plan = {
        "schema_version": "controlled-relational-record-plan-v1.2",
        "version_label": "controlled_relational_record_v1_2",
        "design": {"expected_total_requests": 64 * pair_count_per_scenario},
        "claim_boundary": "synthetic source",
    }
    write_json(source_plan_path, source_plan)

    requests = []
    private = []
    source_arms = (
        "Aggregate-Control",
        "Relational-LinkBroken",
        "Relational-Linked",
        "FullTrace-Control",
    )
    for scenario in ("C1", "C2"):
        for pair_index in range(pair_count_per_scenario):
            pair_id = f"{scenario.lower()}_pair_{pair_index:02d}"
            for arm in source_arms:
                for paraphrase in ("P0", "P1"):
                    for order in ("AB", "BA"):
                        for repeat in (0, 1):
                            request_id = (
                                f"rel_test__{scenario}__{pair_id}__{arm}__"
                                f"{paraphrase}__{order}__R{repeat}"
                            )
                            prompt = (
                                "Select the policy from anonymous relational evidence.\n"
                                f"Scenario {scenario}; representation {arm}; wording "
                                f"{paraphrase}; order {order}.\n"
                                "POLICY A EVIDENCE: anonymous factual sequence A\n"
                                "POLICY B EVIDENCE: anonymous factual sequence B"
                            )
                            requests.append(
                                {
                                    "request_id": request_id,
                                    "scenario_id": scenario,
                                    "concept_id": scenario,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase,
                                    "order": order,
                                    "repeat": repeat,
                                    "response_schema": "semantic-selection-response-v1",
                                    "prompt": prompt,
                                    "prompt_hash": stable_hash(prompt),
                                }
                            )
                            private.append(
                                {
                                    "request_id": request_id,
                                    "scenario_id": scenario,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase,
                                    "order": order,
                                    "repeat": repeat,
                                    "alias_a_policy_id": "private_a",
                                    "alias_b_policy_id": "private_b",
                                    "oracle_status": "DETERMINATE",
                                    "expected_winner": "A" if order == "AB" else "B",
                                }
                            )
    public_evidence = [{"representation": "synthetic-public-evidence"}]
    integrity = {
        "passed": True,
        "gates": {"synthetic_integrity": True},
    }
    write_jsonl(source / "judge_requests.jsonl", requests)
    write_jsonl(source / "judge_private_map.jsonl", private)
    write_jsonl(source / "public_evidence.jsonl", public_evidence)
    write_json(source / "relational_integrity_audit.json", integrity)
    summary = {
        "status": "READY_FOR_RELATIONAL_JUDGE",
        "request_count": len(requests),
        "pair_count": 2 * pair_count_per_scenario,
        "request_hash": stable_hash(requests),
        "private_map_hash": stable_hash(private),
        "public_evidence_hash": stable_hash(public_evidence),
        "integrity_audit_hash": stable_hash(integrity),
        "plan_hash": stable_hash(source_plan),
        "gates": {"synthetic_gate": True},
        "material_passport": {
            "version_label": "controlled_relational_record_v1_2"
        },
    }
    write_json(source / "summary.json", summary)

    selected = [
        row
        for row in requests
        if row["arm"] in {"Relational-LinkBroken", "Relational-Linked"}
        and row["repeat"] == 0
    ]
    selected_ids = {row["request_id"] for row in selected}
    selected_private = [row for row in private if row["request_id"] in selected_ids]
    config = {
        "version_label": "synthetic_relational_deepseek_v1",
        "source_bundle": "synthetic-source",
        "source_plan": "synthetic-plan",
        "source_plan_file_sha256": file_sha256(source_plan_path),
        "source_hashes_sha256": {
            name: file_sha256(source / name)
            for name in (
                "judge_requests.jsonl",
                "judge_private_map.jsonl",
                "public_evidence.jsonl",
                "relational_integrity_audit.json",
                "summary.json",
            )
        },
        "source_bundle_checks": {
            "required_status": "READY_FOR_RELATIONAL_JUDGE",
            "required_version_label": "controlled_relational_record_v1_2",
            "expected_source_plan_stable_hash": stable_hash(source_plan),
            "expected_total_request_count": len(requests),
            "expected_total_pair_count": 2 * pair_count_per_scenario,
            "expected_source_request_stable_hash": stable_hash(requests),
            "expected_source_private_map_stable_hash": stable_hash(private),
            "expected_source_public_evidence_stable_hash": stable_hash(
                public_evidence
            ),
            "expected_source_integrity_audit_stable_hash": stable_hash(integrity),
        },
        "analysis_code_freeze": {
            "purpose": "synthetic code freeze",
            "files_sha256": {
                "src/behavioral_records/deepseek_relational_replication.py": file_sha256(
                    replication_module.__file__
                ),
                "src/behavioral_records/controlled_confirmatory.py": file_sha256(
                    controlled_module.__file__
                ),
                "src/behavioral_records/semantic_pilot.py": file_sha256(
                    semantic_module.__file__
                ),
                "src/behavioral_records/canonical.py": file_sha256(
                    canonical_module.__file__
                ),
                "src/behavioral_records/io.py": file_sha256(io_module.__file__),
            },
        },
        "selection": {
            "scenarios": ["C1", "C2"],
            "pair_count_per_scenario": pair_count_per_scenario,
            "total_unique_pairs": 2 * pair_count_per_scenario,
            "arms": ["Relational-LinkBroken", "Relational-Linked"],
            "paraphrases": ["P0", "P1"],
            "orders": ["AB", "BA"],
            "repeats": [0],
            "contexts_per_pair_arm": 4,
            "expected_total_requests": len(selected),
            "expected_selected_request_stable_hash": stable_hash(selected),
            "expected_selected_private_map_stable_hash": stable_hash(
                selected_private
            ),
        },
        "judge_freeze": {
            "provider": "deepseek",
            "model": "deepseek-test",
            "base_url": "https://judge.invalid",
            "runner_file": "src/behavioral_records/judge_runner.py",
            "runner_file_sha256": file_sha256(judge_runner_module.__file__),
            "fixed_system_instruction": (
                "Use only the supplied evidence. Return exactly one JSON object and "
                "no other final-answer text."
            ),
            "temperature": 0.0,
            "max_output_tokens": 128,
            "maximum_prompt_characters": 2000,
            "response_format_type": "json_object",
            "timeout_seconds": 10.0,
            "max_concurrency": 1,
            "input_accounting_rate_usd_per_million_tokens": 1.0,
            "output_accounting_rate_usd_per_million_tokens": 1.0,
            "maximum_accounted_cost_usd": 1.0,
            "stop_on_failure_error_substrings": ["Judge HTTP 429"],
        },
        "primary_analysis": {
            "alpha": 0.05,
            "bootstrap_seed": 101,
            "randomization_seed": 102,
            "missingness_worst_case_bootstrap_seed": 103,
            "missingness_worst_case_randomization_seed": 104,
            "expected_determinate_pairs_by_scenario": {
                "C1": pair_count_per_scenario,
                "C2": pair_count_per_scenario,
            },
            "expected_pooled_determinate_pair_count": 2
            * pair_count_per_scenario,
            "minimum_valid_response_rate_per_scenario_arm_all_requested_pairs": 0.8,
            "minimum_valid_response_rate_per_scenario_arm_determinate_primary_pairs": 0.8,
        },
        "claim_boundary": "synthetic test only",
    }
    return source, source_plan_path, config


def _provenance_config(config: dict) -> JudgeConfig:
    judge = config["judge_freeze"]
    return JudgeConfig(
        provider=judge["provider"],
        base_url=judge["base_url"],
        model=judge["model"],
        api_key="unused",
        max_requests=config["selection"]["expected_total_requests"],
        max_cost_usd=judge["maximum_accounted_cost_usd"],
        input_cost_per_million=judge[
            "input_accounting_rate_usd_per_million_tokens"
        ],
        output_cost_per_million=judge[
            "output_accounting_rate_usd_per_million_tokens"
        ],
        max_output_tokens=judge["max_output_tokens"],
        max_prompt_chars=judge["maximum_prompt_characters"],
        timeout_seconds=judge["timeout_seconds"],
        temperature=judge["temperature"],
        max_concurrency=judge["max_concurrency"],
        response_format_type=judge["response_format_type"],
    )


def _synthetic_responses(
    public: list[dict],
    private: list[dict],
    config: dict,
    *,
    broken_correct: bool = False,
):
    public_by_id = {row["request_id"]: row for row in public}
    provenance = _provenance_config(config)
    rows = []
    for private_row in private:
        request_id = private_row["request_id"]
        winner = (
            private_row["expected_winner"]
            if private_row["arm"] == "Relational-Linked" or broken_correct
            else "ABSTAIN"
        )
        rows.append(
            {
                "request_id": request_id,
                "provider": config["judge_freeze"]["provider"],
                "model": config["judge_freeze"]["model"],
                "cache_key": cache_key(public_by_id[request_id], provenance),
                "response": {
                    "winner": winner,
                    "confidence": 0.8,
                    "evidence_refs": ["event"],
                    "reason": "synthetic response",
                },
            }
        )
    return rows


def test_builder_selects_all_pairs_with_public_rule_and_separates_private_map(
    tmp_path: Path,
) -> None:
    source, source_plan, config = _make_source_and_config(tmp_path)
    output = tmp_path / "replication"
    manifest = prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    assert manifest["status"] == "READY_FOR_DEEPSEEK_RELATIONAL_REPLICATION"
    assert manifest["selected_request_count"] == 96
    assert manifest["selection_uses_public_fields_only"] is True
    assert manifest["oracle_filter_used_for_selection"] is False
    assert manifest["source_model_response_files_are_not_inputs"] is True
    public = read_jsonl(output / "bundle" / "judge_requests.jsonl")
    private = read_jsonl(output / "private" / "evaluation_map.jsonl")
    assert {row["repeat"] for row in public} == {0}
    assert {row["arm"] for row in public} == {
        "Relational-LinkBroken",
        "Relational-Linked",
    }
    assert all("expected_winner" not in row for row in public)
    assert any("expected_winner" in row for row in private)
    assert prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    ) == manifest

    tampered_config = copy.deepcopy(config)
    tampered_config["analysis_code_freeze"]["files_sha256"][
        "src/behavioral_records/controlled_confirmatory.py"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="analysis code hash mismatch"):
        prepare_deepseek_relational_replication(
            source_bundle=source,
            source_plan_path=source_plan,
            output_dir=tmp_path / "tampered-code-freeze",
            config=tampered_config,
        )


def test_no_key_execution_skips_without_calling_transport(tmp_path: Path) -> None:
    source, source_plan, config = _make_source_and_config(tmp_path)
    output = tmp_path / "replication"
    prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    called = False

    def forbidden_transport(endpoint, headers, payload, timeout):
        nonlocal called
        called = True
        raise AssertionError("No transport call is allowed without a key")

    result = execute_or_skip_deepseek_relational_replication(
        output_dir=output,
        config=config,
        api_key="",
        execute=True,
        transport=forbidden_transport,
    )
    assert result["status"] == "SKIPPED_MISSING_DEEPSEEK_API_KEY"
    assert result["network_request_count"] == 0
    assert called is False
    assert not (output / "run" / "judge_attempts.jsonl").exists()
    no_network_preflight = execute_or_skip_deepseek_relational_replication(
        output_dir=output,
        config=config,
        api_key="",
        execute=False,
        transport=forbidden_transport,
    )
    manifest = prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    with pytest.raises(RuntimeError, match="execution-ready preflight"):
        validate_deepseek_relational_run_preflight(
            preflight=no_network_preflight,
            manifest=manifest,
            config=config,
        )
    write_jsonl(
        output / "run" / "judge_attempts.jsonl",
        [{"request_id": "frozen-attempt", "status": "COMPLETE"}],
    )
    with pytest.raises(RuntimeError, match="immutable"):
        execute_or_skip_deepseek_relational_replication(
            output_dir=output,
            config=config,
            api_key="",
            execute=False,
            transport=forbidden_transport,
        )


def test_pooled_pair_level_evaluation_and_scenario_descriptives(tmp_path: Path) -> None:
    source, source_plan, config = _make_source_and_config(tmp_path)
    output = tmp_path / "replication"
    prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    public = read_jsonl(output / "bundle" / "judge_requests.jsonl")
    private = read_jsonl(output / "private" / "evaluation_map.jsonl")
    responses = _synthetic_responses(public, private, config)
    result = evaluate_deepseek_relational_replication(
        public_requests=public,
        private_map=private,
        responses=responses,
        config=config,
    )
    assert result["status"] == "DEEPSEEK_RELATIONAL_REPLICATION_COMPLETE"
    assert result["primary"]["determinate_pair_count"] == 12
    assert result["primary"]["mean_difference"] == 1.0
    assert result["primary"]["familywise_adjustment"] == (
        "none_one_prespecified_pooled_primary"
    )
    assert result["by_scenario"]["C1"]["role"] == "descriptive_only"
    assert all(result["quality_gates"].values())
    assert result["support"]["cross_model_relational_replication_supported"]


def test_validity_gate_and_response_provenance_are_enforced(tmp_path: Path) -> None:
    source, source_plan, config = _make_source_and_config(tmp_path)
    output = tmp_path / "replication"
    prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    public = read_jsonl(output / "bundle" / "judge_requests.jsonl")
    private = read_jsonl(output / "private" / "evaluation_map.jsonl")
    responses = _synthetic_responses(public, private, config)

    tampered = [dict(row) for row in responses]
    tampered[0] = {**tampered[0], "provider": "wrong-provider"}
    with pytest.raises(ValueError, match="provider/model"):
        evaluate_deepseek_relational_replication(
            public_requests=public,
            private_map=private,
            responses=tampered,
            config=config,
        )

    incomplete = [
        row
        for row in responses
        if not (
            "__C1__" in row["request_id"]
            and "__Relational-LinkBroken__" in row["request_id"]
        )
    ]
    failed = evaluate_deepseek_relational_replication(
        public_requests=public,
        private_map=private,
        responses=incomplete,
        config=config,
    )
    assert not failed["quality_gates"][
        "minimum_all_pair_scenario_arm_valid_rate_met"
    ]
    assert not failed["support"]["cross_model_relational_replication_supported"]


def test_link_broken_only_missingness_cannot_create_replication_support(
    tmp_path: Path,
) -> None:
    source, source_plan, config = _make_source_and_config(
        tmp_path, pair_count_per_scenario=12
    )
    output = tmp_path / "replication"
    prepare_deepseek_relational_replication(
        source_bundle=source,
        source_plan_path=source_plan,
        output_dir=output,
        config=config,
    )
    public = read_jsonl(output / "bundle" / "judge_requests.jsonl")
    private = read_jsonl(output / "private" / "evaluation_map.jsonl")
    responses = _synthetic_responses(
        public, private, config, broken_correct=True
    )
    private_by_id = {row["request_id"]: row for row in private}
    removed_pairs: set[str] = set()
    retained = []
    for response in responses:
        metadata = private_by_id[response["request_id"]]
        should_remove = bool(
            metadata["scenario_id"] == "C1"
            and metadata["arm"] == "Relational-LinkBroken"
            and metadata["paraphrase"] == "P0"
            and metadata["order"] == "AB"
            and len(removed_pairs) < 9
            and metadata["pair_id"] not in removed_pairs
        )
        if should_remove:
            removed_pairs.add(metadata["pair_id"])
        else:
            retained.append(response)
    assert len(removed_pairs) == 9

    result = evaluate_deepseek_relational_replication(
        public_requests=public,
        private_map=private,
        responses=retained,
        config=config,
    )
    assert result["quality_gates"][
        "minimum_all_pair_scenario_arm_valid_rate_met"
    ]
    assert result["quality_gates"][
        "minimum_determinate_scenario_arm_valid_rate_met"
    ]
    assert result["support"]["primary_statistical_condition_met"]
    assert not result["support"]["missingness_worst_case_condition_met"]
    assert not result["support"]["cross_model_relational_replication_supported"]
