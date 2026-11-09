from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .canonical import stable_hash
from .controlled_confirmatory import (
    _bootstrap_mean_difference,
    _paired_sign_flip_p,
)
from .io import read_json, read_jsonl, write_json, write_jsonl
from .judge_runner import (
    JudgeConfig,
    Transport,
    cache_key,
    openai_compatible_transport,
    run_judge_requests,
)
from .semantic_pilot import strict_parse_semantic_response


PUBLIC_REQUIRED_FIELDS = frozenset(
    {
        "request_id",
        "scenario_id",
        "concept_id",
        "pair_id",
        "arm",
        "paraphrase",
        "order",
        "repeat",
        "response_schema",
        "prompt",
        "prompt_hash",
    }
)
FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "expected_winner",
    "oracle",
    "policy_id",
    "lineage",
    "method_identity",
    "training_curriculum",
    "held_out",
)
FORBIDDEN_PROMPT_PATTERNS = (
    re.compile(r"expected_winner", re.IGNORECASE),
    re.compile(r"oracle_status", re.IGNORECASE),
    re.compile(r"policy_id", re.IGNORECASE),
    re.compile(r"\b(?:c1p|c2p)_\d+_[ab]\b", re.IGNORECASE),
    re.compile(r"\blineage\b", re.IGNORECASE),
    re.compile(r"method[-_ ]identity", re.IGNORECASE),
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recursive_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result = [str(key) for key in value]
        for item in value.values():
            result.extend(_recursive_keys(item))
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        result: list[str] = []
        for item in value:
            result.extend(_recursive_keys(item))
        return result
    return []


def _selection_match(row: Mapping[str, Any], selection: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("scenario_id"))
        in {str(value) for value in selection["scenarios"]}
        and str(row.get("arm")) in {str(value) for value in selection["arms"]}
        and str(row.get("paraphrase"))
        in {str(value) for value in selection["paraphrases"]}
        and str(row.get("order"))
        in {str(value) for value in selection["orders"]}
        and int(row.get("repeat", -1))
        in {int(value) for value in selection["repeats"]}
    )


def select_public_relational_replication_requests(
    requests: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Select the replication batch using public request metadata only."""

    return [dict(row) for row in requests if _selection_match(row, selection)]


def _source_hash_checks(
    source_bundle: Path, config: Mapping[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for relative, expected in config["source_hashes_sha256"].items():
        path = source_bundle / str(relative)
        checks[str(relative)] = path.is_file() and file_sha256(path) == str(expected)
    return checks


def _validate_runner_contract(config: Mapping[str, Any]) -> None:
    judge = config["judge_freeze"]
    runner_path = Path(__file__).with_name("judge_runner.py")
    if file_sha256(runner_path) != str(judge["runner_file_sha256"]):
        raise RuntimeError("Judge runner file differs from the frozen execution contract")
    expected_system_instruction = (
        "Use only the supplied evidence. Return exactly one JSON object and no other "
        "final-answer text."
    )
    if str(judge["fixed_system_instruction"]) != expected_system_instruction:
        raise RuntimeError("Frozen system instruction differs from the judge runner")


def _validate_analysis_code_contract(config: Mapping[str, Any]) -> dict[str, bool]:
    repository_root = Path(__file__).resolve().parents[2]
    checks: dict[str, bool] = {}
    for relative, expected in config["analysis_code_freeze"][
        "files_sha256"
    ].items():
        path = repository_root / str(relative)
        checks[str(relative)] = path.is_file() and file_sha256(path) == str(expected)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("Frozen analysis code hash mismatch: " + ", ".join(failed))
    return checks


def _validate_source_bundle(
    *,
    source_bundle: Path,
    source_plan_path: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not source_plan_path.is_file():
        raise RuntimeError("Frozen relational source plan is missing")
    if file_sha256(source_plan_path) != str(config["source_plan_file_sha256"]):
        raise RuntimeError("Frozen relational source plan file hash mismatch")
    source_plan = read_json(source_plan_path)
    checks = config["source_bundle_checks"]
    if stable_hash(source_plan) != str(checks["expected_source_plan_stable_hash"]):
        raise RuntimeError("Frozen relational source plan stable hash mismatch")
    if str(source_plan.get("version_label")) != str(
        checks["required_version_label"]
    ):
        raise RuntimeError("Unexpected relational source plan version")

    hash_checks = _source_hash_checks(source_bundle, config)
    if not all(hash_checks.values()):
        failed = sorted(name for name, passed in hash_checks.items() if not passed)
        raise RuntimeError("Frozen relational source hash mismatch: " + ", ".join(failed))

    summary = read_json(source_bundle / "summary.json")
    integrity = read_json(source_bundle / "relational_integrity_audit.json")
    requests = read_jsonl(source_bundle / "judge_requests.jsonl")
    private_map = read_jsonl(source_bundle / "judge_private_map.jsonl")
    public_evidence = read_jsonl(source_bundle / "public_evidence.jsonl")

    if summary.get("status") != checks["required_status"]:
        raise RuntimeError("The relational source bundle did not pass its offline gate")
    if summary.get("material_passport", {}).get("version_label") != checks[
        "required_version_label"
    ]:
        raise RuntimeError("The relational source summary has an unexpected version")
    if summary.get("plan_hash") != stable_hash(source_plan):
        raise RuntimeError("The relational source summary disagrees with its plan")
    if int(summary.get("request_count", -1)) != int(
        checks["expected_total_request_count"]
    ):
        raise RuntimeError("Unexpected relational source request count")
    if int(summary.get("pair_count", -1)) != int(
        checks["expected_total_pair_count"]
    ):
        raise RuntimeError("Unexpected relational source pair count")
    if not summary.get("gates") or not all(summary["gates"].values()):
        raise RuntimeError("A relational source summary integrity gate is false")
    if not integrity.get("passed") or not all(integrity.get("gates", {}).values()):
        raise RuntimeError("The relational source integrity audit did not pass")

    stable_checks = {
        "source_request_stable_hash": stable_hash(requests)
        == str(checks["expected_source_request_stable_hash"])
        == str(summary.get("request_hash")),
        "source_private_map_stable_hash": stable_hash(private_map)
        == str(checks["expected_source_private_map_stable_hash"])
        == str(summary.get("private_map_hash")),
        "source_public_evidence_stable_hash": stable_hash(public_evidence)
        == str(checks["expected_source_public_evidence_stable_hash"])
        == str(summary.get("public_evidence_hash")),
        "source_integrity_audit_stable_hash": stable_hash(integrity)
        == str(checks["expected_source_integrity_audit_stable_hash"])
        == str(summary.get("integrity_audit_hash")),
    }
    if not all(stable_checks.values()):
        failed = sorted(name for name, passed in stable_checks.items() if not passed)
        raise RuntimeError(
            "Relational source content disagrees with its freeze: " + ", ".join(failed)
        )

    request_ids = [str(row.get("request_id", "")) for row in requests]
    private_ids = [str(row.get("request_id", "")) for row in private_map]
    if (
        not all(request_ids)
        or len(request_ids) != len(set(request_ids))
        or not all(private_ids)
        or len(private_ids) != len(set(private_ids))
        or set(request_ids) != set(private_ids)
    ):
        raise RuntimeError("Relational source public/private request IDs do not align")

    audit = {
        "source_file_hashes": hash_checks,
        "source_stable_hashes": stable_checks,
        "source_plan_file_hash_matches": True,
        "source_plan_stable_hash_matches": True,
        "source_summary_and_integrity_gates_pass": True,
        "source_public_private_ids_align": True,
    }
    return requests, private_map, audit


def _validate_public_selection(
    *,
    source_requests: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selection = config["selection"]
    if len(selected) != int(selection["expected_total_requests"]):
        raise RuntimeError("The public selection has an unexpected request count")
    if stable_hash(selected) != str(
        selection["expected_selected_request_stable_hash"]
    ):
        raise RuntimeError("The public selection differs from the frozen request hash")

    request_ids = [str(row.get("request_id", "")) for row in selected]
    if not all(request_ids) or len(request_ids) != len(set(request_ids)):
        raise RuntimeError("Selected public request IDs must be unique and non-empty")

    prompt_lengths: list[int] = []
    for row in selected:
        missing = PUBLIC_REQUIRED_FIELDS - set(row)
        if missing:
            raise RuntimeError(
                f"Public request {row.get('request_id')} lacks {sorted(missing)}"
            )
        lowered_keys = [key.lower() for key in _recursive_keys(row)]
        leaked_keys = sorted(
            {
                key
                for key in lowered_keys
                if any(fragment in key for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS)
            }
        )
        if leaked_keys:
            raise RuntimeError(
                f"Private key leaked into public request {row['request_id']}: {leaked_keys}"
            )
        prompt = str(row["prompt"])
        if stable_hash(prompt) != str(row["prompt_hash"]):
            raise RuntimeError(f"Prompt hash mismatch for {row['request_id']}")
        prompt_matches = [
            pattern.pattern
            for pattern in FORBIDDEN_PROMPT_PATTERNS
            if pattern.search(prompt)
        ]
        if prompt_matches:
            raise RuntimeError(
                f"Private identifier leaked into prompt {row['request_id']}: "
                + ", ".join(prompt_matches)
            )
        prompt_lengths.append(len(prompt))
    if max(prompt_lengths, default=0) > int(
        config["judge_freeze"]["maximum_prompt_characters"]
    ):
        raise RuntimeError("A selected prompt exceeds the frozen character budget")

    scenarios = [str(value) for value in selection["scenarios"]]
    arms = [str(value) for value in selection["arms"]]
    paraphrases = [str(value) for value in selection["paraphrases"]]
    orders = [str(value) for value in selection["orders"]]
    expected_pairs = int(selection["pair_count_per_scenario"])
    expected_contexts = int(selection["contexts_per_pair_arm"])
    source_pairs: dict[str, set[str]] = defaultdict(set)
    selected_pairs: dict[str, set[str]] = defaultdict(set)
    cell_counts = Counter()
    pair_arm_counts = Counter()
    for row in source_requests:
        scenario = str(row.get("scenario_id"))
        if scenario in scenarios:
            source_pairs[scenario].add(str(row.get("pair_id")))
    for row in selected:
        scenario = str(row["scenario_id"])
        pair_id = str(row["pair_id"])
        arm = str(row["arm"])
        selected_pairs[scenario].add(pair_id)
        cell_counts[(scenario, arm, str(row["paraphrase"]), str(row["order"]))] += 1
        pair_arm_counts[(scenario, pair_id, arm)] += 1

    for scenario in scenarios:
        if len(source_pairs[scenario]) != expected_pairs:
            raise RuntimeError(f"Unexpected source pair count for {scenario}")
        if selected_pairs[scenario] != source_pairs[scenario]:
            raise RuntimeError(f"Not every source pair was selected in {scenario}")
        for arm in arms:
            for paraphrase in paraphrases:
                for order in orders:
                    if cell_counts[(scenario, arm, paraphrase, order)] != expected_pairs:
                        raise RuntimeError("The selected request grid is not balanced")
            for pair_id in selected_pairs[scenario]:
                if pair_arm_counts[(scenario, pair_id, arm)] != expected_contexts:
                    raise RuntimeError("A selected pair-arm cell has the wrong context count")

    return {
        "request_count": len(selected),
        "unique_request_count": len(set(request_ids)),
        "pair_counts": {
            scenario: len(selected_pairs[scenario]) for scenario in scenarios
        },
        "prompt_character_range": [
            min(prompt_lengths, default=0),
            max(prompt_lengths, default=0),
        ],
        "prompt_hashes_match": True,
        "all_source_pairs_retained": True,
        "factorial_balance_passed": True,
        "contexts_per_pair_arm_match": True,
        "forbidden_public_leakage_absent": True,
    }


def _validate_private_alignment(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_private_map: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_by_id = {str(row["request_id"]): row for row in selected}
    source_private_by_id: dict[str, dict[str, Any]] = {}
    for row in source_private_map:
        request_id = str(row.get("request_id", ""))
        if request_id in source_private_by_id:
            raise RuntimeError(f"Duplicate source private request ID: {request_id}")
        source_private_by_id[request_id] = dict(row)
    if not set(selected_by_id) <= set(source_private_by_id):
        raise RuntimeError("A selected public request lacks a private evaluation row")
    selected_private = [
        dict(row)
        for row in source_private_map
        if str(row.get("request_id", "")) in selected_by_id
    ]
    if len(selected_private) != len(selected):
        raise RuntimeError("Selected public/private request counts differ")
    if stable_hash(selected_private) != str(
        config["selection"]["expected_selected_private_map_stable_hash"]
    ):
        raise RuntimeError("The selected private map differs from its frozen hash")
    shared_fields = (
        "scenario_id",
        "pair_id",
        "arm",
        "paraphrase",
        "order",
        "repeat",
    )
    for private in selected_private:
        public = selected_by_id[str(private["request_id"])]
        if any(private.get(field) != public.get(field) for field in shared_fields):
            raise RuntimeError("A public/private request mapping disagrees on public fields")
    return selected_private


def _validate_existing_bundle(
    output: Path, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    plan_path = output / "bundle" / "frozen_plan.json"
    request_path = output / "bundle" / "judge_requests.jsonl"
    private_path = output / "private" / "evaluation_map.jsonl"
    manifest_path = output / "bundle_manifest.json"
    paths = (plan_path, request_path, private_path, manifest_path)
    present = [path.exists() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("Refusing a partially existing DeepSeek relational bundle")
    manifest = read_json(manifest_path)
    frozen_plan = read_json(plan_path)
    requests = read_jsonl(request_path)
    private_map = read_jsonl(private_path)
    if frozen_plan != config or stable_hash(frozen_plan) != manifest.get("config_hash"):
        raise RuntimeError("Existing bundle uses a different frozen config")
    if stable_hash(requests) != manifest.get("selected_request_hash"):
        raise RuntimeError("Existing public bundle stable hash mismatch")
    if stable_hash(private_map) != manifest.get("selected_private_map_hash"):
        raise RuntimeError("Existing private map stable hash mismatch")
    file_checks = {
        "frozen_plan": file_sha256(plan_path)
        == manifest.get("frozen_plan_file_sha256"),
        "public_requests": file_sha256(request_path)
        == manifest.get("public_request_file_sha256"),
        "private_map": file_sha256(private_path)
        == manifest.get("private_evaluation_map_file_sha256"),
    }
    if not all(file_checks.values()):
        raise RuntimeError("An existing DeepSeek relational bundle file was modified")
    return manifest


def prepare_deepseek_relational_replication(
    *,
    source_bundle: str | Path,
    source_plan_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create or read-only revalidate the frozen relational replication bundle."""

    source = Path(source_bundle)
    source_plan = Path(source_plan_path)
    output = Path(output_dir)
    _validate_runner_contract(config)
    analysis_code_checks = _validate_analysis_code_contract(config)
    source_requests, source_private, source_audit = _validate_source_bundle(
        source_bundle=source,
        source_plan_path=source_plan,
        config=config,
    )
    existing = _validate_existing_bundle(output, config)
    if existing is not None:
        return dict(existing)

    selected = select_public_relational_replication_requests(
        source_requests, config["selection"]
    )
    public_audit = _validate_public_selection(
        source_requests=source_requests,
        selected=selected,
        config=config,
    )
    selected_private = _validate_private_alignment(
        selected=selected,
        source_private_map=source_private,
        config=config,
    )

    plan_path = output / "bundle" / "frozen_plan.json"
    request_path = output / "bundle" / "judge_requests.jsonl"
    private_path = output / "private" / "evaluation_map.jsonl"
    write_json(plan_path, config)
    write_jsonl(request_path, selected)
    write_jsonl(private_path, selected_private)
    manifest = {
        "schema_version": "controlled-relational-deepseek-replication-bundle-v1",
        "status": "READY_FOR_DEEPSEEK_RELATIONAL_REPLICATION",
        "version_label": config["version_label"],
        "config_hash": stable_hash(config),
        "source_bundle": str(config["source_bundle"]),
        "source_plan": str(config["source_plan"]),
        "source_audit": source_audit,
        "selection_uses_public_fields_only": True,
        "all_source_pairs_selected_before_private_map_join": True,
        "all_96_pairs_selected_before_private_map_join": int(
            config["selection"]["total_unique_pairs"]
        )
        == 96,
        "oracle_filter_used_for_selection": False,
        "source_model_response_files_are_not_inputs": True,
        "source_prompts_copied_without_modification": True,
        "runner_file_sha256": config["judge_freeze"]["runner_file_sha256"],
        "fixed_system_instruction_hash": stable_hash(
            config["judge_freeze"]["fixed_system_instruction"]
        ),
        "analysis_code_freeze_hash": stable_hash(config["analysis_code_freeze"]),
        "analysis_code_hash_checks": analysis_code_checks,
        "selected_request_count": len(selected),
        "selected_request_hash": stable_hash(selected),
        "selected_private_map_hash": stable_hash(selected_private),
        "frozen_plan_file_sha256": file_sha256(plan_path),
        "public_request_file_sha256": file_sha256(request_path),
        "private_evaluation_map_file_sha256": file_sha256(private_path),
        "public_audit": public_audit,
        "api_calls_during_bundle_construction": 0,
        "api_key_recorded": False,
        "private_map_sent_to_provider": False,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(output / "bundle_manifest.json", manifest)
    return manifest


def _judge_config(config: Mapping[str, Any], *, api_key: str) -> JudgeConfig:
    judge = config["judge_freeze"]
    return JudgeConfig(
        provider=str(judge["provider"]),
        base_url=str(judge["base_url"]),
        model=str(judge["model"]),
        api_key=api_key,
        max_requests=int(config["selection"]["expected_total_requests"]),
        max_cost_usd=float(judge["maximum_accounted_cost_usd"]),
        input_cost_per_million=float(
            judge["input_accounting_rate_usd_per_million_tokens"]
        ),
        output_cost_per_million=float(
            judge["output_accounting_rate_usd_per_million_tokens"]
        ),
        max_output_tokens=int(judge["max_output_tokens"]),
        max_prompt_chars=int(judge["maximum_prompt_characters"]),
        timeout_seconds=float(judge["timeout_seconds"]),
        temperature=float(judge["temperature"]),
        max_concurrency=int(judge["max_concurrency"]),
        response_format_type=str(judge["response_format_type"]),
    )


def execute_or_skip_deepseek_relational_replication(
    *,
    output_dir: str | Path,
    config: Mapping[str, Any],
    api_key: str,
    execute: bool,
    transport: Transport = openai_compatible_transport,
) -> dict[str, Any]:
    """Write a no-network preflight, skip without a key, or run once."""

    output = Path(output_dir)
    _validate_runner_contract(config)
    analysis_code_checks = _validate_analysis_code_contract(config)
    manifest = _validate_existing_bundle(output, config)
    if manifest is None:
        raise RuntimeError("Prepare the frozen replication bundle before execution")
    run_dir = output / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "judge_attempts.jsonl"
    responses_path = run_dir / "judge_responses.jsonl"
    if attempts_path.exists() or responses_path.exists():
        raise RuntimeError(
            "Existing attempts or responses are immutable; do not overwrite the "
            "execution preflight or implicitly resume"
        )
    key_present = bool(api_key.strip())
    status = (
        "PREPARED_NO_NETWORK_CALL"
        if not execute
        else "READY_FOR_BOUNDED_EXECUTION"
        if key_present
        else "SKIPPED_MISSING_DEEPSEEK_API_KEY"
    )
    judge = config["judge_freeze"]
    preflight = {
        "schema_version": "controlled-relational-deepseek-preflight-v1",
        "status": status,
        "execute_requested": bool(execute),
        "provider": judge["provider"],
        "model": judge["model"],
        "base_url": judge["base_url"],
        "runner_file_sha256": judge["runner_file_sha256"],
        "fixed_system_instruction_hash": stable_hash(
            judge["fixed_system_instruction"]
        ),
        "analysis_code_freeze_hash": stable_hash(config["analysis_code_freeze"]),
        "analysis_code_hash_checks": analysis_code_checks,
        "selected_request_count": manifest["selected_request_count"],
        "selected_request_hash": manifest["selected_request_hash"],
        "config_hash": manifest["config_hash"],
        "maximum_accounted_cost_usd": judge["maximum_accounted_cost_usd"],
        "input_accounting_rate_usd_per_million_tokens": judge[
            "input_accounting_rate_usd_per_million_tokens"
        ],
        "output_accounting_rate_usd_per_million_tokens": judge[
            "output_accounting_rate_usd_per_million_tokens"
        ],
        "api_key_configured": key_present,
        "api_key_recorded": False,
        "send_only_fixed_system_instruction_and_public_prompt": True,
        "private_map_sent_to_provider": False,
        "source_model_responses_sent_to_provider": False,
        "automatic_retries": 0,
        "network_call_made_by_preflight": False,
    }
    write_json(run_dir / "judge_preflight.json", preflight)
    if not execute:
        return preflight
    if not key_present:
        skipped = {
            "schema_version": "controlled-relational-deepseek-run-summary-v1",
            "status": "SKIPPED_MISSING_DEEPSEEK_API_KEY",
            "reason": "DEEPSEEK_API_KEY is not configured; no network request was made",
            "selected_request_count": manifest["selected_request_count"],
            "api_key_recorded": False,
            "network_request_count": 0,
        }
        write_json(run_dir / "judge_run_summary.json", skipped)
        return skipped

    requests = read_jsonl(output / "bundle" / "judge_requests.jsonl")
    return run_judge_requests(
        requests,
        output_dir=run_dir,
        config=_judge_config(config, api_key=api_key),
        transport=transport,
        response_parser=strict_parse_semantic_response,
        stop_on_failure_error_substrings=judge["stop_on_failure_error_substrings"],
    )


def validate_deepseek_relational_run_preflight(
    *,
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    judge = config["judge_freeze"]
    expected = {
        "provider": judge["provider"],
        "model": judge["model"],
        "base_url": judge["base_url"],
        "runner_file_sha256": judge["runner_file_sha256"],
        "fixed_system_instruction_hash": stable_hash(
            judge["fixed_system_instruction"]
        ),
        "analysis_code_freeze_hash": stable_hash(config["analysis_code_freeze"]),
        "selected_request_count": manifest["selected_request_count"],
        "selected_request_hash": manifest["selected_request_hash"],
        "config_hash": manifest["config_hash"],
        "maximum_accounted_cost_usd": judge["maximum_accounted_cost_usd"],
    }
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise RuntimeError("DeepSeek run preflight differs from the frozen plan")
    if preflight.get("api_key_recorded") is not False:
        raise RuntimeError("DeepSeek run preflight must not record the API key")
    if preflight.get("private_map_sent_to_provider") is not False:
        raise RuntimeError("Private evaluation data must not be sent to DeepSeek")
    if preflight.get("send_only_fixed_system_instruction_and_public_prompt") is not True:
        raise RuntimeError("DeepSeek preflight does not enforce the public content boundary")
    if (
        preflight.get("execute_requested") is not True
        or preflight.get("api_key_configured") is not True
        or preflight.get("status") != "READY_FOR_BOUNDED_EXECUTION"
    ):
        raise RuntimeError("Responses require a successful execution-ready preflight")


def _validate_response_provenance(
    *,
    public_requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    request_by_id = {str(row["request_id"]): row for row in public_requests}
    if len(request_by_id) != len(public_requests):
        raise ValueError("Frozen public requests contain duplicate IDs")
    response_ids = [str(row.get("request_id", "")) for row in responses]
    if not all(response_ids) or len(response_ids) != len(set(response_ids)):
        raise ValueError("DeepSeek responses must have unique non-empty request IDs")
    unexpected = sorted(set(response_ids) - set(request_by_id))
    if unexpected:
        raise ValueError(
            "DeepSeek responses contain IDs outside the frozen request bundle: "
            + ", ".join(unexpected[:5])
        )
    judge = config["judge_freeze"]
    provenance_config = _judge_config(
        config, api_key="not-used-for-offline-provenance-check"
    )
    for row in responses:
        request_id = str(row["request_id"])
        if row.get("provider") != judge["provider"] or row.get("model") != judge[
            "model"
        ]:
            raise ValueError("DeepSeek response provider/model differs from the freeze")
        if row.get("cache_key") != cache_key(
            request_by_id[request_id], provenance_config
        ):
            raise ValueError("DeepSeek response cache provenance does not match its request")
    return {
        "response_count": len(responses),
        "unique_response_ids": True,
        "all_response_ids_in_frozen_bundle": True,
        "provider_model_match": True,
        "cache_keys_match": True,
    }


def evaluate_deepseek_relational_replication(
    *,
    public_requests: Sequence[Mapping[str, Any]],
    private_map: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the one pooled, pair-level relational replication endpoint."""

    public_rows = [dict(row) for row in public_requests]
    private_rows = [dict(row) for row in private_map]
    response_rows = [dict(row) for row in responses]
    _validate_runner_contract(config)
    analysis_code_checks = _validate_analysis_code_contract(config)
    provenance = _validate_response_provenance(
        public_requests=public_rows,
        responses=response_rows,
        config=config,
    )
    public_ids = {str(row["request_id"]) for row in public_rows}
    private_ids = [str(row.get("request_id", "")) for row in private_rows]
    if (
        not all(private_ids)
        or len(private_ids) != len(set(private_ids))
        or set(private_ids) != public_ids
    ):
        raise ValueError("Frozen replication public/private IDs do not align")

    response_by_id = {str(row["request_id"]): row for row in response_rows}
    calls: list[dict[str, Any]] = []
    for private in private_rows:
        response = response_by_id.get(str(private["request_id"]))
        parsed = response.get("response") if response else None
        observed = parsed.get("winner") if isinstance(parsed, Mapping) else None
        valid = observed in {"A", "B", "TIE", "ABSTAIN"}
        calls.append(
            {
                **private,
                "observed_winner": observed,
                "valid": valid,
                "correct": bool(valid and observed == private["expected_winner"]),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        grouped[(str(row["scenario_id"]), str(row["pair_id"]), str(row["arm"]))].append(row)
    expected_contexts = int(config["selection"]["contexts_per_pair_arm"])
    pair_rows: list[dict[str, Any]] = []
    for (scenario, pair_id, arm), rows in sorted(grouped.items()):
        statuses = {str(row["oracle_status"]) for row in rows}
        if len(statuses) != 1:
            raise ValueError("Oracle status differs within a frozen pair-arm cell")
        pair_rows.append(
            {
                "scenario_id": scenario,
                "pair_id": pair_id,
                "arm": arm,
                "oracle_status": next(iter(statuses)),
                "determinate": next(iter(statuses)) == "DETERMINATE",
                "request_count": len(rows),
                "valid_request_count": sum(row["valid"] for row in rows),
                "oracle_label_mass": fmean(row["correct"] for row in rows),
                "missing_as_correct_oracle_label_mass": fmean(
                    row["correct"] if row["valid"] else True for row in rows
                ),
                "strict_all_context_correct": len(rows) == expected_contexts
                and all(row["correct"] for row in rows),
            }
        )

    scenarios = [str(value) for value in config["selection"]["scenarios"]]
    arms = [str(value) for value in config["selection"]["arms"]]
    linked, broken = "Relational-Linked", "Relational-LinkBroken"
    cell_valid_rates: dict[str, float] = {}
    determinate_cell_valid_rates: dict[str, float] = {}
    scenario_effects: dict[str, list[float]] = defaultdict(list)
    worst_case_scenario_effects: dict[str, list[float]] = defaultdict(list)
    by_scenario: dict[str, Any] = {}
    all_effects: list[float] = []
    worst_case_effects: list[float] = []
    for scenario in scenarios:
        scenario_rows = [row for row in pair_rows if row["scenario_id"] == scenario]
        determinate = [row for row in scenario_rows if row["determinate"]]
        index = {
            (str(row["pair_id"]), str(row["arm"])): row for row in determinate
        }
        pair_ids = sorted({str(row["pair_id"]) for row in determinate})
        for pair_id in pair_ids:
            if (pair_id, linked) not in index or (pair_id, broken) not in index:
                raise ValueError("A determinate pair lacks one relational arm")
            effect = float(index[(pair_id, linked)]["oracle_label_mass"]) - float(
                index[(pair_id, broken)]["oracle_label_mass"]
            )
            scenario_effects[scenario].append(effect)
            all_effects.append(effect)
            worst_case_effect = float(
                index[(pair_id, linked)]["oracle_label_mass"]
            ) - float(
                index[(pair_id, broken)][
                    "missing_as_correct_oracle_label_mass"
                ]
            )
            worst_case_scenario_effects[scenario].append(worst_case_effect)
            worst_case_effects.append(worst_case_effect)

        by_arm: dict[str, Any] = {}
        for arm in arms:
            all_arm_rows = [row for row in scenario_rows if row["arm"] == arm]
            determinate_arm_rows = [row for row in determinate if row["arm"] == arm]
            denominator = sum(row["request_count"] for row in all_arm_rows)
            valid_rate = (
                sum(row["valid_request_count"] for row in all_arm_rows) / denominator
                if denominator
                else 0.0
            )
            cell_valid_rates[f"{scenario}::{arm}"] = valid_rate
            determinate_denominator = sum(
                row["request_count"] for row in determinate_arm_rows
            )
            determinate_valid_rate = (
                sum(row["valid_request_count"] for row in determinate_arm_rows)
                / determinate_denominator
                if determinate_denominator
                else 0.0
            )
            determinate_cell_valid_rates[f"{scenario}::{arm}"] = (
                determinate_valid_rate
            )
            by_arm[arm] = {
                "determinate_pair_count": len(determinate_arm_rows),
                "context_marginalized_oracle_label_mass": fmean(
                    row["oracle_label_mass"] for row in determinate_arm_rows
                )
                if determinate_arm_rows
                else None,
                "strict_all_context_correct_pairs": sum(
                    row["strict_all_context_correct"] for row in determinate_arm_rows
                ),
                "valid_request_rate_all_requested_pairs": valid_rate,
                "valid_request_rate_determinate_primary_pairs": (
                    determinate_valid_rate
                ),
            }
        by_scenario[scenario] = {
            "role": "descriptive_only",
            "determinate_pair_count": len(pair_ids),
            "mean_difference_descriptive": fmean(scenario_effects[scenario])
            if scenario_effects[scenario]
            else None,
            "by_arm": by_arm,
        }

    primary = {
        "effect_definition": (
            "Relational-Linked minus Relational-LinkBroken; positive favors "
            "Relational-Linked"
        ),
        "unit_of_inference": "unique_policy_pair",
        "determinate_pair_count": len(all_effects),
        "mean_difference": fmean(all_effects) if all_effects else None,
        "pair_bootstrap_95_ci": _bootstrap_mean_difference(
            all_effects,
            seed=int(config["primary_analysis"]["bootstrap_seed"]),
        )
        if all_effects
        else None,
        "paired_sign_flip_p": _paired_sign_flip_p(
            all_effects,
            seed=int(config["primary_analysis"]["randomization_seed"]),
        )
        if all_effects
        else None,
        "linked_better_pairs": sum(value > 0 for value in all_effects),
        "equal_pairs": sum(value == 0 for value in all_effects),
        "link_broken_better_pairs": sum(value < 0 for value in all_effects),
        "scenario_mean_differences_descriptive": {
            scenario: fmean(values) if values else None
            for scenario, values in scenario_effects.items()
        },
        "familywise_adjustment": "none_one_prespecified_pooled_primary",
    }
    missingness_worst_case = {
        "completion_rule": (
            "missing Relational-Linked response scores incorrect; missing "
            "Relational-LinkBroken response scores correct"
        ),
        "role": "conservative_fixed_sample_support_gate",
        "determinate_pair_count": len(worst_case_effects),
        "mean_difference_lower_bound": fmean(worst_case_effects)
        if worst_case_effects
        else None,
        "pair_bootstrap_95_ci": _bootstrap_mean_difference(
            worst_case_effects,
            seed=int(
                config["primary_analysis"][
                    "missingness_worst_case_bootstrap_seed"
                ]
            ),
        )
        if worst_case_effects
        else None,
        "paired_sign_flip_p": _paired_sign_flip_p(
            worst_case_effects,
            seed=int(
                config["primary_analysis"][
                    "missingness_worst_case_randomization_seed"
                ]
            ),
        )
        if worst_case_effects
        else None,
        "scenario_mean_differences_descriptive": {
            scenario: fmean(values) if values else None
            for scenario, values in worst_case_scenario_effects.items()
        },
    }

    expected_determinate = {
        str(key): int(value)
        for key, value in config["primary_analysis"][
            "expected_determinate_pairs_by_scenario"
        ].items()
    }
    observed_determinate = {
        scenario: int(by_scenario[scenario]["determinate_pair_count"])
        for scenario in scenarios
    }
    pair_arm_counts_exact = all(
        int(row["request_count"]) == expected_contexts for row in pair_rows
    )
    quality_gates = {
        "expected_request_count": len(private_rows)
        == int(config["selection"]["expected_total_requests"]),
        "all_pair_arm_cells_present": len(pair_rows)
        == int(config["selection"]["total_unique_pairs"]) * len(arms),
        "contexts_per_pair_arm_exact": pair_arm_counts_exact,
        "determinate_pair_counts_match_freeze": observed_determinate
        == expected_determinate
        and len(all_effects)
        == int(config["primary_analysis"]["expected_pooled_determinate_pair_count"]),
        "minimum_all_pair_scenario_arm_valid_rate_met": min(
            cell_valid_rates.values(), default=0.0
        )
        >= float(
            config["primary_analysis"][
                "minimum_valid_response_rate_per_scenario_arm_all_requested_pairs"
            ]
        ),
        "minimum_determinate_scenario_arm_valid_rate_met": min(
            determinate_cell_valid_rates.values(), default=0.0
        )
        >= float(
            config["primary_analysis"][
                "minimum_valid_response_rate_per_scenario_arm_determinate_primary_pairs"
            ]
        ),
        "response_provenance_validated": all(
            value is True
            for key, value in provenance.items()
            if key != "response_count"
        ),
    }
    alpha = float(config["primary_analysis"]["alpha"])
    statistical_condition = bool(
        primary["mean_difference"] is not None
        and primary["mean_difference"] > 0.0
        and primary["paired_sign_flip_p"] is not None
        and primary["paired_sign_flip_p"] < alpha
    )
    missingness_robust_condition = bool(
        missingness_worst_case["mean_difference_lower_bound"] is not None
        and missingness_worst_case["mean_difference_lower_bound"] > 0.0
        and missingness_worst_case["paired_sign_flip_p"] is not None
        and missingness_worst_case["paired_sign_flip_p"] < alpha
    )
    support = {
        "primary_statistical_condition_met": statistical_condition,
        "missingness_worst_case_condition_met": missingness_robust_condition,
        "cross_model_relational_replication_supported": bool(
            all(quality_gates.values())
            and statistical_condition
            and missingness_robust_condition
        ),
        "interpretation_if_not_supported": (
            "Do not claim independent cross-model replication of the relation-link "
            "contribution."
        ),
    }
    return {
        "schema_version": "controlled-relational-deepseek-replication-evaluation-v1",
        "status": "DEEPSEEK_RELATIONAL_REPLICATION_COMPLETE"
        if all(quality_gates.values())
        else "DEEPSEEK_RELATIONAL_REPLICATION_QUALITY_GATE_FAILED",
        "response_count": len(response_rows),
        "expected_request_count": len(private_rows),
        "response_provenance": provenance,
        "analysis_code_hash_checks": analysis_code_checks,
        "analysis_code_freeze_hash": stable_hash(config["analysis_code_freeze"]),
        "primary": primary,
        "missingness_worst_case": missingness_worst_case,
        "by_scenario": by_scenario,
        "cell_valid_rates": cell_valid_rates,
        "determinate_cell_valid_rates": determinate_cell_valid_rates,
        "quality_gates": quality_gates,
        "support": support,
        "pair_rows": pair_rows,
        "analysis_seeds": {
            "bootstrap": int(config["primary_analysis"]["bootstrap_seed"]),
            "randomization": int(config["primary_analysis"]["randomization_seed"]),
            "missingness_worst_case_bootstrap": int(
                config["primary_analysis"][
                    "missingness_worst_case_bootstrap_seed"
                ]
            ),
            "missingness_worst_case_randomization": int(
                config["primary_analysis"][
                    "missingness_worst_case_randomization_seed"
                ]
            ),
        },
        "claim_boundary": config["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-09-01",
            "verification_status": "ANALYZED",
            "version_label": str(config["version_label"]),
        },
    }
