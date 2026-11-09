from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import stable_hash
from .extractor import BehaviorRecordExtractor
from .io import read_json
from .views import (
    build_compact_behavior_view,
    build_matched_raw_trajectory_view,
    public_evidence,
    public_evidence_chars,
)


FORBIDDEN_RECORD_KEYS = {
    "objective",
    "objective_id",
    "oracle",
    "oracle_score",
    "heldout",
    "heldout_score",
    "preference",
    "alignment_score",
    "curriculum",
    "model_path",
    "method",
    "lineage",
}


def extractor_signature_is_objective_free() -> bool:
    parameters = list(inspect.signature(BehaviorRecordExtractor.extract).parameters)
    return parameters == ["self", "policy_id", "trajectories"]


def prompt_view_signatures_are_objective_free() -> bool:
    forbidden_parameters = {"objective", "objective_id", "preference", "judge"}
    functions = (build_compact_behavior_view, build_matched_raw_trajectory_view)
    return all(
        forbidden_parameters.isdisjoint(inspect.signature(function).parameters)
        for function in functions
    )


def find_forbidden_keys(value: Any, *, path: str = "record") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_RECORD_KEYS:
                hits.append(f"{path}.{key}")
            hits.extend(find_forbidden_keys(item, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(find_forbidden_keys(item, path=f"{path}[{index}]"))
    return hits


def audit_record(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_hash = stable_hash(protocol)
    forbidden = find_forbidden_keys(record)
    checks = {
        "extractor_signature_objective_free": extractor_signature_is_objective_free(),
        "protocol_hash_matches": record.get("protocol_hash") == expected_hash,
        "record_hash_matches": record.get("record_hash") == stable_hash(
            {key: value for key, value in record.items() if key != "record_hash"}
        ),
        "no_forbidden_keys": not forbidden,
        "all_steps_tokenized": bool(record.get("coverage", {}).get("all_steps_tokenized", False)),
        "coverage_reported": "coverage" in record,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_key_paths": forbidden,
        "protocol_hash": expected_hash,
        "record_hash": record.get("record_hash"),
    }


def audit_prompt_view_pair(
    compact_view: Mapping[str, Any],
    raw_view: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_view_config = protocol["prompt_view"]
    compact_public = public_evidence(compact_view)
    raw_public = public_evidence(raw_view)
    compact_chars = public_evidence_chars(compact_view)
    raw_chars = public_evidence_chars(raw_view)
    slack = compact_chars - raw_chars
    raw_exhausted = raw_view.get("view_coverage", {}).get("step_fraction") == 1.0
    forbidden_compact = find_forbidden_keys(compact_public, path="compact_public")
    forbidden_raw = find_forbidden_keys(raw_public, path="raw_public")
    raw_text = str(raw_public).lower()
    expected_protocol_hash = stable_hash(protocol)
    checks = {
        "view_builder_signatures_objective_free": prompt_view_signatures_are_objective_free(),
        "policy_ids_match": compact_view.get("policy_id") == raw_view.get("policy_id"),
        "protocol_hashes_match": (
            compact_view.get("protocol_hash") == expected_protocol_hash
            and raw_view.get("protocol_hash") == expected_protocol_hash
        ),
        "compact_view_hash_matches": compact_view.get("view_hash") == stable_hash(
            {key: value for key, value in compact_view.items() if key != "view_hash"}
        ),
        "raw_view_hash_matches": raw_view.get("view_hash") == stable_hash(
            {key: value for key, value in raw_view.items() if key != "view_hash"}
        ),
        "compact_within_hard_cap": compact_chars <= int(prompt_view_config["per_policy_char_budget"]),
        "raw_not_larger_than_matched_compact": raw_chars <= compact_chars,
        "raw_match_slack_within_tolerance_or_exhausted": (
            slack <= int(prompt_view_config["raw_control"]["max_match_slack_chars"])
            or raw_exhausted
        ),
        "no_forbidden_public_keys": not forbidden_compact and not forbidden_raw,
        "compact_has_context_evidence": bool(compact_view.get("context_groups")),
        "compact_has_sequential_evidence": bool(
            compact_view.get("transitions")
            or compact_view.get("trace_patterns")
            or compact_view.get("prefix_automaton", {}).get("edges")
        ),
        "raw_hides_episode_identity_and_metadata": (
            "episode_id" not in raw_text and "metadata" not in raw_text
        ),
        "coverage_reported": "view_coverage" in compact_view and "view_coverage" in raw_view,
    }
    return {
        "passed": all(checks.values()),
        "policy_id": compact_view.get("policy_id"),
        "checks": checks,
        "compact_public_chars": compact_chars,
        "raw_public_chars": raw_chars,
        "match_slack_chars": slack,
        "raw_source_exhausted": raw_exhausted,
        "forbidden_key_paths": forbidden_compact + forbidden_raw,
    }


def build_protocol_manifest(protocol_path: str | Path, schema_paths: Iterable[str | Path] = ()) -> dict[str, Any]:
    protocol_file = Path(protocol_path)
    protocol = read_json(protocol_file)
    files = [{"path": str(protocol_file), "sha256": stable_hash(protocol)}]
    for schema_path in schema_paths:
        path = Path(schema_path)
        files.append({"path": str(path), "sha256": stable_hash(read_json(path))})
    return {
        "protocol_hash": stable_hash(protocol),
        "extractor_signature_objective_free": extractor_signature_is_objective_free(),
        "prompt_view_signatures_objective_free": prompt_view_signatures_are_objective_free(),
        "files": files,
    }
