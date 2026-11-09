from __future__ import annotations

import copy
import hashlib
import inspect
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, stable_hash
from .controlled_confirmatory import (
    _bootstrap_mean_difference,
    _forbidden_terms,
    _paired_sign_flip_p,
    _scenario_required_facts,
)
from .io import read_json, read_jsonl, write_json, write_jsonl
from .semantic_pilot import build_semantic_pair_prompt


SCENARIOS = ("C1", "C2")
RELATIONAL_ARMS = (
    "Aggregate-Control",
    "Relational-LinkBroken",
    "Relational-Linked",
    "FullTrace-Control",
)
PUBLIC_ARM_LABEL = "DeterministicBehavioralEvidence"
MISSING_VALUE = "<NOT_OBSERVED>"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, list[Any]]:
    """Encode only changed fields while distinguishing absent from null."""

    result: dict[str, list[Any]] = {}
    for key in sorted(set(previous) | set(current)):
        before = previous[key] if key in previous else MISSING_VALUE
        after = current[key] if key in current else MISSING_VALUE
        if before != after:
            result[str(key)] = [copy.deepcopy(before), copy.deepcopy(after)]
    return result


def _step_observation(
    step: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    state = dict(step.get("state", {}))
    predicates = dict(step.get("predicates", {}))
    if previous is None:
        state_relation = {"mode": "snapshot", "values": state}
        predicate_relation = {"mode": "snapshot", "values": predicates}
    else:
        state_relation = {
            "mode": "delta",
            "values": _mapping_delta(dict(previous.get("state", {})), state),
        }
        predicate_relation = {
            "mode": "delta",
            "values": _mapping_delta(
                dict(previous.get("predicates", {})), predicates
            ),
        }
    return {
        "events": copy.deepcopy(list(step.get("events", []))),
        "predicate_relation": predicate_relation,
        "reward": copy.deepcopy(step.get("reward", 0.0)),
        "state_relation": state_relation,
    }


def _compile_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = [dict(row) for row in episode.get("steps", [])]
    if not steps:
        raise ValueError("A FullTrace episode must contain at least one step")
    times = [int(row["t"]) for row in steps]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError("FullTrace step times must be unique and increasing")
    declared_length = int(episode.get("length", len(steps)))
    if declared_length != len(steps):
        raise ValueError("FullTrace episode length disagrees with its step count")

    relations = []
    previous: Mapping[str, Any] | None = None
    for step in steps:
        relations.append(
            {
                "action": copy.deepcopy(step.get("action", "unknown")),
                "observation": _step_observation(step, previous),
                "t": int(step["t"]),
            }
        )
        previous = step
    return {
        "context": copy.deepcopy(dict(episode.get("context", {}))),
        "behavior": {
            "length": declared_length,
            "outcome": str(episode.get("outcome", "unknown")),
            "relations": relations,
            "return": copy.deepcopy(episode.get("return", 0.0)),
            "terminated": bool(episode.get("terminated", False)),
            "truncated": bool(episode.get("truncated", False)),
        },
    }


class RelationalBehaviorCompiler:
    """Compile a requirement-blind, task-agnostic relational behavior record.

    The compiler accepts only already-public FullTrace evidence. It never sees a
    natural-language requirement, oracle label, policy identity, or judge output.
    """

    schema_version = "objective-agnostic-relational-behavior-record-v1"

    def compile(self, full_trace: Mapping[str, Any]) -> dict[str, Any]:
        if full_trace.get("representation") != "deterministic_raw_validation_trajectories":
            raise ValueError("Relational compilation requires frozen FullTrace evidence")
        raw_episodes = list(full_trace.get("episodes", []))
        if not raw_episodes:
            raise ValueError("Relational compilation requires at least one episode")

        counts: Counter[str] = Counter()
        values: dict[str, dict[str, Any]] = {}
        total_steps = 0
        for raw_episode in raw_episodes:
            episode = _compile_episode(raw_episode)
            key = canonical_json(episode)
            counts[key] += 1
            values[key] = episode
            total_steps += int(episode["behavior"]["length"])

        patterns = []
        for key in sorted(values):
            episode = copy.deepcopy(values[key])
            patterns.append(
                {
                    "context": episode["context"],
                    "behavior": episode["behavior"],
                    "multiplicity": counts[key],
                }
            )
        return {
            "representation": self.schema_version,
            "provenance": {
                "compiler": "RelationalBehaviorCompiler",
                "compiler_version": "1.0",
                "source_representation": "deterministic_raw_validation_trajectories",
                "source_scope": "public_validation_evidence_only",
                "objective_access": False,
                "state_timing": "post_action_observation",
            },
            "completeness": {
                "episode_set": "All episodes in the supplied public validation evidence are included.",
                "step_set": "All recorded steps are included in their recorded order before exact-pattern deduplication.",
                "action_event_reward_fields": "Every recorded action, event, and reward is retained.",
                "state_predicate_fields": "The first recorded snapshot and every subsequent observed field change are retained.",
                "zero_value_semantics": "Zero, false, null, and empty collections are observed values, not missing observations.",
                "missing_value_semantics": "Only <NOT_OBSERVED> denotes an absent endpoint in a field transition.",
            },
            "coverage": {
                "observed_episode_count": len(raw_episodes),
                "observed_step_count": total_steps,
                "unique_episode_pattern_count": len(patterns),
            },
            "semantics": {
                "observation_scope": "Each relation binds a recorded action to same-step reward, events, state information, and predicate information.",
                "relation_scope": "Relations report recorded adjacency and co-occurrence, not causal attribution.",
            },
            "patterns": patterns,
            "loss_ledger": {
                "omitted": [
                    "policy identity",
                    "episode identity",
                    "generator metadata",
                    "repeated unchanged state and predicate values",
                    "natural-language requirement",
                    "oracle and held-out outcomes",
                ],
                "retained": [
                    "all public validation contexts",
                    "all actions in order",
                    "all rewards and events",
                    "first recorded state and predicate snapshots",
                    "all subsequent state and predicate changes",
                    "all public episode outcomes, returns, lengths, and termination flags",
                    "multiplicity of duplicate episode patterns",
                ],
            },
        }


def validate_relational_record_contract(record: Mapping[str, Any]) -> list[str]:
    """Validate the frozen v1 contract without adding a JSON-Schema dependency."""

    errors: list[str] = []
    required_top = {
        "representation",
        "provenance",
        "completeness",
        "coverage",
        "semantics",
        "patterns",
        "loss_ledger",
    }
    if set(record) != required_top:
        errors.append("top_level_fields")
    if record.get("representation") != RelationalBehaviorCompiler.schema_version:
        errors.append("representation")

    provenance = record.get("provenance")
    expected_provenance = {
        "compiler": "RelationalBehaviorCompiler",
        "compiler_version": "1.0",
        "source_representation": "deterministic_raw_validation_trajectories",
        "source_scope": "public_validation_evidence_only",
        "objective_access": False,
        "state_timing": "post_action_observation",
    }
    if provenance != expected_provenance:
        errors.append("provenance")

    required_completeness = {
        "episode_set",
        "step_set",
        "action_event_reward_fields",
        "state_predicate_fields",
        "zero_value_semantics",
        "missing_value_semantics",
    }
    completeness = record.get("completeness")
    if not isinstance(completeness, Mapping) or set(completeness) != required_completeness:
        errors.append("completeness_fields")
    elif not all(isinstance(value, str) and value for value in completeness.values()):
        errors.append("completeness_values")

    coverage = record.get("coverage")
    required_coverage = {
        "observed_episode_count",
        "observed_step_count",
        "unique_episode_pattern_count",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != required_coverage:
        errors.append("coverage_fields")
        coverage = {}
    elif not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in coverage.values()
    ):
        errors.append("coverage_values")

    semantics = record.get("semantics")
    if not isinstance(semantics, Mapping) or set(semantics) != {
        "observation_scope",
        "relation_scope",
    }:
        errors.append("semantics_fields")
    elif not all(isinstance(value, str) and value for value in semantics.values()):
        errors.append("semantics_values")

    ledger = record.get("loss_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != {"retained", "omitted"}:
        errors.append("loss_ledger_fields")
    elif not all(
        isinstance(ledger[key], list)
        and all(isinstance(value, str) and value for value in ledger[key])
        for key in ("retained", "omitted")
    ):
        errors.append("loss_ledger_values")

    patterns = record.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        errors.append("patterns")
        patterns = []
    episode_total = 0
    step_total = 0
    for pattern_index, pattern in enumerate(patterns):
        prefix = f"pattern_{pattern_index}"
        if not isinstance(pattern, Mapping) or set(pattern) != {
            "context",
            "behavior",
            "multiplicity",
        }:
            errors.append(f"{prefix}_fields")
            continue
        if not isinstance(pattern["context"], Mapping):
            errors.append(f"{prefix}_context")
        multiplicity = pattern["multiplicity"]
        if not (
            isinstance(multiplicity, int)
            and not isinstance(multiplicity, bool)
            and multiplicity > 0
        ):
            errors.append(f"{prefix}_multiplicity")
            multiplicity = 0
        behavior = pattern["behavior"]
        required_behavior = {
            "length",
            "outcome",
            "relations",
            "return",
            "terminated",
            "truncated",
        }
        if not isinstance(behavior, Mapping) or set(behavior) != required_behavior:
            errors.append(f"{prefix}_behavior_fields")
            continue
        length = behavior["length"]
        if not (
            isinstance(length, int) and not isinstance(length, bool) and length > 0
        ):
            errors.append(f"{prefix}_length")
            length = 0
        if not isinstance(behavior["outcome"], str):
            errors.append(f"{prefix}_outcome")
        if isinstance(behavior["return"], bool) or not isinstance(
            behavior["return"], (int, float)
        ):
            errors.append(f"{prefix}_return")
        if not isinstance(behavior["terminated"], bool) or not isinstance(
            behavior["truncated"], bool
        ):
            errors.append(f"{prefix}_termination")
        relations = behavior["relations"]
        if not isinstance(relations, list) or len(relations) != length:
            errors.append(f"{prefix}_relations_length")
            relations = []
        times = []
        for relation_index, relation in enumerate(relations):
            relation_prefix = f"{prefix}_relation_{relation_index}"
            if not isinstance(relation, Mapping) or set(relation) != {
                "t",
                "action",
                "observation",
            }:
                errors.append(f"{relation_prefix}_fields")
                continue
            time = relation["t"]
            if not (
                isinstance(time, int) and not isinstance(time, bool) and time >= 0
            ):
                errors.append(f"{relation_prefix}_time")
            else:
                times.append(time)
            action = relation["action"]
            if isinstance(action, bool) or not isinstance(action, (str, int)):
                errors.append(f"{relation_prefix}_action")
            observation = relation["observation"]
            required_observation = {
                "events",
                "predicate_relation",
                "reward",
                "state_relation",
            }
            if not isinstance(observation, Mapping) or set(observation) != required_observation:
                errors.append(f"{relation_prefix}_observation_fields")
                continue
            events = observation["events"]
            if not isinstance(events, list) or not all(
                isinstance(value, str) for value in events
            ):
                errors.append(f"{relation_prefix}_events")
            if isinstance(observation["reward"], bool) or not isinstance(
                observation["reward"], (int, float)
            ):
                errors.append(f"{relation_prefix}_reward")
            for field in ("predicate_relation", "state_relation"):
                relation_value = observation[field]
                if not isinstance(relation_value, Mapping) or set(relation_value) != {
                    "mode",
                    "values",
                }:
                    errors.append(f"{relation_prefix}_{field}_fields")
                    continue
                if relation_value["mode"] not in {"snapshot", "delta"}:
                    errors.append(f"{relation_prefix}_{field}_mode")
                values = relation_value["values"]
                if not isinstance(values, Mapping):
                    errors.append(f"{relation_prefix}_{field}_values")
                elif relation_value["mode"] == "delta" and not all(
                    isinstance(value, list) and len(value) == 2
                    for value in values.values()
                ):
                    errors.append(f"{relation_prefix}_{field}_delta")
                expected_mode = "snapshot" if relation_index == 0 else "delta"
                if relation_value["mode"] != expected_mode:
                    errors.append(f"{relation_prefix}_{field}_sequence_mode")
        if times and (times != sorted(times) or len(times) != len(set(times))):
            errors.append(f"{prefix}_time_order")
        episode_total += int(multiplicity)
        step_total += int(multiplicity) * int(length)

    if coverage:
        if coverage.get("unique_episode_pattern_count") != len(patterns):
            errors.append("coverage_pattern_count")
        if coverage.get("observed_episode_count") != episode_total:
            errors.append("coverage_episode_count")
        if coverage.get("observed_step_count") != step_total:
            errors.append("coverage_step_count")
    return errors


def _rotate(values: Sequence[Any], shift: int = 1) -> list[Any]:
    rows = [copy.deepcopy(value) for value in values]
    if len(rows) <= 1:
        return rows
    offset = shift % len(rows)
    if offset == 0:
        offset = 1
    return rows[offset:] + rows[:offset]


def relation_broken_control(linked: Mapping[str, Any]) -> dict[str, Any]:
    """Rewire links while preserving every displayed atomic fact and byte budget.

    Complete behavior payloads are rotated across context rows. Within each
    behavior, interior action labels are rotated across fixed time/observation
    rows while boundary actions remain fixed. Snapshot/delta structure therefore
    stays valid and cannot reveal the control through a formatting artifact.
    """

    result = copy.deepcopy(dict(linked))
    if result.get("representation") != RelationalBehaviorCompiler.schema_version:
        raise ValueError("Link breaking requires a relational behavior record")
    patterns = list(result.get("patterns", []))
    behaviors = _rotate([row["behavior"] for row in patterns])
    for row, behavior in zip(patterns, behaviors):
        relations = list(behavior.get("relations", []))
        interior = relations[1:-1]
        actions = _rotate([relation["action"] for relation in interior])
        for relation, action in zip(interior, actions):
            relation["action"] = action
        behavior["relations"] = relations
        row["behavior"] = behavior
    result["patterns"] = patterns
    return result


def _pattern_components(record: Mapping[str, Any]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {
        "contexts": Counter(),
        "multiplicities": Counter(),
        "behavior_summaries": Counter(),
        "actions": Counter(),
        "times": Counter(),
        "observations": Counter(),
    }
    for pattern in record.get("patterns", []):
        result["contexts"][canonical_json(pattern.get("context", {}))] += 1
        result["multiplicities"][canonical_json(pattern.get("multiplicity"))] += 1
        behavior = dict(pattern.get("behavior", {}))
        relations = list(behavior.pop("relations", []))
        result["behavior_summaries"][canonical_json(behavior)] += 1
        for relation in relations:
            result["actions"][canonical_json(relation.get("action"))] += 1
            result["times"][canonical_json(relation.get("t"))] += 1
            result["observations"][canonical_json(relation.get("observation", {}))] += 1
    return result


def audit_relation_break(
    linked: Mapping[str, Any], broken: Mapping[str, Any]
) -> dict[str, Any]:
    linked_patterns = list(linked.get("patterns", []))
    broken_patterns = list(broken.get("patterns", []))
    linked_components = _pattern_components(linked)
    broken_components = _pattern_components(broken)
    component_matches = {
        key: linked_components[key] == broken_components[key]
        for key in linked_components
    }

    context_behavior_changed = 0
    action_observation_changed = 0
    eligible_action_bindings = 0
    for source, transformed in zip(linked_patterns, broken_patterns):
        source_behavior = dict(source.get("behavior", {}))
        transformed_behavior = dict(transformed.get("behavior", {}))
        source_relations = list(source_behavior.pop("relations", []))
        transformed_relations = list(transformed_behavior.pop("relations", []))
        source_signature = {
            "summary": source_behavior,
            "action_multiset": sorted(
                canonical_json(row.get("action")) for row in source_relations
            ),
            "time_axis": [row.get("t") for row in source_relations],
            "observation_multiset": sorted(
                canonical_json(row.get("observation", {}))
                for row in source_relations
            ),
        }
        transformed_signature = {
            "summary": transformed_behavior,
            "action_multiset": sorted(
                canonical_json(row.get("action")) for row in transformed_relations
            ),
            "time_axis": [row.get("t") for row in transformed_relations],
            "observation_multiset": sorted(
                canonical_json(row.get("observation", {}))
                for row in transformed_relations
            ),
        }
        if canonical_json(source_signature) != canonical_json(transformed_signature):
            context_behavior_changed += 1
        for source_relation, transformed_relation in zip(
            source_relations, transformed_relations
        ):
            eligible_action_bindings += 1
            if canonical_json(
                {
                    "action": source_relation.get("action"),
                    "observation": source_relation.get("observation", {}),
                }
            ) != canonical_json(
                {
                    "action": transformed_relation.get("action"),
                    "observation": transformed_relation.get("observation", {}),
                }
            ):
                action_observation_changed += 1

    linked_top = {key: value for key, value in linked.items() if key != "patterns"}
    broken_top = {key: value for key, value in broken.items() if key != "patterns"}
    return {
        "top_level_schema_and_ledger_exact": linked_top == broken_top,
        "pattern_count_exact": len(linked_patterns) == len(broken_patterns),
        "context_multiset_exact": component_matches["contexts"],
        "multiplicity_multiset_exact": component_matches["multiplicities"],
        "behavior_summary_multiset_exact": component_matches["behavior_summaries"],
        "action_multiset_exact": component_matches["actions"],
        "time_axis_multiset_exact": component_matches["times"],
        "observation_payload_multiset_exact": component_matches["observations"],
        "serialized_character_count_exact": len(canonical_json(linked))
        == len(canonical_json(broken)),
        "at_least_one_relation_assignment_changed": bool(
            context_behavior_changed or action_observation_changed
        ),
        "changed_context_behavior_assignments": context_behavior_changed,
        "changed_action_observation_assignments": action_observation_changed,
        "eligible_action_observation_assignments": eligible_action_bindings,
        "linked_characters": len(canonical_json(linked)),
        "broken_characters": len(canonical_json(broken)),
    }


def _expected_winner(oracle: Mapping[str, Any], order: str) -> str:
    if oracle["status"] == "AMBIGUOUS":
        return "ABSTAIN"
    label = str(oracle["label"])
    if label == "TIE":
        return "TIE"
    if order == "AB":
        return label
    return "B" if label == "A" else "A"


def _audit_boolean_fields(audit: Mapping[str, Any]) -> bool:
    ignored = {
        "changed_context_behavior_assignments",
        "changed_action_observation_assignments",
        "eligible_action_observation_assignments",
        "linked_characters",
        "broken_characters",
    }
    return all(bool(value) for key, value in audit.items() if key not in ignored)


RELATIONAL_BUNDLE_FILES = (
    "public_evidence.jsonl",
    "judge_requests.jsonl",
    "judge_private_map.jsonl",
    "relational_integrity_audit.json",
    "summary.json",
)


def validate_controlled_relational_bundle(
    bundle_dir: str | Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed if a frozen relational bundle is partial or has drifted."""

    bundle = Path(bundle_dir)
    present = {name: (bundle / name).is_file() for name in RELATIONAL_BUNDLE_FILES}
    if not all(present.values()):
        missing = sorted(name for name, exists in present.items() if not exists)
        raise RuntimeError("Incomplete relational bundle: " + ", ".join(missing))
    summary = read_json(bundle / "summary.json")
    evidence = read_jsonl(bundle / "public_evidence.jsonl")
    requests = read_jsonl(bundle / "judge_requests.jsonl")
    private_map = read_jsonl(bundle / "judge_private_map.jsonl")
    audit = read_json(bundle / "relational_integrity_audit.json")
    checks = {
        "status": summary.get("status") == "READY_FOR_RELATIONAL_JUDGE",
        "plan_hash": summary.get("plan_hash") == stable_hash(plan),
        "request_hash": summary.get("request_hash") == stable_hash(requests),
        "private_map_hash": summary.get("private_map_hash")
        == stable_hash(private_map),
        "public_evidence_hash": summary.get("public_evidence_hash")
        == stable_hash(evidence),
        "integrity_audit_hash": summary.get("integrity_audit_hash")
        == stable_hash(audit),
        "integrity_audit_passed": audit.get("passed") is True,
        "all_summary_gates_passed": all(
            bool(value) for value in summary.get("gates", {}).values()
        ),
        "request_count": len(requests)
        == int(plan["design"]["expected_total_requests"]),
        "private_count": len(private_map) == len(requests),
        "request_id_alignment": {
            str(row.get("request_id")) for row in requests
        }
        == {str(row.get("request_id")) for row in private_map},
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("Frozen relational bundle validation failed: " + ", ".join(failed))
    return dict(summary)


def build_controlled_relational_record(
    *, output_dir: str | Path, plan_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repository_root)
    plan = read_json(plan_path)
    source = root / str(plan["source_bundle"])
    scenario_config_path = root / str(plan["scenario_config"])
    record_schema_path = root / str(plan["record_schema"])
    scenario_config = read_json(scenario_config_path)
    compiler = RelationalBehaviorCompiler()

    source_hash_checks = {
        relative: _file_sha256(root / relative) == str(expected).lower()
        for relative, expected in plan["source_hashes_sha256"].items()
    }
    config_hash_matches = _file_sha256(scenario_config_path) == str(
        plan["scenario_config_file_sha256"]
    ).lower()
    record_schema_hash_matches = _file_sha256(record_schema_path) == str(
        plan["record_schema_file_sha256"]
    ).lower()
    existing = {
        name: (output / name).is_file() for name in RELATIONAL_BUNDLE_FILES
    }
    if any(existing.values()):
        if not all(existing.values()):
            missing = sorted(name for name, present in existing.items() if not present)
            raise RuntimeError(
                "Refusing to overwrite a partial relational bundle: "
                + ", ".join(missing)
            )
        if not (
            all(source_hash_checks.values())
            and config_hash_matches
            and record_schema_hash_matches
        ):
            raise RuntimeError("Frozen relational source or schema hash drift detected")
        return validate_controlled_relational_bundle(output, plan)

    all_requests: list[dict[str, Any]] = []
    all_private: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    relation_audits: list[dict[str, Any]] = []
    schema_contract_audits: list[dict[str, Any]] = []
    compression_ratios: list[float] = []
    policy_ids: set[str] = set()
    pair_count = 0
    required_fact_gate = True
    forbidden_term_gate = True

    for scenario in SCENARIOS:
        scenario_source = source / scenario
        source_rows = read_jsonl(scenario_source / "public_evidence.jsonl")
        by_policy: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in source_rows:
            by_policy[str(row["policy_id"])][str(row["arm"])] = copy.deepcopy(
                row["public_evidence"]
            )
        blueprint = read_json(scenario_source / "pair_manifest_private.json")
        oracle = read_json(scenario_source / "oracle_adjudication.json")
        requirements = scenario_config["scenarios"][scenario]["requirement_paraphrases"]
        pair_count += len(blueprint)
        policy_ids.update(by_policy)

        evidence_by_policy: dict[str, dict[str, Any]] = {}
        for policy_id, source_arms in sorted(by_policy.items()):
            aggregate = source_arms["B1"]
            full = source_arms["B2-FullTrace"]
            linked = compiler.compile(full)
            broken = relation_broken_control(linked)
            linked_contract_errors = validate_relational_record_contract(linked)
            broken_contract_errors = validate_relational_record_contract(broken)
            schema_contract_audits.append(
                {
                    "scenario_id": scenario,
                    "policy_id": policy_id,
                    "linked_errors": linked_contract_errors,
                    "broken_errors": broken_contract_errors,
                    "passed": not linked_contract_errors
                    and not broken_contract_errors,
                }
            )
            audit = audit_relation_break(linked, broken)
            relation_audits.append(
                {"scenario_id": scenario, "policy_id": policy_id, **audit}
            )
            compression_ratios.append(
                len(canonical_json(linked)) / len(canonical_json(full))
            )
            evidence_by_policy[policy_id] = {
                "Aggregate-Control": aggregate,
                "Relational-LinkBroken": broken,
                "Relational-Linked": linked,
                "FullTrace-Control": full,
            }
            for arm, evidence in evidence_by_policy[policy_id].items():
                all_evidence.append(
                    {
                        "scenario_id": scenario,
                        "policy_id": policy_id,
                        "arm": arm,
                        "public_evidence": evidence,
                        "characters": len(canonical_json(evidence)),
                        "evidence_hash": stable_hash(evidence),
                    }
                )

        required = tuple(term.lower() for term in _scenario_required_facts(scenario))
        forbidden = tuple(term.lower() for term in _forbidden_terms(scenario))
        for pair in blueprint:
            left = str(pair["policy_a"])
            right = str(pair["policy_b"])
            linked_text = (
                canonical_json(evidence_by_policy[left]["Relational-Linked"])
                + canonical_json(evidence_by_policy[right]["Relational-Linked"])
            ).lower()
            required_fact_gate = required_fact_gate and all(
                term in linked_text for term in required
            )
            forbidden_term_gate = forbidden_term_gate and not any(
                term in linked_text for term in forbidden
            )

            pair_id = str(pair["pair_id"])
            pair_oracle = oracle[pair_id]
            for arm in RELATIONAL_ARMS:
                for paraphrase_index, requirement in enumerate(requirements):
                    paraphrase = f"P{paraphrase_index}"
                    for order in ("AB", "BA"):
                        alias_a, alias_b = (
                            (left, right) if order == "AB" else (right, left)
                        )
                        prompt = build_semantic_pair_prompt(
                            requirement,
                            evidence_by_policy[alias_a][arm],
                            evidence_by_policy[alias_b][arm],
                            arm=PUBLIC_ARM_LABEL,
                        )
                        for repeat in range(int(plan["design"]["repeats_per_cell"])):
                            request_id = (
                                f"{plan['request_namespace']}__{scenario}__{pair_id}__"
                                f"{arm}__{paraphrase}__{order}__R{repeat}"
                            )
                            all_requests.append(
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
                            all_private.append(
                                {
                                    "request_id": request_id,
                                    "scenario_id": scenario,
                                    "pair_id": pair_id,
                                    "arm": arm,
                                    "paraphrase": paraphrase,
                                    "order": order,
                                    "repeat": repeat,
                                    "alias_a_policy_id": alias_a,
                                    "alias_b_policy_id": alias_b,
                                    "oracle_status": pair_oracle["status"],
                                    "expected_winner": _expected_winner(pair_oracle, order),
                                }
                            )

    prompts = [str(row["prompt"]) for row in all_requests]
    hidden_literals = {
        "oracle_status",
        "expected_winner",
        "semantic_violation_rate",
        "preferred policy",
        "correct_strategy",
    }
    no_private_leakage = not any(
        literal in prompt.lower() for prompt in prompts for literal in hidden_literals
    ) and not any(policy_id in prompt for prompt in prompts for policy_id in policy_ids)
    all_audits_pass = all(_audit_boolean_fields(row) for row in relation_audits)
    expected_requests = int(plan["design"]["expected_total_requests"])
    objective_blind = not {
        "objective",
        "requirement",
        "preference",
        "oracle",
        "judge_response",
    }.intersection(inspect.signature(RelationalBehaviorCompiler.compile).parameters)
    gates = {
        "source_hashes_match": all(source_hash_checks.values())
        and config_hash_matches
        and record_schema_hash_matches,
        "all_96_source_pairs_retained": pair_count == 96,
        "all_four_arms_present": set(row["arm"] for row in all_requests)
        == set(RELATIONAL_ARMS),
        "expected_request_count": len(all_requests) == expected_requests,
        "compiler_requirement_blind_by_signature": objective_blind,
        "linked_vs_broken_fact_and_length_audits_pass": all_audits_pass,
        "all_relational_records_match_frozen_schema_contract": all(
            bool(row["passed"]) for row in schema_contract_audits
        ),
        "required_public_facts_retained": required_fact_gate,
        "forbidden_semantic_labels_absent": forbidden_term_gate,
        "relational_record_is_smaller_than_fulltrace_for_every_policy": max(
            compression_ratios, default=2.0
        )
        < 1.0,
        "maximum_prompt_characters_within_frozen_budget": max(
            map(len, prompts), default=0
        )
        <= int(plan["offline_gates"]["maximum_prompt_characters"]),
        "no_hidden_oracle_or_policy_identity_leakage": no_private_leakage,
    }

    write_jsonl(output / "public_evidence.jsonl", all_evidence)
    write_jsonl(output / "judge_requests.jsonl", all_requests)
    write_jsonl(output / "judge_private_map.jsonl", all_private)
    changed_action_links = sum(
        int(row["changed_action_observation_assignments"])
        for row in relation_audits
    )
    eligible_action_links = sum(
        int(row["eligible_action_observation_assignments"])
        for row in relation_audits
    )
    integrity_audit = {
            "schema_version": "controlled-relational-integrity-audit-v1",
            "source_hash_checks": source_hash_checks,
            "scenario_config_hash_matches": config_hash_matches,
            "record_schema_path": str(plan["record_schema"]),
            "record_schema_hash_matches": record_schema_hash_matches,
            "policy_audits": relation_audits,
            "schema_contract_audits": schema_contract_audits,
            "relation_rewiring_summary": {
                "policy_count": len(relation_audits),
                "policies_with_at_least_one_changed_relation": sum(
                    bool(row["at_least_one_relation_assignment_changed"])
                    for row in relation_audits
                ),
                "policies_with_changed_context_behavior_assignment": sum(
                    int(row["changed_context_behavior_assignments"]) > 0
                    for row in relation_audits
                ),
                "changed_action_observation_assignments": changed_action_links,
                "eligible_action_observation_assignments": eligible_action_links,
                "changed_action_observation_fraction": (
                    changed_action_links / eligible_action_links
                    if eligible_action_links
                    else None
                ),
            },
            "compression_ratio_linked_over_fulltrace": {
                "minimum": min(compression_ratios, default=None),
                "mean": fmean(compression_ratios) if compression_ratios else None,
                "maximum": max(compression_ratios, default=None),
            },
            "minimum_prompt_characters": min(map(len, prompts), default=0),
            "maximum_prompt_characters": max(map(len, prompts), default=0),
            "gates": gates,
            "passed": all(gates.values()),
        }
    write_json(output / "relational_integrity_audit.json", integrity_audit)
    summary = {
        "schema_version": "controlled-relational-record-bundle-v1",
        "status": "READY_FOR_RELATIONAL_JUDGE"
        if all(gates.values())
        else "OFFLINE_GATE_FAILED",
        "pair_count": pair_count,
        "policy_count": len(policy_ids),
        "arms": list(RELATIONAL_ARMS),
        "request_count": len(all_requests),
        "request_hash": stable_hash(all_requests),
        "private_map_hash": stable_hash(all_private),
        "public_evidence_hash": stable_hash(all_evidence),
        "integrity_audit_hash": stable_hash(integrity_audit),
        "plan_hash": stable_hash(plan),
        "gates": gates,
        "claim_boundary": plan["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "prepare",
            "origin_date": "2026-09-01",
            "verification_status": "VERIFIED_OFFLINE"
            if all(gates.values())
            else "FAILED",
            "version_label": str(plan["version_label"]),
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def _pair_effects(
    pair_rows: Sequence[Mapping[str, Any]], reference: str, comparator: str
) -> tuple[list[float], dict[str, list[float]]]:
    effects: list[float] = []
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for scenario in SCENARIOS:
        rows = [
            row
            for row in pair_rows
            if row["scenario_id"] == scenario and row["determinate"]
        ]
        index = {(str(row["pair_id"]), str(row["arm"])): row for row in rows}
        for pair_id in sorted({str(row["pair_id"]) for row in rows}):
            effect = float(index[(pair_id, reference)]["oracle_label_mass"]) - float(
                index[(pair_id, comparator)]["oracle_label_mass"]
            )
            effects.append(effect)
            by_scenario[scenario].append(effect)
    return effects, by_scenario


def evaluate_controlled_relational_record(
    *,
    private_map: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    private_rows = [dict(row) for row in private_map]
    response_rows = [dict(row) for row in responses]
    response_ids = [str(row["request_id"]) for row in response_rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Relational responses contain duplicate request IDs")
    private_ids = {str(row["request_id"]) for row in private_rows}
    unexpected = sorted(set(response_ids) - private_ids)
    if unexpected:
        raise ValueError(
            "Relational responses contain request IDs outside the frozen bundle: "
            + ", ".join(unexpected[:5])
        )
    response_by_id = {str(row["request_id"]): row for row in response_rows}
    calls = []
    for row in private_rows:
        response = response_by_id.get(str(row["request_id"]))
        parsed = response.get("response") if response else None
        observed = parsed.get("winner") if isinstance(parsed, Mapping) else None
        valid = observed in {"A", "B", "TIE", "ABSTAIN"}
        calls.append(
            {
                **row,
                "observed_winner": observed,
                "valid": valid,
                "correct": bool(valid and observed == row["expected_winner"]),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        grouped[(str(row["scenario_id"]), str(row["pair_id"]), str(row["arm"]))].append(row)
    pair_rows = []
    for (scenario, pair_id, arm), rows in sorted(grouped.items()):
        pair_rows.append(
            {
                "scenario_id": scenario,
                "pair_id": pair_id,
                "arm": arm,
                "oracle_status": rows[0]["oracle_status"],
                "determinate": rows[0]["oracle_status"] == "DETERMINATE",
                "request_count": len(rows),
                "valid_request_count": sum(row["valid"] for row in rows),
                "oracle_label_mass": fmean(row["correct"] for row in rows),
                "strict_all_context_correct": len(rows) == 8
                and all(row["correct"] for row in rows),
            }
        )

    by_scenario: dict[str, Any] = {}
    cell_valid_rates: dict[str, float] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in pair_rows if row["scenario_id"] == scenario]
        determinate = [row for row in scenario_rows if row["determinate"]]
        by_arm = {}
        for arm in RELATIONAL_ARMS:
            arm_rows = [row for row in determinate if row["arm"] == arm]
            all_arm_rows = [row for row in scenario_rows if row["arm"] == arm]
            denominator = sum(row["request_count"] for row in all_arm_rows)
            valid_rate = (
                sum(row["valid_request_count"] for row in all_arm_rows) / denominator
                if denominator
                else 0.0
            )
            cell_valid_rates[f"{scenario}::{arm}"] = valid_rate
            by_arm[arm] = {
                "determinate_pair_count": len(arm_rows),
                "context_marginalized_oracle_label_mass": fmean(
                    row["oracle_label_mass"] for row in arm_rows
                )
                if arm_rows
                else None,
                "strict_all_context_correct_pairs": sum(
                    row["strict_all_context_correct"] for row in arm_rows
                ),
                "valid_request_rate": valid_rate,
            }
        by_scenario[scenario] = {"by_arm": by_arm}

    contrast_specs = {
        "primary_relational_link_contribution": (
            "Relational-Linked",
            "Relational-LinkBroken",
        ),
        "secondary_relational_over_aggregate": (
            "Relational-Linked",
            "Aggregate-Control",
        ),
        "secondary_fulltrace_over_relational": (
            "FullTrace-Control",
            "Relational-Linked",
        ),
        "secondary_fulltrace_over_aggregate": (
            "FullTrace-Control",
            "Aggregate-Control",
        ),
    }
    contrasts = {}
    for index, (name, (reference, comparator)) in enumerate(contrast_specs.items()):
        effects, scenario_effects = _pair_effects(pair_rows, reference, comparator)
        contrasts[name] = {
            "effect_definition": f"{reference} minus {comparator}; positive favors {reference}",
            "determinate_pair_count": len(effects),
            "mean_difference": fmean(effects) if effects else None,
            "pair_bootstrap_95_ci": _bootstrap_mean_difference(
                effects, seed=20260901 + index
            )
            if effects
            else None,
            "paired_sign_flip_p": _paired_sign_flip_p(
                effects, seed=20260911 + index
            )
            if effects
            else None,
            "reference_better_pairs": sum(value > 0 for value in effects),
            "equal_pairs": sum(value == 0 for value in effects),
            "comparator_better_pairs": sum(value < 0 for value in effects),
            "scenario_mean_differences_descriptive": {
                scenario: fmean(values) if values else None
                for scenario, values in scenario_effects.items()
            },
        }

    primary = contrasts["primary_relational_link_contribution"]
    alpha = float(plan["primary_analysis"]["alpha"])
    quality_gates = {
        "expected_request_cells": len(private_rows)
        == int(plan["design"]["expected_total_requests"]),
        "all_384_pair_arm_cells_present": len(pair_rows) == 384,
        "minimum_scenario_arm_valid_rate_met": min(
            cell_valid_rates.values(), default=0.0
        )
        >= float(
            plan["primary_analysis"][
                "minimum_valid_response_rate_per_scenario_arm"
            ]
        ),
    }
    primary_statistical_condition_met = bool(
        primary["mean_difference"] is not None
        and primary["mean_difference"] > 0
        and primary["paired_sign_flip_p"] is not None
        and primary["paired_sign_flip_p"] < alpha
    )
    support = {
        "primary_statistical_condition_met": primary_statistical_condition_met,
        "relational_link_contribution_supported": bool(
            all(quality_gates.values()) and primary_statistical_condition_met
        ),
        "interpretation_if_not_supported": (
            "Do not claim that preserved relation bindings improve semantic policy selection."
        ),
    }
    return {
        "schema_version": "controlled-relational-record-evaluation-v1",
        "status": "RELATIONAL_EXPERIMENT_COMPLETE"
        if all(quality_gates.values())
        else "RELATIONAL_EXPERIMENT_QUALITY_GATE_FAILED",
        "response_count": len(response_by_id),
        "expected_request_count": len(private_rows),
        "by_scenario": by_scenario,
        "contrasts": contrasts,
        "support": support,
        "quality_gates": quality_gates,
        "cell_valid_rates": cell_valid_rates,
        "pair_rows": pair_rows,
        "claim_boundary": plan["claim_boundary"],
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-09-01",
            "verification_status": "ANALYZED",
            "version_label": str(plan["version_label"]),
        },
    }
