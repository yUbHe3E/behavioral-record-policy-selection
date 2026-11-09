from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, stable_hash
from .predictive_record import (
    predictive_compiler_signature_is_objective_free,
    reconstruct_trace_variant,
)
from .views import public_evidence_chars


@dataclass(frozen=True)
class Fact:
    tier: str
    family: str
    arguments_json: str
    value: float

    @classmethod
    def make(
        cls, tier: str, family: str, arguments: Mapping[str, Any], value: int | float
    ) -> "Fact":
        return cls(
            tier=tier,
            family=family,
            arguments_json=canonical_json(arguments),
            value=float(value),
        )

    @property
    def fact_id(self) -> str:
        return stable_hash(["tpbr-fact-v0", self.tier, self.family, self.arguments_json], 24)


def _distribution_facts(
    facts: list[Fact],
    tier: str,
    family_prefix: str,
    arguments: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> None:
    for statistic in ("count", "mean", "min", "max"):
        value = distribution.get(statistic)
        if value is not None:
            facts.append(
                Fact.make(
                    tier,
                    f"{family_prefix}_{statistic}",
                    arguments,
                    value,
                )
            )
    for probability, value in distribution.get("quantiles", []):
        if value is not None:
            facts.append(
                Fact.make(
                    tier,
                    f"{family_prefix}_quantile",
                    dict(arguments) | {"probability": float(probability)},
                    value,
                )
            )


def profile_facts(profile: Mapping[str, Any]) -> list[Fact]:
    facts: list[Fact] = []
    for name in ("episodes", "steps", "typed_symbol_occurrences", "contexts"):
        facts.append(Fact.make("core", "sample", {"field": name}, profile["sample"][name]))
    for row in profile["outcomes"]:
        facts.append(Fact.make("core", "outcome_rate", {"outcome": row["name"]}, row["rate"]))
        facts.append(Fact.make("core", "outcome_count", {"outcome": row["name"]}, row["count"]))
    for row in profile["symbol_profiles"]:
        symbol = row["symbol"]
        arguments = {"symbol": symbol}
        for family, key in (
            ("symbol_occurrence_count", "occurrence_count"),
            ("symbol_episode_count", "episode_count"),
            ("symbol_episode_rate", "episode_rate"),
            ("symbol_repeat_rate_ge_2", "repeat_rate_ge_2"),
            ("symbol_repeat_rate_ge_3", "repeat_rate_ge_3"),
        ):
            facts.append(Fact.make("core", family, arguments, row[key]))
        _distribution_facts(facts, "core", "symbol_count", arguments, row["count_per_episode"])
        _distribution_facts(
            facts,
            "core",
            "symbol_first_hit_given_present",
            arguments,
            row["first_hit_step_given_present"],
        )
        for horizon, value in row["first_hit_cdf"]:
            facts.append(
                Fact.make(
                    "core",
                    "symbol_first_hit_cdf",
                    {"symbol": symbol, "horizon": int(horizon)},
                    value,
                )
            )
        for gamma, value in row["discounted_count_mean"]:
            facts.append(
                Fact.make(
                    "core",
                    "symbol_discounted_count_mean",
                    {"symbol": symbol, "gamma": float(gamma)},
                    value,
                )
            )
        for outcome, count, rate in row["outcome_given_present"]:
            facts.append(
                Fact.make(
                    "core",
                    "outcome_given_symbol_count",
                    {"symbol": symbol, "outcome": outcome},
                    count,
                )
            )
            facts.append(
                Fact.make(
                    "core",
                    "outcome_given_symbol_rate",
                    {"symbol": symbol, "outcome": outcome},
                    rate,
                )
            )
    for row in profile["signal_profiles"]:
        signal = row["signal"]
        arguments = {"signal": signal}
        facts.append(Fact.make("core", "signal_sample_count", arguments, row["sample_count"]))
        facts.append(
            Fact.make("core", "signal_missing_episode_count", arguments, row["missing_episode_count"])
        )
        if row["kind"] == "categorical":
            for value, count in row["values"]:
                facts.append(
                    Fact.make(
                        "core",
                        "signal_categorical_count",
                        {"signal": signal, "value": value},
                        count,
                    )
                )
            facts.append(Fact.make("core", "signal_residual_count", arguments, row["residual_count"]))
        else:
            _distribution_facts(facts, "core", "signal_value", arguments, row["value"])
            if "episode_mean" in row:
                _distribution_facts(
                    facts, "extended", "signal_episode_mean", arguments, row["episode_mean"]
                )
            if "delta" in row:
                _distribution_facts(facts, "extended", "signal_delta", arguments, row["delta"])
            facts.append(Fact.make("core", "signal_change_rate", arguments, row["change_rate"]))
    for row in profile["context_profiles"]:
        base = {"field": row["field"], "value": row["value"]}
        facts.append(Fact.make("core", "context_episode_count", base, row["episode_count"]))
        for outcome, count, rate in row["outcome_rates"]:
            facts.append(
                Fact.make("core", "context_outcome_count", base | {"outcome": outcome}, count)
            )
            facts.append(
                Fact.make("core", "context_outcome_rate", base | {"outcome": outcome}, rate)
            )
        for symbol, count, rate in row["symbol_episode_rates"]:
            facts.append(
                Fact.make("extended", "context_symbol_episode_count", base | {"symbol": symbol}, count)
            )
            facts.append(
                Fact.make("extended", "context_symbol_episode_rate", base | {"symbol": symbol}, rate)
            )
    for row in profile["temporal_profiles"]:
        base = {"left": row["left"], "right": row["right"]}
        for family, key in (
            ("temporal_activations", "activations"),
            ("temporal_activation_episodes", "activation_episodes"),
            ("temporal_satisfied_eventually", "satisfied_eventually"),
            ("temporal_violations", "violations"),
            ("temporal_vacuous_episodes", "vacuous_episodes"),
            ("temporal_response_rate", "response_rate"),
            ("temporal_both_episode_count", "both_episode_count"),
        ):
            facts.append(Fact.make("extended", family, base, row[key]))
        for horizon, value in row["within_response_rate"]:
            facts.append(
                Fact.make(
                    "extended",
                    "temporal_within_response_rate",
                    base | {"horizon": int(horizon)},
                    value,
                )
            )
        for order_name, value in row["first_order_counts"].items():
            facts.append(
                Fact.make(
                    "extended",
                    "temporal_first_order_count",
                    base | {"order": order_name},
                    value,
                )
            )
    for row in profile["conditional_dynamics"]:
        base = {"context": row["context"]}
        facts.append(Fact.make("extended", "conditional_support", base, row["support"]))
        facts.append(
            Fact.make(
                "extended", "conditional_information_gain_bits", base, row["information_gain_bits"]
            )
        )
        for symbol, count, rate in row["next"]:
            facts.append(
                Fact.make("extended", "conditional_next_count", base | {"next": symbol}, count)
            )
            facts.append(
                Fact.make("extended", "conditional_next_rate", base | {"next": symbol}, rate)
            )
        facts.append(
            Fact.make("extended", "conditional_residual_next_count", base, row["residual_next_count"])
        )
    return facts


def _expand_distribution(row: list[Any]) -> dict[str, Any]:
    return {
        "count": row[0],
        "mean": row[1],
        "min": row[2],
        "max": row[3],
        "quantiles": row[4],
    }


def view_facts(view: Mapping[str, Any]) -> list[Fact]:
    profile: dict[str, Any] = {
        "sample": view["sample"],
        "outcomes": [
            {"name": row[0], "count": row[1], "rate": row[2], "rate_ci95": row[3]}
            for row in view["outcomes"]
        ],
        "symbol_profiles": [],
        "signal_profiles": [],
        "context_profiles": [],
        "temporal_profiles": [],
        "conditional_dynamics": [],
    }
    for row in view["symbols"]:
        profile["symbol_profiles"].append(
            {
                "symbol": row[0],
                "category": row[1],
                "occurrence_count": row[2],
                "episode_count": row[3],
                "episode_rate": row[4],
                "episode_rate_ci95": row[5],
                "count_per_episode": _expand_distribution(row[6]),
                "first_hit_step_given_present": _expand_distribution(row[7]),
                "first_hit_cdf": row[8],
                "repeat_rate_ge_2": row[9],
                "repeat_rate_ge_3": row[10],
                "discounted_count_mean": row[11],
                "outcome_given_present": row[12],
            }
        )
    for row in view["signals"]:
        if row[1] == "categorical":
            profile["signal_profiles"].append(
                {
                    "signal": row[0],
                    "kind": row[1],
                    "sample_count": row[2],
                    "missing_episode_count": row[3],
                    "values": row[4],
                    "residual_count": row[5],
                }
            )
        else:
            profile["signal_profiles"].append(
                {
                    "signal": row[0],
                    "kind": row[1],
                    "sample_count": row[2],
                    "missing_episode_count": row[3],
                    "value": _expand_distribution(row[4]),
                    "change_rate": row[5],
                }
            )
    for row in view["contexts"]:
        profile["context_profiles"].append(
            {
                "field": row[0],
                "value": row[1],
                "episode_count": row[2],
                "outcome_rates": row[3],
                "symbol_episode_rates": row[4],
            }
        )
    for row in view["temporal"]:
        profile["temporal_profiles"].append(
            {
                "left": row[0],
                "right": row[1],
                "activations": row[2],
                "activation_episodes": row[3],
                "satisfied_eventually": row[4],
                "violations": row[5],
                "vacuous_episodes": row[6],
                "response_rate": row[7],
                "within_response_rate": row[8],
                "both_episode_count": row[9],
                "first_order_counts": row[10],
            }
        )
    for row in view["conditional_dynamics"]:
        profile["conditional_dynamics"].append(
            {
                "context": row[0],
                "order": row[1],
                "support": row[2],
                "information_gain_bits": row[3],
                "next": row[4],
                "residual_next_count": row[5],
            }
        )
    return profile_facts(profile)


def _fact_map(facts: Iterable[Fact]) -> dict[str, Fact]:
    result: dict[str, Fact] = {}
    for fact in facts:
        if fact.fact_id in result and result[fact.fact_id] != fact:
            raise ValueError(f"Conflicting TPBR fact {fact.fact_id}")
        result[fact.fact_id] = fact
    return result


def _tier_report(
    source: Mapping[str, Fact], observed: Mapping[str, Fact], tier: str, tolerance: float
) -> dict[str, Any]:
    expected = {key: fact for key, fact in source.items() if fact.tier == tier}
    included = {key: fact for key, fact in observed.items() if fact.tier == tier and key in expected}
    accurate = {
        key: fact
        for key, fact in included.items()
        if abs(fact.value - expected[key].value) <= tolerance
    }
    errors = [
        {
            "fact_id": key,
            "family": expected[key].family,
            "arguments": expected[key].arguments_json,
            "expected": expected[key].value,
            "observed": included[key].value,
            "absolute_difference": abs(included[key].value - expected[key].value),
        }
        for key in sorted(included)
        if key not in accurate
    ]
    return {
        "source_fact_count": len(expected),
        "included_fact_count": len(included),
        "accurate_fact_count": len(accurate),
        "coverage": len(included) / max(1, len(expected)),
        "included_accuracy": len(accurate) / max(1, len(included)),
        "missing_fact_count": len(expected) - len(included),
        "errors": errors[:20],
    }


def audit_predictive_recoverability(
    profiles: Mapping[str, Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["audit"]["numeric_tolerance"])
    required_core = float(config["audit"]["required_core_fact_coverage"])
    required_accuracy = float(config["audit"]["required_included_fact_accuracy"])
    policy_rows: list[dict[str, Any]] = []
    for policy_id in sorted(profiles):
        profile = profiles[policy_id]
        view = views[policy_id]
        source = _fact_map(profile_facts(profile))
        observed = _fact_map(view_facts(view))
        core = _tier_report(source, observed, "core", tolerance)
        extended = _tier_report(source, observed, "extended", tolerance)
        hash_valid = stable_hash({key: value for key, value in profile.items() if key != "profile_hash"}) == profile["profile_hash"]
        view_hash_valid = stable_hash({key: value for key, value in view.items() if key != "view_hash"}) == view["view_hash"]
        within_cap = public_evidence_chars(view) <= int(config["view"]["per_policy_char_budget"])
        view_grammar = [
            {"rule": row[0], "expansion": row[1]}
            for row in view.get("trace_grammar", [])
        ]
        trace_reconstruction_valid = all(
            stable_hash(reconstruct_trace_variant(row[0], view_grammar)) == row[9]
            and bool(row[10])
            for row in view.get("trace_variants", [])
        )
        passed = (
            hash_valid
            and view_hash_valid
            and within_cap
            and trace_reconstruction_valid
            and core["coverage"] >= required_core
            and core["included_accuracy"] >= required_accuracy
            and extended["included_accuracy"] >= required_accuracy
        )
        policy_rows.append(
            {
                "policy_id": policy_id,
                "passed": passed,
                "profile_hash_valid": hash_valid,
                "view_hash_valid": view_hash_valid,
                "within_character_cap": within_cap,
                "trace_reconstruction_valid": trace_reconstruction_valid,
                "public_characters": public_evidence_chars(view),
                "core": core,
                "extended": extended,
                "loss_ledger": view["loss_ledger"],
            }
        )
    core_coverages = [row["core"]["coverage"] for row in policy_rows]
    core_accuracies = [row["core"]["included_accuracy"] for row in policy_rows]
    extended_coverages = [row["extended"]["coverage"] for row in policy_rows]
    extended_accuracies = [row["extended"]["included_accuracy"] for row in policy_rows]
    return {
        "schema_version": "tpbr-recoverability-audit-v0",
        "passed": bool(policy_rows) and all(row["passed"] for row in policy_rows),
        "objective_free_signature": predictive_compiler_signature_is_objective_free(),
        "policy_count": len(policy_rows),
        "character_cap": int(config["view"]["per_policy_char_budget"]),
        "thresholds": {
            "required_core_fact_coverage": required_core,
            "required_included_fact_accuracy": required_accuracy,
            "numeric_tolerance": tolerance,
        },
        "aggregate": {
            "core_coverage_mean": fmean(core_coverages) if core_coverages else 0.0,
            "core_coverage_min": min(core_coverages) if core_coverages else 0.0,
            "core_included_accuracy_mean": fmean(core_accuracies) if core_accuracies else 0.0,
            "extended_coverage_mean": fmean(extended_coverages) if extended_coverages else 0.0,
            "extended_coverage_min": min(extended_coverages) if extended_coverages else 0.0,
            "extended_included_accuracy_mean": fmean(extended_accuracies) if extended_accuracies else 0.0,
            "public_characters_mean": fmean(row["public_characters"] for row in policy_rows)
            if policy_rows
            else 0.0,
            "public_characters_min": min((row["public_characters"] for row in policy_rows), default=0),
            "public_characters_max": max((row["public_characters"] for row in policy_rows), default=0),
        },
        "policies": policy_rows,
        "claim_boundary": (
            "The audit proves exact recovery of included declared TPBR facts and measures omitted "
            "declared facts. It does not prove sufficiency for arbitrary future requirements or LLM selection gain."
        ),
    }
