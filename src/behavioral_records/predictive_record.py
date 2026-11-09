from __future__ import annotations

import inspect
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, json_ready, stable_hash
from .models import Trajectory
from .views import public_evidence_chars


FORBIDDEN_COMPILER_KEYS = {
    "objective",
    "objective_id",
    "preference",
    "requirement",
    "judge",
    "oracle_score",
    "heldout_score",
}


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float], quantiles: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    result: dict[str, Any] = {
        "count": len(rows),
        "mean": fmean(rows) if rows else None,
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
        "quantiles": [],
    }
    result["quantiles"] = [
        [float(probability), _quantile(rows, float(probability))]
        for probability in quantiles
    ]
    return result


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _flatten_scalars(value: Any, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten_scalars(value[key], child))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.update(_flatten_scalars(item, f"{prefix}[{index}]"))
        return rows
    if value is None or isinstance(value, (str, bool, int, float)):
        rows[prefix] = value
    return rows


def _atom(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value)


def _symbol_category(symbol: str) -> str:
    return symbol.split("::", 1)[0]


def _run_length(tokens: Iterable[str]) -> tuple[tuple[str, int], ...]:
    rows: list[tuple[str, int]] = []
    for token in tokens:
        if rows and rows[-1][0] == token:
            prior, count = rows[-1]
            rows[-1] = (prior, count + 1)
        else:
            rows.append((token, 1))
    return tuple(rows)


def reconstruct_trace_variant(
    runs: Iterable[Sequence[Any]], grammar: Iterable[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Losslessly expand one selected trace variant from its deterministic grammar."""

    rules = {str(row["rule"]): tuple(str(value) for value in row["expansion"]) for row in grammar}

    def expand(symbol: str, active: frozenset[str]) -> list[str]:
        if symbol not in rules:
            return [symbol]
        if symbol in active:
            raise ValueError(f"Cyclic TPBR trace grammar at {symbol}")
        next_active = active | {symbol}
        result: list[str] = []
        for child in rules[symbol]:
            result.extend(expand(child, next_active))
        return result

    reconstructed: list[str] = []
    for raw_symbol, raw_repeat in runs:
        symbol = str(raw_symbol)
        repeat = int(raw_repeat)
        if repeat < 1:
            raise ValueError("Trace run repeat must be positive")
        expanded = expand(symbol, frozenset())
        for _ in range(repeat):
            reconstructed.extend(expanded)
    return tuple(reconstructed)


@dataclass(frozen=True)
class TypedEpisode:
    policy_id: str
    episode_id: str
    context: Mapping[str, Any]
    outcome: str
    total_return: float
    length: int
    terminated: bool
    truncated: bool
    sequence: tuple[str, ...]
    timed_symbols: tuple[tuple[int, str], ...]
    signal_series: Mapping[str, tuple[Any, ...]]


def type_trajectory(trajectory: Trajectory, symbol_sources: Iterable[str]) -> TypedEpisode:
    """Map one rollout to a domain-neutral typed event/signal interface.

    The mapping uses only observed actions, events, state/predicate changes and
    terminal outcomes. It has no objective, preference or judge input.
    """

    sources = set(symbol_sources)
    sequence: list[str] = []
    timed: list[tuple[int, str]] = []
    signal_values: dict[str, list[Any]] = defaultdict(list)
    prior_boolean: dict[str, bool] = {}

    for step in trajectory.steps:
        step_symbols: list[str] = []
        if "actions" in sources:
            step_symbols.append(f"action::{_atom(step.action)}")
        if "events" in sources:
            step_symbols.extend(f"event::{event}" for event in step.events)

        flat_state = _flatten_scalars(step.state, "state")
        flat_predicates = _flatten_scalars(step.predicates, "predicate")
        flat_signals = {**flat_state, **flat_predicates, "step.reward": step.reward}
        for name, value in sorted(flat_signals.items()):
            signal_values[name].append(value)
            if "boolean_transitions" not in sources or not isinstance(value, bool):
                continue
            previous = prior_boolean.get(name)
            if previous is not None and previous != value:
                direction = "false_to_true" if value else "true_to_false"
                step_symbols.append(f"boolean_transition::{name}:{direction}")
            prior_boolean[name] = value

        for symbol in step_symbols:
            sequence.append(symbol)
            timed.append((step.t, symbol))

    if "outcomes" in sources:
        outcome_symbol = f"outcome::{trajectory.outcome}"
        sequence.append(outcome_symbol)
        timed.append((trajectory.steps[-1].t + 1, outcome_symbol))

    signal_values["episode.return"].append(trajectory.total_return)
    signal_values["episode.length"].append(trajectory.length)
    signal_values["episode.terminated"].append(trajectory.terminated)
    signal_values["episode.truncated"].append(trajectory.truncated)
    return TypedEpisode(
        policy_id=trajectory.policy_id,
        episode_id=trajectory.episode_id,
        context=json_ready(dict(trajectory.context)),
        outcome=trajectory.outcome,
        total_return=trajectory.total_return,
        length=trajectory.length,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        sequence=tuple(sequence),
        timed_symbols=tuple(timed),
        signal_series={key: tuple(values) for key, values in sorted(signal_values.items())},
    )

def type_episode_evidence(
    row: Mapping[str, Any], symbol_sources: Iterable[str]
) -> TypedEpisode:
    """Map a compact objective-blind event log to the shared typed interface.

    Compact logs retain ordered events and episode-level signals but may omit raw
    actions, states, and absolute event timestamps. In that case temporal indices
    are event ordinals; the compiler records that limitation in its loss ledger.
    """

    def reject_forbidden(value: Any, path: str = "episode") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in FORBIDDEN_COMPILER_KEYS:
                    raise ValueError(f"Forbidden objective-conditioned input at {path}.{key}")
                reject_forbidden(item, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                reject_forbidden(item, f"{path}[{index}]")

    reject_forbidden(row)
    sources = set(str(value) for value in symbol_sources)
    events = row.get("events", ())
    if isinstance(events, str):
        events = (events,) if events else ()
    actions = row.get("actions", ())
    if isinstance(actions, (str, int)):
        actions = (actions,)

    sequence: list[str] = []
    timed: list[tuple[int, str]] = []
    cursor = 0
    if "actions" in sources:
        for action in actions:
            symbol = f"action::{_atom(action)}"
            sequence.append(symbol)
            timed.append((cursor, symbol))
            cursor += 1
    if "events" in sources:
        for event in events:
            symbol = f"event::{event}"
            sequence.append(symbol)
            timed.append((cursor, symbol))
            cursor += 1

    outcome = str(row.get("outcome", "unknown"))
    if "outcomes" in sources:
        symbol = f"outcome::{outcome}"
        sequence.append(symbol)
        timed.append((cursor, symbol))

    total_return = float(row.get("return", 0.0))
    length = int(row.get("length", max(1, cursor)))
    terminated = bool(row.get("terminated", False))
    truncated = bool(row.get("truncated", False))
    signal_values: dict[str, list[Any]] = defaultdict(list)
    for name, value in sorted(_flatten_scalars(row.get("signals", {}), "signal").items()):
        signal_values[name].append(value)
    signal_values["episode.return"].append(total_return)
    signal_values["episode.length"].append(length)
    signal_values["episode.terminated"].append(terminated)
    signal_values["episode.truncated"].append(truncated)
    return TypedEpisode(
        policy_id=str(row["policy_id"]),
        episode_id=str(row["episode_id"]),
        context=json_ready(dict(row.get("context", {}))),
        outcome=outcome,
        total_return=total_return,
        length=length,
        terminated=terminated,
        truncated=truncated,
        sequence=tuple(sequence),
        timed_symbols=tuple(timed),
        signal_series={key: tuple(values) for key, values in sorted(signal_values.items())},
    )


class PredictiveBehaviorCompiler:
    """Compile objective-blind, probability-oriented policy evidence."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = json_ready(dict(config))
        self.config_hash = stable_hash(self.config)
        self._validate_config()

    def _validate_config(self) -> None:
        if set(self.config) != {"schema_version", "compiler", "view", "audit"}:
            raise ValueError("TPBR config must contain schema_version, compiler, view, and audit")

        def visit(value: Any, path: str = "config") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key).lower() in FORBIDDEN_COMPILER_KEYS:
                        raise ValueError(f"Forbidden objective-conditioned key at {path}.{key}")
                    visit(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(self.config)
        compiler = self.config["compiler"]
        for key in (
            "symbol_sources",
            "first_hit_horizons",
            "discount_factors",
            "quantiles",
            "temporal_horizons",
        ):
            if not compiler.get(key):
                raise ValueError(f"TPBR compiler config requires non-empty {key}")

    def compile(self, policy_id: str, trajectories: Iterable[Trajectory]) -> dict[str, Any]:
        raw = sorted(list(trajectories), key=lambda row: row.episode_id)
        if not raw:
            raise ValueError("At least one trajectory is required")
        if any(row.policy_id != policy_id for row in raw):
            raise ValueError("TPBR input contains another policy ID")
        cfg = self.config["compiler"]
        episodes = [type_trajectory(row, cfg["symbol_sources"]) for row in raw]
        return self._compile_typed(
            policy_id,
            episodes,
            semantic_adapter=(
                "Objective-blind raw trajectory adapter; typed channels="
                + ",".join(str(value) for value in cfg["symbol_sources"])
                + "; absolute step timestamps, state, predicates, actions and rewards are retained "
                "when present in the trajectory contract."
            ),
        )

    def compile_episode_evidence(
        self, policy_id: str, rows: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Compile an objective-blind compact event log without expanding raw steps."""

        raw = sorted((dict(row) for row in rows), key=lambda row: str(row["episode_id"]))
        if not raw:
            raise ValueError("At least one compact episode is required")
        if any(str(row["policy_id"]) != policy_id for row in raw):
            raise ValueError("TPBR compact input contains another policy ID")
        cfg = self.config["compiler"]
        episodes = [type_episode_evidence(row, cfg["symbol_sources"]) for row in raw]
        return self._compile_typed(
            policy_id,
            episodes,
            semantic_adapter=(
                "Objective-blind compact event-log adapter; ordered events, context, outcome, "
                "return and episode length retained; raw actions/states and absolute event step "
                "timestamps unavailable; temporal indices denote ordered event positions."
            ),
        )

    def _compile_typed(
        self,
        policy_id: str,
        episodes: list[TypedEpisode],
        *,
        semantic_adapter: str,
    ) -> dict[str, Any]:
        if any(row.policy_id != policy_id for row in episodes):
            raise ValueError("TPBR typed input contains another policy ID")
        cfg = self.config["compiler"]
        symbol_profiles = self._symbol_profiles(episodes)
        signal_profiles = self._signal_profiles(episodes)
        temporal_symbols = self._temporal_symbols(symbol_profiles)
        temporal_rows = self._temporal_profiles(episodes, temporal_symbols)
        context_rows, context_ledger = self._context_profiles(episodes, symbol_profiles)
        conditional_rows, conditional_ledger = self._conditional_dynamics(episodes)
        trace_rows, trace_grammar, trace_ledger = self._trace_variants(episodes)
        outcomes = Counter(row.outcome for row in episodes)
        profile: dict[str, Any] = {
            "representation": "task_general_predictive_behavioral_record",
            "schema_version": self.config["schema_version"],
            "policy_id": policy_id,
            "config_hash": self.config_hash,
            "sample": {
                "episodes": len(episodes),
                "steps": sum(row.length for row in episodes),
                "typed_symbol_occurrences": sum(len(row.sequence) for row in episodes),
                "contexts": len({canonical_json(row.context) for row in episodes}),
            },
            "outcomes": [
                {
                    "name": name,
                    "count": count,
                    "rate": count / len(episodes),
                    "rate_ci95": _wilson_interval(count, len(episodes)),
                }
                for name, count in sorted(outcomes.items())
            ],
            "symbol_profiles": symbol_profiles,
            "signal_profiles": signal_profiles,
            "context_profiles": context_rows,
            "temporal_profiles": temporal_rows,
            "conditional_dynamics": conditional_rows,
            "trace_grammar": trace_grammar,
            "trace_variants": trace_rows,
            "loss_ledger": {
                "semantic_adapter": semantic_adapter,
                "raw_episode_count": len(episodes),
                "raw_step_count": sum(row.length for row in episodes),
                "typed_symbol_vocabulary": len({symbol for row in episodes for symbol in row.sequence}),
                "temporal_symbol_candidates": len(symbol_profiles),
                "temporal_symbols_included": len(temporal_symbols),
                "temporal_symbols_omitted": max(0, len(symbol_profiles) - len(temporal_symbols)),
                "temporal_relation_rows": len(temporal_rows),
                **context_ledger,
                **conditional_ledger,
                **trace_ledger,
            },
        }
        profile["profile_hash"] = stable_hash(profile)
        return profile

    def _symbol_profiles(self, episodes: list[TypedEpisode]) -> list[dict[str, Any]]:
        cfg = self.config["compiler"]
        horizons = [int(value) for value in cfg["first_hit_horizons"]]
        discounts = [float(value) for value in cfg["discount_factors"]]
        quantiles = [float(value) for value in cfg["quantiles"]]
        vocabulary = sorted({symbol for row in episodes for symbol in row.sequence})
        rows: list[dict[str, Any]] = []
        for symbol in vocabulary:
            counts: list[int] = []
            first_hits: list[int] = []
            discounted: dict[float, list[float]] = {gamma: [] for gamma in discounts}
            outcomes_present: Counter[str] = Counter()
            for episode in episodes:
                times = [time for time, observed in episode.timed_symbols if observed == symbol]
                counts.append(len(times))
                if times:
                    first_hits.append(min(times))
                    outcomes_present[episode.outcome] += 1
                for gamma in discounts:
                    discounted[gamma].append(sum(gamma**time for time in times))
            episode_count = sum(count > 0 for count in counts)
            occurrence_count = sum(counts)
            rows.append(
                {
                    "symbol": symbol,
                    "category": _symbol_category(symbol),
                    "occurrence_count": occurrence_count,
                    "episode_count": episode_count,
                    "episode_rate": episode_count / len(episodes),
                    "episode_rate_ci95": _wilson_interval(episode_count, len(episodes)),
                    "count_per_episode": _distribution(counts, quantiles),
                    "first_hit_step_given_present": _distribution(first_hits, quantiles),
                    "first_hit_cdf": [
                        [horizon, sum(time < horizon for time in first_hits) / len(episodes)]
                        for horizon in horizons
                    ],
                    "repeat_rate_ge_2": sum(count >= 2 for count in counts) / len(episodes),
                    "repeat_rate_ge_3": sum(count >= 3 for count in counts) / len(episodes),
                    "discounted_count_mean": [
                        [gamma, fmean(discounted[gamma])] for gamma in discounts
                    ],
                    "outcome_given_present": [
                        [outcome, count, count / max(1, episode_count)]
                        for outcome, count in sorted(outcomes_present.items())
                    ],
                }
            )
        rows.sort(key=lambda row: (-row["episode_count"], -row["occurrence_count"], row["symbol"]))
        return rows

    def _signal_profiles(self, episodes: list[TypedEpisode]) -> list[dict[str, Any]]:
        cfg = self.config["compiler"]
        quantiles = [float(value) for value in cfg["quantiles"]]
        max_categories = int(cfg["max_categorical_values"])
        signal_names = sorted({name for row in episodes for name in row.signal_series})
        rows: list[dict[str, Any]] = []
        for name in signal_names:
            flattened: list[Any] = []
            episode_means: list[float] = []
            deltas: list[float] = []
            changed = 0
            comparable = 0
            missing_episodes = 0
            for episode in episodes:
                values = list(episode.signal_series.get(name, ()))
                if not values:
                    missing_episodes += 1
                    continue
                flattened.extend(values)
                numeric = [float(value) for value in values if isinstance(value, (bool, int, float))]
                if len(numeric) == len(values):
                    episode_means.append(fmean(numeric))
                    for left, right in zip(numeric, numeric[1:]):
                        comparable += 1
                        if left != right:
                            changed += 1
                        deltas.append(right - left)
            numeric_all = [float(value) for value in flattened if isinstance(value, (bool, int, float))]
            if len(numeric_all) == len(flattened):
                kind = "boolean" if flattened and all(isinstance(value, bool) for value in flattened) else "numeric"
                rows.append(
                    {
                        "signal": name,
                        "kind": kind,
                        "sample_count": len(flattened),
                        "missing_episode_count": missing_episodes,
                        "value": _distribution(numeric_all, quantiles),
                        "episode_mean": _distribution(episode_means, quantiles),
                        "delta": _distribution(deltas, quantiles),
                        "change_rate": changed / max(1, comparable),
                    }
                )
            else:
                counts = Counter(canonical_json(value) for value in flattened)
                ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                selected = ordered[:max_categories]
                rows.append(
                    {
                        "signal": name,
                        "kind": "categorical",
                        "sample_count": len(flattened),
                        "missing_episode_count": missing_episodes,
                        "values": [[json.loads(value), count] for value, count in selected],
                        "residual_count": sum(count for _, count in ordered[max_categories:]),
                    }
                )
        return rows

    def _temporal_symbols(self, profiles: list[dict[str, Any]]) -> list[str]:
        cfg = self.config["compiler"]
        categories = set(cfg["temporal_symbol_categories"])
        candidates = [row for row in profiles if row["category"] in categories]
        limit = int(cfg["max_temporal_symbols"])
        if len(candidates) <= limit:
            return sorted(row["symbol"] for row in candidates)
        frequent_count = (limit + 1) // 2
        frequent = sorted(
            candidates,
            key=lambda row: (-row["episode_count"], -row["occurrence_count"], row["symbol"]),
        )[:frequent_count]
        selected = {row["symbol"] for row in frequent}
        rare = sorted(
            candidates,
            key=lambda row: (row["episode_count"], row["occurrence_count"], row["symbol"]),
        )
        for row in rare:
            if len(selected) >= limit:
                break
            selected.add(row["symbol"])
        return sorted(selected)

    def _temporal_profiles(
        self, episodes: list[TypedEpisode], symbols: list[str]
    ) -> list[dict[str, Any]]:
        horizons = [int(value) for value in self.config["compiler"]["temporal_horizons"]]
        accumulator: dict[tuple[str, str], dict[str, Any]] = {}
        for left in symbols:
            for right in symbols:
                if left == right:
                    continue
                accumulator[(left, right)] = {
                    "activations": 0,
                    "activation_episodes": 0,
                    "satisfied_eventually": 0,
                    "within": Counter(),
                    "both_episodes": 0,
                    "left_first": 0,
                    "right_first": 0,
                    "simultaneous_first": 0,
                }
        for episode in episodes:
            positions: dict[str, list[int]] = defaultdict(list)
            for time, symbol in episode.timed_symbols:
                if symbol in symbols:
                    positions[symbol].append(time)
            for left in symbols:
                left_positions = positions.get(left, [])
                if not left_positions:
                    continue
                for right in symbols:
                    if left == right:
                        continue
                    row = accumulator[(left, right)]
                    row["activations"] += len(left_positions)
                    row["activation_episodes"] += 1
                    right_positions = positions.get(right, [])
                    if right_positions:
                        row["both_episodes"] += 1
                        if left_positions[0] < right_positions[0]:
                            row["left_first"] += 1
                        elif left_positions[0] > right_positions[0]:
                            row["right_first"] += 1
                        else:
                            row["simultaneous_first"] += 1
                    for left_time in left_positions:
                        index = bisect_right(right_positions, left_time)
                        if index >= len(right_positions):
                            continue
                        gap = right_positions[index] - left_time
                        row["satisfied_eventually"] += 1
                        for horizon in horizons:
                            if gap <= horizon:
                                row["within"][horizon] += 1
        results: list[dict[str, Any]] = []
        episode_count = len(episodes)
        for (left, right), value in accumulator.items():
            activations = int(value["activations"])
            if activations == 0:
                continue
            results.append(
                {
                    "left": left,
                    "right": right,
                    "activations": activations,
                    "activation_episodes": int(value["activation_episodes"]),
                    "satisfied_eventually": int(value["satisfied_eventually"]),
                    "violations": activations - int(value["satisfied_eventually"]),
                    "vacuous_episodes": episode_count - int(value["activation_episodes"]),
                    "response_rate": value["satisfied_eventually"] / activations,
                    "within_response_rate": [
                        [horizon, value["within"][horizon] / activations]
                        for horizon in horizons
                    ],
                    "both_episode_count": int(value["both_episodes"]),
                    "first_order_counts": {
                        "left_before_right": int(value["left_first"]),
                        "right_before_left": int(value["right_first"]),
                        "same_time": int(value["simultaneous_first"]),
                    },
                }
            )
        results.sort(
            key=lambda row: (
                -row["activation_episodes"],
                -row["satisfied_eventually"],
                row["left"],
                row["right"],
            )
        )
        return results

    def _context_profiles(
        self,
        episodes: list[TypedEpisode],
        symbol_profiles: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        cfg = self.config["compiler"]
        max_values = int(cfg["max_context_values_per_field"])
        max_symbols = int(cfg["max_context_symbols"])
        context_fields = sorted({key for row in episodes for key in row.context})
        selected_symbols = [row["symbol"] for row in symbol_profiles[:max_symbols]]
        rows: list[dict[str, Any]] = []
        total_values = 0
        omitted_values = 0
        for field in context_fields:
            groups: dict[str, list[TypedEpisode]] = defaultdict(list)
            for episode in episodes:
                groups[canonical_json(episode.context.get(field))].append(episode)
            ordered = sorted(groups, key=lambda key: (-len(groups[key]), key))
            total_values += len(ordered)
            omitted_values += max(0, len(ordered) - max_values)
            for value_key in ordered[:max_values]:
                group = groups[value_key]
                outcomes = Counter(row.outcome for row in group)
                symbol_counts = Counter(
                    symbol
                    for episode in group
                    for symbol in set(episode.sequence)
                    if symbol in selected_symbols
                )
                rows.append(
                    {
                        "field": field,
                        "value": json.loads(value_key),
                        "episode_count": len(group),
                        "outcome_rates": [
                            [name, count, count / len(group)] for name, count in sorted(outcomes.items())
                        ],
                        "symbol_episode_rates": [
                            [symbol, symbol_counts[symbol], symbol_counts[symbol] / len(group)]
                            for symbol in selected_symbols
                            if symbol_counts[symbol]
                        ],
                    }
                )
        return rows, {
            "context_value_groups_total": total_values,
            "context_value_groups_included": len(rows),
            "context_value_groups_omitted": omitted_values,
        }

    def _conditional_dynamics(
        self, episodes: list[TypedEpisode]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config["compiler"]
        max_order = int(cfg["variable_order_max"])
        min_support = int(cfg["variable_order_min_support"])
        max_rows = int(cfg["max_conditional_contexts"])
        max_next = int(cfg["max_next_symbols_per_context"])
        global_next: Counter[str] = Counter()
        contexts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for episode in episodes:
            sequence = list(episode.sequence)
            global_next.update(sequence)
            for index, next_symbol in enumerate(sequence):
                for order in range(1, min(max_order, index) + 1):
                    context = tuple(sequence[index - order : index])
                    contexts[context][next_symbol] += 1
        total_global = sum(global_next.values())
        candidates: list[dict[str, Any]] = []
        for context, next_counts in contexts.items():
            support = sum(next_counts.values())
            if support < min_support:
                continue
            information_gain = 0.0
            for symbol, count in next_counts.items():
                conditional = count / support
                marginal = global_next[symbol] / max(1, total_global)
                information_gain += count * math.log2(conditional / max(marginal, 1e-300))
            ordered_next = sorted(next_counts.items(), key=lambda item: (-item[1], item[0]))
            selected_next = ordered_next[:max_next]
            candidates.append(
                {
                    "context": list(context),
                    "order": len(context),
                    "support": support,
                    "information_gain_bits": information_gain,
                    "next": [
                        [symbol, count, count / support] for symbol, count in selected_next
                    ],
                    "residual_next_count": sum(count for _, count in ordered_next[max_next:]),
                }
            )
        candidates.sort(
            key=lambda row: (-row["information_gain_bits"], -row["support"], canonical_json(row["context"]))
        )
        selected = candidates[:max_rows]
        selected_support = sum(row["support"] for row in selected)
        total_support = sum(row["support"] for row in candidates)
        return selected, {
            "conditional_context_candidates": len(candidates),
            "conditional_contexts_included": len(selected),
            "conditional_contexts_omitted": max(0, len(candidates) - len(selected)),
            "conditional_context_support_fraction": selected_support / max(1, total_support),
        }

    @staticmethod
    def _replace_pair(sequence: list[str], pair: tuple[str, str], replacement: str) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(replacement)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def _grammar_compress(
        self, sequences: list[list[str]]
    ) -> tuple[list[list[str]], list[dict[str, Any]]]:
        cfg = self.config["compiler"]
        max_rules = int(cfg["max_trace_grammar_rules"])
        min_pair_count = int(cfg["min_trace_pair_count"])
        current = [list(sequence) for sequence in sequences]
        rules: list[dict[str, Any]] = []
        for index in range(max_rules):
            counts: Counter[tuple[str, str]] = Counter()
            for sequence in current:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, count = min(
                counts.items(),
                key=lambda item: (-item[1], canonical_json(item[0])),
            )
            if count < min_pair_count:
                break
            rule = f"grammar::G{index + 1:03d}"
            current = [self._replace_pair(sequence, pair, rule) for sequence in current]
            rules.append(
                {
                    "rule": rule,
                    "expansion": [pair[0], pair[1]],
                    "pair_occurrences_before_replacement": count,
                }
            )
        return current, rules

    def _trace_variants(
        self, episodes: list[TypedEpisode]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        cfg = self.config["compiler"]
        max_common = int(cfg["max_trace_variants"])
        max_rare = int(cfg["max_rare_variants"])
        groups: dict[tuple[tuple[str, int], ...], list[TypedEpisode]] = defaultdict(list)
        for episode in episodes:
            groups[_run_length(episode.sequence)].append(episode)
        common_keys = sorted(groups, key=lambda key: (-len(groups[key]), canonical_json(key)))[:max_common]
        selected = set(common_keys)
        rare_keys = sorted(groups, key=lambda key: (len(groups[key]), stable_hash(key), canonical_json(key)))
        for key in rare_keys:
            if len(selected) >= max_common + max_rare:
                break
            selected.add(key)
        ordered = sorted(selected, key=lambda key: (-len(groups[key]), canonical_json(key)))
        expanded_sequences = [
            [symbol for symbol, repeat in key for _ in range(repeat)] for key in ordered
        ]
        compressed_sequences, grammar = self._grammar_compress(expanded_sequences)
        rows: list[dict[str, Any]] = []
        for key, compressed in zip(ordered, compressed_sequences):
            members = groups[key]
            outcomes = Counter(row.outcome for row in members)
            compressed_runs = [[symbol, repeat] for symbol, repeat in _run_length(compressed)]
            source_sequence = tuple(
                symbol for symbol, repeat in key for _ in range(repeat)
            )
            reconstructed = reconstruct_trace_variant(compressed_runs, grammar)
            rows.append(
                {
                    "runs": compressed_runs,
                    "source_symbol_count": sum(repeat for _, repeat in key),
                    "compressed_symbol_count": len(compressed),
                    "source_sequence_hash": stable_hash(source_sequence),
                    "reconstruction_verified": reconstructed == source_sequence,
                    "episode_count": len(members),
                    "episode_rate": len(members) / len(episodes),
                    "outcomes": [[name, count] for name, count in sorted(outcomes.items())],
                    "return": _distribution(
                        [row.total_return for row in members],
                        self.config["compiler"]["quantiles"],
                    ),
                    "length": _distribution(
                        [row.length for row in members],
                        self.config["compiler"]["quantiles"],
                    ),
                    "evidence_id": stable_hash(
                        ["tpbr-variant-v0", *sorted(row.episode_id for row in members)], 16
                    ),
                }
            )
        represented = sum(len(groups[key]) for key in selected)
        source_symbols = sum(row["source_symbol_count"] for row in rows)
        compressed_symbols = sum(row["compressed_symbol_count"] for row in rows)
        return rows, grammar, {
            "trace_variants_total": len(groups),
            "trace_variants_included": len(rows),
            "trace_variants_omitted": max(0, len(groups) - len(rows)),
            "trace_variant_episode_coverage": represented / len(episodes),
            "trace_grammar_rules": len(grammar),
            "trace_selected_source_symbols": source_symbols,
            "trace_selected_compressed_symbols": compressed_symbols,
            "trace_symbol_compression_ratio": compressed_symbols / max(1, source_symbols),
            "trace_selected_reconstruction_verified": all(
                row["reconstruction_verified"] for row in rows
            ),
        }


def predictive_compiler_signature_is_objective_free() -> bool:
    parameters = set(inspect.signature(PredictiveBehaviorCompiler.compile).parameters)
    return not parameters.intersection(FORBIDDEN_COMPILER_KEYS)


def compile_predictive_profiles(
    trajectories_by_policy: Mapping[str, Iterable[Trajectory]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    compiler = PredictiveBehaviorCompiler(config)
    return {
        policy_id: compiler.compile(policy_id, rows)
        for policy_id, rows in sorted(trajectories_by_policy.items())
    }


def _compact_distribution(row: Mapping[str, Any]) -> list[Any]:
    return [row.get("count"), row.get("mean"), row.get("min"), row.get("max"), row.get("quantiles", [])]


def _compact_profile_view(
    profile: Mapping[str, Any],
    view_config: Mapping[str, Any],
    limits: Mapping[str, int],
) -> dict[str, Any]:
    symbols = list(profile["symbol_profiles"])[: limits["symbol_profiles"]]
    signals = list(profile["signal_profiles"])[: limits["signal_profiles"]]
    contexts = list(profile["context_profiles"])[: limits["context_rows"]]
    temporal = list(profile["temporal_profiles"])[: limits["temporal_rows"]]
    conditional = list(profile["conditional_dynamics"])[: limits["conditional_rows"]]
    variants = list(profile["trace_variants"])[: limits["trace_variants"]]
    grammar_by_rule = {row["rule"]: row for row in profile.get("trace_grammar", [])}
    referenced_rules = {
        symbol
        for row in variants
        for symbol, _ in row["runs"]
        if str(symbol).startswith("grammar::")
    }
    queue = list(referenced_rules)
    while queue:
        rule = queue.pop()
        expansion = grammar_by_rule.get(rule, {}).get("expansion", [])
        for symbol in expansion:
            if str(symbol).startswith("grammar::") and symbol not in referenced_rules:
                referenced_rules.add(symbol)
                queue.append(symbol)
    selected_grammar = [
        grammar_by_rule[rule] for rule in sorted(referenced_rules) if rule in grammar_by_rule
    ]
    max_context_symbols = int(view_config["max_context_symbols_per_row"])
    result: dict[str, Any] = {
        "representation": "tpbr_compact_judge_view",
        "schema_version": view_config["schema_version"],
        "policy_id": profile["policy_id"],
        "config_hash": profile["config_hash"],
        "source_profile_hash": profile["profile_hash"],
        "char_cap": int(view_config["per_policy_char_budget"]),
        "notation": {
            "symbol": [
                "name", "category", "occurrences", "episode_count", "episode_rate", "ci95",
                "count_dist", "first_hit_dist", "first_hit_cdf", "repeat_ge_2", "repeat_ge_3",
                "discounted_count_means", "outcome_given_present"
            ],
            "numeric_distribution": ["count", "mean", "min", "max", "quantiles"],
            "temporal": [
                "left", "right", "activations", "activation_episodes", "satisfied_eventually",
                "violations", "vacuous_episodes", "response_rate", "within_response_rate",
                "both_episode_count", "first_order_counts"
            ],
            "probability_denominators": (
                "episode_rate uses all episodes; response_rate uses left activations; "
                "conditional next probabilities use context support"
            ),
            "missingness": "UNKNOWN/TRUNCATED is never interpreted as zero",
        },
        "sample": profile["sample"],
        "outcomes": [
            [row["name"], row["count"], row["rate"], row["rate_ci95"]]
            for row in profile["outcomes"]
        ],
        "symbols": [
            [
                row["symbol"], row["category"], row["occurrence_count"], row["episode_count"],
                row["episode_rate"], row["episode_rate_ci95"],
                _compact_distribution(row["count_per_episode"]),
                _compact_distribution(row["first_hit_step_given_present"]),
                row["first_hit_cdf"], row["repeat_rate_ge_2"], row["repeat_rate_ge_3"],
                row["discounted_count_mean"], row["outcome_given_present"],
            ]
            for row in symbols
        ],
        "signals": [],
        "contexts": [
            [
                row["field"], row["value"], row["episode_count"],
                row["outcome_rates"], row["symbol_episode_rates"][:max_context_symbols],
            ]
            for row in contexts
        ],
        "temporal": [
            [
                row["left"], row["right"], row["activations"], row["activation_episodes"],
                row["satisfied_eventually"], row["violations"], row["vacuous_episodes"],
                row["response_rate"], row["within_response_rate"], row["both_episode_count"],
                row["first_order_counts"],
            ]
            for row in temporal
        ],
        "conditional_dynamics": [
            [
                row["context"], row["order"], row["support"], row["information_gain_bits"],
                row["next"], row["residual_next_count"],
            ]
            for row in conditional
        ],
        "trace_grammar": [
            [row["rule"], row["expansion"], row["pair_occurrences_before_replacement"]]
            for row in selected_grammar
        ],
        "trace_variants": [
            [
                row["runs"], row["episode_count"], row["episode_rate"], row["outcomes"],
                _compact_distribution(row["return"]), _compact_distribution(row["length"]),
                row["evidence_id"], row["source_symbol_count"], row["compressed_symbol_count"],
                row["source_sequence_hash"], row["reconstruction_verified"],
            ]
            for row in variants
        ],
        "loss_ledger": dict(profile["loss_ledger"]),
    }
    for row in signals:
        if row["kind"] == "categorical":
            result["signals"].append(
                [
                    row["signal"], row["kind"], row["sample_count"], row["missing_episode_count"],
                    row["values"], row["residual_count"],
                ]
            )
        else:
            result["signals"].append(
                [
                    row["signal"], row["kind"], row["sample_count"], row["missing_episode_count"],
                    _compact_distribution(row["value"]),
                    row["change_rate"],
                ]
            )
    result["loss_ledger"]["judge_view"] = {
        "included_symbol_profiles": len(symbols),
        "source_symbol_profiles": len(profile["symbol_profiles"]),
        "included_signal_profiles": len(signals),
        "source_signal_profiles": len(profile["signal_profiles"]),
        "included_context_rows": len(contexts),
        "source_context_rows": len(profile["context_profiles"]),
        "included_context_symbol_rate_rows": sum(
            min(len(row["symbol_episode_rates"]), max_context_symbols) for row in contexts
        ),
        "source_context_symbol_rate_rows": sum(
            len(row["symbol_episode_rates"]) for row in profile["context_profiles"]
        ),
        "included_temporal_rows": len(temporal),
        "source_temporal_rows": len(profile["temporal_profiles"]),
        "included_conditional_rows": len(conditional),
        "source_conditional_rows": len(profile["conditional_dynamics"]),
        "included_trace_variants": len(variants),
        "source_trace_variants": len(profile["trace_variants"]),
        "included_trace_grammar_rules": len(selected_grammar),
        "source_trace_grammar_rules": len(profile.get("trace_grammar", [])),
        "public_characters": 0,
    }
    return result


def build_predictive_view(
    profile: Mapping[str, Any], view_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one deterministic fixed-budget TPBR view without an objective."""

    source_names = {
        "symbol_profiles": "symbol_profiles",
        "signal_profiles": "signal_profiles",
        "context_rows": "context_profiles",
        "temporal_rows": "temporal_profiles",
        "conditional_rows": "conditional_dynamics",
        "trace_variants": "trace_variants",
    }
    limits = {
        key: min(len(profile[source]), int(view_config[f"max_{key}"]))
        for key, source in source_names.items()
    }
    floors = {
        key: min(limits[key], int(view_config["minimum_rows"][key]))
        for key in limits
    }
    shrink_order = (
        "conditional_rows",
        "trace_variants",
        "temporal_rows",
        "context_rows",
        "signal_profiles",
        "symbol_profiles",
    )
    decrement = {
        "symbol_profiles": 4,
        "signal_profiles": 4,
        "context_rows": 4,
        "temporal_rows": 16,
        "conditional_rows": 8,
        "trace_variants": 2,
    }
    char_cap = int(view_config["per_policy_char_budget"])
    while True:
        view = _compact_profile_view(profile, view_config, limits)
        for _ in range(4):
            measured = public_evidence_chars(view)
            if view["loss_ledger"]["judge_view"]["public_characters"] == measured:
                break
            view["loss_ledger"]["judge_view"]["public_characters"] = measured
        if public_evidence_chars(view) <= char_cap:
            break
        changed = False
        for key in shrink_order:
            if limits[key] > floors[key]:
                limits[key] = max(floors[key], limits[key] - decrement[key])
                changed = True
                break
        if not changed:
            minimum_size = public_evidence_chars(view)
            raise ValueError(
                f"TPBR minimum judge view needs {minimum_size} characters, exceeding cap {char_cap}"
            )
    view["view_hash"] = stable_hash(view)
    return view


def build_predictive_views(
    profiles: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        policy_id: build_predictive_view(profile, config["view"])
        for policy_id, profile in sorted(profiles.items())
    }


def build_predictive_pair_prompt(
    natural_language_requirement: str,
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
) -> str:
    """Let the model interpret one fixed evidence record and make the final choice."""

    from .views import public_evidence

    return "\n".join(
        [
            "Select the policy that best satisfies the natural-language requirement using only the fixed TPBR evidence.",
            "Treat all hard constraints as a conjunction. A preference cannot rescue a failed constraint.",
            "If neither policy satisfies every hard constraint, return TIE. If both do, apply preferences in order.",
            "Use the stated denominators, support counts, uncertainty, and loss ledger. Missing or truncated is UNKNOWN, not zero.",
            "The evidence compiler did not receive this requirement and did not rank the policies.",
            f"REQUIREMENT: {natural_language_requirement}",
            f"POLICY A TPBR: {canonical_json(public_evidence(view_a))}",
            f"POLICY B TPBR: {canonical_json(public_evidence(view_b))}",
            "Return exactly one JSON object with winner, confidence, and reason.",
            "In reason, briefly state the hard-constraint checks before any preference comparison.",
            'winner must be "A", "B", or "TIE".',
            "confidence must be a JSON number between 0 and 1, not a word or string.",
        ]
    )
