from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .canonical import stable_hash


def select_disjoint_new_pairs(
    policy_ids: Iterable[str],
    excluded_pairs: Iterable[tuple[str, str]],
    *,
    count: int,
) -> list[dict[str, str]]:
    """Select a deterministic matching without consulting objectives or outcomes."""
    policies = sorted({str(policy_id) for policy_id in policy_ids})
    excluded = {tuple(sorted((str(left), str(right)))) for left, right in excluded_pairs}
    candidates = []
    for left, right in combinations(policies, 2):
        if (left, right) in excluded:
            continue
        candidates.append(
            {
                "pair_id": stable_hash(["pair-graph-v1", left, right], 16),
                "left_policy_id": left,
                "right_policy_id": right,
            }
        )
    candidates.sort(key=lambda row: (row["pair_id"], row["left_policy_id"], row["right_policy_id"]))
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for row in candidates:
        left = row["left_policy_id"]
        right = row["right_policy_id"]
        if left in used or right in used:
            continue
        selected.append(row)
        used.update((left, right))
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected)} disjoint unseen pairs are available; {count} requested"
    )


def mcnemar_exact_two_sided(proposed_only: int, baseline_only: int) -> float:
    discordant = int(proposed_only) + int(baseline_only)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * 0.5**discordant
        for index in range(min(int(proposed_only), int(baseline_only)) + 1)
    )
    return min(1.0, 2.0 * tail)


def paired_binary_summary(
    baseline: Sequence[bool], proposed: Sequence[bool]
) -> dict[str, float | int]:
    if len(baseline) != len(proposed) or not baseline:
        raise ValueError("Paired binary outcomes must be non-empty and aligned")
    proposed_only = sum(right and not left for left, right in zip(baseline, proposed))
    baseline_only = sum(left and not right for left, right in zip(baseline, proposed))
    return {
        "count": len(baseline),
        "baseline_success": sum(baseline),
        "proposed_success": sum(proposed),
        "baseline_rate": sum(baseline) / len(baseline),
        "proposed_rate": sum(proposed) / len(proposed),
        "proposed_minus_baseline_rate": (
            sum(int(right) - int(left) for left, right in zip(baseline, proposed))
            / len(baseline)
        ),
        "proposed_only_success": proposed_only,
        "baseline_only_success": baseline_only,
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(
            proposed_only, baseline_only
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise adjusted p-values; valid without independence."""
    ordered = sorted(
        ((str(key), float(value)) for key, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[key] = min(1.0, running)
    return adjusted


def exact_cluster_sign_flip_p(cluster_differences: Iterable[int]) -> float:
    """Exact two-sided sign-flip p-value using dynamic programming over clusters."""
    values = [int(value) for value in cluster_differences if int(value) != 0]
    if not values:
        return 1.0
    distribution: dict[int, int] = {0: 1}
    for value in values:
        updated: dict[int, int] = defaultdict(int)
        magnitude = abs(value)
        for total, ways in distribution.items():
            updated[total + magnitude] += ways
            updated[total - magnitude] += ways
        distribution = dict(updated)
    observed = abs(sum(values))
    extreme = sum(ways for total, ways in distribution.items() if abs(total) >= observed)
    return extreme / (2 ** len(values))


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]
