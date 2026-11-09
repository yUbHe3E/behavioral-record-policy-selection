from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, stable_hash
from .views import public_evidence


def build_pair_prompt(
    natural_language_objective: str,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    *,
    evidence_arm: str = "P_behavioral_record",
) -> str:
    """Build a judge prompt without policy identity or hidden-oracle output."""
    return "\n".join(
        [
            "You are selecting between two reinforcement-learning policies.",
            f"EVIDENCE ARM: {evidence_arm}",
            "Use only the factual evidence below. Both policies use the same frozen evidence settings.",
            "Do not infer training method, policy identity, or hidden evaluation results.",
            "If the evidence does not distinguish the policies for the requirement, return TIE.",
            "",
            f"REQUIREMENT: {natural_language_objective}",
            "",
            f"POLICY A EVIDENCE: {canonical_json(public_evidence(record_a))}",
            "",
            f"POLICY B EVIDENCE: {canonical_json(public_evidence(record_b))}",
            "",
            "Return exactly one JSON object with keys winner, confidence, and reason.",
            'winner must be "A", "B", or "TIE"; confidence must be between 0 and 1.',
        ]
    )


def make_pair_requests(
    objective: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    evidence_arm: str = "P_behavioral_record",
    pairs: Iterable[tuple[str, str]] | None = None,
    orders: Iterable[str] = ("AB", "BA"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_pairs = list(pairs) if pairs is not None else list(combinations(sorted(records), 2))
    for left_id, right_id in selected_pairs:
        if left_id not in records or right_id not in records:
            raise ValueError(f"Pair contains an unknown policy: {(left_id, right_id)}")
        if left_id == right_id:
            raise ValueError("Self-pairs are not permitted")
        left_id, right_id = sorted((left_id, right_id))
        pair_id = stable_hash(["pair-graph-v1", left_id, right_id], 16)
        for order in orders:
            if order == "AB":
                a_id, b_id = left_id, right_id
            elif order == "BA":
                a_id, b_id = right_id, left_id
            else:
                raise ValueError(f"Unsupported pair order: {order}")
            rows.append(
                {
                    "request_id": stable_hash([objective["id"], pair_id, evidence_arm, order], 20),
                    "pair_id": pair_id,
                    "objective_id": str(objective["id"]),
                    "evidence_arm": evidence_arm,
                    "order": order,
                    "policy_a_id": a_id,
                    "policy_b_id": b_id,
                    "prompt": build_pair_prompt(
                        str(objective["natural_language"]),
                        records[a_id],
                        records[b_id],
                        evidence_arm=evidence_arm,
                    ),
                }
            )
    return rows
