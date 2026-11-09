from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def _response_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("response", row)
    if not isinstance(payload, Mapping):
        raise ValueError("Judge response payload must be a JSON object")
    return payload


def aggregate_pairwise_judgments(
    requests: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map AB/BA answers to canonical policy-pair preferences."""
    request_by_id = {str(row["request_id"]): row for row in requests}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for response_row in responses:
        request_id = str(response_row.get("request_id", ""))
        if request_id not in request_by_id:
            raise ValueError(f"Unknown request_id in judge response: {request_id}")
        request = request_by_id[request_id]
        payload = _response_payload(response_row)
        winner = str(payload.get("winner", "")).upper()
        if winner not in {"A", "B", "TIE"}:
            raise ValueError(f"Invalid winner for request {request_id}: {winner}")
        confidence = float(payload.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence outside [0, 1] for request {request_id}")

        left_id, right_id = sorted((str(request["policy_a_id"]), str(request["policy_b_id"])))
        if winner == "TIE":
            signed_vote = 0.0
            winner_policy_id = None
        else:
            winner_policy_id = str(request["policy_a_id"] if winner == "A" else request["policy_b_id"])
            signed_vote = confidence if winner_policy_id == left_id else -confidence
        key = (str(request["objective_id"]), str(request["evidence_arm"]), str(request["pair_id"]))
        groups[key].append(
            {
                "request_id": request_id,
                "order": str(request["order"]),
                "left_policy_id": left_id,
                "right_policy_id": right_id,
                "winner_policy_id": winner_policy_id,
                "signed_vote": signed_vote,
                "confidence": confidence,
            }
        )

    rows: list[dict[str, Any]] = []
    for (objective_id, evidence_arm, pair_id), votes in sorted(groups.items()):
        left_ids = {row["left_policy_id"] for row in votes}
        right_ids = {row["right_policy_id"] for row in votes}
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise ValueError(f"Inconsistent policy mapping for pair {pair_id}")
        signed_preference = sum(row["signed_vote"] for row in votes) / len(votes)
        non_tie_signs = {
            1 if row["signed_vote"] > 0 else -1
            for row in votes
            if row["signed_vote"] != 0
        }
        rows.append(
            {
                "objective_id": objective_id,
                "evidence_arm": evidence_arm,
                "pair_id": pair_id,
                "left_policy_id": next(iter(left_ids)),
                "right_policy_id": next(iter(right_ids)),
                "response_count": len(votes),
                "orders_observed": sorted({row["order"] for row in votes}),
                "order_consistent": len(non_tie_signs) <= 1,
                "signed_preference_for_left": signed_preference,
                "left_win_probability": (1.0 + signed_preference) / 2.0,
                "mean_confidence": sum(row["confidence"] for row in votes) / len(votes),
            }
        )
    return rows


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Pairwise ranking system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _is_connected(policy_ids: list[str], pairs: Iterable[Mapping[str, Any]]) -> bool:
    if not policy_ids:
        return False
    adjacency = {policy_id: set() for policy_id in policy_ids}
    for row in pairs:
        left = str(row["left_policy_id"])
        right = str(row["right_policy_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = {policy_ids[0]}
    queue: deque[str] = deque([policy_ids[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return len(visited) == len(policy_ids)


def rank_pairwise_consensus(
    consensus_rows: Iterable[Mapping[str, Any]],
    *,
    regularization: float = 0.1,
    probability_clip: float = 0.05,
) -> list[dict[str, Any]]:
    """Fit one fixed regularized log-odds ranking per objective and arm."""
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be between 0 and 0.5")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        groups[(str(row["objective_id"]), str(row["evidence_arm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (objective_id, evidence_arm), pairs in sorted(groups.items()):
        policy_ids = sorted(
            {str(row["left_policy_id"]) for row in pairs}
            | {str(row["right_policy_id"]) for row in pairs}
        )
        index = {policy_id: position for position, policy_id in enumerate(policy_ids)}
        matrix = [[0.0 for _ in policy_ids] for _ in policy_ids]
        vector = [0.0 for _ in policy_ids]
        for position in range(len(policy_ids)):
            matrix[position][position] = regularization
        for row in pairs:
            left = index[str(row["left_policy_id"])]
            right = index[str(row["right_policy_id"])]
            probability = min(
                1.0 - probability_clip,
                max(probability_clip, float(row["left_win_probability"])),
            )
            target_margin = math.log(probability / (1.0 - probability))
            weight = float(row["response_count"])
            matrix[left][left] += weight
            matrix[right][right] += weight
            matrix[left][right] -= weight
            matrix[right][left] -= weight
            vector[left] += weight * target_margin
            vector[right] -= weight * target_margin
        scores = _solve_linear_system(matrix, vector)
        centered = [score - sum(scores) / len(scores) for score in scores]
        ranking = sorted(zip(policy_ids, centered), key=lambda row: (-row[1], row[0]))
        connected = _is_connected(policy_ids, pairs)
        consistent_fraction = sum(bool(row["order_consistent"]) for row in pairs) / len(pairs)
        for rank, (policy_id, score) in enumerate(ranking, start=1):
            output.append(
                {
                    "objective_id": objective_id,
                    "evidence_arm": evidence_arm,
                    "rank": rank,
                    "policy_id": policy_id,
                    "score": score,
                    "selected": rank == 1 and connected,
                    "pair_count": len(pairs),
                    "response_graph_connected": connected,
                    "order_consistent_pair_fraction": consistent_fraction,
                    "ranking_method": "regularized pairwise log-odds least squares v1",
                }
            )
    return output
