from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .canonical import stable_hash


def build_candidate_chain_map(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map every policy checkpoint to its indivisible PPO training chain."""

    result: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        profile = str(row["training_profile_private"])
        seed = int(row["training_seed_private"])
        chain_id = f"{profile}|seed={seed}"
        if candidate_id in result:
            raise ValueError(f"Duplicate LunarLander candidate: {candidate_id}")
        result[candidate_id] = {
            "training_chain_id": chain_id,
            "training_profile": profile,
            "training_seed": seed,
            "checkpoint_timestep": int(row["checkpoint_timestep_private"]),
        }
    if not result:
        raise ValueError("No LunarLander candidates were supplied")
    return result


def _pair_policies(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(policy_id)
        for row in rows
        for policy_id in (row["policy_a"], row["policy_b"])
    }


def _chain_ids(
    policies: set[str], candidate_chain_map: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    missing = sorted(policies - set(candidate_chain_map))
    if missing:
        raise KeyError(f"Policies missing from training-chain map: {missing}")
    return {
        str(candidate_chain_map[policy_id]["training_chain_id"])
        for policy_id in policies
    }


def summarize_pair_coverage(
    pair_rows: Sequence[Mapping[str, Any]],
    candidate_chain_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe pair, policy, and training-chain coverage without inference."""

    rows = [dict(row) for row in pair_rows]
    determinate = [row for row in rows if row["oracle_status"] == "DETERMINATE"]
    all_policies = _pair_policies(rows) if rows else set()
    determinate_policies = _pair_policies(determinate) if determinate else set()
    pool_policies = set(candidate_chain_map)
    all_chains = _chain_ids(all_policies, candidate_chain_map) if rows else set()
    determinate_chains = (
        _chain_ids(determinate_policies, candidate_chain_map) if determinate else set()
    )
    label_counts = Counter(str(row["oracle_label"]) for row in rows)
    decisive_fields = Counter(
        str(row.get("oracle_detail_private", {}).get("decisive_field"))
        for row in determinate
    )
    return {
        "eligible_pair_rows": len(rows),
        "unique_pair_ids": len({str(row["pair_id"]) for row in rows}),
        "determinate_pair_rows": len(determinate),
        "determinate_pair_fraction": len(determinate) / len(rows) if rows else None,
        "ambiguous_pair_rows": len(rows) - len(determinate),
        "oracle_label_counts": dict(sorted(label_counts.items())),
        "candidate_pool_policies": len(pool_policies),
        "eligible_unique_policies": len(all_policies),
        "eligible_policy_pool_fraction": (
            len(all_policies) / len(pool_policies) if pool_policies else None
        ),
        "determinate_unique_policies": len(determinate_policies),
        "determinate_policy_pool_fraction": (
            len(determinate_policies) / len(pool_policies) if pool_policies else None
        ),
        "candidate_pool_training_chains": len(
            {
                str(row["training_chain_id"])
                for row in candidate_chain_map.values()
            }
        ),
        "eligible_training_chains": len(all_chains),
        "determinate_training_chains": len(determinate_chains),
        "decisive_field_counts_among_determinate": dict(sorted(decisive_fields.items())),
    }


def summarize_selected_pair_reuse(
    selected_rows_by_requirement: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_chain_map: Mapping[str, Mapping[str, Any]],
    determinate_rows_by_requirement: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Report policy reuse for pair units, not repeated model calls."""

    per_requirement: dict[str, dict[str, Any]] = {}
    combined_incidence: Counter[str] = Counter()
    combined_rows: list[Mapping[str, Any]] = []
    for requirement_id, source_rows in sorted(selected_rows_by_requirement.items()):
        rows = [dict(row) for row in source_rows]
        incidence = Counter(
            str(policy_id)
            for row in rows
            for policy_id in (row["policy_a"], row["policy_b"])
        )
        maximum = max(incidence.values(), default=0)
        policies = set(incidence)
        chains = _chain_ids(policies, candidate_chain_map) if policies else set()
        determinate_rows = (
            [dict(row) for row in determinate_rows_by_requirement[requirement_id]]
            if determinate_rows_by_requirement is not None
            else []
        )
        determinate_policies = (
            _pair_policies(determinate_rows) if determinate_rows else set()
        )
        per_requirement[requirement_id] = {
            "selected_requirement_pair_units": len(rows),
            "selected_unique_physical_pair_ids": len(
                {str(row["pair_id"]) for row in rows}
            ),
            "all_selected_pairs_are_determinate": all(
                row["oracle_status"] == "DETERMINATE" for row in rows
            ),
            "selected_unique_policies": len(policies),
            "selected_training_chains": len(chains),
            "maximum_pair_appearances_for_one_policy": maximum,
            "policies_at_maximum": sorted(
                policy_id for policy_id, count in incidence.items() if count == maximum
            ),
            "selected_pair_coverage_of_determinate": (
                len(rows) / len(determinate_rows) if determinate_rows else None
            ),
            "selected_policy_coverage_of_determinate": (
                len(policies & determinate_policies) / len(determinate_policies)
                if determinate_policies
                else None
            ),
        }
        combined_incidence.update(incidence)
        combined_rows.extend(rows)

    maximum_combined = max(combined_incidence.values(), default=0)
    physical_pair_requirements: dict[str, set[str]] = defaultdict(set)
    for row in combined_rows:
        physical_pair_requirements[str(row["pair_id"])].add(
            str(row["requirement_id"])
        )
    return {
        "unit_definition": (
            "One policy appearance in one selected requirement-pair unit. "
            "AB/BA orders, paraphrases, repeats, evidence arms, and API calls are excluded."
        ),
        "by_requirement": per_requirement,
        "combined_across_requirements": {
            "selected_requirement_pair_units": len(combined_rows),
            "unique_physical_pair_ids": len(physical_pair_requirements),
            "physical_pair_ids_reused_across_requirements": sum(
                len(requirements) > 1
                for requirements in physical_pair_requirements.values()
            ),
            "selected_unique_policies": len(combined_incidence),
            "maximum_requirement_pair_appearances_for_one_policy": maximum_combined,
            "policies_at_maximum": sorted(
                policy_id
                for policy_id, count in combined_incidence.items()
                if count == maximum_combined
            ),
        },
    }


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1], dtype=float)
    count_zero = max(1, int(np.sum(y == 0)))
    count_one = max(1, int(np.sum(y == 1)))
    sample_weight = np.where(y == 1, 0.5 / count_one, 0.5 / count_zero)
    for _ in range(int(iterations)):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - y) * sample_weight)
        gradient[1:] += float(l2) * weights[1:] / max(1, len(x))
        weights -= float(learning_rate) * gradient
    return weights


def training_chain_group_disjoint_cv(
    pair_rows: Sequence[Mapping[str, Any]],
    candidate_chain_map: Mapping[str, Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a descriptive metric probe with entire PPO chains held out.

    Folds are blocked by training seed when multiple seeds exist. Consequently,
    all checkpoints from every profile/seed PPO chain remain on one side of a
    train/test split. Cross-boundary pairs are omitted from that fold.
    """

    determinate = [
        dict(row) for row in pair_rows if row["oracle_status"] == "DETERMINATE"
    ]
    policies = _pair_policies(determinate) if determinate else set()
    _chain_ids(policies, candidate_chain_map)
    seeds = sorted(
        {
            int(candidate_chain_map[policy_id]["training_seed"])
            for policy_id in policies
        }
    )
    if len(seeds) < 2:
        return {
            "status": "INSUFFICIENT_TRAINING_SEED_GROUPS",
            "inference_authorized": False,
            "eligible_determinate_pairs": len(determinate),
            "reason": "At least two training-seed blocks are required.",
        }

    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, seed in enumerate(seeds):
        test_groups = {
            str(info["training_chain_id"])
            for info in candidate_chain_map.values()
            if int(info["training_seed"]) == seed
        }

        def groups(row: Mapping[str, Any]) -> tuple[str, str]:
            return (
                str(candidate_chain_map[str(row["policy_a"])]["training_chain_id"]),
                str(candidate_chain_map[str(row["policy_b"])]["training_chain_id"]),
            )

        train = [
            row for row in determinate if all(group not in test_groups for group in groups(row))
        ]
        test = [
            row for row in determinate if all(group in test_groups for group in groups(row))
        ]
        crossing = len(determinate) - len(train) - len(test)
        train_policies = _pair_policies(train) if train else set()
        test_policies = _pair_policies(test) if test else set()
        train_groups = _chain_ids(train_policies, candidate_chain_map) if train else set()
        observed_test_groups = (
            _chain_ids(test_policies, candidate_chain_map) if test else set()
        )
        train_labels = {str(row["oracle_label"]) for row in train}
        status = "EVALUATED"
        if not test:
            status = "SKIPPED_NO_WITHIN_BLOCK_TEST_PAIRS"
        elif train_labels != {"A", "B"}:
            status = "SKIPPED_TRAINING_SET_LACKS_BOTH_CLASSES"
        if status == "EVALUATED":
            x_train = np.asarray(
                [row["oriented_metric_difference"] for row in train], dtype=float
            )
            y_train = np.asarray(
                [1.0 if row["oracle_label"] == "A" else 0.0 for row in train],
                dtype=float,
            )
            weights = _fit_logistic(
                x_train,
                y_train,
                iterations=int(diagnostic["iterations"]),
                learning_rate=float(diagnostic["learning_rate"]),
                l2=float(diagnostic["l2"]),
            )
            design_test = np.column_stack(
                [
                    np.ones(len(test)),
                    np.asarray(
                        [row["oriented_metric_difference"] for row in test],
                        dtype=float,
                    ),
                ]
            )
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(design_test @ weights, -30.0, 30.0))
            )
            for row, probability in zip(test, probabilities):
                predictions.append(
                    {
                        "pair_id": str(row["pair_id"]),
                        "truth": str(row["oracle_label"]),
                        "prediction": "A" if float(probability) >= 0.5 else "B",
                        "probability_a": float(probability),
                        "fold": fold_index,
                        "heldout_training_seed": seed,
                    }
                )
        fold_rows.append(
            {
                "fold": fold_index,
                "heldout_training_seed": seed,
                "heldout_training_chain_ids": sorted(test_groups),
                "training_pairs": len(train),
                "test_pairs": len(test),
                "cross_boundary_pairs_omitted": crossing,
                "training_class_counts": dict(
                    sorted(Counter(str(row["oracle_label"]) for row in train).items())
                ),
                "test_class_counts": dict(
                    sorted(Counter(str(row["oracle_label"]) for row in test).items())
                ),
                "training_unique_policies": len(train_policies),
                "test_unique_policies": len(test_policies),
                "train_test_policy_overlap": sorted(train_policies & test_policies),
                "train_test_training_chain_overlap": sorted(
                    train_groups & observed_test_groups
                ),
                "status": status,
            }
        )

    recalls: dict[str, float | None] = {}
    for label in ("A", "B"):
        selected = [row for row in predictions if row["truth"] == label]
        recalls[label] = (
            sum(row["prediction"] == label for row in selected) / len(selected)
            if selected
            else None
        )
    balanced_accuracy = (
        (float(recalls["A"]) + float(recalls["B"])) / 2.0
        if recalls["A"] is not None and recalls["B"] is not None
        else None
    )
    tested_pair_ids = {str(row["pair_id"]) for row in predictions}
    tested_source_rows = [
        row for row in determinate if str(row["pair_id"]) in tested_pair_ids
    ]
    tested_policies = _pair_policies(tested_source_rows) if tested_source_rows else set()
    tested_chains = (
        _chain_ids(tested_policies, candidate_chain_map) if tested_policies else set()
    )
    eligible_chains = _chain_ids(policies, candidate_chain_map) if policies else set()
    evaluated_folds = sum(row["status"] == "EVALUATED" for row in fold_rows)
    return {
        "status": (
            "EVALUATED_DESCRIPTIVE_ONLY"
            if balanced_accuracy is not None and evaluated_folds >= 2
            else "INSUFFICIENT_GROUP_DISJOINT_COVERAGE"
        ),
        "fold_unit": "ppo_training_chain_profile_x_seed",
        "fold_partition": "training_seed_blocked",
        "inference_authorized": False,
        "dependence_statement": (
            "Balanced accuracy is descriptive only. Test pairs share policies and "
            "training chains, so pair predictions are not independent observations; "
            "no p-value or confidence interval is reported."
        ),
        "eligible_determinate_pairs": len(determinate),
        "eligible_determinate_unique_policies": len(policies),
        "eligible_determinate_training_chains": len(eligible_chains),
        "candidate_training_chains": len(
            {
                str(info["training_chain_id"])
                for info in candidate_chain_map.values()
            }
        ),
        "evaluated_folds": evaluated_folds,
        "test_predictions": len(predictions),
        "unique_test_pair_ids": len(tested_pair_ids),
        "test_pair_coverage_of_determinate": (
            len(tested_pair_ids) / len(determinate) if determinate else None
        ),
        "unique_test_policies": len(tested_policies),
        "test_policy_coverage_of_determinate": (
            len(tested_policies) / len(policies) if policies else None
        ),
        "unique_test_training_chains": len(tested_chains),
        "test_training_chain_coverage_of_determinate": (
            len(tested_chains) / len(eligible_chains) if eligible_chains else None
        ),
        "class_recall_descriptive": recalls,
        "balanced_accuracy_descriptive": balanced_accuracy,
        "folds": fold_rows,
        "prediction_hash": stable_hash(predictions),
    }
