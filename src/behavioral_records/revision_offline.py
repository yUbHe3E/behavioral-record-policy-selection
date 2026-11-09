"""Revision analyses of archived observations; no model calls or label generation."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import canonical_json
from .lunarlander_reanalysis import _fit_logistic
from .lunarlander_semantic import flatten_matching_features
from .relational_record import RelationalBehaviorCompiler, _compile_episode


def bootstrap_mean(values: Sequence[float], *, seed: int, draws: int = 10000) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("A finite, nonempty set of independent-unit effects is required")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def requirement_effects(pair_rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict:
    cells = defaultdict(dict)
    for row in pair_rows:
        if row["determinate"]:
            key = (row["scenario_id"], row["pair_id"])
            if row["arm"] in cells[key]:
                raise ValueError("Duplicate pair-arm score")
            cells[key][row["arm"]] = float(row["oracle_label_mass"])
    effects = defaultdict(list)
    for (scenario, _), scores in sorted(cells.items()):
        effects[scenario].append(scores["Relational-Linked"] - scores["Relational-LinkBroken"])
    result = {s: bootstrap_mean(x, seed=seed + i) for i, (s, x) in enumerate(sorted(effects.items()))}
    rng = np.random.default_rng(seed + 10)
    x, y = (np.asarray(effects[s]) for s in ("C1", "C2"))
    # Resample the two grids separately; equal suffixes do not imply paired units.
    delta = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    delta -= y[rng.integers(0, len(y), (10000, len(y)))].mean(1)
    return {"by_requirement": result,
            "recovery_minus_necessity": {"mean": float(x.mean() - y.mean()),
                                         "ci95": np.quantile(delta, [.025, .975]).tolist()},
            "analysis_role": "post_review_exploratory_estimation",
            "scope": "Variation over the constructed pair grids, not a sample of arbitrary RL tasks.",
            "calls_are_not_independent_units": True}


def public_full_trace(episodes: Sequence[Mapping[str, Any]]) -> dict:
    """Allowlist observed facts. Excludes identities, seeds, lineage and all metadata."""
    rows = []
    for ep in episodes:
        if ep.get("metadata", {}).get("evaluation_split") not in (None, "validation"):
            raise ValueError("Reference episodes cannot enter judge-visible evidence")
        rows.append({
            "context": copy.deepcopy(dict(ep["context"])),
            "length": int(ep["length"]), "outcome": str(ep["outcome"]),
            "return": float(ep["return"]), "terminated": bool(ep["terminated"]),
            "truncated": bool(ep["truncated"]),
            "steps": [{k: copy.deepcopy(step[k]) for k in
                       ("t", "action", "state", "events", "predicates", "reward") if k in step}
                      for step in ep["steps"]]})
    return {"representation": "deterministic_raw_validation_trajectories", "episodes": rows}


def compression_components(full: Mapping[str, Any]) -> dict:
    """2x2 payload factorial, distinguished from the production record's overhead."""
    snapshots = list(full["episodes"])
    deltas = [_compile_episode(ep) for ep in snapshots]
    sizes = {}
    for encoding, episodes in (("snapshot", snapshots), ("delta", deltas)):
        for dedup in (False, True):
            if dedup:
                counts = Counter(canonical_json(ep) for ep in episodes)
                import json
                payload = [{"episode": json.loads(k), "multiplicity": n}
                           for k, n in sorted(counts.items())]
            else:
                payload = [{"episode": ep, "multiplicity": 1} for ep in episodes]
            sizes[f"{encoding}_{'dedup' if dedup else 'all'}"] = len(canonical_json(payload))
    linked = RelationalBehaviorCompiler().compile(full)
    raw_chars, linked_chars = len(canonical_json(full)), len(canonical_json(linked))
    return {"payload_characters": sizes, "full_characters": raw_chars,
            "linked_characters": linked_chars, "linked_to_full_ratio": linked_chars / raw_chars,
            "episode_count": len(snapshots), "unique_patterns": len(linked["patterns"]),
            "steps": sum(int(x["length"]) for x in snapshots),
            "character_counts_are_not_token_counts": True}


def train_only_metric_probe(pair_rows, chain_map, records, feature_names) -> dict:
    """Re-fit feature scales inside each seed-blocked fold, before pair subtraction.

    Historical eligible-pair selection is retained, so this is a diagnostic
    correction and does not retroactively create a prospective held-out study.
    """
    rows = [r for r in pair_rows if r["oracle_status"] == "DETERMINATE"]
    vectors = {k: np.asarray(flatten_matching_features(v, feature_names)) for k, v in records.items()}
    preds, folds = [], []
    for seed in sorted({x["training_seed"] for x in chain_map.values()}):
        test_ids = {k for k, v in chain_map.items() if v["training_seed"] == seed}
        train_ids = set(chain_map) - test_ids
        train = [r for r in rows if {r["policy_a"], r["policy_b"]} <= train_ids]
        test = [r for r in rows if {r["policy_a"], r["policy_b"]} <= test_ids]
        used_train = {r[k] for r in train for k in ("policy_a", "policy_b")}
        used_test = {r[k] for r in test for k in ("policy_a", "policy_b")}
        if not train or not test or {r["oracle_label"] for r in train} != {"A", "B"}:
            folds.append({"seed": seed, "status": "INSUFFICIENT_FOLD_DATA"})
            continue
        raw = np.stack([vectors[k] for k in sorted(used_train)])
        mean, scale = raw.mean(0), raw.std(0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        def features(rs):
            return np.stack([((vectors[r["policy_a"]] - mean) / scale) -
                             ((vectors[r["policy_b"]] - mean) / scale) for r in rs])
        y = np.asarray([r["oracle_label"] == "A" for r in train], dtype=float)
        weights = _fit_logistic(features(train), y, iterations=2000, learning_rate=.05, l2=1.)
        logits = np.column_stack([np.ones(len(test)), features(test)]) @ weights
        preds.extend({"pair_id": r["pair_id"], "truth": r["oracle_label"],
                      "prediction": "A" if z >= 0 else "B"} for r, z in zip(test, logits))
        folds.append({"seed": seed, "train_pairs": len(train), "test_pairs": len(test),
                      "scale_fit_policies": sorted(used_train), "test_policies": sorted(used_test),
                      "overlap": sorted(used_train & used_test), "scale": scale.tolist(),
                      "status": "EVALUATED"})
    recalls = [np.mean([r["truth"] == r["prediction"] for r in preds if r["truth"] == k])
               for k in ("A", "B") if any(r["truth"] == k for r in preds)]
    return {"balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
            "test_pairs": len(preds), "folds": folds, "predictions": preds,
            "role": "diagnostic_on_historical_pair_selection_no_population_inference"}


def leave_group_out_effects(pair_scores, pair_manifest, chain_map, group_field) -> dict:
    cells = defaultdict(dict)
    for r in pair_scores:
        cells[r["pair_id"]][r["arm"]] = r["oracle_label_mass"]
    items = []
    for p in pair_manifest:
        scores = cells[p["pair_id"]]
        items.append((float(scores["B2-DenseTrace"] - scores["B1"]),
                      {str(chain_map[p[k]][group_field]) for k in ("policy_a", "policy_b")}))
    out = []
    for g in sorted(set().union(*(groups for _, groups in items))):
        keep = [effect for effect, groups in items if g not in groups]
        out.append({"omitted_group": g, "remaining_pairs": len(keep),
                    "mean": float(np.mean(keep)) if keep else None})
    return {"full_mean": float(np.mean([x for x, _ in items])), "leave_one_group_out": out,
            "role": "dependence_sensitivity_not_a_confidence_interval",
            "all_pairs_incident_to_omitted_group_removed": True}
