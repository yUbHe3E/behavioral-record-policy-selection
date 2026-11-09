"""Evaluate the fixed three-generation extension at the retention-seed level."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from behavioral_records.canonical import canonical_json, stable_hash
from behavioral_records.io import read_json, read_jsonl, write_json
from behavioral_records.revision_offline import bootstrap_mean
from behavioral_records.semantic_pilot import compare_boolean_profiles, adjudicate_oracle

LINKED = 'Relational-Linked-LLM-PBT'
BROKEN = 'Relational-LinkBroken-LLM-PBT'


def seed_contrast(rows, generation):
    index = {(r['scenario_id'], r['outer_seed'], r['selector_arm']): r
             for r in rows if r['generation'] == generation}
    seeds = sorted({r['outer_seed'] for r in rows})
    differences = []; requirement = defaultdict(list)
    for seed in seeds:
        effects = []
        for scenario in ('C1', 'C2'):
            left = index[(scenario, seed, LINKED)]
            right = index[(scenario, seed, BROKEN)]
            delta = (right['selected_parent_mean_semantic_violation_rate']
                     - left['selected_parent_mean_semantic_violation_rate'])
            effects.append(delta); requirement[scenario].append(delta)
        differences.append(fmean(effects))
    result = bootstrap_mean(differences, seed=20261104)
    result['by_requirement'] = {k: bootstrap_mean(v, seed=20261105)
                                 for k, v in requirement.items()}
    result['seed_effects'] = dict(zip(map(str, seeds), differences))
    observed = abs(fmean(differences))
    randomizations = [abs(fmean(s*x for s, x in zip(signs, differences)))
                      for signs in product((-1, 1), repeat=len(seeds))]
    result['exact_two_sided_sign_flip_p'] = fmean(x >= observed-1e-12 for x in randomizations)
    return result


def reference_label(tests_a, tests_b):
    """Require agreement between intervention and behavioral layers."""
    decisions = []
    for layer in ('causal::', 'behavior::'):
        names = sorted(k for k in tests_a if k.startswith(layer))
        if not names or set(names) != {k for k in tests_b if k.startswith(layer)}:
            raise ValueError('REFERENCE_LAYER_MISMATCH')
        decisions.append(compare_boolean_profiles({k: tests_a[k] for k in names},
                                                  {k: tests_b[k] for k in names}))
    adjudication = adjudicate_oracle(*decisions)
    return adjudication['label'] if adjudication['status']=='DETERMINATE' else 'AMBIGUOUS'


def generation_zero_sensitivity(folder):
    """Use all scheduled initial edges; no extra calls or pair selection."""
    compiled = read_json(folder/'compilations_private.json')
    private = read_jsonl(folder/'judge_private_map.jsonl')
    responses = read_jsonl(folder/'judge_minimax'/'judge_responses.jsonl')
    observed = {r['request_id']: r['response']['winner'] for r in responses}
    if len(observed) != len(responses): raise ValueError('DUPLICATE_RESPONSES')
    groups = defaultdict(list); counts = defaultdict(int)
    for row in private:
        s, seed, arm = row['scenario_id'], row['outer_seed'], row['selector_arm']
        data = compiled[f'{s}__s{seed}__{arm}']
        a, b = row['alias_a_policy_id'], row['alias_b_policy_id']
        # The original aggregate panel includes all declared action/event counts.
        same = canonical_json(data['evidence'][a]['B1']) == canonical_json(data['evidence'][b]['B1'])
        label = reference_label(data['hidden_oracle'][a]['tests'], data['hidden_oracle'][b]['tests'])
        stratum = 'exact_aggregate_match' if same else 'unequal_aggregate_panels'
        counts[f'{s}::{stratum}::{label}'] += 1
        if label != 'AMBIGUOUS':
            groups[(stratum, s, seed, arm)].append(float(observed.get(row['request_id']) == label))
    reports = {}
    for stratum in ('exact_aggregate_match', 'unequal_aggregate_panels'):
        seed_values = defaultdict(list); cell_rows = []
        for (name, scenario, seed, arm), scores in sorted(groups.items()):
            if name != stratum or arm != LINKED: continue
            other = groups[(name, scenario, seed, BROKEN)]
            if len(scores) != len(other): raise ValueError('MISMATCHED_STRATUM_DENOMINATOR')
            delta = fmean(scores)-fmean(other)
            seed_values[seed].append(delta)
            cell_rows.append({'scenario': scenario, 'seed': seed, 'planned_calls_per_arm': len(scores),
                              'linked_rate': fmean(scores), 'disrupted_rate': fmean(other), 'difference': delta})
        effects = [fmean(v) for _, v in sorted(seed_values.items())]
        reports[stratum] = {'cells': cell_rows, 'seed_effects': effects,
                            'estimate': bootstrap_mean(effects, seed=20261106) if effects else None}
    return {'role': 'secondary_generation_zero_matching_sensitivity',
            'sampling_unit': 'retention_seed_with_requirements_averaged_within_seed',
            'counts_include_both_arms_and_orders': dict(counts), 'strata': reports,
            'missing_and_invalid_responses_receive_zero_credit': True,
            'exogenous_restoration_is_not_removed_by_this_analysis': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    rows = []; costs = []
    for g in (0, 1, 2):
        folder = args.root/f'retention_generation_{g}'
        batch = folder/'judge_minimax'
        status = read_json(batch/'status.json')
        if status['status'] not in ('COMPLETE', 'COMPLETE_WITH_FAILED_ATTEMPTS'):
            raise SystemExit('INCOMPLETE_RETENTION_GENERATION')
        if stable_hash(read_jsonl(folder/'judge_requests.jsonl')) != read_json(batch/'batch_plan.json')['source_hash']:
            raise SystemExit('RETENTION_REQUEST_HASH_MISMATCH')
        selected = read_json(folder/'selection_private.json')
        expected = {(s, seed, arm, g) for s in ('C1', 'C2') for seed in range(5, 13) for arm in (LINKED, BROKEN)}
        actual = {(r['scenario_id'], r['outer_seed'], r['selector_arm'], r['generation']) for r in selected}
        if actual != expected or len(selected) != 32: raise SystemExit('INCOMPLETE_SELECTION_GROUPS')
        if any(r['valid_request_rate'] < .8 or not r['scheduled_graph_connected'] for r in selected):
            raise SystemExit('RETENTION_QUALITY_GATE_FAILED')
        rows.extend(selected); costs.append(status)
    result = {'schema': 'revision-relational-retention-evaluation-v1',
              'primary_generation_2': seed_contrast(rows, 2),
              'secondary_generations': {str(g): seed_contrast(rows, g) for g in (0, 1)},
              'generation_zero_matching': generation_zero_sensitivity(args.root/'retention_generation_0'),
              'selection_rows': rows, 'batch_statuses': costs,
              'primary_sampling_units': 8, 'original_studies_not_pooled': True}
    write_json(args.output, result)
    print(result['primary_generation_2'])


if __name__ == '__main__': main()
