"""Score the frozen five-arm study, including failed summary generations."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json,read_jsonl,write_json
from behavioral_records.canonical import stable_hash
from behavioral_records.revision_offline import bootstrap_mean
from behavioral_records.controlled_confirmatory import _paired_sign_flip_p


def evaluate(private,responses,attempts,requests):
    index={r['request_id']:r for r in responses}
    if len(index)!=len(responses):raise ValueError('DUPLICATE_RESPONSES')
    allowed={r['request_id'] for r in private}
    if not set(index)<=allowed:raise ValueError('UNEXPECTED_RESPONSE')
    groups=defaultdict(list)
    for row in private:
        observed=index.get(row['request_id'],{}).get('response',{}).get('winner')
        valid=observed in {'A','B','TIE','ABSTAIN'}
        groups[(row['scenario_id'],row['pair_id'],row['arm'])].append({
            **row,'valid':valid,'correct':valid and observed==row['expected_winner'],
            'directional':observed in {'A','B'}})
    pairs=[];cells=defaultdict(list)
    for (scenario,pair,arm),rows in sorted(groups.items()):
        if len(rows)!=8:raise ValueError('INCOMPLETE_PLANNED_PAIR_ARM_CONTEXTS')
        pairs.append({'scenario_id':scenario,'pair_id':pair,'arm':arm,'request_count':len(rows),
                      'determinate':rows[0]['oracle_status']=='DETERMINATE',
                      'oracle_label_mass':fmean(r['correct'] for r in rows),
                      'valid_request_count':sum(r['valid'] for r in rows)})
        cells[(scenario,arm)].extend(rows)
    by_cell={}
    for (scenario,arm),rows in sorted(cells.items()):
        d=[r for r in rows if r['oracle_status']=='DETERMINATE']
        directional=[r for r in d if r['expected_winner'] in {'A','B'}]
        answered=[r for r in directional if r['directional']]
        by_cell[f'{scenario}::{arm}']={
            'reference_label_rate':fmean(r['correct'] for r in d),
            'end_to_end_valid_rate':fmean(r['valid'] for r in rows),
            'directional_coverage':fmean(r['directional'] for r in directional),
            'conditional_alignment':fmean(r['correct'] for r in answered) if answered else None,
            'planned_comparisons':len(rows)}
    arms=sorted({r['arm'] for r in private});contrasts={}
    for arm in arms:
        if arm=='Relational-Linked':continue
        ix={(r['scenario_id'],r['pair_id'],r['arm']):r for r in pairs if r['determinate']}
        vals=[];by_req=defaultdict(list)
        for (scenario,pair,a),r in ix.items():
            if a=='Relational-Linked':
                diff=r['oracle_label_mass']-ix[(scenario,pair,arm)]['oracle_label_mass']
                vals.append(diff);by_req[scenario].append(diff)
        result=bootstrap_mean(vals,seed=20261104)
        result['by_requirement']={k:bootstrap_mean(v,seed=20261105) for k,v in by_req.items()}
        result['role']='primary' if arm=='Relational-LinkBroken' else 'secondary_estimation'
        if arm=='Relational-LinkBroken':
            result['paired_sign_flip_p']=_paired_sign_flip_p(vals,seed=20261104)
        contrasts['Relational-Linked minus '+arm]=result
    req_by_id={r['request_id']:r for r in requests}
    costs=defaultdict(lambda:{'calls':0,'input_tokens':0,'output_tokens':0,'accounted_cost_cny':0.0})
    for a in attempts:
        cell=costs[req_by_id[a['request_id']]['arm']];cell['calls']+=1
        for name in ('input_tokens','output_tokens'):cell[name]+=int(a.get('usage',{}).get(name) or 0)
        cell['accounted_cost_cny']+=a['accounted_cost_cny']
    primary=contrasts['Relational-Linked minus Relational-LinkBroken']
    valid_gate=all(by_cell[f'{s}::{a}']['end_to_end_valid_rate']>=.8
                   for s in ('C1','C2') for a in ('Relational-Linked','Relational-LinkBroken'))
    return {'schema':'revision-baseline-evaluation-v1','pair_rows':pairs,'cells':by_cell,'contrasts':contrasts,
            'primary_response_quality_gate_passed':valid_gate,
            'primary_positive_effect_supported':valid_gate and primary['mean']>0 and primary['paired_sign_flip_p']<.05,
            'costs':dict(costs),'served_models':sorted({r.get('served_model','unknown') for r in responses}),
            'failed_summary_comparisons_remain_in_denominator':True,
            'secondary_contrasts_are_estimation_not_equivalence_tests':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--judge-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    requests=read_jsonl(args.bundle/'judge_requests.jsonl')
    plan=read_json(args.judge_dir/'batch_plan.json')
    if stable_hash(requests)!=plan['source_hash']:raise SystemExit('REQUEST_HASH_MISMATCH')
    status=read_json(args.judge_dir/'status.json')
    if status['status'] not in ('COMPLETE','COMPLETE_WITH_FAILED_ATTEMPTS'):raise SystemExit('BATCH_NOT_FINISHED')
    result=evaluate(read_jsonl(args.bundle/'judge_private_map.jsonl'),
                    read_jsonl(args.judge_dir/'judge_responses.jsonl'),
                    read_jsonl(args.judge_dir/'attempts.jsonl'),requests)
    result['batch_plan']=plan;result['status']=status
    write_json(args.output,result)
    print({k:v['mean'] for k,v in result['contrasts'].items()})


if __name__=='__main__':main()
