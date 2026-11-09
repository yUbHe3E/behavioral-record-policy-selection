"""Evaluate the frozen external comparisons without treating calls as samples.

The operational reference is computed on disjoint raw episodes. Unresolved
responses score zero on determinate pairs; ambiguous references are reported
separately. Dyadic node resampling preserves reuse of both training chains.
Intervals describe this small policy cohort, not population-level confirmation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import fmean
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json,read_jsonl,write_json,write_jsonl
from behavioral_records.canonical import stable_hash

CONTRASTS=[('Relational-Linked','Relational-LinkBroken'),
           ('Relational-Text','Relational-TextBroken'),
           ('Relational-Text','Relational-Linked'),
           ('Relational-Linked','Aggregate'),
           ('Relational-Linked','Event-Caption'),
           ('Relational-Linked','LLM-Summary')]


def score_panels(mapping,responses,arms):
    ids=[r['request_id'] for r in responses]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate response identity')
    by_id={r['request_id']:r for r in responses}
    allowed={r['request_id'] for r in mapping if r['arm'] in arms}
    if not set(by_id)<=allowed:raise ValueError('Response outside frozen design')
    groups=defaultdict(list)
    for row in mapping:
        if row['arm'] in arms:groups[(row['pair_id'],row['scenario_id'],row['arm'])].append(row)
    panels=[]
    for (pair,req,arm),rows in sorted(groups.items()):
        if len(rows)!=4 or {(r['paraphrase'],r['order']) for r in rows}!={('P0','AB'),('P0','BA'),('P1','AB'),('P1','BA')}:
            raise ValueError('Expected four distinct presentation contexts per panel')
        determinate={r['oracle_status']=='DETERMINATE' for r in rows}
        if len(determinate)!=1:raise ValueError('Reference status changes across presentation')
        good=sum(r['request_id'] in by_id for r in rows)
        correct=sum(r['request_id'] in by_id and by_id[r['request_id']]['response']['winner']==r['expected_winner'] for r in rows)
        panels.append({'pair_id':pair,'requirement':req,'arm':arm,'determinate':determinate.pop(),
                       'score':correct/4,'valid_fraction':good/4,'valid_calls':good,'planned_calls':4,
                       'chain_a':rows[0]['chain_a'],'chain_b':rows[0]['chain_b'],
                       'policy_a_private':rows[0]['policy_a_private'],'policy_b_private':rows[0]['policy_b_private']})
    return panels


def balanced_mean(rows,value='effect',weights=None):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['requirement']].append((row[value],1. if weights is None else weights[i]))
    means=[]
    for group in groups.values():
        den=sum(w for _,w in group)
        if den<=0:return None
        means.append(sum(v*w for v,w in group)/den)
    return fmean(means) if means else None


def node_interval(rows,node_map=None,repetitions=10000):
    if not rows:return {'point':None,'interval_95':None,'nodes':0}
    node_map=node_map or {r[k]:r[k] for r in rows for k in ('chain_a','chain_b')}
    nodes=sorted({node_map[r[k]] for r in rows for k in ('chain_a','chain_b')})
    index={v:i for i,v in enumerate(nodes)}
    left=np.array([index[node_map[r['chain_a']]] for r in rows]);right=np.array([index[node_map[r['chain_b']]] for r in rows])
    rng=np.random.default_rng(20261105);samples=[]
    for _ in range(repetitions):
        counts=rng.multinomial(len(nodes),[1/len(nodes)]*len(nodes))
        weights=counts[left]*counts[right]
        # Within a shared seed block, replicate the block once per draw.
        same=left==right;weights[same]=counts[left[same]]
        estimate=balanced_mean(rows,weights=weights)
        if estimate is not None:samples.append(estimate)
    leave=[]
    for node in nodes:
        estimate=balanced_mean([r for r in rows if node_map[r['chain_a']]!=node and node_map[r['chain_b']]!=node])
        if estimate is not None:leave.append(estimate)
    return {'point':balanced_mean(rows),'interval_95':np.quantile(samples,[.025,.975]).tolist() if samples else None,
            'nodes':len(nodes),'resamples':len(samples),'resamples_without_all_requirement_support':repetitions-len(samples),
            'leave_one_node_out_range':[min(leave),max(leave)] if leave else None,
            'interpretation':'Descriptive sensitivity interval from a small reused policy cohort; no significance or broad population claim.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--responses-dir',type=Path,required=True)
    p.add_argument('--provider',choices=['minimax','deepseek'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();prep=read_json(a.prepared/'preparation.json');plan=read_json(a.collection/'collection_plan_private.json')
    requests=read_jsonl(a.prepared/('judge_requests.jsonl' if a.provider=='minimax' else 'deepseek_requests.jsonl'))
    expected_hash=prep['request_hash' if a.provider=='minimax' else 'deepseek_request_hash']
    if stable_hash(requests)!=expected_hash:raise SystemExit('REQUEST_HASH_CHANGED')
    arms=plan['source_plan']['external_evaluation'][a.provider+'_arms']
    mapping=read_jsonl(a.prepared/'judge_private_map.jsonl')
    wanted={r['request_id'] for r in mapping if r['arm'] in arms}
    responses=read_jsonl(a.responses_dir/'judge_responses.jsonl') if (a.responses_dir/'judge_responses.jsonl').exists() else []
    attempts=read_jsonl(a.responses_dir/'attempts.jsonl') if (a.responses_dir/'attempts.jsonl').exists() else []
    attempt_ids=[r['request_id'] for r in attempts]
    if len(attempt_ids)!=len(set(attempt_ids)) or not set(attempt_ids)<={r['request_id'] for r in requests}:
        raise SystemExit('ATTEMPTS_OUTSIDE_FROZEN_REQUESTS_OR_DUPLICATED')
    if not {r['request_id'] for r in responses}<=set(attempt_ids):raise SystemExit('RESPONSE_WITHOUT_ATTEMPT')
    unresolved=read_jsonl(a.prepared/'summary_unresolved_private.jsonl')
    summary_failures={r['request_id'] for r in unresolved if r['reason']=='SUMMARY_GENERATION_FAILED' and r['arm'] in arms}
    complete=wanted<=set(attempt_ids)|summary_failures
    if a.provider=='minimax' and not prep['summary_stage_complete']:complete=False
    if not complete and not a.allow_partial:raise SystemExit('INCOMPLETE_DESIGN_NOT_FINAL_ANALYSIS')
    panels=score_panels(mapping,responses,arms)
    candidates={r['candidate_id']:r for r in plan['candidates']}
    seed_map={r['chain_id']:str(r.get('training_seed_private',r['chain_id'])) for r in candidates.values()}
    results={}
    requirements=sorted({r['requirement'] for r in panels})
    for scope in ['pooled']+requirements:
        selected=[r for r in panels if r['determinate'] and (scope=='pooled' or r['requirement']==scope)]
        accuracies={arm:balanced_mean([r for r in selected if r['arm']==arm],value='score') for arm in arms}
        lookup={(r['pair_id'],r['requirement'],r['arm']):r for r in selected};contrasts=[]
        for left,right in CONTRASTS:
            if left not in arms or right not in arms:continue
            rows=[]
            for r in selected:
                if r['arm']==left:
                    control=lookup[(r['pair_id'],r['requirement'],right)]
                    rows.append({**r,'effect':r['score']-control['score']})
            item={'left':left,'right':right,'pair_requirement_count':len(rows),'chain_resampling':node_interval(rows)}
            if a.provider and plan['environment']=='lunar':item['shared_seed_block_sensitivity']=node_interval(rows,seed_map)
            contrasts.append(item)
        results[scope]={'accuracy':accuracies,'contrasts':contrasts}
    report={'status':'COMPLETE' if complete else 'PROVISIONAL_INCOMPLETE','environment':plan['environment'],
            'provider':a.provider,'planned_calls':len(wanted),'attempted_calls':len(attempts),'valid_calls':len(responses),
            'summary_unavailable_planned_calls':len(summary_failures),'served_models':dict(Counter(r.get('served_model','unknown') for r in responses)),
            'ambiguous_pair_requirements':len({(r['pair_id'],r['requirement']) for r in panels if not r['determinate']}),
            'accuracy_definition':'Four presentation contexts per pair, equally weighted; requirements equally weighted when pooled. Unresolved/TIE/ABSTAIN score zero on determinate references.',
            'reference_scope':'Frozen operational definitions on disjoint raw episodes; no human preference equivalence claim.',
            'primary_contrast':'Relational-Linked minus Relational-LinkBroken; all other contrasts secondary.',
            'dependence':'Calls are not independent samples. Both endpoints of reused chain pairs are resampled together; Lunar also groups chains sharing the same training seed.',
            'results':results,'manuscript_modified':False}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(a.output_dir/'evaluation.json',report);write_jsonl(a.output_dir/'pair_scores_private.jsonl',panels)
    print(json.dumps({k:report[k] for k in ('status','environment','provider','planned_calls','attempted_calls','valid_calls')}))


if __name__=='__main__':main()
