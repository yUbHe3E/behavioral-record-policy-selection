"""Assemble contemporaneous controlled evidence comparisons without model calls."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.canonical import canonical_json, stable_hash
from behavioral_records.io import read_jsonl, write_json, write_jsonl
from behavioral_records.semantic_pilot import build_semantic_pair_prompt


def event_caption(full):
    parts=[]
    for index,e in enumerate(full['episodes']):
        parts.append(f"Episode {index+1}. Context: {canonical_json(e['context'])}. "
                     f"Length: {e['length']}; return: {e['return']}; outcome: {e['outcome']}; "
                     f"terminated: {e['terminated']}; truncated: {e['truncated']}.")
        for s in e['steps']:
            parts.append(f"At step {s['t']}, action {s['action']} has recorded reward {s['reward']}; "
                         f"events: {canonical_json(s.get('events',[]))}; "
                         f"observed state: {canonical_json(s.get('state',{}))}; "
                         f"predicates: {canonical_json(s.get('predicates',{}))}.")
    return {'episodes_in_acquisition_order':'\n'.join(parts),
            'observation_timing':'state and events describe the same step as the action',
            'omitted_fields':[]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--summary-responses',type=Path)
    p.add_argument('--summary-attempts',type=Path)
    args=p.parse_args()
    out=args.output_dir.resolve()
    if out.exists(): raise SystemExit('OUTPUT_ALREADY_EXISTS')
    source=ROOT/'outputs/controlled_relational_record_20260901_v1_2'
    evidence={}
    for r in read_jsonl(source/'public_evidence.jsonl'):
        evidence.setdefault(r['policy_id'],{})[r['arm']]=r['public_evidence']
    summary_requests=[]
    summary_map=[]
    for policy,arms in sorted(evidence.items()):
        full=arms['FullTrace-Control']
        arms['Event-Caption']=event_caption(full)
        request_id='r1_summary_'+stable_hash(policy,16)
        prompt=('Summarize the observed validation episodes of one anonymous policy. '
                'Preserve observable context, action order, events, rewards, state changes, outcomes '
                'and how frequently episode patterns occur. Describe facts without judging the policy '
                'or adding unobserved motives. No task preference is supplied. '
                'Use at most 700 English words. Return exactly one JSON object with one string field '
                'named summary.\nEPISODES: '+canonical_json(full))
        summary_requests.append({'request_id':request_id,'response_schema':'neutral-summary-v1',
                                 'prompt':prompt,'prompt_hash':stable_hash(prompt)})
        summary_map.append({'request_id':request_id,'policy_id_private':policy})
    summaries={};failed_policies=set()
    if args.summary_responses:
        raw={r['request_id']:r for r in read_jsonl(args.summary_responses)}
        attempts={r['request_id']:r for r in read_jsonl(args.summary_attempts)} if args.summary_attempts else {}
        for m in summary_map:
            r=raw.get(m['request_id'],{})
            parsed=r.get('response',{})
            if not isinstance(parsed,dict) or not isinstance(parsed.get('summary'),str):
                if attempts.get(m['request_id'],{}).get('status')=='FAILED':
                    failed_policies.add(m['policy_id_private']);continue
                raise SystemExit('SUMMARY_MISSING_OR_INVALID_WITHOUT_ATTEMPT: do not silently drop candidates')
            summaries[m['policy_id_private']]={'summary':parsed['summary']}
    private={r['request_id']:r for r in read_jsonl(source/'judge_private_map.jsonl')}
    skeleton=[r for r in read_jsonl(source/'judge_requests.jsonl') if r['arm']=='Relational-Linked']
    arms=['Relational-Linked','Relational-LinkBroken','FullTrace-Control','Event-Caption']
    if summaries: arms.append('LLM-Summary')
    requests=[]; mapping=[];skipped=[]
    for old in skeleton:
        pm=private[old['request_id']]
        requirement=old['prompt'].split('REQUIREMENT: ',1)[1].split('\nPOLICY A EVIDENCE:',1)[0]
        a,b=pm['alias_a_policy_id'],pm['alias_b_policy_id']
        for arm in arms:
            request_id='r1_'+stable_hash([old['request_id'],arm],24)
            mapping.append({**pm,'request_id':request_id,'arm':arm})
            if arm=='LLM-Summary' and (a in failed_policies or b in failed_policies):
                skipped.append({'request_id':request_id,'arm':arm,'reason':'SUMMARY_GENERATION_FAILED',
                                'primary_score':0,'candidate_ids_private':[a,b]})
                continue
            ea,eb=(summaries[a],summaries[b]) if arm=='LLM-Summary' else (evidence[a][arm],evidence[b][arm])
            prompt=build_semantic_pair_prompt(requirement,ea,eb,arm='BehavioralEvidence')
            prompt=prompt.replace('The evidence was compiled without access to this requirement and contains factual observations only.',
                                  'The records describe validation behavior and were prepared without access to this requirement.')
            requests.append({**{k:v for k,v in old.items() if k not in ('request_id','prompt','prompt_hash','arm')},
                             'request_id':request_id,'arm':arm,'prompt':prompt,'prompt_hash':stable_hash(prompt)})
    requests.sort(key=lambda r:stable_hash(['schedule',r['request_id']]))
    out.mkdir(parents=True)
    write_jsonl(out/'summary_requests.jsonl',summary_requests)
    write_jsonl(out/'summary_private_map.jsonl',summary_map)
    write_jsonl(out/'judge_requests.jsonl',requests)
    write_jsonl(out/'judge_private_map.jsonl',mapping)
    write_jsonl(out/'unresolved_before_judging_private.jsonl',skipped)
    sizes=[len(r['prompt'].encode('utf-8')) for r in requests]
    summary_bytes=sum(len(r['prompt'].encode('utf-8')) for r in summary_requests)
    # Byte counts are deliberately conservative planning surrogates, not tokenizer measurements.
    report={'schema_version':'revision-baseline-preparation-v1','api_calls':0,'arms':arms,
            'summaries_included':bool(summaries),'summary_request_count':len(summary_requests),
            'planned_comparisons':len(mapping),'summary_generation_failed_policies':len(failed_policies),
            'summary_arm_unresolved_before_judging':len(skipped),
            'summary_failure_scoring':'zero credit in all planned comparison denominators; no retry or candidate removal',
            'judge_request_count':len(requests),'summary_input_bytes':summary_bytes,
            'judge_input_bytes':sum(sizes),'maximum_prompt_bytes':max(sizes),
            'summary_max_output_tokens_per_request':2048,'judge_max_output_tokens_per_request':512,
            'minimax_cny_conservative_planning_ceiling':{
                'summary_stage':summary_bytes*2.10/1e6+len(summary_requests)*2048*8.40/1e6,
                'prepared_judge_arms':sum(sizes)*2.10/1e6+len(requests)*512*8.40/1e6},
            'pricing_basis':'MiniMax-M3 standard <=512K, no cache discount; bytes are planning proxy only',
            'pending':'Generate summaries, then rebuild all five arms before judging.' if not summaries else None,
            'request_hash':stable_hash(requests),'summary_hash':stable_hash(summary_requests),
            'source_hashes':{n:__import__('hashlib').sha256((source/n).read_bytes()).hexdigest()
                             for n in ['public_evidence.jsonl','judge_requests.jsonl','judge_private_map.jsonl']}}
    write_json(out/'preparation.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
