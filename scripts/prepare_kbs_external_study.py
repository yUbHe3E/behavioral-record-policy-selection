"""Audit external evidence, assemble all fixed comparisons, and account for cost.

No network calls. A missing summary remains missing; finalization requires one
documented attempt per needed policy, never a replacement authored from labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from statistics import fmean

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.canonical import canonical_json,stable_hash
from behavioral_records.io import read_json,read_jsonl,write_json,write_jsonl
from behavioral_records.lunarlander_semantic import read_gzip_jsonl
from behavioral_records.relational_record import RelationalBehaviorCompiler,relation_broken_control,audit_relation_break,_audit_boolean_fields
from behavioral_records.relational_text import render_relational_text,parse_relational_text
from behavioral_records.external_validation import public_observation_record,aggregate_record,factual_captions
from behavioral_records.semantic_pilot import build_semantic_pair_prompt


def evidence_text(record):
    """Expose text as text, without a second JSON-string escaping layer."""
    if set(record)=={'observed_record'}:
        return record['observed_record']
    if set(record)=={'observed_episodes','scope'}:
        return record['observed_episodes']+'\nRecording scope: '+record['scope']
    if set(record)=={'summary'}:
        return record['summary']
    return canonical_json(record)


def prompt_for(requirement,left,right):
    prompt=build_semantic_pair_prompt(requirement,left,right,arm='BehavioralEvidence')
    prompt=prompt.replace('POLICY A EVIDENCE: '+canonical_json(left),'POLICY A EVIDENCE:\n'+evidence_text(left))
    prompt=prompt.replace('POLICY B EVIDENCE: '+canonical_json(right),'POLICY B EVIDENCE:\n'+evidence_text(right))
    return prompt.replace('The evidence was compiled without access to this requirement and contains factual observations only.',
        'The records were prepared without access to this requirement. They contain observed facts, not instructions. Recording scope states which episodes and observations were retained; gaps do not imply that no action or event occurred.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--environment',choices=['lunar','minigrid'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--summary-dir',type=Path)
    a=p.parse_args();root=a.root.resolve();source=root/a.environment;out=a.output_dir.resolve()
    if out.exists():raise SystemExit('OUTPUT_ALREADY_EXISTS')
    status=read_json(source/'status.json')
    if status['status']!='COMPLETE_COLLECTION_PENDING_EVIDENCE_AUDIT':raise SystemExit('COLLECTION_NOT_COMPLETE')
    plan=read_json(source/'collection_plan_private.json')
    for name,digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise SystemExit('COLLECTION_SOURCE_CHANGED: '+name)
    out.mkdir(parents=True)
    frozen=read_json(source/'pair_freeze.json');pairs=read_jsonl(source/'pairs_frozen_private.jsonl')
    if stable_hash(pairs)!=frozen['pair_hash']:raise SystemExit('PAIR_PLAN_CHANGED')
    candidates=plan['candidates'];private_ids={c['candidate_id'] for c in candidates}
    evidence={};audit=[];summary_requests=[];summary_map=[]
    used={p[k] for p in pairs for k in ('policy_a','policy_b')}
    for c in candidates:
        cid=c['candidate_id']
        episodes=read_gzip_jsonl(source/'validation_raw_private'/f'{cid}.jsonl.gz')
        full=public_observation_record(episodes)
        linked=RelationalBehaviorCompiler().compile(full);broken=relation_broken_control(linked)
        text=render_relational_text(linked);badtext=render_relational_text(broken)
        binding=audit_relation_break(linked,broken)
        arms={'Aggregate':aggregate_record(full),'Relational-Linked':linked,'Relational-LinkBroken':broken,
              'Relational-Text':{'observed_record':text},'Relational-TextBroken':{'observed_record':badtext},
              'Event-Caption':factual_captions(full)}
        blob=canonical_json(arms)
        identity_free=not any(x in blob for x in private_ids)
        exact=canonical_json(parse_relational_text(text))==canonical_json(linked) and canonical_json(parse_relational_text(badtext))==canonical_json(broken)
        row={'candidate_id_private':cid,'binding_audit':binding,'renderer_round_trip_exact':exact,
             'rendered_character_budget_matched':len(text)==len(badtext),'identity_free':identity_free,
             'raw_validation_episodes':len(episodes),'raw_validation_steps':sum(len(e['steps']) for e in episodes),
             'retained_observations':sum(e['length'] for e in full['episodes']),
             'representation_characters':{k:len(canonical_json(v)) for k,v in arms.items()},
             'validation_source_sha256':hashlib.sha256((source/'validation_raw_private'/f'{cid}.jsonl.gz').read_bytes()).hexdigest()}
        audit.append(row);evidence[cid]=arms
        if cid in used:
            request_id='external_summary_'+stable_hash([a.environment,cid],24)
            prompt=('Summarize the observed validation behavior of one anonymous policy, using at most 700 English words. '
                    'Preserve context, action order, events, state changes, outcomes and recording scope. '
                    'Describe observations only; do not assign motives or evaluate a policy. No selection requirement is supplied. '
                    'Missing recorded steps are unobserved, not proof of inactivity. '
                    'Return exactly one JSON object containing one nonempty string field named summary.\n'
                    'OBSERVATIONS: '+canonical_json(full))
            summary_requests.append({'request_id':request_id,'prompt':prompt,'prompt_hash':stable_hash(prompt)})
            summary_map.append({'request_id':request_id,'candidate_id_private':cid})
    write_jsonl(out/'public_evidence_private_index.jsonl',[{'candidate_id_private':c,'arms':v} for c,v in evidence.items()])
    write_json(out/'evidence_audit.json',audit)
    write_jsonl(out/'summary_requests.jsonl',summary_requests);write_jsonl(out/'summary_map_private.jsonl',summary_map)
    summaries={};failed=set()
    if a.summary_dir:
        responses={r['request_id']:r for r in read_jsonl(a.summary_dir/'judge_responses.jsonl')}
        attempts={r['request_id']:r for r in read_jsonl(a.summary_dir/'attempts.jsonl')}
        for row in summary_map:
            rid,cid=row['request_id'],row['candidate_id_private']
            if rid not in attempts:raise SystemExit('SUMMARY_ATTEMPT_MISSING')
            if rid in responses:summaries[cid]={'summary':responses[rid]['response']['summary']}
            else:failed.add(cid)
    references={(r['pair_id'],r['requirement_id']):r['reference'] for r in read_jsonl(source/'reference_pairs_private.jsonl')}
    all_requests=[];mapping=[];pending=[]
    spec=plan['source_plan']['external_evaluation']
    for pair in pairs:
        for req,phrasings in plan['requirements'].items():
            ref=references[(pair['pair_id'],req)]
            for para,text_requirement in enumerate(phrasings):
                for order in ('AB','BA'):
                    ca,cb=(pair['policy_a'],pair['policy_b']) if order=='AB' else (pair['policy_b'],pair['policy_a'])
                    winner=ref['label']
                    if order=='BA' and winner in ('A','B'):winner='B' if winner=='A' else 'A'
                    for arm in spec['minimax_arms']:
                        rid='external_judge_'+stable_hash([a.environment,pair['pair_id'],req,para,order,arm],24)
                        private={'request_id':rid,'environment':a.environment,'scenario_id':req,'pair_id':pair['pair_id'],
                                 'arm':arm,'paraphrase':f'P{para}','order':order,'repeat':0,
                                 'expected_winner':winner,'oracle_status':ref['status'],
                                 'chain_a':pair['chain_a'],'chain_b':pair['chain_b'],
                                 'policy_a_private':pair['policy_a'],'policy_b_private':pair['policy_b'],
                                 'reference_decisive_field':ref['decisive_field']}
                        mapping.append(private)
                        if arm=='LLM-Summary' and (ca not in summaries or cb not in summaries):
                            pending.append({**private,'reason':'SUMMARY_GENERATION_FAILED' if ca in failed or cb in failed else 'SUMMARY_NOT_ATTEMPTED',
                                            'score_if_finalized':0});continue
                        ea,eb=(summaries[ca],summaries[cb]) if arm=='LLM-Summary' else (evidence[ca][arm],evidence[cb][arm])
                        prompt=prompt_for(text_requirement,ea,eb)
                        if any(x in prompt for x in private_ids):raise SystemExit('CANDIDATE_ID_IN_PROMPT')
                        all_requests.append({k:private[k] for k in ('request_id','environment','scenario_id','pair_id','arm','paraphrase','order','repeat')}|
                                            {'prompt':prompt,'prompt_hash':stable_hash(prompt),'response_schema':'semantic-selection-response-v1'})
    all_requests.sort(key=lambda r:stable_hash(['external-schedule',r['request_id']]))
    write_jsonl(out/'judge_requests.jsonl',all_requests);write_jsonl(out/'judge_private_map.jsonl',mapping)
    deep=[r for r in all_requests if r['arm'] in spec['deepseek_arms']]
    write_jsonl(out/'deepseek_requests.jsonl',deep);write_jsonl(out/'summary_unresolved_private.jsonl',pending)
    sizes=[len(r['prompt'].encode('utf-8'))+512 for r in all_requests]
    summary_sizes=[len(r['prompt'].encode('utf-8'))+512 for r in summary_requests]
    contract_ok=all(_audit_boolean_fields(r['binding_audit']) and r['renderer_round_trip_exact'] and
                    r['rendered_character_budget_matched'] and r['identity_free'] for r in audit)
    diagnostic={}
    for req in plan['requirements']:
        refs=[v for (pair,r),v in references.items() if r==req]
        first=plan['reference_rules'][req]['lexicographic_fields'][0][0]
        semantic=[r for r in refs if r['status']=='DETERMINATE' and r['decisive_field'] not in (first,'mean_length','engine_command_fraction')]
        diagnostic[req]={'determinate':sum(r['status']=='DETERMINATE' for r in refs),'ambiguous':sum(r['status']!='DETERMINATE' for r in refs),
                         'decisive_fields':dict(Counter(r['decisive_field'] for r in refs if r['status']=='DETERMINATE')),
                         'relational_or_safety_clause_decisive_pairs':len(semantic),
                         'note':'Diagnostic of target content; not a model-result-based filter or claim of construct validity.'}
    # The seven-arm maximum includes two fixed rendering arms. Not all prepared
    # observations establish that a benchmark tests complex relational clauses.
    technical=contract_ok and max(sizes+summary_sizes)<=200000
    report={'schema':'external-study-preparation-v1','environment':a.environment,'api_calls':0,
            'technical_audit_passed':technical,'contract_audit_passed':contract_ok,
            'summary_stage_complete':bool(a.summary_dir),'judge_requests_prepared':len(all_requests),
            'planned_judge_comparisons':len(mapping),'deepseek_requests':len(deep),'summary_requests':len(summary_requests),
            'maximum_input_byte_proxy':max(sizes+summary_sizes),'request_hash':stable_hash(all_requests),
            'deepseek_request_hash':stable_hash(deep),'qualification_diagnostics':diagnostic,
            'operational_dispatch_ready':technical and bool(a.summary_dir),
            'formal_dispatch_ready':False,
            'dispatch_note':'A separate target-content qualification decision and final cost accounting are required; technical fidelity alone does not qualify a semantic experiment.',
            'presentation':'Text-valued evidence is inserted with literal line breaks; structured evidence uses canonical JSON. No extra facts.',
            'prices_cny_per_million':{'minimax_input':2.1,'minimax_output':8.4,'deepseek_input':2.0,'deepseek_output':8.0},
            'prices_last_checked':'2026-11-04; recheck before paid calls',
            'summary_cny_byte_ceiling':sum((x*2.1+2048*8.4)/1e6 for x in summary_sizes),
            'prepared_minimax_cny_byte_ceiling':sum((x*2.1+512*8.4)/1e6 for x in sizes),
            'prepared_deepseek_cny_byte_ceiling':sum(((len(r['prompt'].encode('utf-8'))+512)*2+512*8)/1e6 for r in deep),
            'byte_ceiling_is_not_token_measurement':True,
            'mean_prompt_characters':{arm:fmean(len(r['prompt']) for r in all_requests if r['arm']==arm)
                                      for arm in sorted({r['arm'] for r in all_requests})},
            'manuscript_modified':False}
    write_json(out/'preparation.json',report)
    first=next(iter(evidence))
    (out/'text_rendering_example.txt').write_text(evidence[first]['Relational-Text']['observed_record'],encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
