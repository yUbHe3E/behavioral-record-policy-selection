"""Prepare/advance genuine relational retention branches from their own parents."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json,read_jsonl,write_json,write_jsonl
from behavioral_records.canonical import canonical_json,stable_hash
from behavioral_records.controlled_outer import (initial_population,compile_population,build_llm_requests,
    select_llm_parents,summarize_generation_selection,mutate_population)

ARMS=('Relational-Linked-LLM-PBT','Relational-LinkBroken-LLM-PBT')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','advance'])
    p.add_argument('--generation',type=int,choices=[0,1,2],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--judge-dir',type=Path)
    args=p.parse_args();out=args.output_dir.resolve();g=args.generation
    settings=read_json(ROOT/'configs/controlled_outer_confirmatory_v3.json')['design']
    scenarios=read_json(ROOT/'configs/controlled_scenarios_confirmatory_v1.json')['scenarios']
    if args.stage=='prepare':
        if out.exists():raise SystemExit('OUTPUT_ALREADY_EXISTS')
        previous=read_json(args.previous/'next_populations_private.json') if g>0 and args.previous else None
        if g>0 and previous is None:raise SystemExit('PREVIOUS_BRANCH_SELECTION_REQUIRED')
        populations={};compilations={};requests=[];private=[]
        for scenario in ('C1','C2'):
            for seed in settings['outer_seeds']:
                for arm in ARMS:
                    key=f'{scenario}__s{seed}__{arm}'
                    pop=initial_population(scenario,seed,16) if g==0 else previous[key]
                    if any(x['generation']!=g for x in pop):raise ValueError('GENERATION_MISMATCH')
                    compiled=compile_population(scenario,pop,outer_seed=seed,generation=g,include_relational=True)
                    rs,pm=build_llm_requests(scenario=scenario,outer_seed=seed,generation=g,arm=arm,
                        population=pop,evidence=compiled['evidence'],requirement=scenarios[scenario]['requirement_paraphrases'][0])
                    populations[key]=pop;compilations[key]=compiled;requests.extend(rs);private.extend(pm)
        requests.sort(key=lambda r:stable_hash(['revision-retention-schedule',r['request_id']]))
        forbidden=['policy_id','lineage','hidden_oracle','semantic_pass_rate','semantic_violation_rate','"risk_mode"','"collateral_mode"',
                   'disturbance_triggered','reward_triggered','mixed_or','mixed_and']
        forbidden.extend({p['policy_id'] for pop in populations.values() for p in pop})
        if any(term in r['prompt'] for term in forbidden for r in requests):raise ValueError('PRIVATE_FIELD_LEAKAGE')
        if len(requests)!=2048:raise ValueError('UNEXPECTED_REQUEST_COUNT')
        out.mkdir(parents=True)
        write_jsonl(out/'judge_requests.jsonl',requests);write_jsonl(out/'judge_private_map.jsonl',private)
        write_json(out/'populations_private.json',populations);write_json(out/'compilations_private.json',compilations)
        ceiling=sum(((len(r['prompt'].encode('utf-8'))+512)*2.1+512*8.4)/1e6 for r in requests)
        write_json(out/'preparation.json',{'generation':g,'request_count':len(requests),'request_hash':stable_hash(requests),
            'arms':ARMS,'source_settings':settings,'api_calls':0,'minimax_byte_ceiling_cny':ceiling,
            'own_parent_branch_required':True,'maximum_prompt_bytes':max(len(r['prompt'].encode('utf-8')) for r in requests)})
        print({'generation':g,'requests':len(requests),'planning_ceiling_cny':ceiling})
    else:
        if not args.judge_dir:raise SystemExit('JUDGE_DIRECTORY_REQUIRED')
        if (out/'selection_private.json').exists():raise SystemExit('SELECTION_ALREADY_WRITTEN')
        status=read_json(args.judge_dir/'status.json')
        if status['status'] not in ('COMPLETE','COMPLETE_WITH_FAILED_ATTEMPTS'):raise SystemExit('JUDGE_BATCH_NOT_COMPLETE')
        requests=read_jsonl(out/'judge_requests.jsonl')
        if read_json(args.judge_dir/'batch_plan.json')['source_hash']!=stable_hash(requests):raise SystemExit('BATCH_PROVENANCE_MISMATCH')
        populations=read_json(out/'populations_private.json');compiled=read_json(out/'compilations_private.json')
        private=read_jsonl(out/'judge_private_map.jsonl');responses=read_jsonl(args.judge_dir/'judge_responses.jsonl')
        selections=[];next_pops={}
        for scenario in ('C1','C2'):
            for seed in settings['outer_seeds']:
                for arm in ARMS:
                    key=f'{scenario}__s{seed}__{arm}';pop=populations[key]
                    group=[r for r in private if r['scenario_id']==scenario and r['outer_seed']==seed and r['selector_arm']==arm]
                    selection=select_llm_parents(population=pop,private_rows=group,responses=responses,parent_count=4)
                    summary=summarize_generation_selection(selection=selection,compiled=compiled[key])
                    if summary['valid_request_rate']<.8 or not summary['scheduled_graph_connected'] or len(set(summary['selected_policy_ids']))!=4:
                        write_json(out/'advance_stopped.json',{'group':key,'reason':'QUALITY_GATE','summary':summary})
                        raise SystemExit('RETENTION_QUALITY_GATE_FAILED')
                    selections.append({'scenario_id':scenario,'outer_seed':seed,'selector_arm':arm,'generation':g,**summary})
                    if g<2:next_pops[key]=mutate_population(scenario=scenario,population=pop,
                        selected_policy_ids=summary['selected_policy_ids'],outer_seed=seed,next_generation=g+1)
        write_json(out/'selection_private.json',selections)
        if g<2:write_json(out/'next_populations_private.json',next_pops)
        print({'generation':g,'groups':len(selections),'own_branch_next_populations':len(next_pops)})


if __name__=='__main__':main()
