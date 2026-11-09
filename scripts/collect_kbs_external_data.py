"""Prospective external observations and disjoint operational references; no API."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json,write_json,write_jsonl
from behavioral_records.canonical import stable_hash
from behavioral_records.external_validation import (development_summary,choose_pairs,rollout_minigrid,
    minigrid_reference_profile,MINIGRID_REQUIREMENTS,MINIGRID_RULES)
from behavioral_records.lunarlander_semantic import (rollout_hidden_episode,write_gzip_jsonl,
    build_hidden_policy_profile,compare_hidden_profiles)
from behavioral_records.lunarlander_training import rollout_lunarlander_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['lunar','minigrid'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=a.root.resolve();out=root/a.environment
    if out.exists():raise SystemExit('OUTPUT_ALREADY_EXISTS; no silent overwrite')
    out.mkdir(parents=True)
    config=read_json(ROOT/'configs/kbs_external_validation_20261105.json')
    spec=config['external_evaluation'];start=time.monotonic();active_env=None
    def status(value,**extra):
        write_json(out/'status.json',{'status':value,'updated_at':datetime.now(timezone.utc).isoformat(),
                                     'elapsed_seconds':time.monotonic()-start,'api_calls':0,**extra})
    def beat(**extra):
        if time.monotonic()-start>a.timeout:raise TimeoutError('Frozen collection timeout')
        write_json(out/'heartbeat.json',{'updated_at':datetime.now(timezone.utc).isoformat(),
                                        'elapsed_seconds':time.monotonic()-start,**extra})
    if not a.execute:status('PREPARED_NO_COLLECTION');return
    try:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        if a.environment=='lunar':
            source=ROOT/'outputs/lunarlander_transfer/20260823_policy_pool_formal_v1'
            original=read_json(source/'summary.json')['candidate_rows']
            candidates=[{'candidate_id':c['candidate_id'],'chain_id':c['training_profile_private']+'__'+str(c['training_seed_private']),
                         'training_seed_private':c['training_seed_private'],
                         'path':str(source/c['model_path_private']),
                         'checkpoint_private':c['checkpoint_timestep_private']} for c in original]
            transfer=read_json(ROOT/'configs/lunarlander_transfer_v1.json')
            semantic=read_json(ROOT/'configs/lunarlander_semantic_gate_v1.json')
            contexts=[c['context_id'] for c in transfer['environment']['contexts']]
            requirements={k:[v['public_text'],v['independent_paraphrase']] for k,v in transfer['semantic_requirements'].items()}
            rules=semantic['hidden_oracle']
        else:
            training=root/'minigrid_training'
            if read_json(training/'status.json')['status']!='COMPLETE':raise ValueError('Training cohort not complete')
            candidates=[{**c,'path':str(training/c['path'])} for c in read_json(training/'models_private.json')['candidates']]
            from behavioral_records.minigrid_training import make_onehot_cnn_env
            active_env=make_onehot_cnn_env('MiniGrid-ObstructedMaze-1Dlhb-v0')
            contexts=['official_layout_seed'];requirements=MINIGRID_REQUIREMENTS;rules=MINIGRID_RULES
        for c in candidates:c['model_sha256']=hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()
        manifest={'environment':a.environment,'candidates':candidates,'requirements':requirements,'reference_rules':rules,
                  'source_plan':config,'created_at':datetime.now(timezone.utc).isoformat(),
                  'source_sha256':{str(Path(f).relative_to(ROOT)):hashlib.sha256(Path(f).read_bytes()).hexdigest()
                      for f in [Path(__file__),ROOT/'src/behavioral_records/external_validation.py',
                                ROOT/'src/behavioral_records/relational_text.py',ROOT/'src/behavioral_records/relational_record.py']},
                  'reference_is_operational_definition':True,'human_preference_equivalence_claimed':False,
                  'legacy_audit_gate_not_relabelled_as_passed':True,'api_calls':0}
        write_json(out/'collection_plan_private.json',manifest)
        write_json(out/'requirements.json',requirements)
        def collect(model,candidate,split):
            episodes=[]
            if split=='development':seeds=spec['development_seeds']
            elif split=='validation':seeds=spec['public_seeds']
            else:seeds=list(range(spec['reference_seeds_start'],spec['reference_seeds_start']+spec['reference_episodes_per_context']))
            for context_index,context in enumerate(contexts):
                # Exactly one full validation episode for each Lunar context; three in total.
                actual_seeds=[seeds[context_index]] if split=='validation' and a.environment=='lunar' else seeds
                for seed in actual_seeds:
                    beat(candidate=candidate['candidate_id'],phase=split,context=context,seed=seed)
                    if a.environment=='lunar':
                        if split=='reference':
                            ep=rollout_hidden_episode(model=model,transfer_config=transfer,candidate_id=candidate['candidate_id'],
                                                     context_id=context,seed=seed,deterministic=True)
                            ep['split']='reference'
                        else:
                            ep=rollout_lunarlander_policy(model=model,transfer_config=transfer,policy_id=candidate['candidate_id'],
                                                        context_id=context,seed=seed,split=split).to_dict()
                            ep['split']=split
                            if split=='development':ep['outcome']='success' if ep['return']>=200 else ep['outcome']
                    else:
                        ep=rollout_minigrid(model,active_env,seed,candidate['candidate_id'],split)
                    episodes.append(ep)
            return episodes
        development={}
        for i,candidate in enumerate(candidates):
            status('RUNNING_DEVELOPMENT',completed=i,planned=len(candidates))
            model=PPO.load(candidate['path'],device='cpu')
            episodes=collect(model,candidate,'development')
            write_gzip_jsonl(out/'development_raw_private'/f"{candidate['candidate_id']}.jsonl.gz",episodes)
            development[candidate['candidate_id']]=development_summary(episodes)
            write_json(out/'development_summary_private.json',development)
            del model
        pairs=choose_pairs(candidates,development,spec['pairs_per_environment'])
        if len(pairs)!=spec['pairs_per_environment']:raise ValueError('Insufficient independent-chain pairs')
        write_jsonl(out/'pairs_frozen_private.jsonl',pairs)
        write_json(out/'pair_freeze.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),
                                         'pair_hash':stable_hash(pairs),'reference_episodes_collected':0,
                                         'judge_calls':0,'policy_pairs':len(pairs),
                                         'training_chains':len({c['chain_id'] for c in candidates})})
        # No candidate is removed because of success or reference-label outcomes.
        profiles={};counts=[]
        for i,candidate in enumerate(candidates):
            status('RUNNING_VALIDATION_AND_REFERENCE',completed=i,planned=len(candidates))
            model=PPO.load(candidate['path'],device='cpu')
            public=collect(model,candidate,'validation')
            write_gzip_jsonl(out/'validation_raw_private'/f"{candidate['candidate_id']}.jsonl.gz",public)
            reference=collect(model,candidate,'reference')
            write_gzip_jsonl(out/'reference_raw_private'/f"{candidate['candidate_id']}.jsonl.gz",reference)
            profile=build_hidden_policy_profile(candidate['candidate_id'],reference,semantic['factual_operationalization']) if a.environment=='lunar' else minigrid_reference_profile(reference)
            profiles[candidate['candidate_id']]=profile
            write_json(out/'reference_profiles_private.json',profiles)
            counts.append({'candidate_id':candidate['candidate_id'],'validation_episodes':len(public),'reference_episodes':len(reference)})
            write_json(out/'collection_counts.json',counts)
            del model
        labels=[]
        for pair in pairs:
            for requirement,rule in rules.items():
                ref=compare_hidden_profiles(profiles[pair['policy_a']],profiles[pair['policy_b']],rule)
                sensitivity={str(m):compare_hidden_profiles(profiles[pair['policy_a']],profiles[pair['policy_b']],rule,margin_multiplier=m)['label'] for m in (0.8,1.2)}
                labels.append({**pair,'requirement_id':requirement,'reference':ref,'margin_sensitivity_labels':sensitivity})
        write_jsonl(out/'reference_pairs_private.jsonl',labels)
        from collections import Counter
        label_counts={r:dict(Counter(x['reference']['label'] if x['reference']['status']=='DETERMINATE' else 'AMBIGUOUS' for x in labels if x['requirement_id']==r)) for r in requirements}
        decisive={r:dict(Counter(x['reference']['decisive_field'] for x in labels if x['requirement_id']==r and x['reference']['status']=='DETERMINATE')) for r in requirements}
        status('COMPLETE_COLLECTION_PENDING_EVIDENCE_AUDIT',candidates=len(candidates),
               training_chains=len({c['chain_id'] for c in candidates}),pairs=len(pairs),
               label_counts=label_counts,decisive_fields=decisive,
               development_successful_candidates=sum(v['success_rate']>0 for v in development.values()))
    except ImportError as exc:
        status('SKIPPED_DEPENDENCY_MISSING',reason=str(exc))
    except Exception as exc:
        status('STOPPED_TIMEOUT' if isinstance(exc,TimeoutError) else 'FAILED_NO_AUTOMATIC_RETRY',
               error_type=type(exc).__name__,error=str(exc)[:700]);raise
    finally:
        if active_env is not None:active_env.close()


if __name__=='__main__':main()
