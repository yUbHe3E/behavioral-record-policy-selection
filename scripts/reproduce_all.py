"""Recompute and verify paper analyses from archived evidence without API calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_jsonl
from behavioral_records.controlled_outer_evaluation import evaluate_controlled_outer_confirmatory
from behavioral_records.controlled_ablation import evaluate_controlled_representation_ablation
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def main():
    manifest=ROOT/'sha256.json'
    if not manifest.exists():raise SystemExit('Extract evidence-v1.0.0.zip in the repository root first.')
    for name,digest in read(manifest).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('EVIDENCE_HASH_MISMATCH: '+name)
    # No evaluation below uses the network. Do not forward any ambient model keys.
    for key in list(os.environ):
        if key.endswith('API_KEY'):os.environ.pop(key)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    out=ROOT/'reproduced'/stamp;out.mkdir(parents=True,exist_ok=False)
    run('scripts/reproduce_revision_primary.py')
    base=ROOT/'outputs/kbs_revision_r1_20261104'
    checks={}
    outer=ROOT/'outputs/controlled_outer_confirmatory_20260822_minimax_m3_v3'
    result=evaluate_controlled_outer_confirmatory(selection_rows=read_jsonl(outer/'all_selection_results.jsonl'),stage_summary=read(outer/'smoke_summary.json'),frozen_controls=read(outer/'frozen_controls.json'),plan=read(ROOT/'configs/controlled_outer_confirmatory_v3.json'))
    assert result==read(outer/'confirmatory_evaluation.json'),'original retention'
    (out/'original_retention.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    checks['original_retention']='exact full-report match'
    ablation=ROOT/'outputs/controlled_representation_ablation_20260823_v1_1'
    prior=ROOT/'outputs/controlled_representation_ablation_20260823_minimax_m3_v1_1'
    result=evaluate_controlled_representation_ablation(private_map=read_jsonl(ablation/'judge_private_map.jsonl'),responses=read_jsonl(prior/'judge_responses.jsonl'),plan=read(ROOT/'configs/controlled_representation_ablation_v1_1.json'))
    expected=read(prior/'evaluation.json')
    for key in ['primary_contrasts','pair_rows','by_scenario','cell_valid_rates']:
        assert result[key]==expected[key],('representation ablation',key)
    (out/'ablation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    checks['representation_ablation']='exact numerical match'
    for provider,bundle,judge in [('minimax','baseline_final','baselines_minimax'),('deepseek','deepseek_main_bundle','replication_deepseek')]:
        target=out/(provider+'_added.json')
        run('scripts/evaluate_revision_baselines.py','--bundle',base/bundle,'--judge-dir',base/judge,'--output',target)
        actual,expected=read(target),read(base/judge/'evaluation.json')
        for key in ['pair_rows','cells','contrasts']:
            assert actual[key]==expected[key],(provider,key)
        checks[provider+'_additional']='exact numerical match'
    run('scripts/evaluate_revision_retention.py','--root',base,'--output',out/'retention.json')
    actual,expected=read(out/'retention.json'),read(base/'retention_evaluation.json')
    for key in ['primary_generation_2','secondary_generations','generation_zero_matching','selection_rows']:
        assert actual[key]==expected[key],('retention',key)
    checks['compact_retention']='exact numerical match'
    external=ROOT/'outputs/kbs_external_validation_20261105'
    for provider in ['minimax','deepseek']:
        target=out/('lunar_'+provider)
        run('scripts/evaluate_kbs_external_study.py','--collection',external/'lunar','--prepared',external/'lunar_final','--responses-dir',external/('lunar_judge_'+provider),'--provider',provider,'--output-dir',target)
        actual=read(target/'evaluation.json')
        expected=read(external/'lunar_online'/('evaluation_'+provider)/'evaluation.json')
        for key in ['results','planned_calls','attempted_calls','valid_calls','summary_unavailable_planned_calls']:
            assert actual[key]==expected[key],('lunar',provider,key)
        checks['lunar_'+provider]='exact numerical match'
    run('scripts/evaluate_highway.py','--output',out/'highway.json')
    checks['highway']='exact full-report match'
    report={'status':'PASS','network_calls':0,'original_primary':'exact primary contrasts',**checks}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
