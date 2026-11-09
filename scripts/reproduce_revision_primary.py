"""Recompute the original two primary comparisons offline from stored responses."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json,read_jsonl,write_json
from behavioral_records.relational_record import evaluate_controlled_relational_record
from behavioral_records.deepseek_relational_replication import evaluate_deepseek_relational_replication


def main():
    if (ROOT/'sha256.json').exists():
        for name,digest in read_json(ROOT/'sha256.json').items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise SystemExit('ARCHIVE_HASH_MISMATCH: '+name)
    mm=ROOT/'outputs/controlled_relational_record_20260901_v1_2'
    ds=ROOT/'outputs/controlled_relational_deepseek_replication_20260901_v1'
    minimax=evaluate_controlled_relational_record(private_map=read_jsonl(mm/'judge_private_map.jsonl'),
        responses=read_jsonl(mm/'run_minimax_m3/judge_responses.jsonl'),
        plan=read_json(ROOT/'configs/controlled_relational_record_v1.json'))
    deepseek=evaluate_deepseek_relational_replication(public_requests=read_jsonl(ds/'bundle/judge_requests.jsonl'),
        private_map=read_jsonl(ds/'private/evaluation_map.jsonl'),responses=read_jsonl(ds/'run/judge_responses.jsonl'),
        config=read_json(ROOT/'configs/controlled_relational_deepseek_replication_v1.json'))
    out=ROOT/'reproduced_primary';out.mkdir(exist_ok=True)
    write_json(out/'minimax.json',minimax);write_json(out/'deepseek.json',deepseek)
    primary=minimax['contrasts']['primary_relational_link_contribution']
    expected=read_json(mm/'run_minimax_m3/evaluation.json')['contrasts']['primary_relational_link_contribution']
    if primary!=expected:raise SystemExit('MINIMAX_REPRODUCTION_DIFFERS')
    if deepseek['primary']!=read_json(ds/'run/evaluation.json')['primary']:raise SystemExit('DEEPSEEK_REPRODUCTION_DIFFERS')
    print(json.dumps({'status':'EXACT_PRIMARY_REPRODUCTION_PASS','network_calls':0,
                      'MiniMax_primary':primary,'DeepSeek_primary':deepseek['primary']}))


if __name__=='__main__':main()
