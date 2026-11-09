"""Offline horizon/episode-count timing using prefixes of actual trajectories."""
from pathlib import Path
import argparse
import copy
import gzip
import json
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.canonical import canonical_json
from behavioral_records.io import read_json,write_json
from behavioral_records.relational_record import RelationalBehaviorCompiler


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('OUTPUT_ALREADY_EXISTS')
    base=ROOT/'outputs/kbs_revision_r1_20261104'
    lunar=read_json(base/'lunar_preflight/first_candidate_full_public.json')
    with gzip.open(base/'minigrid_records_preflight/epoch_01_records.json.gz','rt',encoding='utf-8') as f:
        mini=json.load(f)['full']
    rows=[]
    for name,source in [('LunarLander-v3',lunar),('MiniGrid-ObstructedMaze-1Dlhb-v0',mini)]:
        episode=max(source['episodes'],key=lambda x:len(x['steps']))
        for length in [16,32,64,128,256]:
            if length>len(episode['steps']):continue
            prefix=copy.deepcopy(episode);prefix['steps']=prefix['steps'][:length];prefix['length']=length
            prefix.update(outcome='timing_prefix',terminated=False,truncated=True,
                          **{'return':sum(x['reward'] for x in prefix['steps'])})
            for multiplicity in (1,4):
                full={'representation':source['representation'],'episodes':[prefix]*multiplicity}
                times=[]
                for _ in range(5):
                    start=time.perf_counter();linked=RelationalBehaviorCompiler().compile(full);times.append(time.perf_counter()-start)
                rows.append({'environment':name,'observed_steps_per_episode':length,'duplicate_episodes':multiplicity,
                    'median_compile_seconds':statistics.median(times),'full_characters':len(canonical_json(full)),
                    'compiled_characters':len(canonical_json(linked))})
    write_json(a.output,{'purpose':'mechanical_scaling_not_policy_outcomes','timing_repetitions':5,
                        'api_calls':0,'prefixes_are_not_used_for_judging':True,'rows':rows})
    print({'rows':len(rows),'output':str(a.output)})


if __name__=='__main__':main()
