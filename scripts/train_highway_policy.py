"""Run one original Highway training chain in a new, independent directory.

The worker preserves the original DQN parameters, checkpoint schedule,
development episodes, and checkpoint qualification. This does not resume any
archived experiment or invoke a language model.
"""
from pathlib import Path
from datetime import datetime, timezone
from statistics import fmean
import argparse
import hashlib
import json
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.bounded_external import make_environment, profile_for, rollout
from behavioral_records.highway_cohort_v2 import highway_schedule, select_checkpoint
from behavioral_records.io import read_json,write_json
from behavioral_records.lunarlander_semantic import write_gzip_jsonl

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def update(path,status,**fields):
    write_json(path,{'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),'api_calls':0,**fields})
def verify_protocol(out):
    data=read_json(out/'protocol.json')
    for name,digest in data['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    return data['plan']

def worker(out, index):
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    plan = verify_protocol(out)
    deadline = read_json(out / "budget_clock.json")["deadline_epoch"]
    job = highway_schedule(plan)[index]
    chain = out / "chains" / job["chain_id"]
    chain.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(plan["runtime"]["torch_threads"])
    last_write = 0.
    def heartbeat(phase, force=False, **extra):
        nonlocal last_write
        if time.time() >= deadline:
            raise TimeoutError("Cumulative compute limit reached")
        if force or time.monotonic() - last_write >= 10:
            update(chain / "heartbeat.json", phase, chain=job["chain_id"], **extra)
            last_write = time.monotonic()
    class FixedCheckpoints(BaseCallback):
        def _on_step(self):
            heartbeat("TRAINING", timesteps=self.num_timesteps, maximum=50000)
            if self.num_timesteps in job["checkpoints"]:
                self.model.save(chain / f"checkpoint_{self.num_timesteps}.zip")
            return True
    env = None
    started = time.monotonic()
    try:
        update(chain / "status.json", "TRAINING", job=job)
        env = Monitor(make_environment(plan, "highway", job["context"]))
        model = DQN("MlpPolicy", env, seed=job["seed"], device="cuda", verbose=0, **plan["highway"]["dqn"])
        model.learn(total_timesteps=50000, callback=FixedCheckpoints())
        assert model.num_timesteps == 50000
        model.save(chain / "checkpoint_50000.zip")
        env.close(); env = None
        del model
        torch.cuda.empty_cache()
        evaluations = []
        for checkpoint in job["checkpoints"]:
            path = chain / f"checkpoint_{checkpoint}.zip"
            model = DQN.load(path, device="cpu")
            episodes = []
            for ci, context in enumerate(plan["highway"]["contexts"]):
                base = (plan["splits"]["development"]["base_seed"] + plan["splits"]["environment_seed_offset"]
                        + ci * plan["splits"]["context_seed_offset"])
                for i in range(plan["splits"]["development"]["episodes_per_context"]):
                    heartbeat("EVALUATING", force=True, checkpoint=checkpoint, episodes=len(episodes))
                    episodes.append(rollout(model, plan, "highway", context["context_id"], base+i,
                                           job["chain_id"], "development", lambda: heartbeat("EVALUATING")))
            write_gzip_jsonl(chain / f"development_{checkpoint}.jsonl.gz", episodes)
            evaluations.append({"checkpoint": checkpoint, "mean_official_return": fmean(e["return"] for e in episodes),
                "profile": profile_for(episodes, plan, "highway", ROOT), "model_sha256": sha(path),
                "path": path.relative_to(out).as_posix()})
            write_json(chain / "checkpoint_evaluations.json", evaluations)
            del model
        selected = select_checkpoint(evaluations, plan)
        result = {**job, "qualified": selected is not None, "selection_rule": plan["selection"]["checkpoint_rule"],
                  "elapsed_seconds": time.monotonic()-started,
                  "diagnostic_best_return_checkpoint": max(evaluations, key=lambda r:r["mean_official_return"])["checkpoint"]}
        if selected: result.update(selected)
        else: result.update(checkpoint=None, profile=None, path=None)
        write_json(chain / "result.json", result)
        update(chain / "status.json", "COMPLETE", qualified=result["qualified"], selected_checkpoint=result["checkpoint"])
    except Exception as exc:
        update(chain / "status.json", "STOPPED_WORKER", error=str(exc))
        raise
    finally:
        if env is not None: env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chain-index',type=int,choices=range(12),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--max-hours',type=float,default=12.)
    a=p.parse_args()
    plan=read_json(ROOT/'configs/kbs_highway_external_v2_20261106.json')
    print(json.dumps({'job':highway_schedule(plan)[a.chain_index],'training_steps':50000,'api_calls':0,'execute':a.execute}))
    if not a.execute:return
    if a.max_hours<=0:raise SystemExit('max-hours must be positive')
    try:
        import torch
        import stable_baselines3
        import highway_env
    except ImportError as e:
        print('SKIPPED_DEPENDENCY_MISSING: '+str(e));return
    if not torch.cuda.is_available():
        print('SKIPPED_CUDA_UNAVAILABLE: original training uses CUDA');return
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    source_files=list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'configs/kbs_highway_external_v2_20261106.json']
    write_json(out/'protocol.json',{'plan':plan,'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in source_files}})
    write_json(out/'budget_clock.json',{'deadline_epoch':time.time()+a.max_hours*3600})
    worker(out,a.chain_index)
if __name__=='__main__':main()
