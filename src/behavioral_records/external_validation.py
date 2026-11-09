"""Factual adapters and fixed-budget external policy evaluation utilities.

Public evidence and independent reference evaluation have separate entry points.
No public compiler takes a requirement or reference score.
"""
from __future__ import annotations
import copy
import math
from collections import Counter, defaultdict
from itertools import combinations
from statistics import fmean
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, stable_hash

OBSERVATION_LIMIT = 48
DECIMALS = 4


def rounded(value):
    if isinstance(value, dict): return {str(k): rounded(v) for k,v in value.items()}
    if isinstance(value, list): return [rounded(v) for v in value]
    if isinstance(value, tuple): return [rounded(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError('Nonfinite observation')
        return round(value, DECIMALS)
    return copy.deepcopy(value)


def evenly(values, count):
    values = sorted(set(values))
    if len(values) <= count: return values
    if count <= 0: return []
    if count == 1: return [values[0]]
    return [values[round(i*(len(values)-1)/(count-1))] for i in range(count)]


def observation_indices(steps: Sequence[Mapping[str, Any]], limit=OBSERVATION_LIMIT):
    """Fixed factual novelty/action-change/uniform selection, independent of goals."""
    n=len(steps)
    if n<=limit:return list(range(n))
    event_times=defaultdict(list)
    for i,s in enumerate(steps):
        for event in s.get('events',[]):event_times[str(event)].append(i)
    anchors=evenly([i for v in event_times.values() for i in (v[0],v[-1])],12)
    chosen={0,n-1,*anchors}
    for i in anchors:
        chosen.update(j for j in (i-1,i+1) if 0<=j<n)
    changes=[i for i in range(1,n) if steps[i]['action']!=steps[i-1]['action']]
    for i in evenly(changes,10):
        if len(chosen)<limit:chosen.add(i)
    remaining=[i for i in range(n) if i not in chosen]
    chosen.update(evenly(remaining,limit-len(chosen)))
    return sorted(chosen)


def public_observation_record(episodes: Sequence[Mapping[str, Any]]) -> dict:
    """Three supplied validation episodes; all arms share the same observation budget."""
    out=[]
    for ep in episodes:
        if ep.get('split')!='validation':raise ValueError('Only validation episodes may become public')
        indices=observation_indices(ep['steps'])
        steps=[]
        for i in indices:
            raw=ep['steps'][i]
            steps.append(rounded({k:raw.get(k, [] if k=='events' else {} if k in ('state','predicates') else None)
                                  for k in ('t','action','reward','events','state','predicates')}))
        context=rounded(ep['context'])
        context['recording_scope']={
            'complete_episode_step_count':len(ep['steps']),
            'recorded_observation_count':len(indices),
            'maximum_observations_per_episode':OBSERVATION_LIMIT,
            'numeric_decimals':DECIMALS,
            'observations_are_post_action':True,
            'selection':'first/last event occurrence neighborhoods, action changes, and uniform coverage; fixed without requirement access',
            'gaps':'Missing time indices are unobserved here. State deltas span consecutive recorded observations, possibly several environment steps; they do not attribute the difference to one action.',
            'episode_summary':'Return, termination and complete length describe the full episode; action/event counts in this record describe retained observations only.',
        }
        out.append({'context':context,'length':len(steps),'outcome':str(ep['outcome']),
                    'return':round(float(ep['return']),DECIMALS),'terminated':bool(ep['terminated']),
                    'truncated':bool(ep['truncated']),'steps':steps})
    return {'representation':'deterministic_raw_validation_trajectories','episodes':out}


def aggregate_record(full):
    """Ordinary factual panel from exactly the same supplied observations."""
    rows=[]
    for ep in full['episodes']:
        states=defaultdict(list)
        for s in ep['steps']:
            for key,value in s['state'].items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):states[key].append(float(value))
        rows.append({'context':ep['context'],'return':ep['return'],'outcome':ep['outcome'],
                     'terminated':ep['terminated'],'truncated':ep['truncated'],
                     'recorded_observations':ep['length'],
                     'action_counts':dict(sorted(Counter(s['action'] for s in ep['steps']).items())),
                     'event_counts':dict(sorted(Counter(e for s in ep['steps'] for e in s['events']).items())),
                     'state_marginals':{k:{'min':min(v),'mean':round(fmean(v),DECIMALS),'max':max(v)} for k,v in sorted(states.items())}})
    return {'episode_panels':rows,'omitted':['order within episode','step-specific action-to-observation binding','nonnumeric state marginals'],
            'scope':'same recorded observations and full-episode summary as the other conditions'}


def factual_captions(full):
    lines=[]
    for i,ep in enumerate(full['episodes'],1):
        lines.append(f"Episode {i}. Context: {canonical_json(ep['context'])}. Return: {ep['return']}; outcome: {ep['outcome']}; terminated: {ep['terminated']}; truncated: {ep['truncated']}.")
        for s in ep['steps']:
            lines.append(f"At recorded time {s['t']}, action {canonical_json(s['action'])}; observed reward {s['reward']}; events {canonical_json(s['events'])}; state {canonical_json(s['state'])}; predicates {canonical_json(s['predicates'])}.")
    return {'observed_episodes':'\n'.join(lines),'scope':'Same supplied observations; no additional raw steps or inferred motivations.'}


def development_summary(episodes):
    return {'success_rate':fmean(ep['outcome']=='success' for ep in episodes),
            'mean_return':fmean(float(ep['return']) for ep in episodes),
            'mean_length':fmean(len(ep['steps']) for ep in episodes),
            'unique_action_sequences':len({stable_hash([s['action'] for s in ep['steps']]) for ep in episodes})}


def choose_pairs(candidates, summaries, count=40):
    """Balance training-chain pairs; no references or model decisions are accepted."""
    by_id={r['candidate_id']:r for r in candidates}
    groups=defaultdict(list)
    for a,b in combinations(sorted(by_id),2):
        ca,cb=by_id[a]['chain_id'],by_id[b]['chain_id']
        if ca==cb:continue
        sa,sb=summaries[a],summaries[b]
        distance=(abs(sa['success_rate']-sb['success_rate']),
                  abs(sa['mean_return']-sb['mean_return']),abs(sa['mean_length']-sb['mean_length']))
        groups[tuple(sorted((ca,cb)))].append((distance,stable_hash([a,b]),a,b))
    for v in groups.values():v.sort()
    schedule=sorted(groups,key=lambda x:stable_hash(['chain-pair',x]))
    selected=[];used=Counter();round_index=0
    while len(selected)<count:
        progressed=False
        for group in schedule:
            available=groups[group]
            if not available:continue
            # Match ordinary performance, with a deterministic degree-balancing priority.
            chosen=min(available,key=lambda r:(max(used[r[2]],used[r[3]]),r[0],r[1]))
            available.remove(chosen)
            _,_,a,b=chosen
            if int(stable_hash(['orientation',a,b],8),16)%2:a,b=b,a
            selected.append({'pair_id':'external_pair_'+stable_hash([a,b],16),'policy_a':a,'policy_b':b,
                             'chain_a':by_id[a]['chain_id'],'chain_b':by_id[b]['chain_id'],
                             'development_distance':list(chosen[0])})
            used[a]+=1;used[b]+=1;progressed=True
            if len(selected)==count:break
        if not progressed:break
        round_index+=1
    return selected


def minigrid_snapshot(env):
    """Stable cell-addressed simulator facts; walls are static context, not repeated."""
    from .standard_env import _object_snapshot
    base=env.unwrapped
    state={'agent_x':int(base.agent_pos[0]),'agent_y':int(base.agent_pos[1]),
           'agent_direction':int(base.agent_dir),'carrying':_object_snapshot(base.carrying)}
    for y in range(base.grid.height):
        for x in range(base.grid.width):
            obj=base.grid.get(x,y)
            if obj is not None and obj.type!='wall':state[f'cell({x},{y})']=_object_snapshot(obj)
    return state


def minigrid_events(before,after,action):
    events=[]
    if before['carrying']!=after['carrying']:
        if before['carrying'] is None:events.append('object_picked_up')
        elif after['carrying'] is None:events.append('object_dropped')
        else:events.append('carried_object_changed')
    for key in set(before)|set(after):
        left,right=before.get(key),after.get(key)
        if not key.startswith('cell('):continue
        if isinstance(left,dict) and left.get('type')=='door' and isinstance(right,dict):
            if left.get('is_locked') and not right.get('is_locked'):events.append('door_unlocked')
            if not left.get('is_open') and right.get('is_open'):events.append('door_opened')
            if left.get('is_open') and not right.get('is_open'):events.append('door_closed')
        if isinstance(left,dict) and left.get('type')=='box' and left!=right:events.append('box_contents_exposed')
    if action in ('pickup','drop','toggle','forward') and before==after:events.append('command_without_state_change')
    return sorted(events)


def rollout_minigrid(model,env,seed,candidate_id,split):
    observation,_=env.reset(seed=int(seed));base=env.unwrapped
    before=minigrid_snapshot(env)
    walls=[[x,y] for y in range(base.grid.height) for x in range(base.grid.width)
           if base.grid.get(x,y) is not None and base.grid.get(x,y).type=='wall']
    context={'environment':'MiniGrid-ObstructedMaze-1Dlhb-v0','mission':str(base.mission),
             'width':base.grid.width,'height':base.grid.height,'walls':walls,'initial_state':copy.deepcopy(before),
             'state_source':'full simulator-state logging; policy uses its configured full observation',
             'actions':'left/right rotate; forward moves; pickup/drop manipulate the front cell; toggle operates a front door or box'}
    steps=[];total=0.;done=False
    while not done:
        act,_=model.predict(observation,deterministic=True);act=int(act)
        action=base.actions(act).name
        observation,reward,terminated,truncated,_=env.step(act)
        after=minigrid_snapshot(env);events=minigrid_events(before,after,action)
        if terminated:events.append('environment_terminated')
        if truncated:events.append('time_limit_reached')
        if reward>0:events.append('positive_environment_reward')
        steps.append({'t':len(steps),'action':action,'action_id':act,'reward':float(reward),
                      'state':after,'events':events,'predicates':{}})
        before=after;total+=float(reward);done=terminated or truncated
    return {'candidate_id':candidate_id,'seed_private':int(seed),'split':split,'context':context,
            'length':len(steps),'return':total,'outcome':'success' if total>0 else 'time_limit',
            'terminated':bool(terminated),'truncated':bool(truncated),'steps':steps}


MINIGRID_REQUIREMENTS={
    'MG-R1':[
        'Complete the requested object pickup. Judge whether interactions respond to the current situation: attempts that reveal an unmet prerequisite are acceptable, but repeatedly issuing the same ineffective command without changing that situation is undesirable. Prioritize reliable completion; among comparably successful policies, prefer adaptation over persistent ineffective retries, and use speed only as a final tie-breaker.',
        'Prefer a policy that obtains the requested object and changes its approach when an interaction cannot work yet. A brief unsuccessful attempt is acceptable; continuing the same ineffective action in an unchanged situation is not. Compare completion first, then persistent retry behavior, then time when these are comparable.'
    ],
    'MG-R2':[
        'Complete the requested object pickup while preserving access gained along the way. Opening a box, moving an obstruction, or using a key is acceptable when needed to reach the target. Among policies with comparable completion, prefer those that keep an opened passage usable instead of closing it again, and avoid repeatedly picking up an object that was already handled and put down; use time only after these considerations.',
        'Favor reliable target retrieval with orderly object handling. Necessary unlocking and obstacle removal are permitted. If completion is comparable, reopening a passage after closing it and repeatedly reacquiring the same handled object count against the policy; efficiency breaks remaining ties.'
    ]
}


def minigrid_reference_profile(episodes):
    """Independent held-out raw trajectories only; no public record input."""
    if any(ep['split']!='reference' for ep in episodes):raise ValueError('Reference split required')
    rows=[]
    for ep in episodes:
        repeats=0;last=None;run=0;rehandled=0;picked=Counter();closed=0
        for s in ep['steps']:
            if 'command_without_state_change' in s['events']:
                key=(s['action'],canonical_json(s['state']))
                run=run+1 if key==last else 1
                if run>=3:repeats+=1
                last=key
            else:last=None;run=0
            if 'door_closed' in s['events']:closed+=1
            if 'object_picked_up' in s['events']:
                obj=s['state']['carrying']
                key=canonical_json(obj)
                rehandled+=int(picked[key]>0);picked[key]+=1
        rows.append({'success':ep['outcome']=='success','persistent_retry':repeats>0,
                     'door_reclosure':closed>0,'object_rehandling':rehandled>0,'length':len(ep['steps'])})
    return {'aggregate':{'completion_rate':fmean(r['success'] for r in rows),
                         'persistent_retry_episode_rate':fmean(r['persistent_retry'] for r in rows),
                         'door_reclosure_episode_rate':fmean(r['door_reclosure'] for r in rows),
                         'object_rehandling_episode_rate':fmean(r['object_rehandling'] for r in rows),
                         'mean_length':fmean(r['length'] for r in rows)},'reference_episode_count':len(rows)}


MINIGRID_RULES={
    'MG-R1':{'lexicographic_fields':[['completion_rate','maximize',0.125],['persistent_retry_episode_rate','minimize',0.125],['mean_length','minimize',10.0]]},
    'MG-R2':{'lexicographic_fields':[['completion_rate','maximize',0.125],['door_reclosure_episode_rate','minimize',0.125],['object_rehandling_episode_rate','minimize',0.125],['mean_length','minimize',10.0]]},
}
