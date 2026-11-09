import copy
import inspect

import pytest

from behavioral_records.external_validation import (observation_indices,public_observation_record,
    choose_pairs,minigrid_reference_profile,aggregate_record)
from behavioral_records.relational_record import RelationalBehaviorCompiler,relation_broken_control,audit_relation_break,_audit_boolean_fields
from behavioral_records.relational_text import render_relational_text,parse_relational_text


def episode(n=100):
    return {'split':'validation','candidate_id':'private-never-expose','seed_private':9999,
            'context':{'wind':1},'outcome':'terminated','return':1.,'terminated':True,'truncated':False,
            'steps':[{'t':i,'action':'left' if i%3 else 'right','reward':0.1234567,
                      'events':['contact'] if i==52 else [],'state':{'x':i/997},'predicates':{}} for i in range(n)]}


def test_fixed_budget_keeps_factual_event_neighborhood_and_original_times():
    ep=episode();idx=observation_indices(ep['steps'])
    assert len(idx)==48 and set([0,99,51,52,53])<=set(idx)
    result=public_observation_record([ep])
    assert len(result['episodes'][0]['steps'])==48
    assert [s['t'] for s in result['episodes'][0]['steps']]==idx
    assert 'private-never-expose' not in str(result) and '9999' not in str(result)
    assert result['episodes'][0]['context']['recording_scope']['complete_episode_step_count']==100
    assert list(inspect.signature(public_observation_record).parameters)==['episodes']


def test_reference_and_development_cannot_enter_public_compiler():
    for split in ('reference','development'):
        ep=episode();ep['split']=split
        with pytest.raises(ValueError):public_observation_record([ep])


def test_original_compiler_and_text_keep_budgeted_records_valid_and_matched():
    ep=episode();other=copy.deepcopy(ep);other['context']={'wind':2}
    other['steps'][3]['action']='wait'
    full=public_observation_record([ep,other]);linked=RelationalBehaviorCompiler().compile(full)
    broken=relation_broken_control(linked)
    assert _audit_boolean_fields(audit_relation_break(linked,broken))
    assert parse_relational_text(render_relational_text(linked))==linked
    assert len(render_relational_text(linked))==len(render_relational_text(broken))
    panel=aggregate_record(full)
    assert sum(panel['episode_panels'][0]['action_counts'].values())==48


def test_pair_selection_uses_multiple_chains_and_never_requires_reference_labels():
    candidates=[{'candidate_id':f'c{i}','chain_id':f'chain{i//3}'} for i in range(18)]
    summaries={c['candidate_id']:{'success_rate':0.5,'mean_return':i,'mean_length':100+i} for i,c in enumerate(candidates)}
    pairs=choose_pairs(candidates,summaries,40)
    assert len(pairs)==40 and len({r['pair_id'] for r in pairs})==40
    assert all(p['chain_a']!=p['chain_b'] for p in pairs)
    assert pairs==choose_pairs(candidates,summaries,40)


def test_retry_reference_depends_on_unchanged_situation_not_just_action_count():
    ep=episode(4);ep['split']='reference';ep['outcome']='success'
    for s in ep['steps']:
        s.update(action='pickup',state={'agent_x':1,'carrying':None},events=['command_without_state_change'])
    profile=minigrid_reference_profile([ep]);assert profile['aggregate']['persistent_retry_episode_rate']==1
    ep['steps'][1]['state']['agent_x']=2
    assert minigrid_reference_profile([ep])['aggregate']['persistent_retry_episode_rate']==0
    ep['split']='validation'
    with pytest.raises(ValueError):minigrid_reference_profile([ep])
