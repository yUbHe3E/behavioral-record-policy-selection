import importlib.util
from pathlib import Path
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_kbs_external_study.py'
    spec=importlib.util.spec_from_file_location('external_evaluator',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_failures_and_order_are_scored_in_the_same_four_context_panel():
    m=module();mapping=[];responses=[]
    for para in ('P0','P1'):
        for order in ('AB','BA'):
            rid=para+order;winner='A' if order=='AB' else 'B'
            mapping.append(dict(request_id=rid,pair_id='pair',scenario_id='req',arm='linked',paraphrase=para,order=order,
                                oracle_status='DETERMINATE',expected_winner=winner,chain_a='c1',chain_b='c2',policy_a_private='p1',policy_b_private='p2'))
            if rid!='P1BA':responses.append(dict(request_id=rid,response={'winner':winner}))
    panels=m.score_panels(mapping,responses,['linked'])
    assert len(panels)==1 and panels[0]['score']==.75 and panels[0]['valid_calls']==3
    with pytest.raises(ValueError):m.score_panels(mapping,responses+responses[:1],['linked'])
    with pytest.raises(ValueError):m.score_panels(mapping[:-1],responses,['linked'])


def test_node_resampling_keeps_shared_pairs_together_and_does_not_count_calls():
    m=module()
    rows=[dict(chain_a=a,chain_b=b,requirement=r,effect=.25) for a,b in [('a','b'),('a','c'),('b','c')] for r in ['r1','r2']]
    result=m.node_interval(rows,repetitions=200)
    assert result['nodes']==3 and result['point']==.25
    assert result['interval_95']==[.25,.25]
    assert result['leave_one_node_out_range']==[.25,.25]
    assert m.balanced_mean([{'requirement':'r1','effect':0},{'requirement':'r2','effect':1},{'requirement':'r2','effect':1}])==.5
