"""Protect seed-level inference and reference disagreement handling."""
import importlib.util
from pathlib import Path


def evaluator():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_revision_retention.py'
    spec=importlib.util.spec_from_file_location('retention_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_reference_disagreement_is_not_a_forced_winner():
    m=evaluator()
    a={'causal::x': True,'behavior::x': False}
    b={'causal::x': False,'behavior::x': True}
    assert m.reference_label(a,b)=='AMBIGUOUS'
    assert m.reference_label(a,a)=='TIE'
    assert m.reference_label({'causal::x':True,'behavior::x':True},b)=='AMBIGUOUS'
    assert m.reference_label({'causal::x':True,'behavior::x':True},
                             {'causal::x':False,'behavior::x':False})=='A'


def test_retention_endpoint_resamples_seeds_not_scenarios_or_calls():
    m=evaluator();rows=[]
    for seed in range(5,13):
        for scenario in ('C1','C2'):
            for arm in (m.LINKED,m.BROKEN):
                rows.append({'outer_seed':seed,'scenario_id':scenario,'selector_arm':arm,'generation':2,
                             'selected_parent_mean_semantic_violation_rate':.5 if arm==m.BROKEN else 0.})
    r=m.seed_contrast(rows,2)
    assert r['n']==8
    assert r['mean']==.5
    assert r['ci95']==[.5,.5]
    assert r['exact_two_sided_sign_flip_p']==2/256


def test_equal_pass_counts_with_different_failures_remain_ambiguous():
    m=evaluator()
    a={'causal::x':True,'causal::y':False,'behavior::x':True,'behavior::y':False}
    b={'causal::x':False,'causal::y':True,'behavior::x':False,'behavior::y':True}
    assert m.reference_label(a,b)=='AMBIGUOUS'
