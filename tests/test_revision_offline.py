import copy
import pytest
from behavioral_records.revision_offline import public_full_trace, requirement_effects, leave_group_out_effects


def test_public_adapter_excludes_private_identity_and_rejects_reference():
    ep = {"context": {"gravity": -10}, "length": 1, "outcome": "truncated",
          "return": 0, "terminated": False, "truncated": True, "policy_id": "SECRET",
          "episode_id": "SECRET", "metadata": {"evaluation_split": "validation", "lineage": "SECRET"},
          "steps": [{"t": 0, "action": "wait", "state": {"x": 0}, "oracle": "SECRET"}]}
    result = public_full_trace([ep])
    assert "SECRET" not in str(result)
    assert result["episodes"][0]["steps"][0]["state"]["x"] == 0
    heldout = copy.deepcopy(ep)
    heldout["metadata"]["evaluation_split"] = "heldout"
    with pytest.raises(ValueError, match="Reference"):
        public_full_trace([heldout])


def test_effects_are_paired_and_requests_do_not_inflate_sample_size():
    rows=[]
    for s, a, b in [("C1", .75, .25), ("C2", .25, .5)]:
        for i in range(3):
            for arm,value in [("Relational-Linked",a),("Relational-LinkBroken",b)]:
                rows.append(dict(scenario_id=s,pair_id=str(i),arm=arm,oracle_label_mass=value,determinate=True,request_count=100))
    r=requirement_effects(rows,seed=4)
    assert r['by_requirement']['C1']['n']==3
    assert r['recovery_minus_necessity']['ci95']==[.75,.75]


def test_leave_group_out_removes_both_endpoints_not_only_first():
    manifest=[dict(pair_id='a',policy_a='p1',policy_b='p2'),dict(pair_id='b',policy_a='p2',policy_b='p3')]
    scores=[dict(pair_id=p,arm=a,oracle_label_mass=v) for p,effect in [('a',1),('b',0)]
            for a,v in [('B1',0),('B2-DenseTrace',effect)]]
    chains={p:{'chain':c} for p,c in [('p1','1'),('p2','2'),('p3','3')]}
    r=leave_group_out_effects(scores,manifest,chains,'chain')
    middle=next(x for x in r['leave_one_group_out'] if x['omitted_group']=='2')
    assert middle['remaining_pairs']==0 and middle['mean'] is None
