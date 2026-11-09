"""Reproduce the fixed five-policy Highway analysis from retained responses."""
from pathlib import Path
from collections import Counter
from statistics import fmean
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from behavioral_records.io import read_json, read_jsonl, write_json
from evaluate_kbs_external_study import CONTRASTS, node_interval, score_panels

def group_summary(panels, pair_ids, arms):
    selected = [r for r in panels if r["pair_id"] in pair_ids]
    valid = [r for r in selected if r["determinate"]]
    lookup = {(r["pair_id"], r["arm"]): r for r in valid}
    contrasts = []
    for left, right in CONTRASTS:
        if left not in arms or right not in arms:
            continue
        rows = [{**r, "effect": r["score"]-lookup[(r["pair_id"], right)]["score"]} for r in valid if r["arm"] == left]
        contrasts.append({"left": left, "right": right, "pair_count": len(rows), "sensitivity": node_interval(rows)})
    return {"planned_pairs": len(pair_ids), "determinate_pairs": len({r["pair_id"] for r in valid}),
        "ambiguous_pairs": sorted({r["pair_id"] for r in selected if not r["determinate"]}),
        "accuracy": {a: fmean([r["score"] for r in valid if r["arm"] == a]) if any(r["arm"] == a for r in valid) else None for a in arms},
        "contrasts": contrasts, "pair_scores": selected}


def presentation_summary(mapping, responses, arms, pair_ids):
    by_id = {r["request_id"]: r["response"]["winner"] for r in responses}
    rows = [r for r in mapping if r["arm"] in arms and r["pair_id"] in pair_ids and r["oracle_status"] == "DETERMINATE"]
    output = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        means = {}
        for field, levels in (("order", ("AB", "BA")), ("paraphrase", ("P0", "P1"))):
            for level in levels:
                subset = [r for r in selected if r[field] == level]
                means[field+"_"+level] = {"correct": sum(by_id.get(r["request_id"]) == r["expected_winner"] for r in subset), "planned": len(subset)}
        output[arm] = means
    return output


def analyze(out, destination):
    plan = read_json(out / 'protocol.json')['plan']
    frozen = read_json(out / 'fixed_cohort.json')
    mapping = read_jsonl(out / "prepared/judge_private_map.jsonl")
    subgroup = set(frozen["development_relation_pair_ids"])
    qualification = read_json(out / "highway_qualification.json")
    all_ids = {p["pair_id"] for p in qualification["pairs"]}
    refs = {r["pair_id"]: r["reference"] for r in read_jsonl(out / "collection/reference_pairs_private.jsonl")}
    transitions = []
    for pair in qualification["pairs"]:
        ref = refs[pair["pair_id"]]
        transitions.append({"pair_id": pair["pair_id"], "fixed_development_relation_subgroup": pair["pair_id"] in subgroup,
            "development": pair["development_reference"], "independent_reference": ref,
            "label_changed": pair["development_reference"]["label"] != ref["label"],
            "decisive_field_changed": pair["development_reference"]["decisive_field"] != ref["decisive_field"]})
    results = {}
    total_attempts, total_spent = 0, 0.
    for provider in ("minimax", "deepseek"):
        folder = out / ("judge_"+provider)
        attempts = read_jsonl(folder / "attempts.jsonl") if (folder / "attempts.jsonl").exists() else []
        responses = read_jsonl(folder / "judge_responses.jsonl") if (folder / "judge_responses.jsonl").exists() else []
        requests = read_jsonl(out / "prepared" / ("judge_requests.jsonl" if provider == "minimax" else "deepseek_requests.jsonl"))
        attempted = [r["request_id"] for r in attempts]
        if len(set(attempted)) != len(attempted) or not set(attempted) <= {r["request_id"] for r in requests}:
            raise ValueError("Invalid attempt identities")
        if not {r["request_id"] for r in responses} <= set(attempted):
            raise ValueError("Response without attempt")
        arms = plan["judges"][provider+"_arms"]
        panels = score_panels(mapping, responses, arms)
        groups = {"all_pairs": all_ids, "fixed_development_relation_pairs": subgroup, "remaining_pairs": all_ids-subgroup}
        results[provider] = {"status": "COMPLETE" if len(attempts) == len(requests) else "INCOMPLETE",
            "attempted": len(attempts), "valid": len(responses), "planned": len(requests),
            "accounted_cost_cny": sum(r["accounted_cost_cny"] for r in attempts),
            "served_models": dict(Counter(r.get("served_model", "unknown") for r in responses)),
            "groups": {name: group_summary(panels, ids, arms) for name, ids in groups.items()},
            "presentation": {name: presentation_summary(mapping, responses, arms, ids) for name, ids in groups.items()}}
        total_attempts += len(attempts)
        total_spent += results[provider]["accounted_cost_cny"]
    if total_attempts > 240:
        raise ValueError("Request budget violated")
    report = {"status": "COMPLETE" if total_attempts == 240 else "CLOSED_INCOMPLETE",
        "providers": results, "attempted": total_attempts, "accounted_cost_cny": total_spent,
        "reference_transitions": transitions, "fixed_development_relation_pair_ids": sorted(subgroup),
        "scope": "Fixed five-policy auxiliary study; one semantic requirement with two paraphrases. Descriptive reused-policy sensitivity, not broad population inference.",
        "no_additional_training": True, "no_further_experiments": True, "manuscript_modified": False}
    write_json(destination, report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,default=ROOT/'outputs/kbs_highway_fixed_assessment_20261108')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/highway.json')
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=analyze(a.study,a.output)
    expected=read_json(a.study/'final_analysis.json')
    if result!=expected: raise SystemExit('HIGHWAY_REPRODUCTION_DIFFERS')
    print('EXACT_HIGHWAY_REPRODUCTION_PASS')
