from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .experiment import group_by_policy
from .io import read_json, read_trajectories, write_json, write_jsonl
from .predictive_record import build_predictive_views, compile_predictive_profiles
from .predictive_recoverability import audit_predictive_recoverability


def run_predictive_offline_audit(
    trajectory_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    validation = [
        row
        for row in read_trajectories(trajectory_path)
        if row.context.get("split") == "validation"
    ]
    if not validation:
        raise ValueError("TPBR v0 requires validation trajectories")
    groups = group_by_policy(validation)
    profiles = compile_predictive_profiles(groups, config)
    views = build_predictive_views(profiles, config)
    audit = audit_predictive_recoverability(profiles, views, config)
    audit["material_passport"] = {
        "origin_skill": "academic-research-suite/experiment-agent",
        "origin_mode": "run+validate",
        "origin_date": datetime.now(timezone.utc).isoformat(),
        "verification_status": "VERIFIED" if audit["passed"] else "ANALYZED",
        "version_label": "tpbr_v0",
    }
    write_jsonl(output / "tpbr_profiles.jsonl", profiles.values())
    write_jsonl(output / "tpbr_judge_views.jsonl", views.values())
    write_json(output / "tpbr_recoverability_audit.json", audit)
    summary = {
        "status": "COMPLETE_OFFLINE_ONLY" if audit["passed"] else "FAILED_REPRESENTATION_GATE",
        "policy_count": len(groups),
        "validation_episode_count": len(validation),
        "config_file": str(Path(config_path).resolve()),
        "config_hash": next(iter(profiles.values()))["config_hash"],
        "recoverability_passed": audit["passed"],
        "core_fact_coverage_min": audit["aggregate"]["core_coverage_min"],
        "core_included_accuracy_mean": audit["aggregate"]["core_included_accuracy_mean"],
        "extended_fact_coverage_mean": audit["aggregate"]["extended_coverage_mean"],
        "extended_included_accuracy_mean": audit["aggregate"]["extended_included_accuracy_mean"],
        "public_characters_mean": audit["aggregate"]["public_characters_mean"],
        "public_characters_max": audit["aggregate"]["public_characters_max"],
        "claims": (
            "Offline representation evidence only. No LLM/model-judge call and no policy-selection "
            "performance claim were produced."
        ),
    }
    write_json(output / "tpbr_summary.json", summary)
    return summary
