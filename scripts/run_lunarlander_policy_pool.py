from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from behavioral_records.canonical import canonical_json, stable_hash  # noqa: E402
from behavioral_records.io import read_json, write_json, write_trajectories  # noqa: E402
from behavioral_records.lunarlander_training import (  # noqa: E402
    evaluation_seeds,
    make_lunarlander_vector_env,
    rollout_lunarlander_policy,
    summarize_validation_trajectories,
    training_candidate_lookup,
    validate_lunarlander_training_config,
)
from behavioral_records.lunarlander_transfer import (  # noqa: E402
    lunarlander_runtime_inventory,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row))
        handle.write("\n")


def relative_to_output(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen official-reward LunarLander PPO pool and export "
            "objective-blind validation trajectories."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lunarlander_policy_pool_training_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    training_config = read_json(config_path)
    source_path = Path(training_config["source_transfer_config"]["path"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    transfer_config = read_json(source_path)
    validation = validate_lunarlander_training_config(
        training_config, transfer_config
    )
    if not validation["passed"]:
        raise SystemExit(
            "FROZEN_TRAINING_CONFIG_INVALID: "
            + json.dumps(validation["gates"], sort_keys=True)
        )

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"OUTPUT_DIRECTORY_NOT_EMPTY_AND_OVERWRITE_IS_DISABLED: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "training_progress.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"
    status_path = output_dir / "run_status.json"

    runtime = lunarlander_runtime_inventory()
    requested_device = args.device or str(training_config["training"]["device"])
    if requested_device == "cuda" and not runtime["cuda_available"]:
        write_json(
            status_path,
            {
                "status": "SKIPPED_CUDA_UNAVAILABLE",
                "requested_device": requested_device,
                "runtime": runtime,
                "api_calls": 0,
            },
        )
        raise SystemExit("SKIPPED_CUDA_UNAVAILABLE")

    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "SKIPPED_MISSING_TRAINING_DEPENDENCY",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "api_calls": 0,
            },
        )
        raise SystemExit(
            f"SKIPPED_MISSING_TRAINING_DEPENDENCY: {type(exc).__name__}: {exc}"
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class HeartbeatCallback(BaseCallback):
        def __init__(self, job: dict[str, Any], seconds: float):
            super().__init__(verbose=0)
            self.job = dict(job)
            self.seconds = float(seconds)
            self.last_write = 0.0

        def _on_step(self) -> bool:
            now = time.monotonic()
            if now - self.last_write >= self.seconds:
                write_json(
                    heartbeat_path,
                    {
                        "status": "RUNNING_TRAINING",
                        "updated_at_utc": utc_now(),
                        "model_timesteps": int(self.model.num_timesteps),
                        **self.job,
                        "api_calls": 0,
                    },
                )
                self.last_write = now
            return True

    if args.smoke:
        smoke = training_config["smoke_override"]
        profiles = [str(value) for value in smoke["profiles"]]
        seeds = [int(value) for value in smoke["seeds"]]
        checkpoints = [int(value) for value in smoke["checkpoint_timesteps"]]
    else:
        profiles = [str(value) for value in training_config["training"]["profiles"]]
        seeds = [int(value) for value in training_config["training"]["seeds"]]
        checkpoints = [
            int(value)
            for value in training_config["training"]["checkpoint_timesteps"]
        ]
    seed_map = evaluation_seeds(training_config, smoke=args.smoke)
    candidate_lookup = (
        training_candidate_lookup(transfer_config) if not args.smoke else {}
    )
    ppo_parameters = dict(training_config["training"]["ppo"])
    vector_environments = int(training_config["training"]["vector_environments"])
    heartbeat_seconds = float(training_config["monitoring"]["heartbeat_seconds"])

    manifest = {
        "schema_version": "lunarlander-policy-pool-run-manifest-v1",
        "created_at_utc": utc_now(),
        "mode": "smoke" if args.smoke else "formal",
        "training_config_path": str(config_path.resolve()),
        "training_config_hash": stable_hash(training_config),
        "source_transfer_config_path": str(source_path.resolve()),
        "source_transfer_config_hash": stable_hash(transfer_config),
        "config_validation": validation,
        "profiles": profiles,
        "training_seeds": seeds,
        "checkpoint_timesteps": checkpoints,
        "validation_seed_hash": stable_hash(seed_map),
        "heldout_trajectories_collected": False,
        "requested_device": requested_device,
        "runtime": runtime,
        "stable_baselines3_version": getattr(
            stable_baselines3, "__version__", "unknown"
        ),
        "official_reward_modified": False,
        "semantic_reward_shaping": False,
        "automatic_retry": False,
        "api_calls": 0,
        "claim": (
            training_config["smoke_override"]["claim_status"]
            if args.smoke
            else training_config["claim_boundary"]
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(
        status_path,
        {
            "status": "RUNNING",
            "started_at_utc": utc_now(),
            "completed_candidates": 0,
            "expected_candidates": len(profiles) * len(seeds) * len(checkpoints),
            "api_calls": 0,
        },
    )

    candidate_rows: list[dict[str, Any]] = []
    started_run = time.monotonic()
    active_job: dict[str, Any] | None = None
    try:
        for profile in profiles:
            for training_seed in seeds:
                vector_env = make_lunarlander_vector_env(
                    transfer_config,
                    profile,
                    training_seed=training_seed,
                    n_envs=vector_environments,
                )
                chain_started = time.monotonic()
                try:
                    model = PPO(
                        str(training_config["training"]["policy"]),
                        vector_env,
                        seed=training_seed,
                        device=requested_device,
                        verbose=0,
                        **ppo_parameters,
                    )
                    previous_checkpoint = 0
                    for checkpoint in checkpoints:
                        candidate_id = (
                            candidate_lookup[(profile, training_seed, checkpoint)]
                            if not args.smoke
                            else f"ll_smoke_{stable_hash({'profile': profile, 'seed': training_seed, 'checkpoint': checkpoint}, 12)}"
                        )
                        active_job = {
                            "training_profile": profile,
                            "training_seed": training_seed,
                            "target_checkpoint": checkpoint,
                            "candidate_id": candidate_id,
                        }
                        checkpoint_started = time.monotonic()
                        callback = HeartbeatCallback(active_job, heartbeat_seconds)
                        model.learn(
                            total_timesteps=checkpoint - previous_checkpoint,
                            reset_num_timesteps=False,
                            progress_bar=False,
                            callback=callback,
                        )
                        if int(model.num_timesteps) != checkpoint:
                            raise RuntimeError(
                                "PPO did not stop on the frozen checkpoint boundary: "
                                f"expected {checkpoint}, observed {model.num_timesteps}"
                            )

                        model_path = (
                            output_dir
                            / "models"
                            / profile
                            / f"seed_{training_seed}"
                            / f"{candidate_id}_step_{checkpoint}"
                        )
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        model.save(model_path)
                        model_zip_path = model_path.with_suffix(".zip")

                        write_json(
                            heartbeat_path,
                            {
                                "status": "RUNNING_VALIDATION",
                                "updated_at_utc": utc_now(),
                                "model_timesteps": int(model.num_timesteps),
                                **active_job,
                                "api_calls": 0,
                            },
                        )
                        trajectories = []
                        for validation_context in sorted(seed_map):
                            for evaluation_seed in seed_map[validation_context]:
                                trajectories.append(
                                    rollout_lunarlander_policy(
                                        model=model,
                                        transfer_config=transfer_config,
                                        policy_id=candidate_id,
                                        context_id=validation_context,
                                        seed=evaluation_seed,
                                        deterministic=bool(
                                            training_config["validation"][
                                                "deterministic_actions"
                                            ]
                                        ),
                                        split="validation",
                                    )
                                )
                        trajectory_path = (
                            output_dir
                            / "validation_trajectories"
                            / f"{candidate_id}.jsonl"
                        )
                        write_trajectories(trajectory_path, trajectories)
                        metrics = summarize_validation_trajectories(trajectories)
                        candidate_row = {
                            "candidate_id": candidate_id,
                            "training_profile_private": profile,
                            "training_seed_private": training_seed,
                            "checkpoint_timestep_private": checkpoint,
                            "actual_model_timesteps": int(model.num_timesteps),
                            "model_path_private": relative_to_output(
                                model_zip_path, output_dir
                            ),
                            "validation_trajectory_path": relative_to_output(
                                trajectory_path, output_dir
                            ),
                            "validation_trajectory_hash": stable_hash(
                                [trajectory.to_dict() for trajectory in trajectories]
                            ),
                            "validation": metrics,
                            "checkpoint_seconds": time.monotonic()
                            - checkpoint_started,
                            "chain_elapsed_seconds": time.monotonic() - chain_started,
                            "actual_device": str(model.device),
                            "heldout_evaluated": False,
                            "api_calls": 0,
                        }
                        candidate_rows.append(candidate_row)
                        append_jsonl(progress_path, candidate_row)
                        write_json(
                            status_path,
                            {
                                "status": "RUNNING",
                                "updated_at_utc": utc_now(),
                                "completed_candidates": len(candidate_rows),
                                "expected_candidates": len(profiles)
                                * len(seeds)
                                * len(checkpoints),
                                "last_completed_candidate": candidate_id,
                                "api_calls": 0,
                            },
                        )
                        previous_checkpoint = checkpoint
                finally:
                    vector_env.close()

        success_rates = [
            float(row["validation"]["official_return_200_success_rate"])
            for row in candidate_rows
        ]
        unique_behavior_signatures = len(
            {
                str(row["validation"]["behavior_signature"])
                for row in candidate_rows
            }
        )
        saturated_fraction = (
            sum(value >= 0.95 for value in success_rates) / len(success_rates)
            if success_rates
            else 1.0
        )
        frozen_gates = transfer_config["offline_qualification_gates"]
        training_gates = {
            "minimum_candidates": len(candidate_rows)
            >= int(frozen_gates["minimum_candidates"]),
            "minimum_unique_behavior_hashes": unique_behavior_signatures
            >= int(frozen_gates["minimum_unique_behavior_hashes"]),
            "minimum_nonzero_policy_success_rate": bool(success_rates)
            and max(success_rates)
            >= float(frozen_gates["minimum_nonzero_policy_success_rate"]),
            "maximum_pool_saturation_rate": saturated_fraction
            <= float(frozen_gates["maximum_pool_saturation_rate"]),
            "heldout_untouched": True,
            "api_calls_before_all_gates_pass": True,
        }
        completed = {
            "schema_version": "lunarlander-policy-pool-training-summary-v1",
            "status": (
                "SMOKE_PASS_NO_RESEARCH_CLAIM"
                if args.smoke
                else "PASS_TRAINING_POOL_GATES_PENDING_PAIR_AND_ORACLE_GATES"
                if all(training_gates.values())
                else "STOP_TRAINING_POOL_GATE_FAILED"
            ),
            "completed_at_utc": utc_now(),
            "mode": "smoke" if args.smoke else "formal",
            "candidate_count": len(candidate_rows),
            "training_chain_count": len(profiles) * len(seeds),
            "unique_behavior_signatures": unique_behavior_signatures,
            "official_return_200_success_rate_range": (
                [min(success_rates), max(success_rates)] if success_rates else None
            ),
            "saturated_candidate_fraction": saturated_fraction,
            "training_gates": training_gates,
            "gates_not_yet_evaluated": [
                "minimum_metric_matched_pairs_per_requirement",
                "minimum_hidden_oracle_determinate_coverage",
                "maximum_metric_only_balanced_accuracy",
            ],
            "candidate_rows": candidate_rows,
            "wall_time_seconds": time.monotonic() - started_run,
            "heldout_trajectories_collected": False,
            "api_calls": 0,
            "claim": manifest["claim"],
        }
        write_json(output_dir / "summary.json", completed)
        write_json(
            status_path,
            {
                "status": completed["status"],
                "completed_at_utc": completed["completed_at_utc"],
                "completed_candidates": len(candidate_rows),
                "api_calls": 0,
            },
        )
        print(
            json.dumps(
                {
                    "output": str((output_dir / "summary.json").resolve()),
                    "status": completed["status"],
                    "candidate_count": completed["candidate_count"],
                    "unique_behavior_signatures": unique_behavior_signatures,
                    "success_rate_range": completed[
                        "official_return_200_success_rate_range"
                    ],
                    "wall_time_seconds": completed["wall_time_seconds"],
                    "api_calls": 0,
                },
                indent=2,
            )
        )
    except Exception as exc:
        failure = {
            "status": "FAILED_NO_AUTOMATIC_RETRY",
            "failed_at_utc": utc_now(),
            "completed_candidates": len(candidate_rows),
            "active_job": active_job,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "partial_progress_path": str(progress_path.resolve()),
            "automatic_retry": False,
            "heldout_trajectories_collected": False,
            "api_calls": 0,
        }
        write_json(status_path, failure)
        raise


if __name__ == "__main__":
    main()
