"""Objective-agnostic behavioral records for policy selection."""

from .extractor import BehaviorRecordExtractor
from .models import Step, Trajectory
from .oracle import ObjectiveOracle, OracleResult, rank_policies
from .predictive_record import (
    PredictiveBehaviorCompiler,
    build_predictive_pair_prompt,
    build_predictive_view,
    type_episode_evidence,
)

__all__ = [
    "BehaviorRecordExtractor",
    "ObjectiveOracle",
    "OracleResult",
    "PredictiveBehaviorCompiler",
    "Step",
    "Trajectory",
    "build_predictive_pair_prompt",
    "build_predictive_view",
    "rank_policies",
    "type_episode_evidence",
]

__version__ = "0.1.0"
