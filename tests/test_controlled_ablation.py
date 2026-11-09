from __future__ import annotations

from behavioral_records.canonical import canonical_json
from behavioral_records.controlled_ablation import (
    COMPACT_MASK_TOKENS,
    audit_state_mask,
    audit_temporal_shuffle,
    state_masked_view,
    temporal_shuffle_view,
)


def _evidence() -> dict:
    return {
        "representation": "deterministic_raw_validation_trajectories",
        "episodes": [
            {
                "evidence_id": "episode-1",
                "context": {"condition": "held-fixed"},
                "length": 3,
                "outcome": "success",
                "return": 1.0,
                "terminated": True,
                "truncated": False,
                "steps": [
                    {
                        "t": 0,
                        "action": "start",
                        "reward": 0.0,
                        "events": [],
                        "state": {"progress": 0, "safe": True},
                        "predicates": {"advanced": False},
                    },
                    {
                        "t": 1,
                        "action": "advance",
                        "reward": 0.0,
                        "events": ["checkpoint"],
                        "state": {"progress": 1, "safe": True},
                        "predicates": {"advanced": True},
                    },
                    {
                        "t": 2,
                        "action": "finish",
                        "reward": 1.0,
                        "events": ["goal"],
                        "state": {"progress": 2, "safe": True},
                        "predicates": {"advanced": True},
                    },
                ],
            }
        ],
    }


def test_temporal_shuffle_is_deterministic_character_and_content_matched() -> None:
    source = _evidence()
    first = temporal_shuffle_view(source)
    second = temporal_shuffle_view(source)
    assert first == second
    assert first != source
    audit = audit_temporal_shuffle(source, first)
    assert all(
        value
        for key, value in audit.items()
        if key not in {"source_characters", "transformed_characters"}
    )


def test_state_mask_preserves_non_state_trace_and_hides_all_values() -> None:
    source = _evidence()
    masked = state_masked_view(source)
    audit = audit_state_mask(source, masked)
    assert all(
        value
        for key, value in audit.items()
        if key not in {"source_characters", "transformed_characters"}
    )
    step = masked["episodes"][0]["steps"][0]
    assert step["action"] == "start"
    assert step["state"] == {
        "progress": "<NUMBER_MASKED>",
        "safe": "<BOOL_MASKED>",
    }
    assert step["predicates"] == {"advanced": "<BOOL_MASKED>"}


def test_compact_state_mask_preserves_type_information_with_fewer_characters() -> None:
    source = _evidence()
    verbose = state_masked_view(source)
    compact = state_masked_view(source, mask_tokens=COMPACT_MASK_TOKENS)
    audit = audit_state_mask(
        source, compact, mask_tokens=COMPACT_MASK_TOKENS
    )
    assert all(
        value
        for key, value in audit.items()
        if key not in {"source_characters", "transformed_characters"}
    )
    step = compact["episodes"][0]["steps"][0]
    assert step["state"] == {"progress": "<N>", "safe": "<B>"}
    assert step["predicates"] == {"advanced": "<B>"}
    assert audit["transformed_characters"] < len(canonical_json(verbose))
