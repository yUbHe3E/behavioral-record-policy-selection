from __future__ import annotations

import json

import pytest

from behavioral_records.io import read_jsonl
from behavioral_records.judge_runner import (
    JudgeConfig,
    _completion_payload,
    cache_key,
    final_answer_content,
    run_judge_requests,
    select_requests,
    strict_parse_judge_response,
)


def _request(request_id: str, arm: str) -> dict[str, str]:
    return {
        "request_id": request_id,
        "pair_id": "pair_1",
        "objective_id": "objective_1",
        "evidence_arm": arm,
        "order": "AB",
        "policy_a_id": "alpha",
        "policy_b_id": "beta",
        "prompt": "choose from factual evidence",
    }


def test_strict_response_parser_rejects_wrappers_and_extra_keys() -> None:
    assert strict_parse_judge_response(
        '{"winner":"A","confidence":0.75,"reason":"safer"}'
    )["winner"] == "A"
    with pytest.raises(ValueError):
        strict_parse_judge_response(
            '```json\n{"winner":"A","confidence":0.75,"reason":"safer"}\n```'
        )
    with pytest.raises(ValueError):
        strict_parse_judge_response(
            '{"winner":"A","confidence":0.75,"reason":"safer","score":2}'
        )


def test_minimax_documented_reasoning_envelope_is_removed_before_strict_parse() -> None:
    final, removed = final_answer_content(
        '<think>private reasoning</think>\n{"winner":"TIE","confidence":0.5,"reason":"same"}',
        "minimax",
    )
    assert removed is True
    assert strict_parse_judge_response(final)["winner"] == "TIE"


def test_minimax_json_object_mode_is_explicit_and_cache_separated() -> None:
    unconstrained = JudgeConfig(
        provider="minimax",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret",
        max_requests=1,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        temperature=0.1,
    )
    structured = JudgeConfig(
        **{
            **unconstrained.__dict__,
            "response_format_type": "json_object",
        }
    )
    assert "response_format" not in _completion_payload("prompt", unconstrained)
    assert _completion_payload("prompt", structured)["response_format"] == {
        "type": "json_object"
    }
    assert cache_key(_request("r1", "P"), unconstrained) != cache_key(
        _request("r1", "P"), structured
    )


def test_runner_caches_valid_responses_without_recording_key(tmp_path) -> None:
    calls = []

    def fake_transport(endpoint, headers, payload, timeout):
        calls.append((endpoint, headers, payload, timeout))
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"winner": "B", "confidence": 0.8, "reason": "better fit"}
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }

    config = JudgeConfig(
        provider="generic",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret-not-for-disk",
        max_requests=1,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    first = run_judge_requests([_request("r1", "P")], output_dir=tmp_path, config=config, transport=fake_transport)
    second = run_judge_requests([_request("r1", "P")], output_dir=tmp_path, config=config, transport=fake_transport)
    assert first["completed_request_count"] == 1
    assert second["cached_request_count"] == 1
    assert len(calls) == 1
    responses = read_jsonl(tmp_path / "judge_responses.jsonl")
    assert responses[0]["response"]["winner"] == "B"
    assert responses[0]["finish_reason"] is None
    assert "secret-not-for-disk" not in (tmp_path / "judge_responses.jsonl").read_text(encoding="utf-8")


def test_request_selection_keeps_arms_adjacent() -> None:
    rows = [_request("b", "P"), _request("a", "B0")]
    selected = select_requests(rows, limit=2)
    assert [row["evidence_arm"] for row in selected] == ["B0", "P"]


def test_runner_accepts_an_explicit_response_parser(tmp_path) -> None:
    def fake_transport(endpoint, headers, payload, timeout):
        return {
            "choices": [{"message": {"content": '{"custom":"ok"}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    config = JudgeConfig(
        provider="generic",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret",
        max_requests=1,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
    )
    summary = run_judge_requests(
        [_request("custom", "Q")],
        output_dir=tmp_path,
        config=config,
        transport=fake_transport,
        response_parser=lambda content: json.loads(content),
    )
    assert summary["completed_request_count"] == 1
    assert read_jsonl(tmp_path / "judge_responses.jsonl")[0]["response"] == {
        "custom": "ok"
    }


def test_concurrent_runner_attempts_each_request_once_and_writes_in_input_order(tmp_path) -> None:
    calls = []

    def fake_transport(endpoint, headers, payload, timeout):
        calls.append(payload["messages"][1]["content"])
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"winner":"A","confidence":0.7,"reason":"fit"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    config = JudgeConfig(
        provider="generic",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret",
        max_requests=3,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        max_concurrency=3,
    )
    requests = [_request(f"r{index}", "P") for index in range(3)]
    summary = run_judge_requests(
        requests,
        output_dir=tmp_path,
        config=config,
        transport=fake_transport,
    )
    assert summary["completed_request_count"] == 3
    assert summary["max_concurrency"] == 3
    assert len(calls) == 3
    assert [row["request_id"] for row in read_jsonl(tmp_path / "judge_attempts.jsonl")] == [
        "r0",
        "r1",
        "r2",
    ]


def test_sequential_runner_stops_after_explicit_failure_pattern(tmp_path) -> None:
    calls = []

    def fake_transport(endpoint, headers, payload, timeout):
        calls.append(payload["messages"][1]["content"])
        raise RuntimeError("Judge HTTP 429: provider limit")

    config = JudgeConfig(
        provider="generic",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret",
        max_requests=3,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        max_concurrency=1,
    )
    requests = [_request(f"r{index}", "P") for index in range(3)]
    summary = run_judge_requests(
        requests,
        output_dir=tmp_path,
        config=config,
        transport=fake_transport,
        stop_on_failure_error_substrings=("HTTP 429",),
    )
    assert summary["status"] == "PARTIAL"
    assert summary["stopped_reason"] == "failure_stop_pattern_matched"
    assert summary["failed_request_count"] == 1
    assert len(calls) == 1


def test_failure_triggered_stop_rejects_concurrent_scheduling(tmp_path) -> None:
    config = JudgeConfig(
        provider="generic",
        base_url="https://judge.invalid/v1",
        model="test-model",
        api_key="secret",
        max_requests=3,
        max_cost_usd=1.0,
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        max_concurrency=2,
    )
    with pytest.raises(ValueError, match="max_concurrency=1"):
        run_judge_requests(
            [_request("r0", "P")],
            output_dir=tmp_path,
            config=config,
            stop_on_failure_error_substrings=("HTTP 429",),
        )
