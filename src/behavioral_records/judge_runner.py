from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .canonical import canonical_json
from .io import read_jsonl, write_json


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]
ResponseParser = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    max_requests: int
    max_cost_usd: float
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    max_output_tokens: int = 512
    max_prompt_chars: int = 200_000
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    free_endpoint: bool = False
    max_concurrency: int = 1
    response_format_type: str | None = None

    def validate_for_execution(self) -> None:
        if self.provider not in {"generic", "deepseek", "minimax"}:
            raise ValueError(f"Unsupported judge provider: {self.provider}")
        if self.provider == "minimax" and not 0.0 < self.temperature <= 1.0:
            raise ValueError("MiniMax temperature must be in (0, 1]")
        if self.response_format_type not in {None, "json_object"}:
            raise ValueError("response_format_type must be None or 'json_object'")
        missing = [
            name
            for name, value in (
                ("base_url", self.base_url),
                ("model", self.model),
                ("api_key", self.api_key),
            )
            if not value.strip()
        ]
        if missing:
            raise ValueError(f"Missing judge configuration: {', '.join(missing)}")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive for execution")
        if self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive for execution")
        if self.max_output_tokens <= 0 or self.max_prompt_chars <= 0:
            raise ValueError("Output-token and prompt-character limits must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("Token prices cannot be negative")
        if not self.free_endpoint and (
            self.input_cost_per_million <= 0 or self.output_cost_per_million <= 0
        ):
            raise ValueError(
                "Paid execution requires positive input and output token prices; "
                "mark a genuinely free/local endpoint explicitly"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def completion_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def cache_key(request: Mapping[str, Any], config: JudgeConfig) -> str:
    payload = {
        "runner_version": "openai-compatible-judge-v1",
        "provider": config.provider,
        "request_id": str(request["request_id"]),
        "prompt_sha256": hashlib.sha256(str(request["prompt"]).encode("utf-8")).hexdigest(),
        "model": config.model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "response_format_type": _effective_response_format(config),
    }
    if "response_schema" in request:
        payload["response_schema"] = str(request["response_schema"])
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _effective_response_format(config: JudgeConfig) -> str | None:
    """Return the actual wire-level response format used by this runner."""
    if config.response_format_type is not None:
        return config.response_format_type
    # Preserve the existing generic/DeepSeek behavior. MiniMax was previously
    # left unconstrained because its documented compatibility surface omitted
    # response_format; formal MiniMax runs now opt in explicitly after preflight.
    return None if config.provider == "minimax" else "json_object"


def strict_parse_judge_response(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not one strict JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    expected = {"winner", "confidence", "reason"}
    if set(parsed) != expected:
        raise ValueError(f"Judge response keys must be exactly {sorted(expected)}")
    winner = parsed["winner"]
    if winner not in {"A", "B", "TIE"}:
        raise ValueError('winner must be "A", "B", or "TIE"')
    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    reason = parsed["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {"winner": winner, "confidence": float(confidence), "reason": reason.strip()}


def final_answer_content(content: str, provider: str) -> tuple[str, bool]:
    """Remove only a provider-documented reasoning envelope, never a JSON wrapper."""
    stripped = content.strip()
    if provider == "minimax" and stripped.startswith("<think>"):
        closing = stripped.rfind("</think>")
        if closing < 0:
            raise ValueError("MiniMax response opened <think> without a closing tag")
        return stripped[closing + len("</think>") :].strip(), True
    return stripped, False


def select_requests(
    requests: Iterable[Mapping[str, Any]],
    *,
    objective_ids: set[str] | None = None,
    evidence_arms: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    def objective_key(row: Mapping[str, Any]) -> str:
        return str(row.get("objective_id", row.get("concept_id", "")))

    def arm_key(row: Mapping[str, Any]) -> str:
        return str(row.get("evidence_arm", row.get("arm", "")))

    selected = [
        dict(row)
        for row in requests
        if (objective_ids is None or objective_key(row) in objective_ids)
        and (evidence_arms is None or arm_key(row) in evidence_arms)
    ]
    # Keeping every evidence arm adjacent for the same objective/pair/order makes
    # capped pilots balanced whenever the limit is a multiple of the arm count.
    selected.sort(
        key=lambda row: (
            objective_key(row),
            str(row.get("pair_id", "")),
            str(row.get("order", "")),
            arm_key(row),
            str(row["request_id"]),
        )
    )
    if limit is not None:
        selected = selected[:limit]
    return selected


def openai_compatible_transport(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=canonical_json(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Judge HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Judge connection failed: {exc.reason}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, Mapping):
        raise RuntimeError("Judge API returned a non-object response")
    return parsed


def _extract_completion(
    api_response: Mapping[str, Any],
) -> tuple[str, int | None, int | None, str | None]:
    try:
        choice = api_response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Judge API response has no choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("Judge API content must be text")
    usage = api_response.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        content,
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
        str(choice.get("finish_reason")) if choice.get("finish_reason") is not None else None,
    )


def _estimated_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _cost(config: JudgeConfig, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * config.input_cost_per_million
        + output_tokens * config.output_cost_per_million
    ) / 1_000_000.0


def _completion_payload(prompt: str, config: JudgeConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "Use only the supplied evidence. Return exactly one JSON object and no other final-answer text.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
    }
    if config.provider == "minimax":
        payload["max_completion_tokens"] = config.max_output_tokens
        payload["thinking"] = {"type": "disabled"}
        payload["reasoning_split"] = True
    else:
        payload["max_tokens"] = config.max_output_tokens
    response_format = _effective_response_format(config)
    if response_format is not None:
        payload["response_format"] = {"type": response_format}
    if config.provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    return payload


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(dict(row)))
        handle.write("\n")


def run_judge_requests(
    requests: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    config: JudgeConfig,
    transport: Transport = openai_compatible_transport,
    response_parser: ResponseParser = strict_parse_judge_response,
    stop_on_failure_error_substrings: Iterable[str] = (),
) -> dict[str, Any]:
    """Run a bounded judge batch, caching valid responses and never retrying failures."""
    config.validate_for_execution()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    responses_path = output / "judge_responses.jsonl"
    attempts_path = output / "judge_attempts.jsonl"
    existing = read_jsonl(responses_path) if responses_path.exists() else []
    cached_keys = {str(row.get("cache_key", "")) for row in existing}
    candidates = select_requests(requests, limit=config.max_requests)

    spent_usd = sum(float(row.get("estimated_cost_usd", 0.0)) for row in existing)
    reserved_usd = spent_usd
    completed = 0
    cached = 0
    failed = 0
    stopped_reason: str | None = None
    stop_patterns = tuple(
        str(value).lower() for value in stop_on_failure_error_substrings if str(value)
    )
    if stop_patterns and config.max_concurrency != 1:
        raise ValueError(
            "Failure-triggered stopping requires max_concurrency=1 so unsent requests "
            "are not scheduled speculatively"
        )
    endpoint = completion_endpoint(config.base_url)

    scheduled: list[tuple[dict[str, Any], str, int]] = []
    for request in candidates:
        key = cache_key(request, config)
        if key in cached_keys:
            cached += 1
            continue
        prompt = str(request["prompt"])
        if len(prompt) > config.max_prompt_chars:
            failed += 1
            _append_jsonl(
                attempts_path,
                {
                    "request_id": request["request_id"],
                    "cache_key": key,
                    "status": "FAILED_PREFLIGHT",
                    "error": f"prompt has {len(prompt)} chars; limit is {config.max_prompt_chars}",
                    "timestamp": _utc_now(),
                },
            )
            continue
        estimated_input = _estimated_tokens(prompt)
        reserved_cost = _cost(config, estimated_input, config.max_output_tokens)
        if reserved_usd + reserved_cost > config.max_cost_usd + 1e-12:
            stopped_reason = "cost_cap_would_be_exceeded"
            break
        reserved_usd += reserved_cost
        scheduled.append((request, key, estimated_input))

    def perform(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
        request, key, estimated_input = task
        prompt = str(request["prompt"])
        payload = _completion_payload(prompt, config)
        started = time.monotonic()
        finish_reason: str | None = None
        try:
            raw = transport(
                endpoint,
                {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                payload,
                config.timeout_seconds,
            )
            content, input_tokens, output_tokens, finish_reason = _extract_completion(raw)
            final_content, reasoning_removed = final_answer_content(content, config.provider)
            parsed = dict(response_parser(final_content))
            billed_input = input_tokens if input_tokens is not None else estimated_input
            billed_output = output_tokens if output_tokens is not None else _estimated_tokens(content)
            request_cost = _cost(config, billed_input, billed_output)
            response_row = {
                "request_id": request["request_id"],
                "cache_key": key,
                "model": config.model,
                "provider": config.provider,
                "response": parsed,
                "provider_reasoning_removed": reasoning_removed,
                "finish_reason": finish_reason,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage_estimated": input_tokens is None or output_tokens is None,
                },
                "estimated_cost_usd": request_cost,
                "timestamp": _utc_now(),
            }
            return {
                "ok": True,
                "cost": request_cost,
                "cache_key": key,
                "response": response_row,
                "attempt": {
                    "request_id": request["request_id"],
                    "cache_key": key,
                    "status": "COMPLETE",
                    "elapsed_seconds": time.monotonic() - started,
                    "timestamp": _utc_now(),
                },
            }
        except Exception as exc:  # one failed request is logged; rerun is always explicit
            return {
                "ok": False,
                "cost": 0.0,
                "cache_key": key,
                "response": None,
                "attempt": {
                    "request_id": request["request_id"],
                    "cache_key": key,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "finish_reason": finish_reason,
                    "elapsed_seconds": time.monotonic() - started,
                    "timestamp": _utc_now(),
                },
            }

    if config.max_concurrency == 1:
        results = map(perform, scheduled)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=config.max_concurrency)
        results = executor.map(perform, scheduled)
    try:
        for result in results:
            _append_jsonl(attempts_path, result["attempt"])
            if result["ok"]:
                response_row = result["response"]
                _append_jsonl(responses_path, response_row)
                cached_keys.add(str(result["cache_key"]))
                spent_usd += float(result["cost"])
                completed += 1
            else:
                failed += 1
                error = str(result["attempt"].get("error", "")).lower()
                if any(pattern in error for pattern in stop_patterns):
                    stopped_reason = "failure_stop_pattern_matched"
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary = {
        "status": "COMPLETE" if stopped_reason is None and failed == 0 else "PARTIAL",
        "runner_version": "openai-compatible-judge-v1",
        "provider": config.provider,
        "model": config.model,
        "selected_request_count": len(candidates),
        "completed_request_count": completed,
        "cached_request_count": cached,
        "failed_request_count": failed,
        "stopped_reason": stopped_reason,
        "estimated_total_cost_usd": spent_usd,
        "max_cost_usd": config.max_cost_usd,
        "max_requests": config.max_requests,
        "api_key_recorded": False,
        "automatic_retries": 0,
        "failure_stop_patterns": list(stop_patterns),
        "max_concurrency": config.max_concurrency,
        "response_format_type": _effective_response_format(config),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": _utc_now(),
            "verification_status": "UNVERIFIED",
            "version_label": "judge_run_v1",
        },
        "completed_at": _utc_now(),
    }
    write_json(output / "judge_run_summary.json", summary)
    return summary
