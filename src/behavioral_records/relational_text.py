"""One fixed, reversible text rendering of the unchanged relational record.

This layer receives only the public compiler output. It adds no policy label,
requirement, causal explanation, filtering, or numerical approximation.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .canonical import canonical_json
from .relational_record import validate_relational_record_contract


HEADER = "Behavioral observations. Same-step observations do not establish causation."
TOP_FIELDS = ("representation", "provenance", "completeness", "coverage", "semantics", "loss_ledger")
LABELS = ("Format", "Source", "Completeness", "Coverage", "Observation semantics", "Retained and omitted information")


def render_relational_text(record: Mapping[str, Any]) -> str:
    """Render all facts and bindings; the signature deliberately has no objective."""
    errors = validate_relational_record_contract(record)
    if errors:
        raise ValueError(f"Invalid relational record: {errors[:5]}")
    lines = [HEADER]
    lines.extend(f"{label}: {canonical_json(record[field])}" for field, label in zip(TOP_FIELDS, LABELS))
    for index, pattern in enumerate(record["patterns"], start=1):
        lines.append(f"Episode pattern {index}.")
        lines.append("Context: " + canonical_json(pattern["context"]))
        lines.append("Occurrences: " + canonical_json(pattern["multiplicity"]))
        behavior = pattern["behavior"]
        lines.append("Episode observations: " + canonical_json({k: v for k, v in behavior.items() if k != "relations"}))
        for relation in behavior["relations"]:
            observation = relation["observation"]
            state, predicate = observation["state_relation"], observation["predicate_relation"]
            lines.append(
                "At recorded time " + canonical_json(relation["t"])
                + ", action " + canonical_json(relation["action"])
                + "; observed reward " + canonical_json(observation["reward"])
                + "; events " + canonical_json(observation["events"])
                + "; state " + state["mode"] + " " + canonical_json(state["values"])
                + "; predicates " + predicate["mode"] + " " + canonical_json(predicate["values"])
                + "."
            )
        lines.append("End episode pattern.")
    lines.append("State/predicate delta pairs give the recorded before and after values, in that order.")
    return "\n".join(lines)


def parse_relational_text(text: str) -> dict[str, Any]:
    """Strict inverse used for factual audits, never to generate model choices."""
    lines = text.splitlines()
    if not lines or lines.pop(0) != HEADER:
        raise ValueError("Unknown text header")
    record: dict[str, Any] = {}
    for field, label in zip(TOP_FIELDS, LABELS):
        prefix = label + ": "
        line = lines.pop(0)
        if not line.startswith(prefix):
            raise ValueError("Missing metadata field")
        record[field] = json.loads(line[len(prefix):])
    record["patterns"] = []
    decoder = json.JSONDecoder()
    while lines and lines[0].startswith("Episode pattern "):
        number = len(record["patterns"]) + 1
        if lines.pop(0) != f"Episode pattern {number}.":
            raise ValueError("Episode sequence mismatch")
        pattern = {}
        for key, label in (("context", "Context"), ("multiplicity", "Occurrences"), ("behavior", "Episode observations")):
            line = lines.pop(0)
            prefix = label + ": "
            if not line.startswith(prefix):
                raise ValueError("Missing episode field")
            pattern[key] = json.loads(line[len(prefix):])
        pattern["behavior"]["relations"] = []
        while lines and lines[0] != "End episode pattern.":
            line = lines.pop(0)
            offset = 0
            def read_after(prefix):
                nonlocal offset
                if not line.startswith(prefix, offset):
                    raise ValueError(f"Invalid relation separator: {prefix}")
                offset += len(prefix)
                value, offset = decoder.raw_decode(line, offset)
                return value
            t = read_after("At recorded time ")
            action = read_after(", action ")
            reward = read_after("; observed reward ")
            events = read_after("; events ")
            state_mode = "snapshot" if line.startswith("; state snapshot ", offset) else "delta"
            state_values = read_after(f"; state {state_mode} ")
            predicate_mode = "snapshot" if line.startswith("; predicates snapshot ", offset) else "delta"
            predicate_values = read_after(f"; predicates {predicate_mode} ")
            if line[offset:] != ".":
                raise ValueError("Unrecognized relation suffix")
            pattern["behavior"]["relations"].append({"t": t, "action": action, "observation": {
                "reward": reward, "events": events,
                "state_relation": {"mode": state_mode, "values": state_values},
                "predicate_relation": {"mode": predicate_mode, "values": predicate_values},
            }})
        if not lines or lines.pop(0) != "End episode pattern.":
            raise ValueError("Unterminated episode")
        record["patterns"].append(pattern)
    if lines != ["State/predicate delta pairs give the recorded before and after values, in that order."]:
        raise ValueError("Unrecognized trailing content")
    errors = validate_relational_record_contract(record)
    if errors:
        raise ValueError(f"Reconstructed record is invalid: {errors[:5]}")
    return record


def audit_text_rendering(record: Mapping[str, Any]) -> dict[str, Any]:
    rendered = render_relational_text(record)
    return {
        "round_trip_exact": canonical_json(parse_relational_text(rendered)) == canonical_json(record),
        "deterministic": rendered == render_relational_text(record),
        "source_characters": len(canonical_json(record)),
        "rendered_characters": len(rendered),
        "source_contains_all_information": True,
        "objective_or_reference_input": False,
    }
