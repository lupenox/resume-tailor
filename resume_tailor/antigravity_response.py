from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .utilities import AntigravityResponseEnvelopeError


_TAILORING_FIELDS = frozenset(
    {
        "status",
        "message",
        "cannot_apply",
        "technical_failure",
        "tailored_resume",
    }
)


@dataclass(frozen=True)
class AntigravityResponseCandidate:
    payload: dict[str, Any]
    envelope_type: str


def _envelope_error(
    envelope_type: str,
    detail: str = "Antigravity returned JSON in an unsupported response format.",
) -> AntigravityResponseEnvelopeError:
    return AntigravityResponseEnvelopeError(
        detail,
        envelope_type=envelope_type,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_exact_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise _envelope_error(f"json-wrapper-invalid-{field}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise _envelope_error(f"json-wrapper-unstructured-{field}") from exc
    if not isinstance(parsed, dict):
        raise _envelope_error(f"json-wrapper-invalid-{field}")
    return parsed


def locate_json_tailoring_candidate(
    payload: Any,
    *,
    expected_schema: dict[str, Any] | None = None,
) -> AntigravityResponseCandidate:
    """Locate one whole-document candidate in Antigravity JSON output.

    This deliberately does not scan prose, Markdown, or string suffixes for
    braces. A string-valued documented result field must itself be one complete
    JSON document.
    """
    if not isinstance(payload, dict):
        raise _envelope_error("json-root-not-object")

    embedded_schema = payload.get("json_schema")
    if embedded_schema is not None:
        if (
            expected_schema is None
            or not isinstance(embedded_schema, dict)
            or _canonical_json(embedded_schema) != _canonical_json(expected_schema)
        ):
            raise _envelope_error("json-wrapper-schema-mismatch")

    candidates: list[AntigravityResponseCandidate] = []
    if _TAILORING_FIELDS <= payload.keys():
        candidates.append(
            AntigravityResponseCandidate(
                payload=payload,
                envelope_type="direct-root",
            )
        )

    for field in ("structured_output", "result", "response"):
        if field not in payload:
            continue
        value = payload[field]
        if field == "response" and not isinstance(value, (dict, str)):
            continue
        try:
            candidate = _parse_exact_object(value, field=field)
        except AntigravityResponseEnvelopeError:
            if field == "response" and candidates:
                # Human-readable response text may accompany one authoritative
                # structured_output/result field. It is never parsed as a second
                # candidate.
                continue
            raise
        candidates.append(
            AntigravityResponseCandidate(
                payload=candidate,
                envelope_type=f"json-wrapper-{field}",
            )
        )

    if not candidates:
        raise _envelope_error("json-wrapper-missing-structured-output")
    if len(candidates) != 1:
        raise _envelope_error(
            "json-wrapper-multiple-structured-candidates",
            "Antigravity returned multiple ambiguous structured-output candidates.",
        )
    return candidates[0]


def parse_json_output(
    text: str,
    *,
    expected_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], AntigravityResponseCandidate]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _envelope_error(
            "malformed-json-output",
            "Antigravity returned malformed JSON output.",
        ) from exc
    if not isinstance(payload, dict):
        raise _envelope_error("json-root-not-object")
    return payload, locate_json_tailoring_candidate(
        payload,
        expected_schema=expected_schema,
    )


def parse_stream_json_output(
    text: str,
    *,
    expected_schema: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], AntigravityResponseCandidate]:
    """Parse Antigravity 1.1.8's documented typed NDJSON terminal result."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _envelope_error(
                "stream-json-malformed-event",
                "Antigravity returned a malformed stream-json event.",
            ) from exc
        if not isinstance(event, dict):
            raise _envelope_error("stream-json-event-not-object")
        events.append(event)
    terminal = [event for event in events if event.get("step_type") == "result"]
    if len(terminal) != 1:
        kind = (
            "stream-json-missing-terminal-result"
            if not terminal
            else "stream-json-multiple-terminal-results"
        )
        raise _envelope_error(kind)

    event = terminal[0]
    embedded_schema = event.get("json_schema")
    if embedded_schema is not None:
        if (
            expected_schema is None
            or not isinstance(embedded_schema, dict)
            or _canonical_json(embedded_schema) != _canonical_json(expected_schema)
        ):
            raise _envelope_error("stream-json-schema-mismatch")

    fields = [
        field for field in ("structured_output", "result") if field in event
    ]
    if len(fields) != 1:
        kind = (
            "stream-json-missing-result-candidate"
            if not fields
            else "stream-json-multiple-result-candidates"
        )
        raise _envelope_error(kind)
    field = fields[0]
    candidate = _parse_exact_object(event[field], field=field)
    return events, AntigravityResponseCandidate(
        payload=candidate,
        envelope_type=f"stream-json-terminal-{field}",
    )
