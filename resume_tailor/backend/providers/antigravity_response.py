from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from resume_tailor.backend.utils.utilities import AntigravityResponseEnvelopeError


_TAILORING_FIELDS = frozenset(
    {
        "status",
        "message",
        "cannot_apply",
        "technical_failure",
        "tailored_resume",
    }
)
_STRUCTURED_FIELDS = ("structured_output", "result", "response")


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
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise _envelope_error("json-candidate-not-canonical") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_nonfinite(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_json_loads(
    text: str,
    *,
    envelope_type: str,
    detail: str = "Antigravity returned malformed JSON output.",
) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _envelope_error(envelope_type, detail) from exc


def _parse_exact_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise _envelope_error(f"json-wrapper-invalid-{field}")
    parsed = _strict_json_loads(
        value,
        envelope_type=f"json-wrapper-unstructured-{field}",
        detail="Antigravity returned JSON in an unsupported response format.",
    )
    if not isinstance(parsed, dict):
        raise _envelope_error(f"json-wrapper-invalid-{field}")
    return parsed


def locate_json_candidate(
    payload: Any,
    *,
    required_fields: frozenset[str],
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
    if required_fields <= payload.keys():
        candidates.append(
            AntigravityResponseCandidate(
                payload=payload,
                envelope_type="direct-root",
            )
        )

    for field in _STRUCTURED_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if field == "response" and not isinstance(value, (dict, str)):
            continue
        envelope_type = f"json-wrapper-{field}"
        if (
            field == "response"
            and isinstance(value, dict)
            and not required_fields <= value.keys()
        ):
            nested_fields = [
                nested
                for nested in ("structured_output", "result")
                if nested in value
            ]
            if nested_fields:
                for nested in nested_fields:
                    candidates.append(
                        AntigravityResponseCandidate(
                            payload=_parse_exact_object(
                                value[nested],
                                field=f"response-{nested}",
                            ),
                            envelope_type=f"json-wrapper-response-{nested}",
                        )
                    )
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
                envelope_type=envelope_type,
            )
        )

    if not candidates:
        raise _envelope_error("json-wrapper-missing-structured-output")
    if len(candidates) != 1:
        canonical_payloads = {
            _canonical_json(candidate.payload) for candidate in candidates
        }
        if len(canonical_payloads) != 1:
            raise _envelope_error(
                "json-wrapper-multiple-structured-candidates",
                "Antigravity returned multiple ambiguous structured-output candidates.",
            )
        preference = {
            "json-wrapper-structured_output": 0,
            "json-wrapper-response-structured_output": 0,
            "json-wrapper-result": 1,
            "json-wrapper-response-result": 1,
            "json-wrapper-response": 2,
            "direct-root": 3,
        }
        return min(
            candidates,
            key=lambda candidate: preference.get(candidate.envelope_type, 10),
        )
    return candidates[0]


def locate_json_tailoring_candidate(
    payload: Any,
    *,
    expected_schema: dict[str, Any] | None = None,
) -> AntigravityResponseCandidate:
    return locate_json_candidate(
        payload,
        required_fields=_TAILORING_FIELDS,
        expected_schema=expected_schema,
    )


def parse_json_output(
    text: str,
    *,
    expected_schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], AntigravityResponseCandidate]:
    payload = _strict_json_loads(
        text,
        envelope_type="malformed-json-output",
    )
    if not isinstance(payload, dict):
        raise _envelope_error("json-root-not-object")
    return payload, locate_json_tailoring_candidate(
        payload,
        expected_schema=expected_schema,
    )


def parse_stream_json_events(text: str) -> list[dict[str, Any]]:
    """Decode complete Antigravity stream-json records without joining fragments."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        event = _strict_json_loads(
            line,
            envelope_type="stream-json-malformed-event",
            detail="Antigravity returned malformed JSON in a stream-json event.",
        )
        if not isinstance(event, dict):
            raise _envelope_error("stream-json-event-not-object")
        events.append(event)
    return events


def locate_stream_json_terminal(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Select exactly one documented or retained legacy terminal result."""
    current_terminal = [event for event in events if event.get("event") == "result"]
    legacy_terminal = [
        event
        for event in events
        if event.get("step_type") == "result" and event.get("event") is None
    ]
    terminal = current_terminal + legacy_terminal
    if len(terminal) != 1:
        kind = (
            "stream-json-missing-terminal-result"
            if not terminal
            else "stream-json-multiple-terminal-results"
        )
        raise _envelope_error(kind)

    event = terminal[0]
    if event.get("event") == "result":
        envelope = event.get("result")
        if not isinstance(envelope, dict):
            raise _envelope_error("stream-json-terminal-result-not-object")
        return envelope, "stream-json-event-result"
    return event, "stream-json-legacy-step-result"


def parse_stream_json_envelope(
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    events = parse_stream_json_events(text)
    envelope, stream_type = locate_stream_json_terminal(events)
    return events, envelope, stream_type


def parse_stream_json_output(
    text: str,
    *,
    expected_schema: dict[str, Any] | None = None,
    required_fields: frozenset[str] = _TAILORING_FIELDS,
) -> tuple[list[dict[str, Any]], AntigravityResponseCandidate]:
    """Parse one complete tailoring payload from one terminal stream event."""
    events, envelope, stream_type = parse_stream_json_envelope(text)
    candidate = locate_json_candidate(
        envelope,
        required_fields=required_fields,
        expected_schema=expected_schema,
    )
    return events, AntigravityResponseCandidate(
        payload=candidate.payload,
        envelope_type=f"{stream_type}:{candidate.envelope_type}",
    )
