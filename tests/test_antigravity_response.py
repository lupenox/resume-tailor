from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.antigravity_response import (
    locate_json_tailoring_candidate,
    parse_json_output,
    parse_stream_json_output,
)
from resume_tailor.antigravity_writer import (
    resolve_tailoring_response,
    resolve_tailoring_response_text_with_envelope,
)
from resume_tailor.docx_extract import extract_resume
from resume_tailor.schemas import load_schema
from resume_tailor.utilities import (
    AntigravityResponseEnvelopeError,
    AntigravityTailoringContractError,
)


def _complete(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "message": "Applied the approved synthetic edit plan.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": content,
    }


def _analysis() -> dict[str, Any]:
    return {"recommended_edits": []}


def _stream(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            json.dumps({"event": "init", "init": {"status": "started"}}),
            json.dumps({"event": "result", "result": result}),
        ]
    )


@pytest.fixture
def synthetic_content(master_resume: Path) -> dict[str, Any]:
    extracted, _ = extract_resume(master_resume)
    return extracted["content"]


@pytest.mark.parametrize(
    ("wrapper", "expected_type"),
    [
        (lambda value: value, "direct-root"),
        (
            lambda value: {"status": "SUCCESS", "structured_output": value},
            "json-wrapper-structured_output",
        ),
        (
            lambda value: {
                "status": "SUCCESS",
                "response": json.dumps(value, ensure_ascii=False),
            },
            "json-wrapper-response",
        ),
        (
            lambda value: {
                "status": "SUCCESS",
                "result": json.dumps(value, ensure_ascii=False),
            },
            "json-wrapper-result",
        ),
    ],
)
def test_documented_json_envelopes_resolve_one_strict_candidate(
    synthetic_content: dict[str, Any],
    wrapper: Any,
    expected_type: str,
) -> None:
    response = wrapper(_complete(copy.deepcopy(synthetic_content)))
    located = locate_json_tailoring_candidate(
        response,
        expected_schema=load_schema("tailored_resume.schema.json"),
    )

    assert located.envelope_type == expected_type
    assert resolve_tailoring_response(
        response,
        approved_analysis=_analysis(),
    ) == synthetic_content


def test_wrapper_may_carry_non_candidate_human_response_text(
    synthetic_content: dict[str, Any],
) -> None:
    response = {
        "status": "SUCCESS",
        "response": "Bounded synthetic summary only.",
        "structured_output": _complete(copy.deepcopy(synthetic_content)),
    }

    located = locate_json_tailoring_candidate(response)

    assert located.envelope_type == "json-wrapper-structured_output"


def test_wrapper_accepts_identical_documented_structured_output_and_response(
    synthetic_content: dict[str, Any],
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    response = {
        "status": "SUCCESS",
        "structured_output": complete,
        "response": json.dumps(complete, ensure_ascii=False),
    }

    located = locate_json_tailoring_candidate(response)

    assert located.envelope_type == "json-wrapper-structured_output"
    assert located.payload == complete


@pytest.mark.parametrize("string_valued", [False, True])
def test_stream_terminal_extracts_one_complete_object_or_json_string(
    synthetic_content: dict[str, Any],
    string_valued: bool,
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    represented: Any = json.dumps(complete) if string_valued else complete
    events, located = parse_stream_json_output(
        _stream({"structured_output": represented}),
        expected_schema=load_schema("tailored_resume.schema.json"),
    )

    assert len(events) == 2
    assert located.envelope_type == (
        "stream-json-event-result:json-wrapper-structured_output"
    )
    assert located.payload == complete


def test_stream_equivalent_duplicate_representations_prefer_structured_output(
    synthetic_content: dict[str, Any],
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    _, located = parse_stream_json_output(
        _stream(
            {
                "structured_output": complete,
                "response": json.dumps(complete),
            }
        )
    )

    assert located.envelope_type.endswith("json-wrapper-structured_output")
    assert located.payload == complete


def test_nested_equivalent_representations_are_not_ambiguous(
    synthetic_content: dict[str, Any],
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    located = locate_json_tailoring_candidate(
        {
            "response": {
                "structured_output": complete,
                "result": json.dumps(complete),
            }
        }
    )

    assert located.envelope_type == "json-wrapper-response-structured_output"


def test_exact_print_wrapper_shape_is_rejected_without_prose_scraping(
    repository_root: Path,
) -> None:
    fixture = json.loads(
        (
            repository_root
            / "tests"
            / "fixtures"
            / "antigravity_print_unstructured_wrapper.json"
        ).read_text(encoding="utf-8")
    )
    fixture["json_schema"] = load_schema("tailored_resume.schema.json")

    with pytest.raises(
        AntigravityResponseEnvelopeError,
        match="unsupported response format",
    ) as raised:
        resolve_tailoring_response(
            fixture,
            approved_analysis=_analysis(),
        )

    assert raised.value.envelope_type == "json-wrapper-unstructured-response"


def test_observed_json_mode_shape_with_four_invalid_fragments_remains_rejected(
) -> None:
    fragment = {
        "status": "complete",
        "message": "Synthetic incomplete fragment.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": None,
    }
    response = {
        "conversation_id": "00000000-0000-4000-8000-000000000000",
        "duration_seconds": 1.0,
        "json_schema": load_schema("tailored_resume.schema.json"),
        "num_turns": 1,
        "response": "Synthetic provider prose "
        + " ".join(json.dumps(fragment) for _ in range(4)),
        "status": "SUCCESS",
        "usage": {"total_tokens": 1},
    }

    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        resolve_tailoring_response_text_with_envelope(
            json.dumps(response),
            output_format="json",
            approved_analysis=_analysis(),
        )

    assert raised.value.envelope_type == "json-wrapper-unstructured-response"


@pytest.mark.parametrize(
    ("payload", "envelope_type"),
    [
        ({"status": "SUCCESS"}, "json-wrapper-missing-structured-output"),
        (
            {
                "structured_output": {"synthetic": True},
                "result": {"synthetic": False},
            },
            "json-wrapper-multiple-structured-candidates",
        ),
        (
            {
                "status": "SUCCESS",
                "response": "```json\n{\"synthetic\": true}\n```",
            },
            "json-wrapper-unstructured-response",
        ),
        (
            {
                "status": "SUCCESS",
                "response": "Prose before {\"synthetic\": true}",
            },
            "json-wrapper-unstructured-response",
        ),
    ],
)
def test_missing_ambiguous_markdown_and_prose_candidates_are_rejected(
    payload: dict[str, Any],
    envelope_type: str,
) -> None:
    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        locate_json_tailoring_candidate(payload)

    assert raised.value.envelope_type == envelope_type


def test_schema_invalid_candidate_is_not_an_envelope_fallback() -> None:
    response = {
        "structured_output": {
            "status": "complete",
            "message": "Synthetic but incomplete.",
        }
    }

    with pytest.raises(AntigravityTailoringContractError):
        resolve_tailoring_response(
            response,
            approved_analysis=_analysis(),
        )


def test_embedded_schema_must_match_the_local_expected_schema(
    synthetic_content: dict[str, Any],
) -> None:
    response = {
        "json_schema": {"type": "object"},
        "structured_output": _complete(copy.deepcopy(synthetic_content)),
    }

    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        locate_json_tailoring_candidate(
            response,
            expected_schema=load_schema("tailored_resume.schema.json"),
        )

    assert raised.value.envelope_type == "json-wrapper-schema-mismatch"


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ('{"event":"init"}\n', "stream-json-missing-terminal-result"),
        (
            '{"event":"result","result":{}}\n'
            '{"event":"result","result":{}}\n',
            "stream-json-multiple-terminal-results",
        ),
        ('{"event":"result","result":', "stream-json-malformed-event"),
        ('{"event":"result","event":"result","result":{}}', "stream-json-malformed-event"),
        ('{"event":"result","result":{"value":NaN}}', "stream-json-malformed-event"),
    ],
)
def test_stream_missing_multiple_malformed_duplicate_and_nonfinite_are_rejected(
    text: str,
    expected_type: str,
) -> None:
    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        parse_stream_json_output(text)

    assert raised.value.envelope_type == expected_type


def test_stream_conflicting_complete_candidates_are_rejected(
    synthetic_content: dict[str, Any],
) -> None:
    first = _complete(copy.deepcopy(synthetic_content))
    second = copy.deepcopy(first)
    second["message"] = "Conflicting synthetic representation."

    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        parse_stream_json_output(
            _stream(
                {
                    "structured_output": first,
                    "response": json.dumps(second),
                }
            )
        )

    assert raised.value.envelope_type == "json-wrapper-multiple-structured-candidates"


def test_stream_unknown_wrapper_and_schema_invalid_payload_are_rejected(
    synthetic_content: dict[str, Any],
) -> None:
    with pytest.raises(AntigravityResponseEnvelopeError):
        parse_stream_json_output(_stream({"data": _complete(synthetic_content)}))

    with pytest.raises(AntigravityTailoringContractError):
        resolve_tailoring_response_text_with_envelope(
            _stream(
                {
                    "structured_output": {
                        "status": "complete",
                        "message": "Synthetic incomplete payload.",
                    }
                }
            ),
            output_format="stream-json",
            approved_analysis=_analysis(),
        )


@pytest.mark.parametrize(
    "candidate",
    [
        'Prose {"status":"complete"}',
        '```json\n{"status":"complete"}\n```',
        '{"status":"complete"}\n{"tailored_resume":{}}',
    ],
)
def test_stream_never_scrapes_prose_fences_or_fragments(candidate: str) -> None:
    with pytest.raises(AntigravityResponseEnvelopeError):
        parse_stream_json_output(_stream({"response": candidate}))


@pytest.mark.parametrize(
    "text",
    [
        '{"structured_output":{"status":"complete"},"structured_output":{}}',
        '{"structured_output":{"status":"complete","score":NaN}}',
    ],
)
def test_stored_json_mode_rejects_duplicate_keys_and_nonfinite_values(
    text: str,
) -> None:
    with pytest.raises(AntigravityResponseEnvelopeError) as raised:
        parse_json_output(text)

    assert raised.value.envelope_type == "malformed-json-output"
