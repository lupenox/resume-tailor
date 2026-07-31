from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.antigravity_response import (
    locate_json_tailoring_candidate,
    parse_stream_json_output,
)
from resume_tailor.antigravity_writer import resolve_tailoring_response
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


def test_stream_json_accepts_current_documented_terminal_result(
    synthetic_content: dict[str, Any],
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    schema = load_schema("tailored_resume.schema.json")
    stream = "\n".join(
        (
            json.dumps({"event": "init", "init": {"conversation_id": "synthetic"}}),
            json.dumps(
                {"event": "step_update", "step_update": {"status": "running"}}
            ),
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "status": "SUCCESS",
                        "structured_output": complete,
                        "response": json.dumps(complete, ensure_ascii=False),
                        "json_schema": schema,
                    },
                },
                ensure_ascii=False,
            ),
        )
    )

    events, candidate = parse_stream_json_output(
        stream,
        expected_schema=load_schema("tailored_resume.schema.json"),
    )

    assert len(events) == 3
    assert (
        candidate.envelope_type
        == "stream-json-event-result:json-wrapper-structured_output"
    )
    assert resolve_tailoring_response(
        candidate.payload,
        approved_analysis=_analysis(),
    ) == synthetic_content


def test_stream_json_retains_legacy_terminal_compatibility(
    synthetic_content: dict[str, Any],
) -> None:
    complete = _complete(copy.deepcopy(synthetic_content))
    stream = json.dumps(
        {
            "step_type": "result",
            "result": complete,
            "json_schema": load_schema("tailored_resume.schema.json"),
        },
        ensure_ascii=False,
    )

    events, candidate = parse_stream_json_output(
        stream,
        expected_schema=load_schema("tailored_resume.schema.json"),
    )

    assert len(events) == 1
    assert (
        candidate.envelope_type
        == "stream-json-legacy-step-result:json-wrapper-result"
    )


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
