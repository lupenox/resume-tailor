from __future__ import annotations

import json
from pathlib import Path

import pytest

import resume_tailor.ollama_writer as writer
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.ollama_capabilities import (
    OllamaModelCapabilities,
    capabilities_for_model,
    estimate_tokens,
    plan_ollama_budget,
)
from resume_tailor.ollama_probe import (
    OBSERVED_WRONG_ROOT_KEYS,
    REQUIRED_ROOT_FIELDS,
    probe_structured_output_support,
)
from resume_tailor.utilities import (
    OllamaBudgetError,
    OllamaCanonicalSchemaError,
    OllamaMalformedJSONError,
    OllamaOutputTruncationError,
    OllamaResponseEnvelopeError,
    OllamaTailoringContractError,
    OllamaTransportSchemaError,
    TailoringPreflightError,
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
WRONG_ROOT_FIXTURE = FIXTURE_DIRECTORY / "ollama_wrong_root_resume_response.json"


def _resolved_analysis(extracted: dict, requirements: dict) -> dict:
    analysis = {
        "role_summary": "Synthetic role",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": [],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": [],
        "missing_or_unsupported_requirements": [],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": ["Unsupported synthetic claim"],
        "content_budget_guidance": [],
        "questions_for_user": [],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in requirements["requirements"]
        ],
    }
    resolved, issues = resolve_analysis_evidence(
        analysis,
        extracted,
        requirements,
    )
    assert issues == []
    return resolved


def _inputs(master_resume: Path) -> tuple[dict, str, dict, dict]:
    extracted, _ = extract_resume(master_resume)
    private_job = (
        "SYNTHETIC_PRIVATE_JOB_MARKER. Build Python validation workflows with "
        "structured outputs and tests."
    )
    requirements = build_job_requirement_catalog(private_job)
    analysis = _resolved_analysis(extracted, requirements)
    return extracted, private_job, requirements, analysis


def _complete(content: dict) -> dict:
    return {
        "status": "complete",
        "message": "Applied the approved synthetic plan.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": content,
    }


def test_qwen_prompt_uses_approved_plan_not_raw_job_description(
    master_resume: Path,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    prompt = writer.build_ollama_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=private_job,
        job_requirements=requirements,
        approved_analysis=analysis,
        company="Synthetic Systems",
        role="Validation Engineer",
    )

    assert "SYNTHETIC_PRIVATE_JOB_MARKER" not in prompt
    assert "Return exactly one" in prompt
    assert "WHAT supported skill" in prompt
    assert "Never add an unsupported" in prompt
    assert "first half of the resume" in prompt


def test_qwen_invocation_uses_schema_mode_and_canonical_validation(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    observed: dict[str, object] = {}

    def fake_request(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        request = kwargs["body"]
        assert isinstance(request, dict)
        return {
            "model": "resume-tailor-qwen:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(extracted["content"])),
            },
            "eval_count": 100,
        }

    monkeypatch.setattr(writer, "run_ollama_request", fake_request)
    tailored = writer.invoke_ollama(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=private_job,
        job_requirements=requirements,
        approved_analysis=analysis,
        company="Synthetic Systems",
        role="Validation Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )

    assert tailored == extracted["content"]
    request = observed["body"]
    assert isinstance(request, dict)
    assert request["model"] == "resume-tailor-qwen"
    assert request["stream"] is False
    assert request["think"] is False
    # The context window is now an explicit declared capability rather than a
    # hardcoded 8192, and the output budget must fit inside it.
    capabilities = capabilities_for_model("resume-tailor-qwen")
    assert request["options"]["num_ctx"] == capabilities.context_window
    assert request["options"]["num_predict"] <= capabilities.max_output_tokens
    assert (
        request["options"]["num_predict"]
        + estimate_tokens(request["messages"][1]["content"])
        < capabilities.context_window
    )
    assert request["format"]["type"] == "object"
    assert "allOf" not in request["format"]
    assert "SYNTHETIC_PRIVATE_JOB_MARKER" not in request["messages"][1]["content"]
    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["provider"] == "qwen"
    assert envelope["runtime"] == "ollama"
    assert envelope["local_only"] is True
    assert envelope["validation_result"] == "PASS"
    assert envelope["validation_path"] == "pass"
    assert envelope["prompt"]["content_logged"] is False
    assert envelope["capabilities"]["context_window"] == capabilities.context_window
    assert envelope["budget"]["requested_output_tokens"] >= 1
    assert envelope["structured_output_probe"]["supported"] is True
    assert envelope["structured_output_probe"]["provider_called"] is False
    assert envelope["generation"]["truncated"] is False


def test_qwen_invalid_json_is_rejected_and_recorded(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-qwen:latest",
            "done": True,
            "message": {"role": "assistant", "content": "not-json"},
        },
    )

    with pytest.raises(OllamaMalformedJSONError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["validation_result"] == "REJECTED"
    assert metadata["validation_path"] == "malformed_json"


def test_qwen_preflight_failure_is_provider_neutral_and_makes_no_request(
    master_resume: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: pytest.fail("Ollama ran before local preflight passed"),
    )

    with pytest.raises(TailoringPreflightError, match="No writer request"):
        writer.build_ollama_tailoring_prompt(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description="",
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
        )


def _fixture_body() -> dict:
    payload = json.loads(WRONG_ROOT_FIXTURE.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["contains_real_resume_content"] is False
    return payload


def test_preserved_wrong_root_response_is_classified_as_transport_schema_failure(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the preserved Step 6 failure.

    Qwen returned complete, parseable JSON and stopped naturally, but the root
    was a bare résumé shape with none of the required envelope fields. That must
    be classified as a transport-schema rejection, not generic malformed JSON,
    not truncation, and not a canonical-schema failure.
    """
    fixture = _fixture_body()
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: json.loads(json.dumps(fixture["chat_body"])),
    )

    with pytest.raises(OllamaTransportSchemaError) as caught:
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )

    # The specific class must remain a contract error for existing handlers.
    assert isinstance(caught.value, OllamaTailoringContractError)
    message = str(caught.value)
    for field in fixture["expected_classification"]["missing_required_root_fields"]:
        assert field in message
    # No résumé-derived value may appear in the sanitized error message.
    assert "Synthetic Candidate" not in message
    assert "synthetic@example.invalid" not in message

    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["validation_result"] == "REJECTED"
    assert (
        envelope["validation_path"]
        == fixture["expected_classification"]["validation_path"]
    )
    assert envelope["content_logged"] is False
    # Truncation must not be blamed: generation stopped naturally under the cap.
    assert envelope["generation"]["done_reason"] == "stop"
    assert envelope["generation"]["output_ceiling_reached"] is False
    assert envelope["generation"]["reason_indicates_length"] is False
    # The context-window overflow is still recorded as the contributing signal.
    assert envelope["generation"]["reported_total_tokens"] == 8735
    assert envelope["validation_message"]


def test_wrong_root_shape_is_rejected_by_the_derived_transport_schema() -> None:
    """The derived transport schema alone must reject the observed root."""
    fixture = _fixture_body()
    schema = writer._ollama_transport_schema("tailored_resume.schema.json")
    wrong_root = json.loads(fixture["chat_body"]["message"]["content"])
    assert sorted(wrong_root) == sorted(OBSERVED_WRONG_ROOT_KEYS)
    with pytest.raises(OllamaTransportSchemaError):
        writer._validate_transport_payload(
            wrong_root,
            transport_schema=schema,
            label="Qwen structured output",
        )


def test_envelope_shaped_output_failing_canonical_rules_is_a_canonical_failure(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correct envelope with a canonical violation is classified separately."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    # status=complete with a null tailored_resume satisfies the transport schema
    # (its cross-field allOf is stripped) but violates the canonical contract.
    payload = _complete(extracted["content"])
    payload["tailored_resume"] = None
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-qwen:latest",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(payload)},
            "prompt_eval_count": 100,
            "eval_count": 50,
        },
    )

    with pytest.raises(OllamaCanonicalSchemaError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["validation_path"] == "canonical_schema"


def test_truncated_output_is_classified_as_truncation_not_malformed_json(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-qwen:latest",
            "done": True,
            "done_reason": "length",
            "message": {"role": "assistant", "content": '{"status":"comp'},
            "prompt_eval_count": 100,
            "eval_count": 4096,
        },
    )

    with pytest.raises(OllamaOutputTruncationError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["validation_path"] == "output_truncation"
    assert envelope["generation"]["truncated"] is True


def test_incomplete_stream_envelope_is_classified_as_response_envelope(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {"model": "resume-tailor-qwen:latest", "done": False},
    )

    with pytest.raises(OllamaResponseEnvelopeError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["validation_path"] == "response_envelope"


def test_oversized_prompt_budget_refuses_before_any_request(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved overflow condition must now fail closed before launch."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: pytest.fail("Ollama ran despite an impossible budget"),
    )

    with pytest.raises(OllamaBudgetError, match="context window"):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            capability_overrides={
                "context_window": 4096,
                "max_output_tokens": 2048,
                "min_output_tokens": 2048,
            },
        )


def test_budget_reserves_output_room_inside_the_declared_window() -> None:
    capabilities = OllamaModelCapabilities(
        context_window=32_768,
        max_output_tokens=8_192,
        min_output_tokens=2_048,
    )
    budget = plan_ollama_budget(prompt="a" * 30_000, capabilities=capabilities)
    assert budget.requested_output_tokens >= capabilities.min_output_tokens
    assert (
        budget.prompt_tokens_estimated
        + budget.overhead_tokens
        + budget.requested_output_tokens
        <= capabilities.context_window
    )
    assert budget.sanitized()["estimated"] is True


def test_declared_capabilities_reject_an_output_ceiling_above_the_window() -> None:
    with pytest.raises(OllamaBudgetError):
        OllamaModelCapabilities(context_window=8_192, max_output_tokens=8_192)


def test_model_capabilities_resolve_by_tagged_and_unknown_name() -> None:
    tagged = capabilities_for_model("resume-tailor-qwen:latest")
    assert tagged.context_window == 32_768
    assert capabilities_for_model("some-unknown-model").supports_json_schema is True
    overridden = capabilities_for_model(
        "resume-tailor-qwen",
        overrides={"context_window": 16_384},
    )
    assert overridden.context_window == 16_384
    assert overridden.max_output_tokens == 8_192


def test_structured_output_probe_covers_required_constructs_offline() -> None:
    """The probe must assert $ref, oneOf, additionalProperties, required roots."""
    schema = writer._ollama_transport_schema("tailored_resume.schema.json")
    result = probe_structured_output_support(schema)
    assert result["provider_called"] is False
    assert result["supported"] is True
    for check in (
        "required_root_fields_declared",
        "required_root_fields_enforced",
        "additional_properties_false",
        "additional_properties_enforced",
        "ref_declared",
        "ref_resolves",
        "oneof_present",
        "status_enum_enforced",
    ):
        assert result["checks"][check] is True, check
    assert set(REQUIRED_ROOT_FIELDS) <= set(schema["required"])


def test_structured_output_probe_detects_a_weakened_schema() -> None:
    """A schema that stopped enforcing the envelope must fail the probe."""
    schema = writer._ollama_transport_schema("tailored_resume.schema.json")
    weakened = json.loads(json.dumps(schema))
    weakened["required"] = ["status"]
    weakened["additionalProperties"] = True
    result = probe_structured_output_support(weakened)
    assert result["supported"] is False
    assert result["checks"]["required_root_fields_declared"] is False
    assert result["checks"]["additional_properties_false"] is False
