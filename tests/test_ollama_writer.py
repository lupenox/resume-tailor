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


def _complete(analysis: dict | None = None, patches: list[dict] | None = None) -> dict:
    if analysis is not None:
        catalog = writer.approved_edit_catalog(analysis)
        catalog_sha256 = writer.canonical_digest(catalog)
        if patches is None:
            patches = [
                {
                    "edit_id": edit["edit_id"],
                    "target_source_id": edit["target_source_id"],
                    "operation": edit.get("operation", "replace"),
                    "replacement_text": edit.get("proposed_text", "Updated text"),
                }
                for edit in catalog
            ]
        return {
            "status": "complete",
            "message": "Applied the approved synthetic plan.",
            "catalog_sha256": catalog_sha256,
            "cannot_apply": None,
            "technical_failure": None,
            "patches": patches,
        }
    return {
        "status": "complete",
        "message": "Applied the approved synthetic plan.",
        "catalog_sha256": "0" * 64,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": patches or [],
    }


def test_ollama_prompt_uses_approved_plan_not_raw_job_description(
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
    assert "Author target-only edits" in prompt
    assert "NO UNSUPPORTED CLAIMS" in prompt
    assert "CATALOG SHA256 DIGEST" in prompt


def test_gemma_invocation_uses_schema_mode_and_canonical_validation(
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
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
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
    assert request["model"] == "resume-tailor-gemma"
    assert request["stream"] is False
    assert request["think"] is False
    # The context window is an explicit declared capability, not a hardcoded value.
    capabilities = capabilities_for_model("resume-tailor-gemma")
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
    assert envelope["provider"] == "gemma"
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


def test_ollama_invalid_json_is_rejected_and_recorded(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma:latest",
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


def test_ollama_preflight_failure_is_provider_neutral_and_makes_no_request(
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
    """Regression for the preserved historical Step 6 Qwen failure.

    The prior Qwen writer returned complete, parseable JSON and stopped
    naturally, but the root was a bare résumé shape with none of the required
    envelope fields. That response is preserved as a historical fixture and must
    continue to be classified as a transport-schema rejection, not generic
    malformed JSON, not truncation, and not a canonical-schema failure.
    This test remains valid regardless of which model is currently active.
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
    for field in ["status", "cannot_apply", "technical_failure"]:
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
    """The derived transport schema alone must reject the observed historical root."""
    fixture = _fixture_body()
    schema = writer._ollama_transport_schema("ollama_tailoring_patch.schema.json")
    wrong_root = json.loads(fixture["chat_body"]["message"]["content"])
    assert sorted(wrong_root) == sorted(OBSERVED_WRONG_ROOT_KEYS)
    with pytest.raises(OllamaTransportSchemaError):
        writer._validate_transport_payload(
            wrong_root,
            transport_schema=schema,
            label="local writer structured output",
        )


def test_envelope_shaped_output_failing_canonical_rules_is_a_canonical_failure(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correct envelope with a canonical violation is classified separately."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    # status=complete with a null patches satisfies the transport schema
    # (its cross-field allOf is stripped) but violates the canonical contract.
    payload = _complete(analysis)
    payload["patches"] = None
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma:latest",
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
            "model": "resume-tailor-gemma:latest",
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
        lambda **kwargs: {"model": "resume-tailor-gemma:latest", "done": False},
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
                "context_window": 1024,
                "max_output_tokens": 1024,
                "min_output_tokens": 1024,
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
    # Active default: Gemma
    tagged_gemma = capabilities_for_model("resume-tailor-gemma:latest")
    assert tagged_gemma.context_window == 32_768
    assert tagged_gemma.supports_json_schema is True
    # Historical Qwen entry still resolves (preserved for regression fixture).
    tagged_qwen = capabilities_for_model("resume-tailor-qwen:latest")
    assert tagged_qwen.context_window == 32_768
    assert capabilities_for_model("some-unknown-model").supports_json_schema is True
    overridden = capabilities_for_model(
        "resume-tailor-gemma",
        overrides={"context_window": 16_384},
    )
    assert overridden.context_window == 16_384
    assert overridden.max_output_tokens == 8_192


def test_structured_output_probe_covers_required_constructs_offline() -> None:
    """The probe must assert $ref, oneOf, additionalProperties, required roots."""
    schema = writer._ollama_transport_schema("ollama_tailoring_patch.schema.json")
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


# ---------------------------------------------------------------------------
# Focused Gemma 4 12B regression tests
# ---------------------------------------------------------------------------


def test_default_model_is_resume_tailor_gemma() -> None:
    """DEFAULT_OLLAMA_MODEL must be resume-tailor-gemma after the migration."""
    assert writer.DEFAULT_OLLAMA_MODEL == "resume-tailor-gemma"


def test_gemma_capability_lookup_resolves_exact_name() -> None:
    """Exact name lookup must return declared Gemma capabilities."""
    from resume_tailor.ollama_capabilities import MODEL_CAPABILITIES

    assert "resume-tailor-gemma" in MODEL_CAPABILITIES
    caps = MODEL_CAPABILITIES["resume-tailor-gemma"]
    assert caps.context_window == 32_768
    assert caps.max_output_tokens == 8_192
    assert caps.min_output_tokens == 2_048
    assert caps.supports_json_schema is True


def test_gemma_capability_lookup_resolves_tagged_name() -> None:
    """Tag-free lookup must resolve resume-tailor-gemma:latest."""
    caps = capabilities_for_model("resume-tailor-gemma:latest")
    assert caps.context_window == 32_768
    assert caps.supports_json_schema is True


def test_gemma_request_body_uses_expected_context_and_output_budgets(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request body must use the Gemma capability window and a fitting budget."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
            },
            "eval_count": 50,
        }

    monkeypatch.setattr(writer, "run_ollama_request", capture)
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
    req = observed["body"]
    assert isinstance(req, dict)
    caps = capabilities_for_model("resume-tailor-gemma")
    assert req["options"]["num_ctx"] == caps.context_window
    assert req["options"]["num_predict"] <= caps.max_output_tokens
    assert req["options"]["num_predict"] >= caps.min_output_tokens


def test_gemma_metadata_does_not_report_qwen_as_provider(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response envelope must name gemma, not qwen, as the provider."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
            },
            "eval_count": 50,
        },
    )
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
    assert envelope["provider"] == "gemma"
    assert envelope["provider"] != "qwen"
    assert envelope["runtime"] == "ollama"
    assert envelope["model"] == "resume-tailor-gemma"


def test_gemma_request_retains_structured_output_schema(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The format field must carry a JSON Schema object, not be absent or null."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
            },
            "eval_count": 50,
        }

    monkeypatch.setattr(writer, "run_ollama_request", capture)
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
    req = observed["body"]
    assert isinstance(req, dict)
    schema = req["format"]
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    # allOf cross-field assertions are stripped for the transport schema only.
    assert "allOf" not in schema
    assert schema.get("additionalProperties") is False


def test_no_automatic_qwen_fallback_when_gemma_is_requested(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoking with resume-tailor-gemma must never silently fall back to Qwen."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    observed_models: list[str] = []

    def capture(**kwargs: object) -> dict[str, object]:
        body = kwargs.get("body", {})
        assert isinstance(body, dict)
        observed_models.append(body.get("model", ""))
        return {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
            },
            "eval_count": 50,
        }

    monkeypatch.setattr(writer, "run_ollama_request", capture)
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
    # Exactly one request, using the Gemma profile — never the Qwen profile.
    assert len(observed_models) == 1
    assert observed_models[0] == "resume-tailor-gemma"
    assert "qwen" not in observed_models[0].lower()


def test_model_override_still_works_with_gemma_default(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing an explicit model kwarg must override the default."""
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "model": "my-custom-model:latest",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(_complete(analysis)),
            },
            "eval_count": 50,
        }

    monkeypatch.setattr(writer, "run_ollama_request", capture)
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
        model="my-custom-model",
    )
    req = observed["body"]
    assert isinstance(req, dict)
    assert req["model"] == "my-custom-model"


def test_unknown_ollama_model_uses_conservative_defaults() -> None:
    """An unregistered model must fall back to the conservative capability defaults."""
    caps = capabilities_for_model("some-never-seen-model:v99")
    assert caps.context_window == 32_768
    assert caps.max_output_tokens == 8_192
    assert caps.min_output_tokens == 2_048
    assert caps.supports_json_schema is True


def test_historical_wrong_root_fixture_still_classifies_as_transport_schema(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserved Qwen fixture must still classify as ollama-transport-schema.

    This is a direct duplicate of the primary regression test expressed as an
    explicit Gemma-era named test so future readers understand the fixture
    remains intentionally Qwen-labelled as a historical record.
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
    envelope = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert envelope["validation_path"] == "transport_schema"
    assert envelope["validation_result"] == "REJECTED"
    assert caught.value is not None
