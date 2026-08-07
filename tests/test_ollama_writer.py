from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import resume_tailor.backend.providers.ollama_writer as writer
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.providers.ollama_capabilities import (
    OllamaModelCapabilities,
    capabilities_for_model,
    estimate_tokens,
    plan_ollama_budget,
)
from resume_tailor.backend.providers.ollama_probe import (
    OBSERVED_WRONG_ROOT_KEYS,
    REQUIRED_ROOT_FIELDS,
    probe_structured_output_support,
)
from resume_tailor.backend.utils.utilities import (
    OllamaBudgetError,
    OllamaCanonicalSchemaError,
    OllamaCannotApplyError,
    OllamaMalformedJSONError,
    OllamaOutputTruncationError,
    OllamaResponseEnvelopeError,
    OllamaTailoringContractError,
    OllamaTechnicalFailureError,
    OllamaTransportSchemaError,
    TailoringPreflightError,
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
WRONG_ROOT_FIXTURE = FIXTURE_DIRECTORY / "ollama_wrong_root_resume_response.json"


def _resolved_analysis(
    extracted: dict,
    requirements: dict,
    *,
    recommended_edits: list[dict] | None = None,
) -> dict:
    analysis = {
        "role_summary": "Synthetic role",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": [],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": [],
        "missing_or_unsupported_requirements": [],
        "recommended_edits": recommended_edits
        if recommended_edits is not None
        else [
            {
                "target_source_id": "professional_summary",
                "proposed_text": "Updated text",
                "alignment_rationale": "Test rationale",
                "evidence_source_ids": ["professional_summary"],
                "operation": "replace",
            }
        ],
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


def _deterministic_edits(extracted: dict) -> list[dict]:
    group = extracted["content"]["skill_groups"][0]
    items = [item.strip() for item in group["text"].split(",")]
    reordered = ", ".join([*items[1:], items[0]])
    return [
        {
            "target_source_id": "skill_groups.0",
            "proposed_text": f"{group['label']}: {reordered}",
            "alignment_rationale": "Reorder authenticated synthetic skills.",
            "evidence_source_ids": ["skill_groups.0"],
            "operation": "replace",
        }
    ]


def _mixed_analysis(extracted: dict, requirements: dict) -> dict:
    return _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=[
            *_deterministic_edits(extracted),
            {
                "target_source_id": "professional_summary",
                "proposed_text": "Updated text",
                "alignment_rationale": "Test rationale",
                "evidence_source_ids": ["professional_summary"],
                "operation": "replace",
            },
        ],
    )


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


def test_ollama_prompt_and_validator_share_canonical_replacement_limit(
    master_resume: Path,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=[
            {
                "target_source_id": "professional_summary",
                "proposed_text": "Cafe\u0301\r\n  “quoted” value",
                "alignment_rationale": "Synthetic canonical prompt test.",
                "evidence_source_ids": ["professional_summary"],
                "operation": "replace",
            }
        ],
    )
    prompt = writer.build_ollama_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=private_job,
        job_requirements=requirements,
        approved_analysis=analysis,
        company="Synthetic Systems",
        role="Validation Engineer",
    )
    descriptor_json = prompt.split(
        "APPROVED EDIT CATALOG & TARGET DESCRIPTORS\n",
        1,
    )[1].split("\n\nAUTHORIZED SOURCE EVIDENCE", 1)[0]
    descriptor = json.loads(descriptor_json)[0]
    hard_limit = next(
        paragraph["content_budget"]["maximum_characters"]
        for paragraph in extracted["paragraphs"]
        if paragraph["content_id"] == "professional_summary"
    )

    assert descriptor["mutable_proposed_body"] == "Café\n  “quoted” value"
    assert descriptor["proposed_replacement_characters"] == len(
        descriptor["mutable_proposed_body"]
    )
    assert descriptor["maximum_replacement_characters"] == hard_limit
    assert "JSON escape syntax does not add characters" in prompt


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

    assert tailored != extracted["content"]
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
    assert envelope["ollama_invoked"] is True
    assert envelope["execution"]["execution_mode"] == "prose_only"
    assert envelope["execution"]["deterministic_patch_count"] == 0
    assert envelope["execution"]["gemma_patch_count"] == 1
    assert envelope["execution"]["ollama_invoked"] is True


def test_mixed_run_persists_sanitized_hybrid_execution_metadata(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _mixed_analysis(extracted, requirements)
    catalog = writer.approved_edit_catalog(analysis)
    deterministic_edits, prose_edits = writer.partition_edit_catalog(catalog)
    full_digest = writer.canonical_digest(catalog)
    prose_digest = writer.canonical_digest(prose_edits)
    provider_calls = 0
    preflight_calls = 0
    real_preflight = writer.preflight_tailoring_inputs

    def tracked_preflight(**kwargs: object) -> list[dict]:
        nonlocal preflight_calls
        preflight_calls += 1
        return real_preflight(**kwargs)

    def fake_request(**kwargs: object) -> dict[str, object]:
        nonlocal provider_calls
        provider_calls += 1
        response = _complete(
            patches=[
                {
                    "edit_id": prose_edits[0]["edit_id"],
                    "target_source_id": prose_edits[0]["target_source_id"],
                    "operation": prose_edits[0]["operation"],
                    "replacement_text": prose_edits[0]["proposed_text"],
                }
            ]
        )
        response["catalog_sha256"] = prose_digest
        return {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(response)},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

    monkeypatch.setattr(writer, "preflight_tailoring_inputs", tracked_preflight)
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

    assert preflight_calls == 1
    assert provider_calls == 1
    assert tailored["skill_groups"][0] != extracted["content"]["skill_groups"][0]
    assert tailored["professional_summary"] == "Updated text"
    metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    execution = metadata["execution"]
    assert execution == {
        "execution_mode": "hybrid",
        "deterministic_patch_count": 1,
        "gemma_patch_count": 1,
        "deterministic_target_ids": [
            edit["target_source_id"] for edit in deterministic_edits
        ],
        "prose_target_ids": [edit["target_source_id"] for edit in prose_edits],
        "full_catalog_digest": full_digest,
        "writer_subset_digest": prose_digest,
        "ollama_invoked": True,
    }
    serialized_execution = json.dumps(execution, sort_keys=True)
    for private_value in (
        private_job,
        "Updated text",
        extracted["content"]["skill_groups"][0]["label"],
        extracted["content"]["professional_summary"],
        "Test rationale",
    ):
        assert private_value not in serialized_execution


@pytest.mark.parametrize(
    ("status", "expected_error", "validation_path"),
    [
        ("cannot_apply", OllamaCannotApplyError, "cannot_apply"),
        ("technical_failure", OllamaTechnicalFailureError, "technical_failure"),
    ],
)
def test_provider_failure_status_preserves_response_and_execution_telemetry(
    status: str,
    expected_error: type[Exception],
    validation_path: str,
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _mixed_analysis(extracted, requirements)
    original = copy.deepcopy(extracted["content"])
    catalog = writer.approved_edit_catalog(analysis)
    deterministic_edits, prose_edits = writer.partition_edit_catalog(catalog)
    full_digest = writer.canonical_digest(catalog)
    prose_digest = writer.canonical_digest(prose_edits)
    provider_marker = f"SYNTHETIC_PROVIDER_{status.upper()}_MARKER"
    response = {
        "status": status,
        "message": provider_marker,
        "catalog_sha256": prose_digest,
        "cannot_apply": (
            {
                "edit_id": prose_edits[0]["edit_id"],
                "reason_code": "unsupported_claim_risk",
                "reason": provider_marker,
            }
            if status == "cannot_apply"
            else None
        ),
        "technical_failure": (
            {
                "reason_code": "other_technical_failure",
                "reason": provider_marker,
            }
            if status == "technical_failure"
            else None
        ),
        "patches": None,
    }

    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma:latest",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(response)},
            "prompt_eval_count": 100,
            "eval_count": 50,
        },
    )

    with pytest.raises(expected_error):
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

    assert extracted["content"] == original
    response_path = tmp_path / writer.OLLAMA_RESPONSE_FILENAME
    assert response_path.is_file()
    assert provider_marker in response_path.read_text(encoding="utf-8")
    metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["validation_result"] == "REJECTED"
    assert metadata["validation_path"] == validation_path
    assert metadata["response"] == {
        "filename": writer.OLLAMA_RESPONSE_FILENAME,
        "sha256": writer.sha256_file(response_path),
    }
    assert metadata["execution"] == {
        "execution_mode": "hybrid",
        "deterministic_patch_count": 1,
        "gemma_patch_count": 0,
        "deterministic_target_ids": [
            edit["target_source_id"] for edit in deterministic_edits
        ],
        "prose_target_ids": [edit["target_source_id"] for edit in prose_edits],
        "full_catalog_digest": full_digest,
        "writer_subset_digest": prose_digest,
        "ollama_invoked": True,
    }
    assert provider_marker not in json.dumps(metadata, sort_keys=True)
    assert not list(tmp_path.glob("tailored-content*.json"))


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


@pytest.mark.parametrize(
    "failure_case",
    [
        "missing_company",
        "missing_role",
        "invalid_job_requirements",
        "mismatched_master_extraction",
        "unresolved_approved_analysis",
    ],
)
def test_deterministic_entrypoint_preflight_rejects_before_compile_or_provider(
    failure_case: str,
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=_deterministic_edits(extracted),
    )
    master_content = extracted["content"]
    company = "Synthetic Systems"
    role = "Validation Engineer"
    supplied_requirements = requirements
    supplied_analysis = analysis

    if failure_case == "missing_company":
        company = ""
    elif failure_case == "missing_role":
        role = ""
    elif failure_case == "invalid_job_requirements":
        supplied_requirements = {}
    elif failure_case == "mismatched_master_extraction":
        master_content = copy.deepcopy(master_content)
        master_content["professional_summary"] = "Mismatched synthetic summary."
    else:
        supplied_analysis = copy.deepcopy(analysis)
        supplied_analysis["recommended_edits"][0].pop("resolved_evidence")

    real_preflight = writer.preflight_tailoring_inputs
    preflight_calls = 0

    def tracked_preflight(**kwargs: object) -> list[dict]:
        nonlocal preflight_calls
        preflight_calls += 1
        return real_preflight(**kwargs)

    monkeypatch.setattr(writer, "preflight_tailoring_inputs", tracked_preflight)
    monkeypatch.setattr(
        writer,
        "compile_deterministic_structured_patches",
        lambda **kwargs: pytest.fail("compilation ran before preflight passed"),
    )
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: pytest.fail("provider ran before preflight passed"),
    )

    with pytest.raises(TailoringPreflightError, match="No writer request"):
        writer.invoke_ollama(
            master_content=master_content,
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=supplied_requirements,
            approved_analysis=supplied_analysis,
            company=company,
            role=role,
            run_directory=tmp_path,
            timeout_seconds=30,
            model="invalid model name",
        )

    assert preflight_calls == 1
    assert not (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).exists()
    assert not (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).exists()


def test_deterministic_only_metadata_is_complete_and_has_no_provider_artifact(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=_deterministic_edits(extracted),
    )
    expected_catalog = writer.approved_edit_catalog(analysis)
    real_preflight = writer.preflight_tailoring_inputs
    preflight_calls = 0

    def tracked_preflight(**kwargs: object) -> list[dict]:
        nonlocal preflight_calls
        preflight_calls += 1
        return real_preflight(**kwargs)

    monkeypatch.setattr(writer, "preflight_tailoring_inputs", tracked_preflight)
    monkeypatch.setattr(
        writer,
        "approved_edit_catalog",
        lambda *_args, **_kwargs: pytest.fail(
            "writer re-derived the catalog after authoritative preflight"
        ),
    )
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: pytest.fail("deterministic-only run invoked Ollama"),
    )

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
        # An unused provider model must not affect deterministic execution.
        model="invalid model name",
    )

    assert preflight_calls == 1
    assert tailored["skill_groups"][0] != extracted["content"]["skill_groups"][0]
    assert not (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).exists()
    metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["provider"] == "deterministic"
    assert metadata["runtime"] == "local"
    assert metadata["model"] is None
    assert metadata["ollama_invoked"] is False
    assert metadata["execution_mode"] == "deterministic-local"
    assert metadata["response_envelope_type"] == "deterministic-local-patches"
    assert metadata["output_format"] == "deterministic-json"
    assert metadata["response"] is None
    assert metadata["validation_result"] == "PASS"
    assert metadata["validation_path"] == "pass"
    assert metadata["execution"] == {
        "execution_mode": "deterministic_only",
        "deterministic_patch_count": 1,
        "gemma_patch_count": 0,
        "deterministic_target_ids": ["skill_groups.0"],
        "prose_target_ids": [],
        "full_catalog_digest": writer.canonical_digest(expected_catalog),
        "writer_subset_digest": None,
        "ollama_invoked": False,
        "writer_skipped": True,
        "writer_skipped_reason": "all_targets_deterministic",
    }


def test_empty_catalog_is_a_deterministic_noop_without_provider_artifacts(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(extracted, requirements, recommended_edits=[])
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: pytest.fail("empty catalog invoked Ollama"),
    )

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
        model="invalid model name",
    )

    assert tailored == extracted["content"]
    assert tailored is not extracted["content"]
    assert not (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).exists()
    metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["response"] is None
    assert metadata["execution"]["deterministic_patch_count"] == 0
    assert metadata["execution"]["gemma_patch_count"] == 0
    assert metadata["execution"]["writer_skipped_reason"] == "empty_catalog"


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
    from resume_tailor.backend.providers.ollama_capabilities import MODEL_CAPABILITIES

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


def _set_hard_budget(
    extracted: dict,
    target_source_id: str,
    maximum_characters: int,
) -> None:
    for paragraph in extracted["paragraphs"]:
        if paragraph["content_id"] == target_source_id:
            paragraph["content_budget"]["maximum_characters"] = maximum_characters
            return
    raise AssertionError(f"Missing synthetic target {target_source_id}")


def _chat_response(payload: dict) -> dict[str, object]:
    return {
        "model": "resume-tailor-gemma:latest",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(payload)},
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


def _patch_payload(
    *,
    catalog_sha256: str,
    patches: list[dict],
) -> dict:
    return {
        "status": "complete",
        "message": "Synthetic bounded patch response.",
        "catalog_sha256": catalog_sha256,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": patches,
    }


def test_patch_exactly_at_hard_limit_passes_without_budget_repair(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    _set_hard_budget(extracted, "professional_summary", 193)
    catalog = writer.approved_edit_catalog(analysis)
    replacement = "E" * 193
    calls: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return _chat_response(
            _patch_payload(
                catalog_sha256=writer.canonical_digest(catalog),
                patches=[
                    {
                        "edit_id": "edit.001",
                        "target_source_id": "professional_summary",
                        "operation": "replace",
                        "replacement_text": replacement,
                    }
                ],
            )
        )

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

    assert tailored["professional_summary"] == replacement
    assert len(calls) == 1
    assert not (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).exists()


def test_one_character_over_invokes_exactly_one_focused_repair(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    _set_hard_budget(extracted, "professional_summary", 193)
    catalog = writer.approved_edit_catalog(analysis)
    initial_text = "O" * 194
    repaired_text = "R" * 193
    requests: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        requests.append(kwargs)
        replacement = initial_text if len(requests) == 1 else repaired_text
        request = kwargs["body"]
        assert isinstance(request, dict)
        digest = request["format"]["properties"]["catalog_sha256"]["enum"][0]
        return _chat_response(
            _patch_payload(
                catalog_sha256=digest,
                patches=[
                    {
                        "edit_id": catalog[0]["edit_id"],
                        "target_source_id": "professional_summary",
                        "operation": "replace",
                        "replacement_text": replacement,
                    }
                ],
            )
        )

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

    assert tailored["professional_summary"] == repaired_text
    assert len(requests) == 2
    primary_metadata = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert primary_metadata["validation_result"] == "PASS"
    assert primary_metadata["budget_repair"]["attempt_count"] == 1
    assert primary_metadata["budget_repair"]["outcome"] == "PASS"
    assert primary_metadata["budget_repair"]["violations"] == [
        {
            "edit_id": "edit.001",
            "target_source_id": "professional_summary",
            "actual_characters": 194,
            "maximum_characters": 193,
        }
    ]
    initial_artifact = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).read_text(encoding="utf-8")
    )
    repair_artifact = json.loads(
        (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert initial_text in initial_artifact["message"]["content"]
    assert repaired_text not in initial_artifact["message"]["content"]
    assert repaired_text in repair_artifact["message"]["content"]
    repair_metadata = json.loads(
        (
            tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_METADATA_FILENAME
        ).read_text(encoding="utf-8")
    )
    serialized_metadata = json.dumps(
        {"primary": primary_metadata, "repair": repair_metadata},
        sort_keys=True,
    )
    assert initial_text not in serialized_metadata
    assert repaired_text not in serialized_metadata


def test_205_character_patch_repairs_to_193_and_leaves_valid_patch_unchanged(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=[
            {
                "target_source_id": "professional_summary",
                "proposed_text": "Approved synthetic summary proposal.",
                "alignment_rationale": "Preserve evidence-backed validation meaning.",
                "evidence_source_ids": ["professional_summary"],
                "operation": "replace",
            },
            {
                "target_source_id": "open_source.bullet",
                "proposed_text": "Stable valid prose patch.",
                "alignment_rationale": "Synthetic unchanged valid patch.",
                "evidence_source_ids": ["open_source.bullet"],
                "operation": "replace",
            },
        ],
    )
    _set_hard_budget(extracted, "professional_summary", 193)
    _set_hard_budget(extracted, "open_source.bullet", 193)
    catalog = writer.approved_edit_catalog(analysis)
    initial_over = "Evidence-backed Python validation".ljust(205, "x")
    repaired = "Evidence-backed Python validation".ljust(193, "y")
    valid_patch = "Stable valid prose patch."
    requests: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        requests.append(kwargs)
        request = kwargs["body"]
        assert isinstance(request, dict)
        digest = request["format"]["properties"]["catalog_sha256"]["enum"][0]
        if len(requests) == 1:
            patches = [
                {
                    "edit_id": catalog[0]["edit_id"],
                    "target_source_id": "professional_summary",
                    "operation": "replace",
                    "replacement_text": initial_over,
                },
                {
                    "edit_id": catalog[1]["edit_id"],
                    "target_source_id": "open_source.bullet",
                    "operation": "replace",
                    "replacement_text": valid_patch,
                },
            ]
        else:
            patches = [
                {
                    "edit_id": catalog[0]["edit_id"],
                    "target_source_id": "professional_summary",
                    "operation": "replace",
                    "replacement_text": repaired,
                }
            ]
        return _chat_response(
            _patch_payload(catalog_sha256=digest, patches=patches)
        )

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

    assert tailored["professional_summary"] == repaired
    assert len(tailored["professional_summary"]) == 193
    assert tailored["open_source"]["bullet"] == valid_patch
    assert len(requests) == 2
    repair_request = requests[1]["body"]
    assert isinstance(repair_request, dict)
    repair_prompt = repair_request["messages"][1]["content"]
    assert initial_over in repair_prompt
    assert "maximum_replacement_characters" in repair_prompt
    assert '"required_evidence_source_ids":["professional_summary"]' in repair_prompt
    assert "Preserve the supported meaning" in repair_prompt
    assert "open_source.bullet" not in repair_prompt
    assert valid_patch not in repair_prompt
    repair_schema = repair_request["format"]
    array_branch = next(
        branch
        for branch in repair_schema["properties"]["patches"]["oneOf"]
        if branch.get("type") == "array"
    )
    assert array_branch["minItems"] == 1
    assert array_branch["maxItems"] == 1


def test_budget_repair_that_adds_forbidden_claim_is_rejected_and_preserved(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    _set_hard_budget(extracted, "professional_summary", 193)
    catalog = writer.approved_edit_catalog(analysis)
    responses = ["I" * 194, "Unsupported synthetic claim"]
    calls = 0

    def fake_request(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        request = kwargs["body"]
        assert isinstance(request, dict)
        digest = request["format"]["properties"]["catalog_sha256"]["enum"][0]
        replacement = responses[calls]
        calls += 1
        return _chat_response(
            _patch_payload(
                catalog_sha256=digest,
                patches=[
                    {
                        "edit_id": catalog[0]["edit_id"],
                        "target_source_id": "professional_summary",
                        "operation": "replace",
                        "replacement_text": replacement,
                    }
                ],
            )
        )

    monkeypatch.setattr(writer, "run_ollama_request", fake_request)
    with pytest.raises(OllamaTailoringContractError, match="forbidden claim"):
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

    assert calls == 2
    assert (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).is_file()
    assert (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).is_file()
    primary = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    repair = json.loads(
        (
            tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_METADATA_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert primary["validation_result"] == "REJECTED"
    assert primary["budget_repair"]["outcome"] == "REJECTED"
    assert repair["validation_result"] == "REJECTED"
    assert not list(tmp_path.glob("tailored-content*.json"))


def test_over_budget_patch_with_existing_contract_violation_is_not_repaired(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    _set_hard_budget(extracted, "professional_summary", 20)
    catalog = writer.approved_edit_catalog(analysis)
    calls = 0

    def fake_request(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _chat_response(
            _patch_payload(
                catalog_sha256=writer.canonical_digest(catalog),
                patches=[
                    {
                        "edit_id": catalog[0]["edit_id"],
                        "target_source_id": "professional_summary",
                        "operation": "replace",
                        "replacement_text": (
                            "Unsupported synthetic claim that is also over budget"
                        ),
                    }
                ],
            )
        )

    monkeypatch.setattr(writer, "run_ollama_request", fake_request)
    with pytest.raises(OllamaTailoringContractError, match="forbidden claim"):
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

    assert calls == 1
    assert not (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).exists()
    primary = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert primary["budget_repair"]["attempted"] is False
    assert primary["budget_repair"]["provider_invoked"] is False
    assert primary["budget_repair"]["attempt_count"] == 0


def test_budget_repair_still_over_limit_stops_after_one_attempt_with_diagnostics(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, analysis = _inputs(master_resume)
    _set_hard_budget(extracted, "professional_summary", 193)
    catalog = writer.approved_edit_catalog(analysis)
    calls = 0

    def fake_request(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        request = kwargs["body"]
        assert isinstance(request, dict)
        digest = request["format"]["properties"]["catalog_sha256"]["enum"][0]
        return _chat_response(
            _patch_payload(
                catalog_sha256=digest,
                patches=[
                    {
                        "edit_id": catalog[0]["edit_id"],
                        "target_source_id": "professional_summary",
                        "operation": "replace",
                        "replacement_text": "B" * 194,
                    }
                ],
            )
        )

    monkeypatch.setattr(writer, "run_ollama_request", fake_request)
    with pytest.raises(OllamaTailoringContractError, match="exceeds target budget"):
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

    assert calls == 2
    assert (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).is_file()
    assert (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).is_file()
    assert (tmp_path / writer.OLLAMA_TAILORING_TRANSPORT_SCHEMA_FILENAME).is_file()
    assert (
        tmp_path / writer.OLLAMA_BUDGET_REPAIR_TRANSPORT_SCHEMA_FILENAME
    ).is_file()
    primary = json.loads(
        (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    repair = json.loads(
        (
            tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_METADATA_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert primary["validation_path"] == "character_budget"
    assert primary["budget_repair"]["attempt_count"] == 1
    assert repair["validation_path"] == "character_budget"
    assert repair["budget_repair"]["maximum_attempts"] == 1


def test_python_owned_structured_patch_never_enters_budget_repair(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=[
            *_deterministic_edits(extracted),
            {
                "target_source_id": "professional_summary",
                "proposed_text": "Updated text",
                "alignment_rationale": "Test rationale",
                "evidence_source_ids": [
                    "professional_summary",
                    "skill_groups.0",
                ],
                "operation": "replace",
            },
        ],
    )
    _set_hard_budget(extracted, "professional_summary", 193)
    catalog = writer.approved_edit_catalog(analysis)
    _, prose_edits = writer.partition_edit_catalog(catalog)
    prose_edit = prose_edits[0]
    requests: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        requests.append(kwargs)
        request = kwargs["body"]
        assert isinstance(request, dict)
        digest = request["format"]["properties"]["catalog_sha256"]["enum"][0]
        replacement = "P" * (194 if len(requests) == 1 else 193)
        return _chat_response(
            _patch_payload(
                catalog_sha256=digest,
                patches=[
                    {
                        "edit_id": prose_edit["edit_id"],
                        "target_source_id": prose_edit["target_source_id"],
                        "operation": prose_edit["operation"],
                        "replacement_text": replacement,
                    }
                ],
            )
        )

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

    assert len(requests) == 2
    repair_request = requests[1]["body"]
    assert isinstance(repair_request, dict)
    repair_prompt = repair_request["messages"][1]["content"]
    assert "skill_groups.0" not in repair_prompt
    structured_exact_text = next(
        block["exact_text"]
        for block in extracted["source_blocks"]
        if block["source_id"] == "skill_groups.0"
    )
    assert structured_exact_text not in repair_prompt
    assert '"source_id":"professional_summary"' in repair_prompt
    assert prose_edit["target_source_id"] in repair_prompt
    assert tailored["skill_groups"][0] != extracted["content"]["skill_groups"][0]


def test_composite_205_vs_193_is_rejected_before_approval_or_provider(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, private_job, requirements, _ = _inputs(master_resume)
    target_id = "skill_groups.2"
    group = extracted["content"]["skill_groups"][2]
    group["label"] = "Synthetic Skills Set"
    assert len(group["label"]) == 20
    source_block = next(
        block for block in extracted["source_blocks"] if block["source_id"] == target_id
    )
    source_block["exact_text"] = f"{group['label']}: {group['text']}"
    paragraph = next(
        item for item in extracted["paragraphs"] if item["content_id"] == target_id
    )
    paragraph["text"] = source_block["exact_text"]
    paragraph["content_budget"]["maximum_characters"] = 193

    body = ", ".join(["JSON Schema", *(["pytest"] * 11), *(["Linux"] * 12)])
    assert len(body) == 183
    proposed = f"{group['label']}: {body}"
    assert len(proposed) == 205
    analysis = _resolved_analysis(
        extracted,
        requirements,
        recommended_edits=[],
    )
    analysis["recommended_edits"] = [
        {
            "target_source_id": "professional_summary",
            "proposed_text": "Updated text",
            "alignment_rationale": "Synthetic prose edit.",
            "evidence_source_ids": ["professional_summary"],
            "operation": "replace",
        },
        {
            "target_source_id": target_id,
            "proposed_text": proposed,
            "alignment_rationale": "Reorder authenticated synthetic skills.",
            "evidence_source_ids": [target_id],
            "operation": "replace",
        },
    ]
    resolved, issues = resolve_analysis_evidence(
        analysis,
        extracted,
        requirements,
    )

    budget_issue = next(
        issue for issue in issues if issue.code == "structured_proposal_over_budget"
    )
    assert budget_issue.location == "recommended_edits[1].proposed_text"
    assert "205 characters" in budget_issue.detail
    assert "maximum of 193" in budget_issue.detail
    assert writer.approved_edit_catalog(resolved)[1]["edit_id"] == "edit.002"

    calls = 0

    def fail_if_provider_called(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("The provider must not run for an impossible plan.")

    monkeypatch.setattr(writer, "run_ollama_request", fail_if_provider_called)
    with pytest.raises(TailoringPreflightError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=private_job,
            job_requirements=requirements,
            approved_analysis=resolved,
            company="Synthetic Systems",
            role="Validation Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )

    assert calls == 0
    assert not (tmp_path / writer.OLLAMA_RESPONSE_FILENAME).exists()
    assert not (tmp_path / writer.OLLAMA_BUDGET_REPAIR_RESPONSE_FILENAME).exists()
