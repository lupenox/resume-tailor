from __future__ import annotations

import json
from pathlib import Path

import pytest

import resume_tailor.ollama_writer as writer
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.utilities import OllamaTailoringContractError, TailoringPreflightError


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
    assert request["options"]["num_ctx"] == 8192
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
    assert envelope["prompt"]["content_logged"] is False


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

    with pytest.raises(OllamaTailoringContractError):
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
