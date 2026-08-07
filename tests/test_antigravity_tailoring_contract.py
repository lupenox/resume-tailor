from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import resume_tailor.backend.providers.antigravity_writer as writer_module
from resume_tailor.backend.providers.antigravity_writer import (
    approved_edit_catalog,
    build_tailoring_prompt,
    invoke_antigravity,
    preflight_tailoring_inputs,
    resolve_tailoring_response,
)
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.utils.schemas import load_schema, validate_payload
from resume_tailor.backend.utils.utilities import (
    AntigravityCannotApplyError,
    AntigravityTailoringContractError,
    AntigravityTailoringPreflightError,
    AntigravityTechnicalFailureError,
    sha256_file,
)


def _inputs(master_resume: Path) -> dict[str, Any]:
    extracted, _ = extract_resume(master_resume)
    job_description = (
        "Build synthetic Python evidence validation and unsupported orbital "
        "telemetry systems."
    )
    requirements = build_job_requirement_catalog(
        job_description,
        structured_job={
            "responsibilities": ["Build synthetic evidence validation systems."],
            "technologies_and_skills": ["Python", "Orbital telemetry"],
        },
    )
    supported_id = next(
        item["requirement_id"]
        for item in requirements["requirements"]
        if item["exact_text"] == "Python"
    )
    raw_analysis = {
        "role_summary": "Synthetic evidence validation role.",
        "fit_assessment": {
            "overall": "Partial fit with an explicit unsupported gap.",
            "strengths": ["Python"],
            "gaps": ["Orbital telemetry is absent"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": supported_id,
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": [
            item["requirement_id"]
            for item in requirements["requirements"]
            if item["requirement_id"] != supported_id
        ],
        "recommended_edits": [
            {
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": (
                    "Synthetic engineer focused on evidence-gated Python systems."
                ),
                "alignment_rationale": "Emphasize supported Python evidence.",
                "evidence_source_ids": [
                    "professional_summary",
                    "skill_groups.0",
                ],
            }
        ],
        "immutable_facts": [],
        "forbidden_claims": ["Orbital telemetry experience"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }
    analysis, issues = resolve_analysis_evidence(
        raw_analysis,
        extracted,
        requirements,
    )
    assert issues == []
    return {
        "master_content": extracted["content"],
        "extracted_resume": extracted,
        "job_description": job_description,
        "job_requirements": requirements,
        "approved_analysis": analysis,
        "company": "Synthetic Systems",
        "role": "Evidence Engineer",
    }


def _complete_payload(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "complete",
        "message": "Applied the approved synthetic edit catalog.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": content,
    }


def test_preserved_waiting_shape_is_rejected_offline_and_provider_text_is_hidden(
    repository_root: Path,
    master_resume: Path,
) -> None:
    payload = json.loads(
        (
            repository_root
            / "tests"
            / "fixtures"
            / "antigravity_waiting_plan_mode.json"
        ).read_text(encoding="utf-8")
    )
    inputs = _inputs(master_resume)

    with pytest.raises(AntigravityTailoringContractError) as raised:
        resolve_tailoring_response(
            payload,
            approved_analysis=inputs["approved_analysis"],
        )

    message = str(raised.value)
    assert "generic request for more information" in message
    assert payload["structured_output"]["message"] not in message
    assert payload["structured_output"]["questions_for_user"][0] not in message
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            payload["structured_output"],
            load_schema("tailored_resume.schema.json"),
        )


def test_compliant_response_resolves_offline_to_content_diff_input(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    inputs = _inputs(master_resume)
    payload = _complete_payload(copy.deepcopy(inputs["master_content"]))
    before_hash = sha256_file(master_resume)

    validate_payload(
        payload,
        "tailored_resume.schema.json",
        label="synthetic compliant tailoring response",
    )
    result = resolve_tailoring_response(
        payload,
        approved_analysis=inputs["approved_analysis"],
    )

    assert result == inputs["master_content"]
    assert result is not inputs["master_content"]
    assert sha256_file(master_resume) == before_hash
    assert list(tmp_path.iterdir()) == []


def test_valid_content_wins_over_incidental_legacy_wrapper_status(
    master_resume: Path,
) -> None:
    inputs = _inputs(master_resume)
    payload = {
        "status": "WAITING",
        "message": "Incidental synthetic wrapper prose.",
        "structured_output": _complete_payload(inputs["master_content"]),
    }

    assert resolve_tailoring_response(
        payload,
        approved_analysis=inputs["approved_analysis"],
    ) == inputs["master_content"]


def test_bounded_cannot_apply_requires_a_local_approved_edit_id(
    master_resume: Path,
) -> None:
    inputs = _inputs(master_resume)
    valid_id = approved_edit_catalog(inputs["approved_analysis"])[0]["edit_id"]
    payload = {
        "status": "cannot_apply",
        "message": "A synthetic approved edit could not be applied safely.",
        "cannot_apply": {
            "edit_id": valid_id,
            "reason_code": "evidence_conflict",
            "reason": "Synthetic bounded reason.",
        },
        "technical_failure": None,
        "tailored_resume": None,
    }

    with pytest.raises(AntigravityCannotApplyError, match=valid_id):
        resolve_tailoring_response(
            payload,
            approved_analysis=inputs["approved_analysis"],
        )

    payload["cannot_apply"]["edit_id"] = "edit.999"
    with pytest.raises(AntigravityTailoringContractError, match="unknown"):
        resolve_tailoring_response(
            payload,
            approved_analysis=inputs["approved_analysis"],
        )


def test_technical_failure_remains_separate_from_factual_input(
    master_resume: Path,
) -> None:
    inputs = _inputs(master_resume)
    payload = {
        "status": "technical_failure",
        "message": "Synthetic provider execution failed.",
        "cannot_apply": None,
        "technical_failure": {
            "reason_code": "structured_output_failure",
            "reason": "Synthetic bounded reason.",
        },
        "tailored_resume": None,
    }

    with pytest.raises(
        AntigravityTechnicalFailureError,
        match="structured_output_failure",
    ) as raised:
        resolve_tailoring_response(
            payload,
            approved_analysis=inputs["approved_analysis"],
        )
    assert "Synthetic bounded reason" not in str(raised.value)


def test_local_preflight_rejects_missing_or_unresolved_inputs_without_provider(
    master_resume: Path,
) -> None:
    inputs = _inputs(master_resume)
    invalid = copy.deepcopy(inputs)
    invalid["job_description"] = ""
    with pytest.raises(AntigravityTailoringPreflightError, match="job description"):
        preflight_tailoring_inputs(**invalid)

    invalid = copy.deepcopy(inputs)
    invalid["approved_analysis"]["questions_for_user"] = [
        "Do you have an unlisted synthetic credential?"
    ]
    with pytest.raises(
        AntigravityTailoringPreflightError,
        match="unanswered factual question",
    ):
        preflight_tailoring_inputs(**invalid)

    invalid = copy.deepcopy(inputs)
    invalid["approved_analysis"]["recommended_edits"][0][
        "existing_text"
    ] = "Fabricated source text"
    with pytest.raises(
        AntigravityTailoringPreflightError,
        match="no longer resolves exactly",
    ):
        preflight_tailoring_inputs(**invalid)


def test_invoke_runs_completeness_preflight_before_transport(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(master_resume)
    inputs["job_description"] = ""
    monkeypatch.setattr(
        writer_module,
        "run_antigravity_prompt",
        lambda **_kwargs: pytest.fail(
            "Antigravity transport ran before local completeness preflight"
        ),
    )

    with pytest.raises(AntigravityTailoringPreflightError):
        invoke_antigravity(
            **inputs,
            run_directory=tmp_path,
            timeout_seconds=30,
            antigravity_duration="30s",
            executable="/synthetic/bin/agy",
        )
    assert not (tmp_path / "antigravity-response.json").exists()


def test_tailoring_prompt_is_execution_only_and_omits_raw_job_text(
    master_resume: Path,
) -> None:
    inputs = _inputs(master_resume)
    prompt = build_tailoring_prompt(**inputs)

    assert "not a planning or factual-" in prompt
    assert "Never return WAITING" in prompt
    assert "Never request missing" in prompt
    assert '"edit_id": "edit.001"' in prompt
    assert inputs["job_description"] not in prompt
