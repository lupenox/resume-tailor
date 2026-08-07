from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import changed_content_ids, resolve_analysis_evidence, validate_tailored_content
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.providers.ollama_writer import (
    _resolve_initial_payload,
    _validate_gemma_structural_contract,
    build_ollama_tailoring_prompt,
)
from resume_tailor.backend.utils.utilities import (
    OllamaCannotApplyError,
    OllamaEvidenceRejectionError,
    OllamaTailoringContractError,
)


def _setup_synthetic_inputs(master_resume_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    extracted, _ = extract_resume(master_resume_path)
    job_desc = "Synthetic Job Description for AI Solutions Engineer position requiring Python, FastAPI, and Linux."
    reqs = build_job_requirement_catalog(job_desc)
    raw_analysis = {
        "role_summary": "Synthetic role summary",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": [],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": [],
        "missing_or_unsupported_requirements": [],
        "immutable_facts": ["Synthetic degree"],
        "forbidden_claims": [],
        "recommended_edits": [
            {
                "target_source_id": "skill_groups.0",
                "operation": "replace",
                "proposed_text": f"{extracted['content']['skill_groups'][0]['label']}: Python, JavaScript, SQL",
                "alignment_rationale": "Align with job requirements",
                "evidence_source_ids": ["skill_groups.0"],
            },
            {
                "target_source_id": "projects.0.bullets.0",
                "operation": "replace",
                "proposed_text": "Designed synthetic AI pipeline in 2024 with zero errors.",
                "alignment_rationale": "Align with job requirements",
                "evidence_source_ids": ["projects.0.bullets.0"],
            },
        ],
        "content_budget_guidance": [],
        "questions_for_user": [],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
    }
    analysis, issues = resolve_analysis_evidence(raw_analysis, extracted, reqs)
    assert issues == []
    return extracted, reqs, analysis


def test_1_skill_group_label_cannot_be_renamed_without_approved_edit(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["skill_groups"][0]["label"] = "Renamed Skill Category"

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("Immutable field changed at skill_groups.0.label" in issue for issue in report.issues)


def test_2_forged_label_target_cannot_override_immutable_policy(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    analysis["recommended_edits"].append({
        "target_source_id": "skill_groups.0.label",
        "operation": "replace",
        "proposed_text": "Renamed Skill Category",
        "evidence_source_ids": ["skill_groups.0"],
    })
    tailored = copy.deepcopy(master)
    tailored["skill_groups"][0]["label"] = "Renamed Skill Category"

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any(
        "Immutable field changed at skill_groups.0.label" in issue
        for issue in report.issues
    )
    with pytest.raises(
        OllamaTailoringContractError, match="modified immutable skill-group label"
    ):
        _validate_gemma_structural_contract(
            master_content=master,
            tailored=tailored,
            approved_analysis=analysis,
        )

def test_3_rewriting_bullet_does_not_reduce_bullet_count(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["projects"][0]["bullets"].pop()

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("changed from" in issue and "bullets" in issue for issue in report.issues)


def test_4_unsupported_structural_operation_cannot_authorize_bullet_removal(
    master_resume: Path,
) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    analysis["recommended_edits"].append({
        "target_source_id": "projects.0",
        "operation": "remove_bullet",
        "proposed_text": "forged structural operation",
        "evidence_source_ids": ["projects.0.bullets.0"],
    })
    tailored = copy.deepcopy(master)
    tailored["projects"][0]["bullets"].pop()

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("changed from" in issue and "bullets" in issue for issue in report.issues)
    with pytest.raises(OllamaTailoringContractError, match="altered bullet count"):
        _validate_gemma_structural_contract(
            master_content=master,
            tailored=tailored,
            approved_analysis=analysis,
        )

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas" / "codex_analysis.schema.json")
        .read_text(encoding="utf-8")
    )
    operations = schema["properties"]["recommended_edits"]["items"]["properties"][
        "operation"
    ]["enum"]
    assert operations == ["replace", "append"]

def test_5_unsupported_inferred_skill_agent_state_machines_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["skill_groups"][0]["text"] += ", agent state machines"

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("agent state machines" in issue for issue in report.issues)


def test_6_unsupported_inferred_phrase_api_integration_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["skill_groups"][2]["text"] += ", API integration"

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("API integration" in issue for issue in report.issues)


def test_7_supported_skill_rewrite_grounded_in_evidence_succeeds(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    # FastAPI is present in sample resume skill_groups.2 and project 1
    tailored["skill_groups"][0]["text"] += ", FastAPI"

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not any("FastAPI" in issue and "lacks verbatim source evidence" in issue for issue in report.issues)


def test_8_new_numeric_claim_6_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["projects"][0]["bullets"][0] = "Designed system with 6-second latency."

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("New numeric or metric claim '6'" in issue for issue in report.issues)


def test_9_existing_authenticated_numeric_claims_remain_valid(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    # 2024 is an existing numeric claim in sample resume
    tailored["projects"][0]["bullets"][0] = "Designed synthetic AI pipeline in 2024 with zero errors."

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not any("New numeric or metric claim '2024'" in issue for issue in report.issues)


def test_10_approved_edit_comparison_succeeds_for_valid_gemma_response(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["projects"][0]["bullets"][0] = "Designed synthetic AI pipeline in 2024 with zero errors."

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert report.passed
    assert not report.issues


def test_11_unrelated_unapproved_change_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["experience"]["bullets"][0] = "Unapproved change to experience bullet."

    report = validate_tailored_content(
        original=master,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert not report.passed
    assert any("Unapproved content target changed: experience.bullets.0" in issue for issue in report.issues)


def test_12_unknown_approved_edit_ids_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    payload = {
        "status": "cannot_apply",
        "message": "Cannot apply edit.999",
        "cannot_apply": {
            "edit_id": "edit.999",
            "reason_code": "unsupported_claim_risk",
            "reason": "Unsupported claim risk in synthetic edit",
        },
        "technical_failure": None,
        "tailored_resume": None,
    }
    with pytest.raises(OllamaEvidenceRejectionError, match="unknown approved edit ID"):
        _resolve_initial_payload(payload, approved_analysis=analysis)


def test_13_missing_or_mismatched_target_paths_rejected(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["skill_groups"][0]["label"] = "Mismatched Label"

    with pytest.raises(OllamaTailoringContractError, match="modified immutable skill-group label"):
        _validate_gemma_structural_contract(
            master_content=master,
            tailored=tailored,
            approved_analysis=analysis,
        )


def test_14_cannot_apply_remains_available_for_known_edit(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    payload = {
        "status": "cannot_apply",
        "message": "Cannot apply edit.001",
        "cannot_apply": {
            "edit_id": "edit.001",
            "reason_code": "unsupported_claim_risk",
            "reason": "Unsupported claim risk in synthetic edit",
        },
        "technical_failure": None,
        "tailored_resume": None,
    }
    with pytest.raises(OllamaCannotApplyError, match="could not apply approved edit.001"):
        _resolve_initial_payload(payload, approved_analysis=analysis)


def test_15_constraint_manifest_is_present_in_writer_prompt(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    prompt = build_ollama_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description="Synthetic Job Description for AI Solutions Engineer position requiring Python, FastAPI, and Linux.",
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic Corp",
        role="AI Solutions Engineer",
    )
    assert "AUTHENTICATED METRICS" in prompt
    assert "CATALOG SHA256 DIGEST" in prompt

def test_16_no_qwen_or_antigravity_fallback_introduced() -> None:
    from resume_tailor.backend.providers.ollama_writer import DEFAULT_OLLAMA_MODEL
    assert "gemma" in DEFAULT_OLLAMA_MODEL.lower()


def test_17_changed_content_ids_handles_differing_keys_without_error(master_resume: Path) -> None:
    extracted, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master = extracted["content"]
    tailored = copy.deepcopy(master)
    tailored["projects"][0]["bullets"].pop()

    # Should return changed target IDs without raising ValueError
    changed = changed_content_ids(master, tailored)
    assert "projects.0.bullets.2" in changed


def test_18_historical_wrong_root_ollama_regression_still_passes(master_resume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from resume_tailor.backend.providers.ollama_capabilities import capabilities_for_model, plan_ollama_budget
    from resume_tailor.backend.providers.ollama_writer import _invoke_payload, _write_transport_schema
    schema, transport_path = _write_transport_schema(
        tmp_path,
        canonical_name="ollama_tailoring_patch.schema.json",
        filename="ollama-tailoring-transport.schema.json",
    )
    capabilities = capabilities_for_model("resume-tailor-gemma")
    budget = plan_ollama_budget(prompt="synthetic prompt", capabilities=capabilities)

    wrong_root_payload = {
        "professional_summary": "Synthetic summary",
        "education": {},
        "skill_groups": [],
        "projects": [],
        "open_source": {},
        "experience": {},
    }

    monkeypatch.setattr(
        "resume_tailor.backend.providers.ollama_writer.run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(wrong_root_payload)},
            "prompt_eval_count": 100,
            "eval_count": 50,
        },
    )

    from resume_tailor.backend.utils.utilities import OllamaTransportSchemaError
    with pytest.raises(OllamaTransportSchemaError, match="ignored the supplied structured-output schema"):
        _invoke_payload(
            model="resume-tailor-gemma",
            prompt="synthetic prompt",
            transport_schema=schema,
            schema_path=transport_path,
            response_filename="ollama-response.json",
            metadata_filename="ollama-response-envelope.json",
            run_directory=tmp_path,
            timeout_seconds=30,
            capabilities=capabilities,
            budget=budget,
        )
