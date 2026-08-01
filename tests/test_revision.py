from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import validate_tailored_content
from resume_tailor.revision import (
    build_revision_prompt,
    invoke_antigravity_revision,
    validate_revision_scope,
)
from resume_tailor.utilities import (
    AntigravityResponseEnvelopeError,
    AntigravityRevisionCannotApplyError,
    AntigravityRevisionContractError,
    AntigravityRevisionTechnicalFailureError,
    RevisionValidationError,
)


def _inputs(master_resume: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    extracted, _ = extract_resume(master_resume)
    source = next(
        block
        for block in extracted["source_blocks"]
        if block["source_id"] == "professional_summary"
    )
    analysis = {
        "recommended_edits": [
            {
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": "Synthetic evidence-backed summary.",
                "alignment_rationale": "Synthetic clarity alignment.",
                "evidence_source_ids": ["professional_summary"],
                "existing_text": source["exact_text"],
                "resume_section": source["section_context"],
                "resolved_evidence": [],
            }
        ],
        "immutable_facts": [],
        "forbidden_claims": ["GraphQL expertise"],
        "supported_ats_keywords": [],
    }
    qa = {
        "status": "material_findings",
        "summary": "Synthetic material finding.",
        "issues": [
            {
                "issue_id": "qa.001",
                "category": "clarity",
                "severity": "medium",
                "description": "The summary needs a bounded clarity correction.",
                "affected_content_id": "professional_summary",
                "evidence_source_ids": ["professional_summary"],
                "correction_action": "improve_clarity",
                "correction_objective": "Improve clarity without adding facts.",
            }
        ],
        "technical_failure": None,
    }
    return extracted, analysis, qa


def _invoke(
    *,
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    extracted, analysis, qa = _inputs(master_resume)
    revised = invoke_antigravity_revision(
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa,
        company="Synthetic Systems",
        role="Evidence Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
        antigravity_duration="30s",
        attempt_number=1,
        executable=str(stubs_on_path / "agy"),
    )
    return extracted, analysis, qa, revised


def test_successful_revision_is_authored_by_antigravity_and_scope_validates(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "agy-invocations.jsonl"
    monkeypatch.setenv("STUB_AGY_INVOCATION_LOG", str(log))
    extracted, analysis, qa, revised = _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
    )

    assert revised != extracted["content"]
    assert revised["professional_summary"].split(" ", 1)[0].endswith(",")
    issue_map = validate_revision_scope(
        initial_content=extracted["content"],
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"professional_summary": ["qa.001"]}
    report = validate_tailored_content(
        original=extracted["content"],
        tailored=revised,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="Evidence Engineer",
    )
    assert report.passed
    invocation = json.loads(log.read_text(encoding="utf-8"))
    assert invocation["role"] == "resume_revision_writer"
    assert invocation["output_format"] == "stream-json"


def test_revision_prompt_contains_only_bounded_authoring_instructions(
    master_resume: Path,
) -> None:
    extracted, analysis, qa = _inputs(master_resume)
    prompt = build_revision_prompt(
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa,
        company="Synthetic Systems",
        role="Evidence Engineer",
    )

    assert "revision attempt 1 of 1" in prompt
    assert "Never request or produce a second revision" in prompt
    assert "Change only targets present" in prompt
    assert "invoke tools" in prompt
    assert "qa.001" in prompt


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("cannot_apply", AntigravityRevisionCannotApplyError),
        ("invalid_issue", AntigravityRevisionContractError),
        ("technical_failure", AntigravityRevisionTechnicalFailureError),
        ("incomplete_stream", AntigravityResponseEnvelopeError),
        ("structural_drift", AntigravityRevisionContractError),
    ],
)
def test_revision_bounded_failures_are_rejected(
    mode: str,
    error_type: type[BaseException],
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_AGY_REVISION_MODE", mode)

    with pytest.raises(error_type):
        _invoke(
            master_resume=master_resume,
            tmp_path=tmp_path,
            stubs_on_path=stubs_on_path,
        )


@pytest.mark.parametrize("mode", ["outside_target", "unsupported_technology", "content_budget"])
def test_revision_local_checks_reject_scope_fact_and_budget_violations(
    mode: str,
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_AGY_REVISION_MODE", mode)
    extracted, analysis, qa, revised = _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
    )
    report = validate_tailored_content(
        original=extracted["content"],
        tailored=revised,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="Evidence Engineer",
    )
    if mode == "outside_target":
        with pytest.raises(RevisionValidationError):
            validate_revision_scope(
                initial_content=extracted["content"],
                revised_content=revised,
                qa_result=qa,
                approved_analysis=analysis,
            )
    else:
        assert not report.passed


def test_second_revision_attempt_is_blocked_before_provider_launch(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, analysis, qa = _inputs(master_resume)
    log = tmp_path / "agy-invocations.jsonl"
    monkeypatch.setenv("STUB_AGY_INVOCATION_LOG", str(log))

    with pytest.raises(RevisionValidationError, match="one.*revision|Exactly one"):
        invoke_antigravity_revision(
            current_tailored_content=copy.deepcopy(extracted["content"]),
            extracted_resume=extracted,
            approved_analysis=analysis,
            qa_result=qa,
            company="Synthetic Systems",
            role="Evidence Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            antigravity_duration="30s",
            attempt_number=2,
            executable=str(stubs_on_path / "agy"),
        )

    assert not log.exists()
