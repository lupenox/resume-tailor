from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import resume_tailor.ollama_writer as ollama_writer_module
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import validate_tailored_content
from resume_tailor.revision import (
    build_revision_prompt,
    invoke_antigravity_revision,
    validate_revision_scope,
)
from resume_tailor.schemas import schema_path
from resume_tailor.utilities import (
    AntigravityResponseEnvelopeError,
    AntigravityRevisionCannotApplyError,
    AntigravityRevisionContractError,
    AntigravityRevisionTechnicalFailureError,
    OllamaRevisionContractError,
    RevisionValidationError,
    sha256_file,
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


def test_successful_prose_only_ollama_revision_is_scoped_and_one_shot(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, analysis, qa = _inputs(master_resume)
    invocation_calls: list[dict[str, Any]] = []
    target_map = {"professional_summary": ["qa.001"]}
    authorization_sha256 = ollama_writer_module.canonical_digest(target_map)
    canonical_schema_path = schema_path("ollama_revision_patch.schema.json")
    canonical_schema_sha256 = sha256_file(canonical_schema_path)
    replacement = (
        "Synthetic evidence-gated engineering profile used only for résumé "
        "workflow tests."
    )

    revision_payload = {
        "status": "complete",
        "message": "Synthetic revision complete.",
        "authorization_sha256": authorization_sha256,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "issue_id": "qa.001",
                "target_source_id": "professional_summary",
                "replacement_text": replacement,
            }
        ],
    }

    def mocked_provider(**kwargs: Any) -> dict[str, Any]:
        invocation_calls.append(kwargs)
        request_body = kwargs["body"]
        assert kwargs["path"] == "/api/chat"
        assert isinstance(request_body, dict)
        schema = request_body["format"]
        schema_file = (
            tmp_path / ollama_writer_module.OLLAMA_REVISION_TRANSPORT_SCHEMA_FILENAME
        )
        assert isinstance(schema, dict)
        assert json.loads(schema_file.read_text(encoding="utf-8")) == schema
        assert schema["properties"]["authorization_sha256"]["enum"] == [
            authorization_sha256
        ]
        patch_array = next(
            branch
            for branch in schema["properties"]["patches"]["oneOf"]
            if branch.get("type") == "array"
        )
        assert patch_array["minItems"] == 1
        assert patch_array["maxItems"] == 1
        patch_properties = patch_array["items"]["properties"]
        assert patch_properties["issue_id"]["enum"] == ["qa.001"]
        assert patch_properties["target_source_id"]["enum"] == [
            "professional_summary"
        ]
        prompt = request_body["messages"][1]["content"]
        assert "skill_groups." not in prompt
        assert "education.coursework" not in prompt
        assert authorization_sha256 in prompt
        return {
            "model": request_body["model"],
            "created_at": "2026-08-03T12:00:00Z",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
            "message": {
                "role": "assistant",
                "content": json.dumps(revision_payload),
            },
        }

    monkeypatch.setattr(
        ollama_writer_module,
        "run_ollama_request",
        mocked_provider,
    )

    revised = ollama_writer_module.invoke_ollama_revision(
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa,
        company="Synthetic Systems",
        role="Evidence Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
        attempt_number=1,
    )

    assert len(invocation_calls) == 1
    assert revised["professional_summary"] == replacement
    assert revised["professional_summary"] != extracted["content"][
        "professional_summary"
    ]
    assert validate_revision_scope(
        initial_content=extracted["content"],
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    ) == target_map
    assert sha256_file(canonical_schema_path) == canonical_schema_sha256
    metadata = json.loads(
        (
            tmp_path
            / ollama_writer_module.OLLAMA_REVISION_RESPONSE_METADATA_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert metadata["validation_result"] == "PASS"
    assert metadata["validation_path"] == "pass"
    assert metadata["response"]["filename"] == (
        ollama_writer_module.OLLAMA_REVISION_RESPONSE_FILENAME
    )

    with pytest.raises(OllamaRevisionContractError, match="Exactly one"):
        ollama_writer_module.invoke_ollama_revision(
            current_tailored_content=revised,
            extracted_resume=extracted,
            approved_analysis=analysis,
            qa_result=qa,
            company="Synthetic Systems",
            role="Evidence Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            attempt_number=2,
        )

    structured_analysis = copy.deepcopy(analysis)
    structured_analysis["recommended_edits"].append(
        {
            **copy.deepcopy(analysis["recommended_edits"][0]),
            "target_source_id": "skill_groups.0",
        }
    )
    structured_qa = copy.deepcopy(qa)
    structured_qa["issues"][0]["affected_content_id"] = "skill_groups.0"
    with pytest.raises(
        OllamaRevisionContractError,
        match="structured_target_requires_new_analysis",
    ):
        ollama_writer_module.invoke_ollama_revision(
            current_tailored_content=revised,
            extracted_resume=extracted,
            approved_analysis=structured_analysis,
            qa_result=structured_qa,
            company="Synthetic Systems",
            role="Evidence Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            attempt_number=1,
        )
    assert len(invocation_calls) == 1
