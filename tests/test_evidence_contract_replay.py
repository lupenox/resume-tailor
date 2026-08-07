from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import (
    diagnose_legacy_analysis_evidence,
    resolve_analysis_evidence,
)
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.engine.orchestration import ApprovalResponse, PipelineHooks
from resume_tailor.backend.utils.schemas import (
    build_codex_analysis_transport_schema,
    validate_payload,
)
from resume_tailor.backend.utils.utilities import ModelError


def _requirements() -> dict[str, Any]:
    return build_job_requirement_catalog(
        "Synthetic AI engineering requirements.",
        structured_job={
            "responsibilities": ["Build evidence-gated AI systems."],
            "technologies_and_skills": [
                "Python",
                "RAG",
                "Kubernetes",
                "rule-based link scoring",
            ],
        },
    )


def _compliant_response() -> dict[str, Any]:
    return {
        "role_summary": "Synthetic evidence-gated role.",
        "fit_assessment": {
            "overall": "Partial fit with explicit unsupported gaps.",
            "strengths": ["Python"],
            "gaps": ["Absent technologies remain unsupported"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": "skill.001",
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": [
            "responsibility.001",
            "skill.002",
            "skill.003",
            "skill.004",
        ],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": ["RAG", "Kubernetes"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


def test_legacy_model_authored_ats_mismatch_replays_exact_old_failure_shape(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    legacy = {
        "ats_keywords": [
            {
                "keyword": label,
                "evidence_source_ids": ["skill_groups.0"],
            }
            for label in (
                "synthetic semantic label one",
                "synthetic semantic label two",
                "synthetic semantic label three",
                "synthetic semantic label four",
            )
        ]
    }

    issues = diagnose_legacy_analysis_evidence(legacy, extracted)

    assert [issue.location for issue in issues] == [
        "ats_keywords[0].evidence_source_ids",
        "ats_keywords[1].evidence_source_ids",
        "ats_keywords[2].evidence_source_ids",
        "ats_keywords[3].evidence_source_ids",
    ]
    assert {issue.code for issue in issues} == {
        "unsupported_ats_keyword_has_evidence"
    }


def test_retired_label_contract_is_rejected_by_canonical_and_transport_schema(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _requirements()
    transport, _, _, _ = build_codex_analysis_transport_schema(
        extracted,
        requirements,
    )
    legacy_response = {
        **_compliant_response(),
        "matched_requirements": ["model-authored Python label"],
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(legacy_response, transport)
    with pytest.raises(ModelError, match="Additional properties"):
        validate_payload(
            legacy_response,
            "codex_analysis.schema.json",
            label="retired synthetic response",
        )


def test_compliant_requirement_ids_reach_stubbed_approval_boundary(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _requirements()
    response = _compliant_response()
    transport, evidence_ids, editable_ids, requirement_ids = (
        build_codex_analysis_transport_schema(extracted, requirements)
    )

    jsonschema.validate(response, transport)
    validate_payload(
        response,
        "codex_analysis.schema.json",
        label="compliant synthetic response",
    )
    resolved, issues = resolve_analysis_evidence(
        response,
        extracted,
        requirements,
    )
    assert issues == []
    supported_ids = {
        item["requirement_id"]
        for item in response["supported_requirement_mappings"]
    }
    unsupported_ids = set(response["unsupported_requirement_ids"])
    assert supported_ids.isdisjoint(unsupported_ids)
    assert supported_ids | unsupported_ids == set(requirement_ids)
    assert all(
        source_id in evidence_ids
        for item in response["supported_requirement_mappings"]
        for source_id in item["evidence_source_ids"]
    )
    assert all(
        edit["target_source_id"] in editable_ids
        for edit in response["recommended_edits"]
    )
    assert resolved["unsupported_ats_keywords"] == [
        "RAG",
        "Kubernetes",
        "rule-based link scoring",
    ]
    assert resolved["questions_for_user"] == []

    requests = []
    downstream_calls: list[str] = []
    hooks = PipelineHooks(
        approval_handler=lambda request: (
            requests.append(request) or ApprovalResponse("approve")
        )
    )
    approval = hooks.approve(
        kind="codex_analysis",
        title="Codex analysis",
        payload=resolved,
        assume_yes=False,
    )
    assert approval.action == "approve"
    assert [request.kind for request in requests] == ["codex_analysis"]
    assert downstream_calls == []
