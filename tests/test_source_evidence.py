from __future__ import annotations

import copy
from typing import Any

import pytest

from resume_tailor.backend.documents.docx_extract import source_blocks_from_paragraphs
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.utils.schemas import validate_payload
from resume_tailor.backend.utils.utilities import ModelError


SUMMARY_TEXT = "Built Python—based validation,\nwith strict evidence and exact IDs."
SKILL_TEXT = "Languages: Python, JavaScript, SQL; validation: JSON Schema"


def _paragraph(
    index: int,
    content_id: str,
    text: str,
    *,
    is_list: bool = False,
) -> dict[str, Any]:
    return {
        "index": index,
        "content_id": content_id,
        "text": text,
        "is_list": is_list,
        "content_budget": {
            "original_characters": len(text),
            "maximum_characters": len(text) + 8,
            "original_words": len(text.split()),
        },
    }


def _extracted() -> dict[str, Any]:
    paragraphs = [
        _paragraph(0, "section.objective_summary", "OBJECTIVE / SUMMARY"),
        _paragraph(1, "professional_summary", SUMMARY_TEXT),
        _paragraph(2, "section.technical_skills", "TECHNICAL SKILLS"),
        _paragraph(3, "skill_groups.0", SKILL_TEXT, is_list=True),
    ]
    return {
        "source": {"filename": "synthetic.docx", "sha256": "a" * 64},
        "paragraphs": paragraphs,
        "source_blocks": source_blocks_from_paragraphs(paragraphs),
        "content": {},
    }


def _requirements() -> dict[str, Any]:
    return build_job_requirement_catalog(
        "Synthetic confirmed posting.",
        structured_job={
            "responsibilities": [
                "Build trustworthy validation systems with strict evidence.",
                "Operate distributed systems at production scale.",
            ],
            "required_qualifications": ["Python", "JSON Schema"],
            "technologies_and_skills": [
                "Python",
                "JSON Schema",
                "Kubernetes",
                "rule-based link scoring",
                "Python-based validation, with strict evidence",
            ],
        },
    )


def _analysis() -> dict[str, Any]:
    supported_ids = {
        "responsibility.001": ["professional_summary"],
        "required.001": ["skill_groups.0"],
        "required.002": ["skill_groups.0"],
        "skill.001": ["skill_groups.0"],
        "skill.002": ["skill_groups.0"],
        "skill.005": ["professional_summary"],
    }
    all_ids = [
        item["requirement_id"] for item in _requirements()["requirements"]
    ]
    return {
        "role_summary": "Synthetic role.",
        "fit_assessment": {
            "overall": "Synthetic fit.",
            "strengths": ["Locally cited validation work"],
            "gaps": ["Unsupported production experience"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": requirement_id,
                "evidence_source_ids": evidence_ids,
                "strength": "strong" if requirement_id != "responsibility.001" else "partial",
            }
            for requirement_id, evidence_ids in supported_ids.items()
        ],
        "unsupported_requirement_ids": [
            requirement_id
            for requirement_id in all_ids
            if requirement_id not in supported_ids
        ],
        "recommended_edits": [
            {
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": "Evidence-gated synthetic engineering profile.",
                "alignment_rationale": "Surface locally supported validation work.",
                "evidence_source_ids": ["professional_summary", "skill_groups.0"],
            }
        ],
        "immutable_facts": [],
        "forbidden_claims": ["Unsupported production scale"],
        "content_budget_guidance": [
            {
                "target_source_id": "professional_summary",
                "guidance": "Remain within the local paragraph budget.",
            }
        ],
        "questions_for_user": [],
    }


@pytest.mark.parametrize(
    "bad_id",
    [
        "OBJECTIVE / SUMMARY",
        "paragraph 3 contains the supporting claim",
        "professional_summary | skill_groups.0",
        "Evidence-gated synthetic engineering profile.",
        "source.does_not_exist",
    ],
)
def test_model_prose_cannot_replace_a_source_id(bad_id: str) -> None:
    analysis = _analysis()
    analysis["recommended_edits"][0]["evidence_source_ids"] = [bad_id]

    _, issues = resolve_analysis_evidence(analysis, _extracted(), _requirements())

    assert "unknown_source_id" in {issue.code for issue in issues}


def test_missing_duplicate_and_inappropriate_source_ids_are_rejected() -> None:
    analysis = _analysis()
    analysis["supported_requirement_mappings"][0]["evidence_source_ids"] = []
    analysis["recommended_edits"][0]["evidence_source_ids"] = [
        "skill_groups.0",
        "skill_groups.0",
        "section.technical_skills",
    ]

    _, issues = resolve_analysis_evidence(analysis, _extracted(), _requirements())
    codes = {issue.code for issue in issues}

    assert {"missing_source_ids", "duplicate_source_id", "inappropriate_evidence_source"} <= codes


def test_duplicate_and_inappropriate_edit_targets_are_rejected_separately() -> None:
    analysis = _analysis()
    analysis["recommended_edits"].append(
        copy.deepcopy(analysis["recommended_edits"][0])
    )
    analysis["content_budget_guidance"].append(
        {
            "target_source_id": "section.objective_summary",
            "guidance": "Synthetic invalid target.",
        }
    )

    _, issues = resolve_analysis_evidence(analysis, _extracted(), _requirements())
    codes = {issue.code for issue in issues}

    assert "duplicate_target_source_id" in codes
    assert "inappropriate_target_source_id" in codes
    assert not any(issue.location.startswith("supported_requirement_mappings") for issue in issues)


def test_model_authored_labels_existing_text_and_copied_evidence_are_not_fields() -> None:
    analysis = _analysis()
    analysis["matched_requirements"] = ["Python"]
    analysis["recommended_edits"][0]["existing_text"] = "Invented existing text"
    analysis["recommended_edits"][0]["exact_supporting_evidence"] = "Invented quotation"

    with pytest.raises(ModelError, match="Additional properties"):
        validate_payload(
            analysis,
            "codex_analysis.schema.json",
            label="synthetic invalid analysis",
        )


def test_valid_ids_resolve_exact_local_text_and_reach_review_shape() -> None:
    resolved, issues = resolve_analysis_evidence(
        _analysis(),
        _extracted(),
        _requirements(),
    )

    assert issues == []
    edit = resolved["recommended_edits"][0]
    assert edit["existing_text"] == SUMMARY_TEXT
    assert edit["resume_section"] == "OBJECTIVE / SUMMARY"
    assert [item["source_id"] for item in edit["resolved_evidence"]] == [
        "professional_summary",
        "skill_groups.0",
    ]
    mapping = next(
        item
        for item in resolved["requirement_assessment"]
        if item["requirement_id"] == "responsibility.001"
    )
    assert mapping["requirement"] == (
        "Build trustworthy validation systems with strict evidence."
    )
    assert mapping["resolved_evidence"][0]["exact_text"] == SUMMARY_TEXT
    assert mapping["status"] == "supported_by_source"
    assert mapping["support_provenance"] == (
        "model_assessed_human_review_required"
    )


def test_requirement_id_contract_rejects_unknown_duplicate_overlap_and_omission() -> None:
    analysis = _analysis()
    analysis["supported_requirement_mappings"].append(
        {
            "requirement_id": "requirement.unknown",
            "evidence_source_ids": ["skill_groups.0"],
            "strength": "weak",
        }
    )
    analysis["supported_requirement_mappings"].append(
        copy.deepcopy(analysis["supported_requirement_mappings"][0])
    )
    overlap = analysis["supported_requirement_mappings"][1]["requirement_id"]
    analysis["unsupported_requirement_ids"].append(overlap)
    analysis["unsupported_requirement_ids"].remove("skill.004")

    _, issues = resolve_analysis_evidence(analysis, _extracted(), _requirements())
    codes = {issue.code for issue in issues}

    assert {
        "unknown_requirement_id",
        "duplicate_requirement_mapping",
        "unsupported_requirement_has_evidence",
        "missing_requirement_classification",
    } <= codes


def test_unsupported_requirements_have_no_evidence_and_ats_status_is_local() -> None:
    resolved, issues = resolve_analysis_evidence(
        _analysis(),
        _extracted(),
        _requirements(),
    )

    assert issues == []
    statuses = {
        item["keyword"]: item["status"]
        for item in resolved["ats_keyword_assessment"]
    }
    assert statuses["Python"] == "present_verbatim"
    assert statuses["JSON Schema"] == "present_verbatim"
    assert statuses["Kubernetes"] == "unsupported"
    assert statuses["rule-based link scoring"] == "unsupported"
    assert statuses["Python-based validation, with strict evidence"] == (
        "present_verbatim"
    )
    unsupported = {
        item["requirement_id"]: item
        for item in resolved["requirement_assessment"]
        if item["status"] == "unsupported"
    }
    assert all(item["evidence_source_ids"] == [] for item in unsupported.values())
    assert all(item["resolved_evidence"] == [] for item in unsupported.values())


def test_punctuation_difference_does_not_become_local_verbatim_proof() -> None:
    requirements = build_job_requirement_catalog(
        "Synthetic punctuation test.",
        structured_job={
            "technologies_and_skills": [
                "Python-based validation with strict evidence"
            ]
        },
    )
    analysis = _analysis()
    analysis["supported_requirement_mappings"] = [
        {
            "requirement_id": "skill.001",
            "evidence_source_ids": ["professional_summary"],
            "strength": "partial",
        }
    ]
    analysis["unsupported_requirement_ids"] = []

    resolved, issues = resolve_analysis_evidence(analysis, _extracted(), requirements)

    assert issues == []
    assessment = resolved["requirement_assessment"][0]
    assert assessment["status"] == "supported_by_source"
    assert assessment["support_provenance"] == (
        "model_assessed_human_review_required"
    )


def test_absent_experience_questions_are_forbidden_but_contradictions_are_allowed() -> None:
    analysis = _analysis()
    analysis["questions_for_user"] = [
        "Do you have Kubernetes experience that is not listed in the résumé?",
        "Could you describe any unlisted production deployment experience?",
        (
            "The supplied source contains conflicting experience dates; "
            "which date is authoritative?"
        ),
    ]

    _, issues = resolve_analysis_evidence(analysis, _extracted(), _requirements())

    forbidden = [
        issue for issue in issues if issue.code == "forbidden_absent_experience_question"
    ]
    assert len(forbidden) == 2


def test_edit_evidence_remains_authoritative_with_typography_differences() -> None:
    analysis = _analysis()
    analysis["recommended_edits"][0]["proposed_text"] = (
        "Built Python-based validation with strict evidence."
    )

    resolved, issues = resolve_analysis_evidence(
        analysis,
        _extracted(),
        _requirements(),
    )

    assert issues == []
    assert resolved["recommended_edits"][0]["existing_text"] == SUMMARY_TEXT
    assert resolved["recommended_edits"][0]["resolved_evidence"][0][
        "exact_text"
    ] == SUMMARY_TEXT
