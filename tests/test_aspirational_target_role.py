"""Synthetic tests for aspirational vs factual target-role classification.

No providers, network calls, or private content are used. The preserved Baker
Tilly live geometry is reconstructed from the public failure description only.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import build_content_diff, validate_tailored_content
from resume_tailor.job_requirements import build_job_requirement_catalog


TARGET_ROLE = "AI Solutions Engineer"


def _base_analysis(master: dict[str, Any], *, approve_summary: bool = True) -> dict[str, Any]:
    edits: list[dict[str, Any]] = []
    if approve_summary:
        edits.append(
            {
                "edit_id": "edit.summary",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": master["professional_summary"],
                "evidence_source_ids": ["professional_summary"],
            }
        )
    return {
        "recommended_edits": edits,
        "supported_ats_keywords": [],
        "forbidden_claims": [],
    }


def _report(
    master_resume: Path,
    *,
    source_summary: str | None = None,
    tailored_summary: str | None = None,
    mutate_tailored: Any = None,
    target_role: str = TARGET_ROLE,
    approve_summary: bool = True,
    approve_extra: list[str] | None = None,
    summary_budget: int | None = None,
):
    extracted, _ = extract_resume(master_resume)
    if summary_budget is not None:
        for paragraph in extracted.get("paragraphs", []):
            if paragraph.get("content_id") == "professional_summary":
                paragraph["content_budget"]["maximum_characters"] = summary_budget
    original = copy.deepcopy(extracted["content"])
    if source_summary is not None:
        original["professional_summary"] = source_summary
    tailored = copy.deepcopy(original)
    if tailored_summary is not None:
        tailored["professional_summary"] = tailored_summary
    if mutate_tailored is not None:
        mutate_tailored(tailored)

    analysis = _base_analysis(original, approve_summary=approve_summary)
    for content_id in approve_extra or []:
        analysis["recommended_edits"].append(
            {
                "edit_id": f"edit.{content_id}",
                "target_source_id": content_id,
                "operation": "replace",
                "proposed_text": "synthetic",
                "evidence_source_ids": [content_id],
            }
        )
    # Re-sync approved proposed_text for summary to the tailored value when changed.
    for edit in analysis["recommended_edits"]:
        if edit.get("target_source_id") == "professional_summary":
            edit["proposed_text"] = tailored["professional_summary"]

    return validate_tailored_content(
        original=original,
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role=target_role,
    ), original, tailored


def test_seeking_ai_engineering_to_ai_solutions_engineering_roles_allowed(
    master_resume: Path,
) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking AI Engineering roles in applied systems.",
        tailored_summary="Seeking AI Solutions Engineering roles in applied systems.",
    )
    assert report.passed
    assert TARGET_ROLE in report.allowed_aspirational_role_references
    assert TARGET_ROLE not in report.introduced_role_labels
    assert not any("Target role label" in issue for issue in report.issues)


def test_seeking_software_roles_to_ai_solutions_engineer_positions_allowed(
    master_resume: Path,
) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="Seeking AI Solutions Engineer positions with production ownership.",
    )
    assert report.passed
    assert TARGET_ROLE in report.allowed_aspirational_role_references
    assert TARGET_ROLE not in report.introduced_role_labels


def test_experienced_ai_solutions_engineer_blocked(master_resume: Path) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="Experienced AI Solutions Engineer building production systems.",
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert TARGET_ROLE not in report.allowed_aspirational_role_references
    assert any("Target role label" in issue for issue in report.issues)


def test_role_with_five_years_experience_blocked(master_resume: Path) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="AI Solutions Engineer with five years of experience in backends.",
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert any("Target role label" in issue for issue in report.issues)


def test_experience_title_changed_to_target_role_blocked(master_resume: Path) -> None:
    def mutate(tailored: dict[str, Any]) -> None:
        tailored["experience"]["role"] = TARGET_ROLE

    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="Seeking software roles with production ownership.",
        mutate_tailored=mutate,
        approve_summary=False,
        approve_extra=["experience.heading"],
    )
    assert not report.passed
    # Immutable experience.role change and/or target-role claim must block.
    assert any(
        "Immutable field changed at experience.role" in issue
        or "Target role label" in issue
        for issue in report.issues
    )
    assert TARGET_ROLE not in report.allowed_aspirational_role_references


def test_worked_as_target_role_blocked(master_resume: Path) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="Worked as an AI Solutions Engineer on production systems.",
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert any("Target role label" in issue for issue in report.issues)


def test_recruiter_seeking_phrase_not_candidate_aspiration(master_resume: Path) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary=(
            "Seeking an AI Solutions Engineer to join our team building platforms."
        ),
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert TARGET_ROLE not in report.allowed_aspirational_role_references


def test_target_company_plus_claimed_employment_blocked(master_resume: Path) -> None:
    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="AI Solutions Engineer at Baker Tilly building AI platforms.",
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert any("Target role label" in issue for issue in report.issues)


def test_aspiration_not_allowed_in_experience_bullet(master_resume: Path) -> None:
    def mutate(tailored: dict[str, Any]) -> None:
        tailored["experience"]["bullets"][0] = (
            "Seeking AI Solutions Engineering roles while supporting bar operations."
        )

    report, _original, _tailored = _report(
        master_resume,
        source_summary="Seeking software roles with production ownership.",
        tailored_summary="Seeking software roles with production ownership.",
        mutate_tailored=mutate,
        approve_summary=False,
        approve_extra=["experience.bullets.0"],
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
    assert TARGET_ROLE not in report.allowed_aspirational_role_references


def test_content_diff_does_not_list_allowed_aspiration_as_blocking_role(
    master_resume: Path,
) -> None:
    report, original, tailored = _report(
        master_resume,
        source_summary="Seeking AI Engineering roles in applied systems.",
        tailored_summary="Seeking AI Solutions Engineering roles in applied systems.",
    )
    assert report.passed
    diff = build_content_diff(original, tailored, report)
    assert "### Allowed aspirational role references" in diff
    assert f"- {TARGET_ROLE}" in diff
    # Must not appear under newly introduced role labels as a blocking claim.
    section = diff.split("### Newly introduced role labels", 1)[1].split(
        "### Allowed aspirational role references", 1
    )[0]
    assert TARGET_ROLE not in section
    blocking = diff.split("### Blocking evidence issues", 1)[1]
    assert "Target role label" not in blocking


def test_preserved_live_geometry_passes_step_7_synthetically(
    master_resume: Path,
) -> None:
    """Reconstruct the Baker Tilly summary geometry that previously blocked Step 7."""
    source = (
        "UW–Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Combining AI software engineering with a "
        "strong Linux, Docker, C/Python backend, and cloud pipeline foundation. "
        "Seeking AI Engineering, Applied AI, or Agentic AI internships."
    )
    proposal = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Expert in Python/C backend systems, Docker, "
        "and cloud pipelines. Seeking AI Solutions Engineering roles to build "
        "scalable, production-ready AI infrastructure."
    )
    report, original, tailored = _report(
        master_resume,
        source_summary=source,
        tailored_summary=proposal,
        target_role="AI Solutions Engineer",
        # Live geometry is longer than the synthetic fixture summary budget;
        # expand only the local budget so this test isolates role classification.
        summary_budget=400,
    )
    assert report.passed, report.issues
    assert "AI Solutions Engineer" in report.allowed_aspirational_role_references
    assert "AI Solutions Engineer" not in report.introduced_role_labels
    assert not any("Target role label" in issue for issue in report.issues)

    diff = build_content_diff(original, tailored, report)
    assert "BLOCKED" not in diff.split("## Section-by-section changes", 1)[0]
    assert "Allowed aspirational role references" in diff


def test_aspiration_without_source_marker_blocked(master_resume: Path) -> None:
    """Proposal aspiration alone is insufficient without source continuity."""
    report, _original, _tailored = _report(
        master_resume,
        source_summary=(
            "UW–Milwaukee CS senior building agentic workflows and local-first AI."
        ),
        tailored_summary=(
            "UW–Milwaukee CS senior building agentic workflows. "
            "Seeking AI Solutions Engineering roles."
        ),
    )
    assert not report.passed
    assert TARGET_ROLE in report.introduced_role_labels
