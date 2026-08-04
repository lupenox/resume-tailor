"""Synthetic tests for minimal one-shot revision scope (initial-baseline)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.docx_extract import extract_resume
from resume_tailor.revision import build_revision_prompt, validate_revision_scope
from resume_tailor.utilities import RevisionValidationError


def _skeleton(master_resume: Path) -> dict[str, Any]:
    extracted, _ = extract_resume(master_resume)
    return copy.deepcopy(extracted["content"])


def _analysis_for(*content_ids: str) -> dict[str, Any]:
    return {
        "recommended_edits": [
            {
                "target_source_id": content_id,
                "operation": "replace",
                "proposed_text": "synthetic",
                "evidence_source_ids": [content_id],
            }
            for content_id in content_ids
        ],
        "immutable_facts": [],
        "forbidden_claims": [],
    }


def _qa_issue(
    *,
    issue_id: str,
    content_id: str,
    description: str,
    objective: str,
    category: str = "unsupported_wording",
    action: str = "remove_unsupported_claim",
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "category": category,
        "severity": "medium",
        "description": description,
        "affected_content_id": content_id,
        "evidence_source_ids": [content_id],
        "correction_action": action,
        "correction_objective": objective,
    }


def test_prompt_states_initial_baseline_not_master(master_resume: Path) -> None:
    extracted, _ = extract_resume(master_resume)
    content = extracted["content"]
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description="Clarity issue in the summary.",
                objective="Improve clarity without adding facts.",
                category="clarity",
                action="improve_clarity",
            )
        ],
        "technical_failure": None,
    }
    prompt = build_revision_prompt(
        current_tailored_content=content,
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa,
        company="Synthetic Corp",
        role="Engineer",
    )
    assert "REVISION BASELINE" in prompt
    assert "not a replacement template" in prompt.casefold() or "EVIDENCE ONLY" in prompt
    assert "minimal wording" in prompt.casefold() or "minimal wording changes" in prompt


def test_expert_phrase_fixed_while_aspiration_preserved(master_resume: Path) -> None:
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    initial["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Expert in Python/C backend systems, Docker, "
        "and cloud pipelines. Seeking AI Solutions Engineering roles to build "
        "scalable, production-ready AI infrastructure."
    )
    revised = copy.deepcopy(initial)
    revised["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Combining AI software engineering with a "
        "strong Linux, Docker, C/Python backend, and cloud pipeline foundation. "
        "Seeking AI Solutions Engineering roles to build scalable, production-ready "
        "AI infrastructure."
    )
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description=(
                    'Professional summary upgrades the master resume’s “strong … '
                    "foundation” framing of Python/C, Docker, and cloud pipeline "
                    'skills to an unsupported “Expert in” proficiency claim.'
                ),
                objective=(
                    "Restore skill-level wording that does not exceed the master "
                    "resume’s supported strength framing for backend, Docker, and "
                    "cloud pipeline capabilities."
                ),
            )
        ],
        "technical_failure": None,
    }
    issue_map = validate_revision_scope(
        initial_content=initial,
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"professional_summary": ["qa.001"]}
    assert "Seeking AI Solutions Engineering roles" in revised["professional_summary"]
    assert "Expert in" not in revised["professional_summary"]


def test_whole_field_master_restore_rejected_when_only_one_clause_flagged(
    master_resume: Path,
) -> None:
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    initial["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Expert in Python/C backend systems, Docker, "
        "and cloud pipelines. Seeking AI Solutions Engineering roles to build "
        "scalable, production-ready AI infrastructure."
    )
    revised = copy.deepcopy(initial)
    # Restores master aspiration + foundation wording — overbroad.
    revised["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Combining AI software engineering with a "
        "strong Linux, Docker, C/Python backend, and cloud pipeline foundation. "
        "Seeking AI Engineering, Applied AI, or Agentic AI internships."
    )
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description=(
                    'Unsupported “Expert in” proficiency claim for Python/C, Docker, '
                    "and cloud pipeline skills."
                ),
                objective=(
                    "Restore supported foundation wording without rewriting unrelated "
                    "clauses."
                ),
            )
        ],
        "technical_failure": None,
    }
    with pytest.raises(RevisionValidationError, match="revision_scope_violation") as exc:
        validate_revision_scope(
            initial_content=initial,
            revised_content=revised,
            qa_result=qa,
            approved_analysis=analysis,
        )
    diagnostic = getattr(exc.value, "diagnostic", {})
    assert diagnostic.get("code") == "revision_scope_violation"
    assert diagnostic.get("content_id") == "professional_summary"
    assert "qa.001" in diagnostic.get("issue_ids", [])
    assert "initial_sha256" in diagnostic and "revised_sha256" in diagnostic
    # Accepted initial generation remains the caller's baseline; validator does
    # not mutate either structure.
    assert "Seeking AI Solutions Engineering roles" in initial["professional_summary"]


def test_project_factual_correction_may_restore_disputed_phrase(
    master_resume: Path,
) -> None:
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    # Keep surrounding unflagged clause; only the disputed phrase is wrong.
    initial["projects"][0]["bullets"][0] = (
        "Built connectors for broader corporate data from active portals with retries."
    )
    revised = copy.deepcopy(initial)
    revised["projects"][0]["bullets"][0] = (
        "Built connectors for corporate domains from LinkedIn URLs and active career "
        "portals with retries."
    )
    analysis = _analysis_for("projects.0.bullets.0")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.002",
                content_id="projects.0.bullets.0",
                description=(
                    'Bullet changes evidenced “corporate domains from LinkedIn URLs” '
                    'and “active career portals” to broader “corporate data” and '
                    '“active portals,” overstating extraction scope.'
                ),
                objective=(
                    "Restore the specific extraction target and portal type supported "
                    "by the master resume."
                ),
                category="factual_integrity",
                action="verify_factual_integrity",
            )
        ],
        "technical_failure": None,
    }
    issue_map = validate_revision_scope(
        initial_content=initial,
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"projects.0.bullets.0": ["qa.002"]}
    assert "with retries" in revised["projects"][0]["bullets"][0]


def test_unrelated_clause_in_same_bullet_preserved(master_resume: Path) -> None:
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    initial["projects"][0]["bullets"][1] = (
        "Calibrated endpointing for candidate pauses. Delivered seamless human-to-AI "
        "interaction across sessions."
    )
    revised = copy.deepcopy(initial)
    # Fixes only the unsupported seamless phrase; keeps endpointing clause.
    revised["projects"][0]["bullets"][1] = (
        "Calibrated endpointing for candidate pauses. Kept turn-taking responsive "
        "during candidate pauses."
    )
    analysis = _analysis_for("projects.0.bullets.1")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.003",
                content_id="projects.0.bullets.1",
                description=(
                    'Bullet replaces concrete endpointing calibration for candidate '
                    'pauses with unsupported qualitative wording about “seamless '
                    'human-to-AI interaction.”'
                ),
                objective=(
                    "Remove unsupported seamless-interaction framing and keep "
                    "endpointing outcome language grounded in candidate pauses."
                ),
            )
        ],
        "technical_failure": None,
    }
    issue_map = validate_revision_scope(
        initial_content=initial,
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"projects.0.bullets.1": ["qa.003"]}
    assert "Calibrated endpointing for candidate pauses" in revised["projects"][0][
        "bullets"
    ][1]
    assert "seamless human-to-AI" not in revised["projects"][0]["bullets"][1]


def test_multiple_issues_allow_union_of_correction_scope(master_resume: Path) -> None:
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    initial["professional_summary"] = (
        "Clause A about systems. Expert in Python backends. Seeking AI Solutions "
        "Engineering roles. Extra unsupported seamless human-to-AI claim."
    )
    revised = copy.deepcopy(initial)
    revised["professional_summary"] = (
        "Clause A about systems. Strong Python backend foundation. Seeking AI "
        "Solutions Engineering roles. Pause-aware endpointing for candidates."
    )
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description='Unsupported “Expert in” proficiency claim for Python backends.',
                objective="Restore foundation-level wording for Python backends.",
            ),
            _qa_issue(
                issue_id="qa.002",
                content_id="professional_summary",
                description=(
                    'Unsupported “seamless human-to-AI” qualitative claim in the '
                    "summary."
                ),
                objective="Remove seamless human-to-AI wording.",
            ),
        ],
        "technical_failure": None,
    }
    issue_map = validate_revision_scope(
        initial_content=initial,
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"professional_summary": ["qa.001", "qa.002"]}
    assert "Seeking AI Solutions Engineering roles" in revised["professional_summary"]
    assert "Clause A about systems" in revised["professional_summary"]


def test_revision_baseline_is_initial_not_master(master_resume: Path) -> None:
    content = _skeleton(master_resume)
    master_summary = content["professional_summary"]
    initial = copy.deepcopy(content)
    # Initial already differs from master with a safe, unflagged aspiration.
    initial["professional_summary"] = (
        f"{master_summary.rstrip('.')} Seeking AI Solutions Engineering roles."
    )
    revised = copy.deepcopy(initial)
    # Valid minimal clarity tweak that keeps the aspiration.
    first, sep, rest = revised["professional_summary"].partition(" ")
    revised["professional_summary"] = (
        f"{first}, {rest}" if sep else revised["professional_summary"]
    )
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description="The summary needs a bounded clarity correction.",
                objective="Improve clarity without adding facts.",
                category="clarity",
                action="improve_clarity",
            )
        ],
        "technical_failure": None,
    }
    issue_map = validate_revision_scope(
        initial_content=initial,
        revised_content=revised,
        qa_result=qa,
        approved_analysis=analysis,
    )
    assert issue_map == {"professional_summary": ["qa.001"]}
    assert "Seeking AI Solutions Engineering roles" in revised["professional_summary"]
    # Must not have silently replaced with pure master text.
    assert revised["professional_summary"] != master_summary


def test_preserved_live_summary_geometry(master_resume: Path) -> None:
    """Preserved Baker Tilly geometry: foundation wording + AI Solutions aspiration."""
    content = _skeleton(master_resume)
    initial = copy.deepcopy(content)
    initial["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Expert in Python/C backend systems, Docker, "
        "and cloud pipelines. Seeking AI Solutions Engineering roles to build "
        "scalable, production-ready AI infrastructure."
    )
    good = copy.deepcopy(initial)
    good["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Combining AI software engineering with a "
        "strong Linux, Docker, C/Python backend, and cloud pipeline foundation. "
        "Seeking AI Solutions Engineering roles to build scalable, production-ready "
        "AI infrastructure."
    )
    bad = copy.deepcopy(initial)
    bad["professional_summary"] = (
        "UW-Milwaukee CS senior building agentic workflows, real-time voice agents, "
        "and local-first AI applications. Combining AI software engineering with a "
        "strong Linux, Docker, C/Python backend, and cloud pipeline foundation. "
        "Seeking AI Engineering, Applied AI, or Agentic AI internships."
    )
    analysis = _analysis_for("professional_summary")
    qa = {
        "status": "material_findings",
        "summary": "x",
        "issues": [
            _qa_issue(
                issue_id="qa.001",
                content_id="professional_summary",
                description=(
                    'Professional summary upgrades the master resume’s “strong … '
                    "foundation” framing of Python/C, Docker, and cloud pipeline "
                    'skills to an unsupported “Expert in” proficiency claim.'
                ),
                objective=(
                    "Restore skill-level wording that does not exceed the master "
                    "resume’s supported strength framing for backend, Docker, and "
                    "cloud pipeline capabilities."
                ),
            )
        ],
        "technical_failure": None,
    }
    assert validate_revision_scope(
        initial_content=initial,
        revised_content=good,
        qa_result=qa,
        approved_analysis=analysis,
    ) == {"professional_summary": ["qa.001"]}
    with pytest.raises(RevisionValidationError, match="revision_scope_violation"):
        validate_revision_scope(
            initial_content=initial,
            revised_content=bad,
            qa_result=qa,
            approved_analysis=analysis,
        )
    assert "AI Solutions Engineering" in good["professional_summary"]
    assert "strong" in good["professional_summary"].casefold()
    assert "foundation" in good["professional_summary"].casefold()
