from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .antigravity_writer import (
    _invoke_antigravity_candidate,
    _write_response_metadata,
    approved_edit_catalog,
)
from .evidence import changed_content_ids, content_values
from .schemas import validate_payload
from .utilities import (
    AntigravityRevisionCannotApplyError,
    AntigravityRevisionContractError,
    AntigravityRevisionTechnicalFailureError,
    RevisionValidationError,
    ModelError,
    require_executable,
)


REVISION_RESPONSE_FILENAME = "antigravity-revision-response.json"
REVISION_RESPONSE_METADATA_FILENAME = "antigravity-revision-response-envelope.json"
REVISION_SCHEMA_NAME = "antigravity_revision.schema.json"


def approved_revision_targets(
    *,
    qa_result: dict[str, Any],
    approved_analysis: dict[str, Any],
) -> dict[str, list[str]]:
    """Map each locally authenticated target to the QA issues that permit it."""
    approved = {
        edit.get("target_source_id")
        for edit in approved_analysis.get("recommended_edits", [])
        if isinstance(edit, dict) and isinstance(edit.get("target_source_id"), str)
    }
    target_map: dict[str, list[str]] = {}
    for issue in qa_result.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_id = issue.get("issue_id")
        content_id = issue.get("affected_content_id")
        if (
            isinstance(issue_id, str)
            and isinstance(content_id, str)
            and content_id in approved
        ):
            target_map.setdefault(content_id, []).append(issue_id)
    return target_map


def build_revision_prompt(
    *,
    current_tailored_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    approved_analysis: dict[str, Any],
    qa_result: dict[str, Any],
    company: str,
    role: str,
) -> str:
    target_map = approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=approved_analysis,
    )
    budgets = [
        {
            "content_id": paragraph["content_id"],
            **paragraph["content_budget"],
        }
        for paragraph in extracted_resume["paragraphs"]
    ]
    return f"""Revise the already-authored resume now. Do not plan, ask for more
information, invoke tools, call another agent, or modify any file. Return exactly
one strict structured result matching the supplied JSON schema.

This is revision attempt 1 of 1. Never request or produce a second revision.
Apply only corrections needed for the supplied QA issue IDs. Preserve every
content value not identified in REVISION TARGET AUTHORIZATION. An issue without
an authorized target cannot permit a wording change. If an issue cannot be
corrected within these evidence and edit boundaries, return cannot_apply with
that issue_id and one bounded reason code.

TARGET
Company: {company}
Role: {role}

CURRENT ANTIGRAVITY-AUTHORED CONTENT
BEGIN_CURRENT_TAILORED_CONTENT
{json.dumps(current_tailored_content, ensure_ascii=False, indent=2)}
END_CURRENT_TAILORED_CONTENT

ORIGINAL IMMUTABLE RESUME SOURCE CATALOG
BEGIN_IMMUTABLE_SOURCE_CATALOG
{json.dumps(extracted_resume['source_blocks'], ensure_ascii=False, indent=2)}
END_IMMUTABLE_SOURCE_CATALOG

ORIGINAL APPROVED EDIT CATALOG
BEGIN_APPROVED_EDIT_CATALOG
{json.dumps(approved_edit_catalog(approved_analysis), ensure_ascii=False, indent=2)}
END_APPROVED_EDIT_CATALOG

IMMUTABLE FACTS
{json.dumps(approved_analysis['immutable_facts'], ensure_ascii=False, indent=2)}

FORBIDDEN CLAIMS
{json.dumps(approved_analysis['forbidden_claims'], ensure_ascii=False, indent=2)}

CONTENT BUDGETS
{json.dumps(budgets, ensure_ascii=False, indent=2)}

LOCALLY VALIDATED CODEX QA ISSUE CATALOG
BEGIN_QA_ISSUES
{json.dumps(qa_result['issues'], ensure_ascii=False, indent=2)}
END_QA_ISSUES

REVISION TARGET AUTHORIZATION
{json.dumps(target_map, ensure_ascii=False, indent=2)}

NON-NEGOTIABLE RULES
- Antigravity is the sole author. Revise the resume content now, not a plan.
- Address only authenticated QA issue IDs and preserve all other content exactly.
- Change only targets present in REVISION TARGET AUTHORIZATION.
- Keep every revision within the original approved edit and evidence boundaries.
- Do not introduce facts, technologies, metrics, credentials, seniority,
  employment, education, availability, accomplishments, or customer impact.
- Do not change contact information, dates, links, section structure, project
  count, bullet count, labels, names, employers, institutions, or template geometry.
- Preserve source-supported technologies and all immutable numeric claims.
- Respect every content budget.
- Return complete with the full revised resume only if the bounded corrections
  can be made safely. Otherwise return cannot_apply for one supplied issue ID.
- Use technical_failure only for a genuine execution or output failure.
"""


def invoke_antigravity_revision(
    *,
    current_tailored_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    approved_analysis: dict[str, Any],
    qa_result: dict[str, Any],
    company: str,
    role: str,
    run_directory: Path,
    timeout_seconds: int,
    antigravity_duration: str,
    attempt_number: int,
    executable: str | None = None,
) -> dict[str, Any]:
    if attempt_number != 1:
        raise RevisionValidationError(
            "Exactly one Antigravity revision attempt is permitted."
        )
    issue_ids = {
        issue["issue_id"]
        for issue in qa_result.get("issues", [])
        if isinstance(issue, dict) and isinstance(issue.get("issue_id"), str)
    }
    if qa_result.get("status") != "material_findings" or not issue_ids:
        raise RevisionValidationError(
            "An Antigravity revision requires authenticated material QA findings."
        )
    agy = executable or require_executable("agy")
    prompt = build_revision_prompt(
        current_tailored_content=current_tailored_content,
        extracted_resume=extracted_resume,
        approved_analysis=approved_analysis,
        qa_result=qa_result,
        company=company,
        role=role,
    )
    candidate, response_path = _invoke_antigravity_candidate(
        executable=agy,
        prompt=prompt,
        prompt_label="Antigravity revision prompt",
        schema_name=REVISION_SCHEMA_NAME,
        response_filename=REVISION_RESPONSE_FILENAME,
        metadata_filename=REVISION_RESPONSE_METADATA_FILENAME,
        run_directory=run_directory,
        timeout_seconds=timeout_seconds,
        antigravity_duration=antigravity_duration,
    )
    try:
        validate_payload(
            candidate.payload,
            REVISION_SCHEMA_NAME,
            label="Antigravity revision output",
        )
    except ModelError as exc:
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type=candidate.envelope_type,
            validation_result="REJECTED",
            schema_name=REVISION_SCHEMA_NAME,
            metadata_filename=REVISION_RESPONSE_METADATA_FILENAME,
        )
        raise AntigravityRevisionContractError(
            "Antigravity violated the one-shot revision response contract. "
            "Provider content was omitted from the exception."
        ) from exc

    payload = candidate.payload
    if payload["status"] == "cannot_apply":
        detail = payload["cannot_apply"]
        if detail["issue_id"] not in issue_ids:
            raise AntigravityRevisionContractError(
                "Antigravity returned an unknown QA issue ID in cannot_apply."
            )
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type=candidate.envelope_type,
            validation_result="REJECTED",
            schema_name=REVISION_SCHEMA_NAME,
            metadata_filename=REVISION_RESPONSE_METADATA_FILENAME,
        )
        raise AntigravityRevisionCannotApplyError(
            "Antigravity could not apply the bounded correction for "
            f"{detail['issue_id']} ({detail['reason_code']}). Provider prose was "
            "omitted; no second revision is permitted."
        )
    if payload["status"] == "technical_failure":
        detail = payload["technical_failure"]
        _write_response_metadata(
            run_directory=run_directory,
            response_path=response_path,
            envelope_type=candidate.envelope_type,
            validation_result="REJECTED",
            schema_name=REVISION_SCHEMA_NAME,
            metadata_filename=REVISION_RESPONSE_METADATA_FILENAME,
        )
        raise AntigravityRevisionTechnicalFailureError(
            "Antigravity revision reported technical failure "
            f"{detail['reason_code']}. Provider prose was omitted; no second "
            "revision is permitted."
        )
    _write_response_metadata(
        run_directory=run_directory,
        response_path=response_path,
        envelope_type=candidate.envelope_type,
        validation_result="PASS",
        schema_name=REVISION_SCHEMA_NAME,
        metadata_filename=REVISION_RESPONSE_METADATA_FILENAME,
    )
    return payload["tailored_resume"]


def validate_revision_scope(
    *,
    initial_content: dict[str, Any],
    revised_content: dict[str, Any],
    qa_result: dict[str, Any],
    approved_analysis: dict[str, Any],
) -> dict[str, list[str]]:
    target_map = approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=approved_analysis,
    )
    try:
        changed = changed_content_ids(initial_content, revised_content)
    except (KeyError, TypeError, ValueError) as exc:
        raise RevisionValidationError(
            "The revised resume structure cannot be compared with the initial output."
        ) from exc
    if not changed:
        raise RevisionValidationError(
            "Antigravity returned complete without changing an authorized QA target."
        )
    unauthorized = [content_id for content_id in changed if content_id not in target_map]
    if unauthorized:
        raise RevisionValidationError(
            "Antigravity changed content outside the authenticated QA target set: "
            + ", ".join(unauthorized)
        )
    return {content_id: target_map[content_id] for content_id in changed}


def build_revision_diff(
    *,
    master_content: dict[str, Any],
    initial_content: dict[str, Any],
    revised_content: dict[str, Any],
    issue_map: dict[str, list[str]],
    master_to_revision_diff: str,
) -> str:
    initial_values = content_values(initial_content)
    revised_values = content_values(revised_content)
    lines = [
        "# Revision 1 Content Diff",
        "",
        "## Master versus revision 1",
        "",
        master_to_revision_diff.rstrip(),
        "",
        "## Initial Antigravity output versus revision 1",
        "",
    ]
    for content_id, issue_ids in issue_map.items():
        lines.extend(
            [
                f"### {content_id}",
                "",
                f"QA issues: {', '.join(issue_ids)}",
                "",
            ]
        )
        diff = difflib.unified_diff(
            [initial_values[content_id]],
            [revised_values[content_id]],
            fromfile="initial",
            tofile="revision-1",
            lineterm="",
        )
        lines.extend(f"    {line}" for line in diff)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
