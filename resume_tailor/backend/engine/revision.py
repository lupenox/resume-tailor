from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from resume_tailor.backend.providers.antigravity_writer import (
    _invoke_antigravity_candidate,
    _write_response_metadata,
    approved_edit_catalog,
)
from resume_tailor.backend.engine.character_budget import CHARACTER_COUNTING_CONTRACT
from resume_tailor.backend.engine.evidence import changed_content_ids, content_values
from resume_tailor.backend.utils.schemas import validate_payload
from resume_tailor.backend.utils.utilities import (
    AntigravityRevisionCannotApplyError,
    AntigravityRevisionContractError,
    AntigravityRevisionTechnicalFailureError,
    RevisionValidationError,
    ModelError,
    require_executable,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./_-]*")
_QUOTE_RE = re.compile(r"[\"“”‘']([^\"“”‘']{3,120})[\"“”‘']")
_ASPIRATION_CLAUSE_RE = re.compile(
    r"\b(?:seeking|targeting|pursuing|interested\s+in|applying\s+for|"
    r"aiming\s+to\s+transition\s+into)\b[^.!?\n]*[.!?]?",
    re.I,
)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "that",
        "this",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "on",
        "as",
        "at",
        "it",
        "its",
        "not",
        "no",
        "so",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "than",
        "then",
        "into",
        "over",
        "under",
        "about",
        "without",
        "within",
        "using",
        "used",
        "use",
        "can",
        "may",
        "must",
        "should",
        "will",
        "would",
        "could",
        "resume",
        "summary",
        "professional",
        "master",
        "source",
        "wording",
        "claim",
        "claims",
        "framing",
        "supported",
        "unsupported",
        "restore",
        "remove",
        "keep",
        "grounded",
    }
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
    provider_name: str = "Antigravity",
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
    provider_label = provider_name.strip() or "The writer"
    provider_token = provider_label.upper().replace(" ", "_")
    return f"""Revise the already-authored resume now. Do not plan, ask for more
information, invoke tools, call another agent, or modify any file. Return exactly
one strict structured result matching the supplied JSON schema.

This is revision attempt 1 of 1. Never request or produce a second revision.
The CURRENT tailored content below is the revision baseline — not the master
résumé. The master/source catalog is evidence only; do not restore whole master
fields unless every clause is implicated by the QA findings.

Apply only the minimal wording changes required to resolve the supplied QA
issue IDs and their correction objectives. Preserve every unflagged clause,
sentence, and unrelated initial-tailoring change exactly. An issue without an
authorized target cannot permit a wording change. If an issue cannot be
corrected within these evidence and edit boundaries, return cannot_apply with
that issue_id and one bounded reason code.

TARGET
Company: {company}
Role: {role}

CURRENT {provider_token}-AUTHORED CONTENT (REVISION BASELINE)
BEGIN_CURRENT_TAILORED_CONTENT
{json.dumps(current_tailored_content, ensure_ascii=False, indent=2)}
END_CURRENT_TAILORED_CONTENT

ORIGINAL IMMUTABLE RESUME SOURCE CATALOG (EVIDENCE ONLY — NOT A REPLACEMENT TEMPLATE)
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

CHARACTER COUNTING CONTRACT
{CHARACTER_COUNTING_CONTRACT}

LOCALLY VALIDATED QA ISSUE CATALOG
BEGIN_QA_ISSUES
{json.dumps(qa_result['issues'], ensure_ascii=False, indent=2)}
END_QA_ISSUES

REVISION TARGET AUTHORIZATION
{json.dumps(target_map, ensure_ascii=False, indent=2)}

NON-NEGOTIABLE RULES
- {provider_label} is the sole author. Revise the resume content now, not a plan.
- Baseline is CURRENT tailored content. Never wholesale-restore master text.
- Address only authenticated QA issue IDs; change only the implicated clauses.
- Preserve unflagged clauses in the same field (including approved aspirational
  targeting such as Seeking … roles) when the issue concerns other wording.
- Change only targets present in REVISION TARGET AUTHORIZATION.
- Keep every revision within the original approved edit and evidence boundaries.
- Do not introduce facts, technologies, metrics, credentials, seniority,
  employment, education, availability, accomplishments, or customer impact.
- Do not change contact information, dates, links, section structure, project
  count, bullet count, labels, names, employers, institutions, or template geometry.
- Preserve source-supported technologies and all immutable numeric claims.
- Respect every content budget under the supplied character-counting contract.
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


def _normalize_compare_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("–", "-").replace("—", "-")).strip()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _significant_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _WORD_RE.finditer(text):
        token = match.group(0).casefold()
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _token_overlap_ratio(span: str, reference: str) -> float:
    span_tokens = _significant_tokens(span)
    if not span_tokens:
        return 0.0
    ref_tokens = _significant_tokens(reference)
    if not ref_tokens:
        return 0.0
    return len(span_tokens & ref_tokens) / len(span_tokens)


def _split_clauses(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = _SENTENCE_SPLIT_RE.split(stripped)
    clauses = [part.strip() for part in parts if part and part.strip()]
    return clauses if clauses else [stripped]


def _issue_blob(issue: dict[str, Any]) -> str:
    return " ".join(
        str(issue.get(key) or "")
        for key in ("description", "correction_objective", "category", "correction_action")
    )


def _quoted_anchors(issue: dict[str, Any]) -> list[str]:
    blob = _issue_blob(issue)
    return [match.group(1).strip() for match in _QUOTE_RE.finditer(blob) if match.group(1).strip()]


def _clause_implicated_by_issue(clause: str, issue: dict[str, Any]) -> bool:
    """Return True when the QA issue clearly concerns this clause."""
    clause_n = _normalize_compare_text(clause)
    if not clause_n:
        return False
    blob = _issue_blob(issue)
    blob_n = _normalize_compare_text(blob)
    for quoted in _quoted_anchors(issue):
        quoted_n = _normalize_compare_text(quoted)
        if len(quoted_n) >= 3 and (
            quoted_n in clause_n or clause_n in quoted_n
        ):
            return True
    # Distinctive multi-word markers often appear in both the disputed clause
    # and the issue description (e.g. "expert in", "seamless human-to-ai").
    for length in (4, 3, 2):
        words = clause_n.split()
        for index in range(0, max(0, len(words) - length + 1)):
            phrase = " ".join(words[index : index + length])
            if any(token in _STOPWORDS for token in phrase.split()):
                # Still allow phrases containing stopwords if the phrase is in blob.
                pass
            if len(phrase) >= 8 and phrase in blob_n:
                return True
    if _token_overlap_ratio(clause, blob) >= 0.45:
        return True
    return False


def _issues_for_content_id(
    qa_result: dict[str, Any],
    content_id: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for issue in qa_result.get("issues", []):
        if (
            isinstance(issue, dict)
            and issue.get("affected_content_id") == content_id
        ):
            issues.append(issue)
    return issues


def _clause_preserved(clause: str, revised_text: str) -> bool:
    clause_n = _normalize_compare_text(clause)
    revised_n = _normalize_compare_text(revised_text)
    if not clause_n:
        return True
    if clause_n in revised_n:
        return True
    # Allow trivial whitespace/punctuation drift via high similarity.
    ratio = difflib.SequenceMatcher(None, clause_n, revised_n).ratio()
    if ratio >= 0.98:
        return True
    # Also accept if a close sentence exists in the revision.
    for revised_clause in _split_clauses(revised_text):
        if (
            difflib.SequenceMatcher(
                None,
                clause_n,
                _normalize_compare_text(revised_clause),
            ).ratio()
            >= 0.92
        ):
            return True
    return False


def _deleted_spans(initial: str, revised: str) -> list[str]:
    matcher = difflib.SequenceMatcher(
        None,
        _normalize_compare_text(initial),
        _normalize_compare_text(revised),
    )
    deleted: list[str] = []
    initial_n = _normalize_compare_text(initial)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"} and i2 > i1:
            span = initial_n[i1:i2].strip()
            if len(span) >= 12:
                deleted.append(span)
    return deleted


def _span_implicated(span: str, issues: list[dict[str, Any]]) -> bool:
    if not span.strip():
        return True
    for issue in issues:
        if _clause_implicated_by_issue(span, issue):
            return True
        blob_n = _normalize_compare_text(_issue_blob(issue))
        span_n = _normalize_compare_text(span)
        if span_n in blob_n or any(
            _normalize_compare_text(q) in span_n for q in _quoted_anchors(issue)
        ):
            return True
        if _token_overlap_ratio(span, _issue_blob(issue)) >= 0.35:
            return True
    return False


def _raise_scope_violation(
    *,
    content_id: str,
    issue_ids: list[str],
    initial_text: str,
    revised_text: str,
    detail: str,
) -> None:
    diagnostic = {
        "code": "revision_scope_violation",
        "detail": detail,
        "content_id": content_id,
        "issue_ids": list(issue_ids),
        "initial_sha256": _text_sha256(initial_text),
        "revised_sha256": _text_sha256(revised_text),
    }
    message = (
        "revision_scope_violation: Revision changed content outside the approved "
        f"QA correction objective (content_id={content_id}; "
        f"issues={','.join(issue_ids) or 'none'}; "
        f"initial_sha256={diagnostic['initial_sha256'][:16]}…; "
        f"revised_sha256={diagnostic['revised_sha256'][:16]}…)."
    )
    error = RevisionValidationError(message)
    setattr(error, "diagnostic", diagnostic)
    raise error


def _validate_minimal_field_correction(
    *,
    content_id: str,
    initial_text: str,
    revised_text: str,
    issues: list[dict[str, Any]],
    issue_ids: list[str],
) -> None:
    """Reject overbroad rewrites that touch clauses not implicated by QA issues."""
    if _normalize_compare_text(initial_text) == _normalize_compare_text(revised_text):
        return

    clauses = _split_clauses(initial_text)
    implicated_indices: set[int] = set()
    for index, clause in enumerate(clauses):
        if any(_clause_implicated_by_issue(clause, issue) for issue in issues):
            implicated_indices.add(index)

    # Deterministic preservation: unflagged clauses from the initial tailored
    # baseline must survive (including aspirational targeting clauses).
    preserved_unflagged: list[str] = []
    for index, clause in enumerate(clauses):
        if index in implicated_indices:
            continue
        is_aspiration = _ASPIRATION_CLAUSE_RE.search(clause) is not None
        preserved = _clause_preserved(clause, revised_text)
        if implicated_indices and not preserved:
            _raise_scope_violation(
                content_id=content_id,
                issue_ids=issue_ids,
                initial_text=initial_text,
                revised_text=revised_text,
                detail=(
                    "Unflagged clause from the initial tailored baseline was removed "
                    "or rewritten outside the QA correction objective."
                    + (" (aspirational targeting)" if is_aspiration else "")
                ),
            )
        if preserved:
            preserved_unflagged.append(clause)
        if (
            not implicated_indices
            and is_aspiration
            and not preserved
            and not any(_clause_implicated_by_issue(clause, issue) for issue in issues)
        ):
            # Even when issue anchors are vague, never drop an initial aspiration
            # clause unless the issue text clearly targets it.
            _raise_scope_violation(
                content_id=content_id,
                issue_ids=issue_ids,
                initial_text=initial_text,
                revised_text=revised_text,
                detail=(
                    "Approved aspirational targeting clause was removed although "
                    "the QA issue did not implicate it."
                ),
            )

    # Span-level guard for multi-clause fields: large deletions must be covered
    # by issue anchors. Ignore SequenceMatcher fragments that belong to clauses
    # we already verified are preserved (alignment artifacts).
    if len(clauses) > 1 and implicated_indices:
        preserved_norms = [
            _normalize_compare_text(clause) for clause in preserved_unflagged
        ]
        for span in _deleted_spans(initial_text, revised_text):
            span_n = _normalize_compare_text(span)
            if any(span_n in clause_n or clause_n in span_n for clause_n in preserved_norms):
                continue
            # Fragments of implicated clauses may be rewritten freely.
            implicated_norms = [
                _normalize_compare_text(clauses[i]) for i in implicated_indices
            ]
            if any(
                span_n in clause_n or clause_n in span_n for clause_n in implicated_norms
            ):
                continue
            if not _span_implicated(span, issues):
                _raise_scope_violation(
                    content_id=content_id,
                    issue_ids=issue_ids,
                    initial_text=initial_text,
                    revised_text=revised_text,
                    detail=(
                        "Revision deleted or replaced initial-tailoring content that "
                        "is not implicated by the QA issue description/objective."
                    ),
                )


def validate_revision_scope(
    *,
    initial_content: dict[str, Any],
    revised_content: dict[str, Any],
    qa_result: dict[str, Any],
    approved_analysis: dict[str, Any],
) -> dict[str, list[str]]:
    """Ensure revision changes only authorized targets with minimal clause scope.

    Baseline is the initial tailored content (not the master résumé). Whole-field
    restoration of master wording is rejected when unflagged initial clauses are
    discarded.
    """
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
            "The writer returned complete without changing an authorized QA target."
        )
    unauthorized = [content_id for content_id in changed if content_id not in target_map]
    if unauthorized:
        raise RevisionValidationError(
            "The writer changed content outside the authenticated QA target set: "
            + ", ".join(unauthorized)
        )

    initial_values = content_values(initial_content)
    revised_values = content_values(revised_content)
    for content_id in changed:
        issue_ids = list(target_map[content_id])
        issues = _issues_for_content_id(qa_result, content_id)
        # Fall back to issues listed in the target map order when payloads omit
        # matching affected_content_id (should not happen for normal QA).
        if not issues:
            for issue in qa_result.get("issues", []):
                if (
                    isinstance(issue, dict)
                    and issue.get("issue_id") in issue_ids
                ):
                    issues.append(issue)
        initial_text = initial_values.get(content_id, "")
        revised_text = revised_values.get(content_id, "")
        if not isinstance(initial_text, str) or not isinstance(revised_text, str):
            continue
        _validate_minimal_field_correction(
            content_id=content_id,
            initial_text=initial_text,
            revised_text=revised_text,
            issues=issues,
            issue_ids=issue_ids,
        )

    return {content_id: target_map[content_id] for content_id in changed}


def build_revision_diff(
    *,
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
        "## Initial writer output versus revision 1",
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
