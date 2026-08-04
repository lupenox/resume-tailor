from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from .character_budget import (
    canonicalize_budget_text,
    composite_label_for_source_id,
    compose_rendered_text,
    count_budget_characters,
    mutable_text_from_composite_proposal,
)
from .utilities import flatten_strings, normalized_text


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?%?|\d+\s*[–—-]\s*\d+(?:\.\d+)?%?)(?![\w.])"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|myself|we|our|ours|ourselves)\b", re.I)
_SENIORITY_RE = re.compile(
    r"\b(?:senior|staff|principal|lead|manager|director|architect|owner|founder)\b",
    re.I,
)
_AVAILABILITY_RE = re.compile(
    r"\b(?:available immediately|immediate availability|available to start|"
    r"start date|open to relocation|willing to relocate)\b",
    re.I,
)
_HIGH_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("RAG", re.compile(r"\bRAG\b|\bretrieval[- ]augmented generation\b", re.I)),
    ("GraphQL", re.compile(r"\bGraphQL\b", re.I)),
    ("observability", re.compile(r"\bobservability\b", re.I)),
    (
        "distributed production scale",
        re.compile(r"\bdistributed\b.{0,40}\bproduction\b|\bproduction[- ]scale\b", re.I),
    ),
    ("IVR platforms", re.compile(r"\bIVR\b|\binteractive voice response\b", re.I)),
    ("Kubernetes", re.compile(r"\bKubernetes\b|\bk8s\b", re.I)),
    ("vector databases", re.compile(r"\bvector database\b|\bPinecone\b|\bWeaviate\b|\bChromaDB?\b", re.I)),
    ("LangChain/LlamaIndex", re.compile(r"\bLangChain\b|\bLlamaIndex\b", re.I)),
    ("Kafka", re.compile(r"\bApache Kafka\b|\bKafka\b", re.I)),
    ("Redis", re.compile(r"\bRedis\b", re.I)),
    ("gRPC", re.compile(r"\bgRPC\b", re.I)),
    ("OpenTelemetry", re.compile(r"\bOpenTelemetry\b", re.I)),
    ("Prometheus/Grafana", re.compile(r"\bPrometheus\b|\bGrafana\b", re.I)),
)

_DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)

_ABSENT_EXPERIENCE_QUESTION_RE = re.compile(
    r"(?:\b(?:do|did|have|can|could|would)\s+you\b.{0,160}"
    r"\b(?:experience|skill|technology|tool|project|metric|certification|"
    r"domain|worked|used|built|implemented|deployed|managed|led)\b)|"
    r"(?:\b(?:please|can|could)\s+(?:you\s+)?(?:confirm|describe|provide|"
    r"share|list)\b.{0,160}\b(?:experience|skill|technology|tool|project|"
    r"metric|certification|domain)\b)|"
    r"(?:\b(?:unlisted|not (?:shown|listed|included|present|represented)|"
    r"absent from (?:the )?r[eé]sum[eé])\b)",
    re.I,
)
_SOURCE_CONTRADICTION_QUESTION_RE = re.compile(
    r"\b(?:conflict(?:ing)?|contradict(?:ion|ory)?|inconsisten(?:t|cy)|"
    r"disagree|two different|which (?:date|value|version).{0,80}authoritative)\b",
    re.I,
)


@dataclass(frozen=True)
class SourceEvidenceIssue:
    code: str
    location: str
    detail: str

    def describe(self) -> str:
        return f"{self.location}: {self.detail} ({self.code})."


@dataclass
class EvidenceReport:
    issues: list[str] = field(default_factory=list)
    introduced_technologies: list[str] = field(default_factory=list)
    introduced_metrics: list[str] = field(default_factory=list)
    introduced_role_labels: list[str] = field(default_factory=list)
    introduced_availability: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


def validate_analysis_evidence(
    analysis: dict[str, Any],
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
) -> list[str]:
    """Compatibility wrapper returning safe source-reference issue summaries."""
    _, issues = resolve_analysis_evidence(
        analysis,
        extracted_resume,
        job_requirements,
    )
    return [issue.describe() for issue in issues]


def _source_reference(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": block["source_id"],
        "section_context": block["section_context"],
        "exact_text": block["exact_text"],
    }


def _source_catalog(
    extracted_resume: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[SourceEvidenceIssue]]:
    from .docx_extract import source_blocks_from_paragraphs

    supplied = extracted_resume.get("source_blocks")
    blocks = (
        copy.deepcopy(supplied)
        if isinstance(supplied, list)
        else source_blocks_from_paragraphs(extracted_resume["paragraphs"])
    )
    index: dict[str, dict[str, Any]] = {}
    issues: list[SourceEvidenceIssue] = []
    for position, block in enumerate(blocks):
        location = f"source_blocks[{position}]"
        if not isinstance(block, dict):
            issues.append(
                SourceEvidenceIssue(
                    "invalid_source_object",
                    location,
                    "the local extraction contains a non-object source block",
                )
            )
            continue
        source_id = block.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            issues.append(
                SourceEvidenceIssue(
                    "missing_source_id",
                    location,
                    "the local extraction contains a source block without an ID",
                )
            )
            continue
        if source_id in index:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_source_id",
                    location,
                    f"the local extraction repeats source ID {source_id!r}",
                )
            )
            continue
        if not isinstance(block.get("exact_text"), str):
            issues.append(
                SourceEvidenceIssue(
                    "missing_source_text",
                    location,
                    f"local source ID {source_id!r} has no exact text",
                )
            )
            continue
        index[source_id] = block
    return blocks, index, issues


def _resolve_source_ids(
    values: Any,
    *,
    location: str,
    source_index: dict[str, dict[str, Any]],
    issues: list[SourceEvidenceIssue],
    required: bool,
    evidence_only: bool,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        issues.append(
            SourceEvidenceIssue(
                "missing_source_ids",
                location,
                "Codex did not provide a source-ID array",
            )
        )
        return []
    if required and not values:
        issues.append(
            SourceEvidenceIssue(
                "missing_source_ids",
                location,
                "at least one source ID is required",
            )
        )
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for position, value in enumerate(values):
        item_location = f"{location}[{position}]"
        if not isinstance(value, str) or not value.strip():
            issues.append(
                SourceEvidenceIssue(
                    "missing_source_id",
                    item_location,
                    "Codex supplied an empty source ID",
                )
            )
            continue
        if value in seen:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_source_id",
                    item_location,
                    "Codex repeated a source ID",
                )
            )
            continue
        seen.add(value)
        block = source_index.get(value)
        if block is None:
            issues.append(
                SourceEvidenceIssue(
                    "unknown_source_id",
                    item_location,
                    "Codex referenced an unknown source ID",
                )
            )
            continue
        if evidence_only and block.get("evidence_allowed") is not True:
            issues.append(
                SourceEvidenceIssue(
                    "inappropriate_evidence_source",
                    item_location,
                    "Codex referenced document context instead of claim evidence",
                )
            )
            continue
        resolved.append(block)
    return resolved


def _typographic_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).translate(_DASH_TRANSLATION)
    normalized = re.sub(r"\s*-\s*", "-", normalized.casefold())
    return " ".join(normalized.split())


def _requirement_support_status(
    requirement: str,
    cited_blocks: list[dict[str, Any]],
) -> str:
    """Derive claim support from cited local text without fuzzy matching."""
    if requirement and any(
        requirement in str(block["exact_text"]) for block in cited_blocks
    ):
        return "present_verbatim"

    normalized_requirement = _typographic_phrase(requirement)
    if normalized_requirement and any(
        normalized_requirement in _typographic_phrase(str(block["exact_text"]))
        for block in cited_blocks
    ):
        return "supported_by_source"
    return "unsupported"


def _present_verbatim(
    requirement: str,
    cited_blocks: list[dict[str, Any]],
) -> bool:
    """Conservatively match case, Unicode dashes, whitespace, and line breaks."""
    normalized_requirement = _typographic_phrase(requirement)
    return bool(normalized_requirement) and any(
        normalized_requirement in _typographic_phrase(str(block["exact_text"]))
        for block in cited_blocks
    )


def _keyword_status(
    keyword: str,
    *,
    eligible_blocks: list[dict[str, Any]],
    cited_blocks: list[dict[str, Any]],
) -> str:
    if any(keyword in str(block["exact_text"]) for block in eligible_blocks):
        return "present_verbatim"
    normalized_keyword = _typographic_phrase(keyword)
    if normalized_keyword and any(
        normalized_keyword in _typographic_phrase(str(block["exact_text"]))
        for block in cited_blocks
    ):
        return "supported_by_source"
    return "unsupported"


def diagnose_legacy_analysis_evidence(
    analysis: dict[str, Any],
    extracted_resume: dict[str, Any],
) -> list[SourceEvidenceIssue]:
    """Replay the retired label-based ATS contract without accepting it live."""
    blocks, source_index, issues = _source_catalog(extracted_resume)
    eligible_blocks = [
        block for block in blocks if block.get("evidence_allowed") is True
    ]
    for position, item in enumerate(analysis.get("ats_keywords", [])):
        if not isinstance(item, dict):
            continue
        location = f"ats_keywords[{position}].evidence_source_ids"
        cited = _resolve_source_ids(
            item.get("evidence_source_ids"),
            location=location,
            source_index=source_index,
            issues=issues,
            required=False,
            evidence_only=True,
        )
        keyword = str(item.get("keyword", ""))
        status = _keyword_status(
            keyword,
            eligible_blocks=eligible_blocks,
            cited_blocks=cited,
        )
        if status == "unsupported" and cited:
            issues.append(
                SourceEvidenceIssue(
                    "unsupported_ats_keyword_has_evidence",
                    location,
                    "an unsupported ATS term was assigned unrelated source evidence",
                )
            )
    return list(
        {
            (issue.code, issue.location, issue.detail): issue
            for issue in issues
        }.values()
    )


def resolve_analysis_evidence(
    analysis: dict[str, Any],
    extracted_resume: dict[str, Any],
    job_requirements: dict[str, Any],
) -> tuple[dict[str, Any], list[SourceEvidenceIssue]]:
    """Resolve both ID catalogs locally; model-authored labels are never authority."""
    from .job_requirements import job_requirement_index

    _, source_index, issues = _source_catalog(extracted_resume)
    requirement_index = job_requirement_index(job_requirements)
    resolved = copy.deepcopy(analysis)
    supported_by_id: dict[str, dict[str, Any]] = {}
    resolved_mappings: list[dict[str, Any]] = []
    for position, item in enumerate(
        analysis.get("supported_requirement_mappings", [])
    ):
        if not isinstance(item, dict):
            issues.append(
                SourceEvidenceIssue(
                    "invalid_requirement_mapping",
                    f"supported_requirement_mappings[{position}]",
                    "Codex returned a non-object requirement mapping",
                )
            )
            continue
        requirement_id = item.get("requirement_id")
        id_location = (
            f"supported_requirement_mappings[{position}].requirement_id"
        )
        requirement = (
            requirement_index.get(requirement_id)
            if isinstance(requirement_id, str)
            else None
        )
        if requirement is None:
            issues.append(
                SourceEvidenceIssue(
                    "unknown_requirement_id",
                    id_location,
                    "Codex referenced an unknown job requirement ID",
                )
            )
        elif requirement_id in supported_by_id:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_requirement_mapping",
                    id_location,
                    "Codex mapped the same job requirement more than once",
                )
            )

        evidence_location = (
            f"supported_requirement_mappings[{position}].evidence_source_ids"
        )
        cited = _resolve_source_ids(
            item.get("evidence_source_ids"),
            location=evidence_location,
            source_index=source_index,
            issues=issues,
            required=True,
            evidence_only=True,
        )
        local_item = copy.deepcopy(item)
        local_item["resolved_evidence"] = [
            _source_reference(block) for block in cited
        ]
        if requirement is not None:
            status = (
                "present_verbatim"
                if _present_verbatim(requirement["exact_text"], cited)
                else "supported_by_source"
            )
            local_item.update(
                {
                    "requirement": requirement["exact_text"],
                    "category": requirement["category"],
                    "support_status": status,
                    "support_provenance": (
                        "local_exact_phrase"
                        if status == "present_verbatim"
                        else "model_assessed_human_review_required"
                    ),
                }
            )
            if requirement_id not in supported_by_id:
                supported_by_id[requirement_id] = local_item
        resolved_mappings.append(local_item)

    unsupported_ids: set[str] = set()
    for position, requirement_id in enumerate(
        analysis.get("unsupported_requirement_ids", [])
    ):
        location = f"unsupported_requirement_ids[{position}]"
        if not isinstance(requirement_id, str) or requirement_id not in requirement_index:
            issues.append(
                SourceEvidenceIssue(
                    "unknown_requirement_id",
                    location,
                    "Codex referenced an unknown unsupported job requirement ID",
                )
            )
            continue
        if requirement_id in unsupported_ids:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_unsupported_requirement",
                    location,
                    "Codex repeated an unsupported job requirement ID",
                )
            )
        unsupported_ids.add(requirement_id)
        if requirement_id in supported_by_id:
            issues.append(
                SourceEvidenceIssue(
                    "unsupported_requirement_has_evidence",
                    location,
                    "an unsupported requirement also has a supported evidence mapping",
                )
            )

    classified_ids = set(supported_by_id) | unsupported_ids
    for requirement_id in requirement_index:
        if requirement_id not in classified_ids:
            issues.append(
                SourceEvidenceIssue(
                    "missing_requirement_classification",
                    f"job_requirements[{requirement_id}]",
                    "Codex omitted this job requirement from both classifications",
                )
            )

    requirement_assessment: list[dict[str, Any]] = []
    keyword_assessment: list[dict[str, Any]] = []
    for requirement_id, requirement in requirement_index.items():
        mapping = supported_by_id.get(requirement_id)
        if mapping is not None and requirement_id not in unsupported_ids:
            assessment = {
                "requirement_id": requirement_id,
                "requirement": requirement["exact_text"],
                "category": requirement["category"],
                "status": mapping["support_status"],
                "support_provenance": mapping["support_provenance"],
                "strength": mapping.get("strength"),
                "evidence_source_ids": list(mapping.get("evidence_source_ids", [])),
                "resolved_evidence": list(mapping.get("resolved_evidence", [])),
            }
        else:
            assessment = {
                "requirement_id": requirement_id,
                "requirement": requirement["exact_text"],
                "category": requirement["category"],
                "status": "unsupported",
                "support_provenance": "local_unsupported_classification",
                "strength": None,
                "evidence_source_ids": [],
                "resolved_evidence": [],
            }
        requirement_assessment.append(assessment)
        if requirement["category"] in {"technology_and_skill", "ai_focus_area"}:
            keyword_assessment.append(
                {
                    "requirement_id": requirement_id,
                    "keyword": requirement["exact_text"],
                    "status": assessment["status"],
                    "support_provenance": assessment["support_provenance"],
                    "evidence_source_ids": assessment["evidence_source_ids"],
                    "resolved_evidence": assessment["resolved_evidence"],
                }
            )

    resolved["supported_requirement_mappings"] = resolved_mappings
    resolved["requirement_assessment"] = requirement_assessment
    resolved["matched_requirements"] = [
        item["requirement"]
        for item in requirement_assessment
        if item["status"] != "unsupported"
    ]
    resolved["evidence_map"] = [
        {
            "requirement_id": item["requirement_id"],
            "requirement": item["requirement"],
            "evidence_source_ids": item["evidence_source_ids"],
            "strength": item["strength"],
            "support_status": item["status"],
            "support_provenance": item["support_provenance"],
            "resolved_evidence": item["resolved_evidence"],
        }
        for item in requirement_assessment
        if item["status"] != "unsupported"
    ]
    resolved["missing_or_unsupported_requirements"] = [
        item["requirement"]
        for item in requirement_assessment
        if item["status"] == "unsupported"
    ]
    resolved["ats_keywords"] = [
        {
            "keyword": item["keyword"],
            "evidence_source_ids": item["evidence_source_ids"],
        }
        for item in keyword_assessment
    ]
    resolved["ats_keyword_assessment"] = keyword_assessment
    resolved["supported_ats_keywords"] = [
        item["keyword"]
        for item in keyword_assessment
        if item["status"] in {"present_verbatim", "supported_by_source"}
    ]
    resolved["unsupported_ats_keywords"] = [
        item["keyword"]
        for item in keyword_assessment
        if item["status"] == "unsupported"
    ]

    for position, question in enumerate(analysis.get("questions_for_user", [])):
        if isinstance(question, str) and _ABSENT_EXPERIENCE_QUESTION_RE.search(
            question
        ) and not _SOURCE_CONTRADICTION_QUESTION_RE.search(question):
            issues.append(
                SourceEvidenceIssue(
                    "forbidden_absent_experience_question",
                    f"questions_for_user[{position}]",
                    "Codex asked the user to supply experience absent from the résumé",
                )
            )

    seen_targets: set[str] = set()
    resolved_edits: list[dict[str, Any]] = []
    discarded_no_op_edit_ids: list[str] = list(analysis.get("discarded_no_op_edit_ids", []))
    normalized_composite_edit_ids: list[str] = list(analysis.get("normalized_composite_edit_ids", []))
    invalid_composite_edit_ids: list[str] = list(analysis.get("invalid_composite_edit_ids", []))

    extracted_content = extracted_resume.get("content")
    if not isinstance(extracted_content, dict):
        extracted_content = {}

    for position, edit in enumerate(analysis.get("recommended_edits", [])):
        target_location = f"recommended_edits[{position}].target_source_id"
        target_id = edit.get("target_source_id")
        target: dict[str, Any] | None = None
        if not isinstance(target_id, str) or not target_id.strip():
            issues.append(
                SourceEvidenceIssue(
                    "missing_target_source_id",
                    target_location,
                    "Codex omitted the edit target source ID",
                )
            )
        elif target_id in seen_targets:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_target_source_id",
                    target_location,
                    "Codex proposed conflicting edits for one source block",
                )
            )
        else:
            seen_targets.add(target_id)
            target = source_index.get(target_id)
            if target is None:
                issues.append(
                    SourceEvidenceIssue(
                        "unknown_target_source_id",
                        target_location,
                        "Codex referenced an unknown edit target",
                    )
                )
            elif target.get("editable") is not True:
                issues.append(
                    SourceEvidenceIssue(
                        "inappropriate_target_source_id",
                        target_location,
                        "Codex referenced a non-editable résumé block",
                    )
                )

        cited = _resolve_source_ids(
            edit.get("evidence_source_ids"),
            location=f"recommended_edits[{position}].evidence_source_ids",
            source_index=source_index,
            issues=issues,
            required=True,
            evidence_only=True,
        )
        local_edit = copy.deepcopy(edit)
        local_edit["resolved_evidence"] = [_source_reference(block) for block in cited]
        if target is not None:
            target_text = target["exact_text"]
            proposed_text = local_edit.get("proposed_text")

            mutable_target_text = target_text
            immutable_label = composite_label_for_source_id(extracted_content, target_id)
            if immutable_label is not None:
                try:
                    mutable_target_text = mutable_text_from_composite_proposal(
                        target_text,
                        immutable_label=immutable_label,
                    )
                except ValueError:
                    pass

                if isinstance(proposed_text, str):
                    try:
                        mutable_text = mutable_text_from_composite_proposal(
                            proposed_text,
                            immutable_label=immutable_label,
                        )
                    except ValueError:
                        mutable_text = proposed_text

                    if mutable_text != proposed_text:
                        # Exact authenticated label was stripped once. A second
                        # matching wrapper is a duplicated/malformed label, not
                        # an ordinary prose edit.
                        try:
                            second_pass = mutable_text_from_composite_proposal(
                                mutable_text,
                                immutable_label=immutable_label,
                            )
                        except ValueError:
                            second_pass = mutable_text
                        if second_pass != mutable_text:
                            issues.append(
                                SourceEvidenceIssue(
                                    "invalid_composite_label",
                                    f"recommended_edits[{position}].proposed_text",
                                    "Codex proposed a changed or malformed immutable label",
                                )
                            )
                            edit_id_to_store = local_edit.get("edit_id")
                            if (
                                edit_id_to_store
                                and edit_id_to_store not in invalid_composite_edit_ids
                            ):
                                invalid_composite_edit_ids.append(edit_id_to_store)
                            continue
                        edit_id_to_store = local_edit.get("edit_id")
                        if (
                            edit_id_to_store
                            and edit_id_to_store not in normalized_composite_edit_ids
                        ):
                            normalized_composite_edit_ids.append(edit_id_to_store)
                        local_edit["proposed_text"] = mutable_text
                        proposed_text = mutable_text
                    else:
                        stripped = proposed_text.strip()
                        colon_idx = stripped.find(":")
                        if colon_idx != -1:
                            prefix = stripped[:colon_idx]
                            # Any non-authenticated label wrapper is rejected
                            # before approval so it cannot survive as prose.
                            if canonicalize_budget_text(prefix) != canonicalize_budget_text(
                                immutable_label
                            ):
                                issues.append(
                                    SourceEvidenceIssue(
                                        "invalid_composite_label",
                                        f"recommended_edits[{position}].proposed_text",
                                        "Codex proposed a changed or malformed immutable label",
                                    )
                                )
                                edit_id_to_store = local_edit.get("edit_id")
                                if (
                                    edit_id_to_store
                                    and edit_id_to_store not in invalid_composite_edit_ids
                                ):
                                    invalid_composite_edit_ids.append(edit_id_to_store)
                                continue
                        elif canonicalize_budget_text(stripped) in {
                            canonicalize_budget_text(immutable_label),
                            canonicalize_budget_text(str(target.get("section_context") or "")),
                        }:
                            # Bare label or section-heading text rewrites structure.
                            issues.append(
                                SourceEvidenceIssue(
                                    "invalid_composite_label",
                                    f"recommended_edits[{position}].proposed_text",
                                    "Codex proposed a changed or malformed immutable label",
                                )
                            )
                            edit_id_to_store = local_edit.get("edit_id")
                            if (
                                edit_id_to_store
                                and edit_id_to_store not in invalid_composite_edit_ids
                            ):
                                invalid_composite_edit_ids.append(edit_id_to_store)
                            continue

            # No-op elimination: compare mutable bodies only (never full rendered
            # value that still includes the immutable label).
            if isinstance(proposed_text, str) and canonicalize_budget_text(
                proposed_text
            ) == canonicalize_budget_text(mutable_target_text):
                edit_id_to_store = local_edit.get("edit_id")
                if edit_id_to_store and edit_id_to_store not in discarded_no_op_edit_ids:
                    discarded_no_op_edit_ids.append(edit_id_to_store)
                continue

            local_edit["existing_text"] = target_text
            local_edit["resume_section"] = target["section_context"]
        resolved_edits.append(local_edit)

    resolved["recommended_edits"] = resolved_edits
    resolved["discarded_no_op_edit_ids"] = discarded_no_op_edit_ids
    resolved["normalized_composite_edit_ids"] = normalized_composite_edit_ids
    resolved["invalid_composite_edit_ids"] = invalid_composite_edit_ids

    # Remove only evidence-map associations that name discarded edit IDs.
    # Do not strip target_source_id values still used as valid evidence.
    discarded_edit_ids = set(discarded_no_op_edit_ids)
    if discarded_edit_ids:
        for item in resolved["evidence_map"]:
            item["evidence_source_ids"] = [
                sid for sid in item["evidence_source_ids"]
                if sid not in discarded_edit_ids
            ]

        resolved["evidence_map"] = [
            item for item in resolved["evidence_map"]
            if item["evidence_source_ids"]
        ]

    budgets = {
        paragraph["content_id"]: paragraph["content_budget"]["maximum_characters"]
        for paragraph in extracted_resume["paragraphs"]
    }
    extracted_content = extracted_resume.get("content")
    if not isinstance(extracted_content, dict):
        extracted_content = {}
    for position, edit in enumerate(resolved_edits):
        target_id = edit.get("target_source_id")
        proposed_text = edit.get("proposed_text")
        if not isinstance(target_id, str) or not isinstance(proposed_text, str):
            continue
        immutable_label = composite_label_for_source_id(
            extracted_content,
            target_id,
        )
        maximum_characters = budgets.get(target_id)
        if immutable_label is None or not isinstance(maximum_characters, int):
            continue
        try:
            mutable_text = mutable_text_from_composite_proposal(
                proposed_text,
                immutable_label=immutable_label,
            )
        except (TypeError, ValueError):
            continue
        # A colon-bearing body whose prefix is not the authenticated label is
        # handled by the existing composite-label/grounding contract. Do not
        # let a derived budget issue mask that more specific failure class.
        if mutable_text == proposed_text and ":" in proposed_text:
            continue
        rendered_text = compose_rendered_text(
            mutable_text,
            immutable_label=immutable_label,
        )
        actual_characters = count_budget_characters(rendered_text)
        if actual_characters > maximum_characters:
            issues.append(
                SourceEvidenceIssue(
                    "structured_proposal_over_budget",
                    f"recommended_edits[{position}].proposed_text",
                    "Codex proposed a Python-owned structured value of "
                    f"{actual_characters} characters for a hard rendered "
                    f"maximum of {maximum_characters}",
                )
            )
    seen_budget_targets: set[str] = set()
    resolved_guidance: list[dict[str, Any]] = []
    for position, guidance in enumerate(analysis.get("content_budget_guidance", [])):
        target_id = guidance.get("target_source_id")
        location = f"content_budget_guidance[{position}].target_source_id"
        if not isinstance(target_id, str) or not target_id.strip():
            issues.append(
                SourceEvidenceIssue(
                    "missing_target_source_id",
                    location,
                    "Codex omitted the budget target source ID",
                )
            )
        elif target_id in seen_budget_targets:
            issues.append(
                SourceEvidenceIssue(
                    "duplicate_target_source_id",
                    location,
                    "Codex repeated budget guidance for one source block",
                )
            )
        else:
            seen_budget_targets.add(target_id)
            block = source_index.get(target_id)
            if block is None:
                issues.append(
                    SourceEvidenceIssue(
                        "unknown_target_source_id",
                        location,
                        "Codex referenced an unknown budget target",
                    )
                )
            elif block.get("editable") is not True or target_id not in budgets:
                issues.append(
                    SourceEvidenceIssue(
                        "inappropriate_target_source_id",
                        location,
                        "Codex referenced a block that cannot receive content guidance",
                    )
                )
        local_guidance = copy.deepcopy(guidance)
        if isinstance(target_id, str) and target_id in budgets:
            local_guidance["maximum_characters"] = budgets[target_id]
        resolved_guidance.append(local_guidance)
    resolved["content_budget_guidance"] = resolved_guidance

    unique_issues = list(
        {
            (issue.code, issue.location, issue.detail): issue
            for issue in issues
        }.values()
    )
    return resolved, unique_issues


def _resume_text(content: dict[str, Any]) -> str:
    return "\n".join(flatten_strings(content))


def _technology_items(content: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for index, group in enumerate(content["skill_groups"]):
        for item in re.split(r"\s*[,•]\s*", group["text"]):
            if item.strip():
                values.append((f"skill_groups.{index}", item.strip().rstrip(".")))
    for index, project in enumerate(content["projects"]):
        for item in re.split(r"\s*[,•]\s*", project["technologies"]):
            if item.strip():
                values.append((f"projects.{index}.technologies", item.strip().rstrip(".")))
    return values


def _paragraph_values(content: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {
        "professional_summary": content["professional_summary"],
        "education.degree": (
            f"{content['education']['institution']} | "
            f"{content['education']['degree_details']}"
        ),
        "education.coursework": (
            f"{content['education']['coursework']['label']}: "
            f"{content['education']['coursework']['text']}"
        ),
        "education.certifications": (
            f"{content['education']['certifications']['label']}: "
            f"{content['education']['certifications']['text']}"
        ),
        "open_source.heading": (
            f"{content['open_source']['name']} | "
            f"{content['open_source']['technologies']}"
        ),
        "open_source.bullet": content["open_source"]["bullet"],
        "experience.heading": (
            f"{content['experience']['role']} | "
            f"{content['experience']['employer_location']} "
            f"({content['experience']['dates']})"
        ),
    }
    for index, group in enumerate(content["skill_groups"]):
        values[f"skill_groups.{index}"] = f"{group['label']}: {group['text']}"
    for project_index, project in enumerate(content["projects"]):
        values[f"projects.{project_index}.heading"] = (
            f"{project['name']} | {project['technologies']}"
        )
        for bullet_index, bullet in enumerate(project["bullets"]):
            values[f"projects.{project_index}.bullets.{bullet_index}"] = bullet
    for bullet_index, bullet in enumerate(content["experience"]["bullets"]):
        values[f"experience.bullets.{bullet_index}"] = bullet
    return values


def changed_content_ids(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Return stable logical content IDs whose exact rendered text changed."""
    before_values = _paragraph_values(before)
    after_values = _paragraph_values(after)
    changed: list[str] = []
    for content_id in sorted(set(before_values.keys()) | set(after_values.keys())):
        val_before = before_values.get(content_id)
        val_after = after_values.get(content_id)
        if val_before != val_after:
            changed.append(content_id)
    return changed


def content_values(content: dict[str, Any]) -> dict[str, str]:
    """Expose exact logical paragraph values for bounded local diff reporting."""
    return dict(_paragraph_values(content))


def _assert_exact(
    report: EvidenceReport,
    *,
    field_name: str,
    original: str,
    tailored: str,
) -> None:
    if tailored != original:
        report.issues.append(
            f"Immutable field changed at {field_name}: {original!r} → {tailored!r}."
        )


def validate_tailored_content(
    *,
    original: dict[str, Any],
    tailored: dict[str, Any],
    extracted_resume: dict[str, Any],
    analysis: dict[str, Any],
    target_role: str,
) -> EvidenceReport:
    report = EvidenceReport()

    education_fields = ("institution", "degree_details")
    for name in education_fields:
        _assert_exact(
            report,
            field_name=f"education.{name}",
            original=original["education"][name],
            tailored=tailored["education"][name],
        )
    for name in ("coursework", "certifications"):
        _assert_exact(
            report,
            field_name=f"education.{name}.label",
            original=original["education"][name]["label"],
            tailored=tailored["education"][name]["label"],
        )

    if len(tailored["skill_groups"]) != len(original["skill_groups"]):
        report.issues.append("The number of technical-skill groups changed.")
    else:
        for index, (before, after) in enumerate(
            zip(original["skill_groups"], tailored["skill_groups"], strict=True)
        ):
            _assert_exact(
                report,
                field_name=f"skill_groups.{index}.label",
                original=before["label"],
                tailored=after["label"],
            )

    if len(tailored["projects"]) != len(original["projects"]):
        report.issues.append("The number of projects changed.")
    else:
        for index, (before, after) in enumerate(
            zip(original["projects"], tailored["projects"], strict=True)
        ):
            _assert_exact(
                report,
                field_name=f"projects.{index}.name",
                original=before["name"],
                tailored=after["name"],
            )
            if len(before["bullets"]) != len(after["bullets"]):
                report.issues.append(
                    f"Project {before['name']!r} changed from "
                    f"{len(before['bullets'])} to {len(after['bullets'])} bullets."
                )

    for name in ("name", "technologies"):
        _assert_exact(
            report,
            field_name=f"open_source.{name}",
            original=original["open_source"][name],
            tailored=tailored["open_source"][name],
        )
    for name in ("role", "employer_location", "dates"):
        _assert_exact(
            report,
            field_name=f"experience.{name}",
            original=original["experience"][name],
            tailored=tailored["experience"][name],
        )
    if len(tailored["experience"]["bullets"]) != len(original["experience"]["bullets"]):
        report.issues.append("The number of employment bullets changed.")

    budgets = {
        paragraph["content_id"]: paragraph["content_budget"]["maximum_characters"]
        for paragraph in extracted_resume["paragraphs"]
    }
    for content_id, value in _paragraph_values(tailored).items():
        maximum = budgets.get(content_id)
        actual_characters = count_budget_characters(value)
        if maximum is not None and actual_characters > maximum:
            report.issues.append(
                f"{content_id} is {actual_characters} characters; its "
                f"template-derived budget is {maximum}."
            )

    source_text = _resume_text(original)
    tailored_text = _resume_text(tailored)
    normalized_source = normalized_text(source_text)

    authorized_evidence_texts = [source_text]
    source_blocks = extracted_resume.get("source_blocks", [])
    if isinstance(source_blocks, list):
        for block in source_blocks:
            if isinstance(block, dict) and isinstance(block.get("exact_text"), str):
                authorized_evidence_texts.append(block["exact_text"])
    for edit in analysis.get("recommended_edits", []):
        if isinstance(edit, dict):
            for block in edit.get("resolved_evidence", []):
                if isinstance(block, dict) and isinstance(block.get("exact_text"), str):
                    authorized_evidence_texts.append(block["exact_text"])

    normalized_authorized_evidence = normalized_text(" ".join(authorized_evidence_texts))

    original_tech_by_location = {
        (location, normalized_text(item))
        for location, item in _technology_items(original)
    }
    for location, item in _technology_items(tailored):
        normalized_item = normalized_text(item)
        if (location, normalized_item) not in original_tech_by_location:
            report.introduced_technologies.append(f"{location}: {item}")
        if normalized_item not in normalized_authorized_evidence:
            report.issues.append(
                f"Technology/skill item lacks verbatim source evidence at "
                f"{location}: {item!r}."
            )

    combined_metric_source = " ".join(authorized_evidence_texts)
    original_metrics = set(_NUMBER_RE.findall(combined_metric_source))
    for metric in _NUMBER_RE.findall(tailored_text):
        if metric not in original_metrics and metric not in report.introduced_metrics:
            report.introduced_metrics.append(metric)
            report.issues.append(
                f"New numeric or metric claim {metric!r} is not present in the master resume."
            )

    for label, pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(tailored_text) and not pattern.search(source_text):
            report.issues.append(
                f"Forbidden unsupported capability introduced: {label}."
            )

    for forbidden in analysis.get("forbidden_claims", []):
        normalized_forbidden = normalized_text(forbidden)
        if (
            len(normalized_forbidden) >= 8
            and normalized_forbidden in normalized_text(tailored_text)
            and normalized_forbidden not in normalized_source
        ):
            report.issues.append(
                f"Codex-marked forbidden claim appears in tailored content: {forbidden!r}."
            )

    if _FIRST_PERSON_RE.search(tailored_text):
        report.issues.append("First-person language was introduced.")

    for match in _SENIORITY_RE.finditer(tailored_text):
        term = match.group(0)
        if not re.search(rf"\b{re.escape(term)}\b", source_text, re.I):
            if term.casefold() not in {item.casefold() for item in report.introduced_role_labels}:
                report.introduced_role_labels.append(term)
            report.issues.append(
                f"Unsupported seniority/leadership role label introduced: {term!r}."
            )

    if target_role and normalized_text(target_role) not in normalized_source:
        if normalized_text(target_role) in normalized_text(tailored_text):
            report.introduced_role_labels.append(target_role)
            report.issues.append(
                f"Target role label {target_role!r} was introduced as a resume claim "
                "without appearing in the source."
            )

    for match in _AVAILABILITY_RE.finditer(tailored_text):
        phrase = match.group(0)
        if not _AVAILABILITY_RE.search(source_text):
            report.introduced_availability.append(phrase)
            report.issues.append(
                f"New availability statement lacks source evidence: {phrase!r}."
            )

    for keyword in analysis.get("supported_ats_keywords", []):
        if not keyword.strip():
            continue
        count = normalized_text(tailored_text).count(normalized_text(keyword))
        original_count = normalized_source.count(normalized_text(keyword))
        if count > max(3, original_count + 2):
            report.issues.append(
                f"Possible keyword stuffing: {keyword!r} appears {count} times "
                f"(source: {original_count})."
            )

    approved_targets = {
        edit.get("target_source_id")
        for edit in analysis.get("recommended_edits", [])
        if isinstance(edit, dict) and isinstance(edit.get("target_source_id"), str)
    }
    try:
        changed_targets = changed_content_ids(original, tailored)
    except (KeyError, TypeError, ValueError):
        report.issues.append(
            "Tailored content could not be compared against the approved edit catalog."
        )
    else:
        for content_id in changed_targets:
            if content_id not in approved_targets:
                report.issues.append(
                    f"Unapproved content target changed: {content_id}."
                )

    report.issues = list(dict.fromkeys(report.issues))
    report.introduced_technologies = list(dict.fromkeys(report.introduced_technologies))
    report.introduced_metrics = list(dict.fromkeys(report.introduced_metrics))
    report.introduced_role_labels = list(dict.fromkeys(report.introduced_role_labels))
    report.introduced_availability = list(dict.fromkeys(report.introduced_availability))
    return report


def _section_lines(content: dict[str, Any]) -> list[tuple[str, list[str]]]:
    education = content["education"]
    sections: list[tuple[str, list[str]]] = [
        ("Professional Summary", [content["professional_summary"]]),
        (
            "Education & Certifications",
            [
                f"{education['institution']} | {education['degree_details']}",
                f"{education['coursework']['label']}: {education['coursework']['text']}",
                f"{education['certifications']['label']}: "
                f"{education['certifications']['text']}",
            ],
        ),
        (
            "Technical Skills",
            [f"{group['label']}: {group['text']}" for group in content["skill_groups"]],
        ),
    ]
    project_lines: list[str] = []
    for project in content["projects"]:
        project_lines.append(f"{project['name']} | {project['technologies']}")
        project_lines.extend(f"• {bullet}" for bullet in project["bullets"])
    sections.append(("AI Engineering Projects", project_lines))
    sections.append(
        (
            "Open Source Contribution",
            [
                f"{content['open_source']['name']} | "
                f"{content['open_source']['technologies']}",
                f"• {content['open_source']['bullet']}",
            ],
        )
    )
    sections.append(
        (
            "Experience",
            [
                f"{content['experience']['role']} | "
                f"{content['experience']['employer_location']} "
                f"({content['experience']['dates']})",
                *(f"• {bullet}" for bullet in content["experience"]["bullets"]),
            ],
        )
    )
    return sections


def _markdown_list(values: Iterable[str]) -> str:
    items = list(values)
    return "\n".join(f"- {item}" for item in items) if items else "- None detected"


def build_content_diff(
    original: dict[str, Any],
    tailored: dict[str, Any],
    report: EvidenceReport,
) -> str:
    before_sections = dict(_section_lines(original))
    after_sections = dict(_section_lines(tailored))
    lines = [
        "# Tailored Resume Content Diff",
        "",
        "## Local evidence-check result",
        "",
        "PASS — no blocking unsupported claims detected."
        if report.passed
        else "BLOCKED — questionable claims require correction; rendering is disabled.",
        "",
        "### Newly introduced technologies or skill placements",
        "",
        _markdown_list(report.introduced_technologies),
        "",
        "### Newly introduced metrics",
        "",
        _markdown_list(report.introduced_metrics),
        "",
        "### Newly introduced role labels",
        "",
        _markdown_list(report.introduced_role_labels),
        "",
        "### Newly introduced availability statements",
        "",
        _markdown_list(report.introduced_availability),
        "",
        "### Blocking evidence issues",
        "",
        _markdown_list(report.issues),
        "",
        "## Section-by-section changes",
        "",
    ]
    for section_name, before in before_sections.items():
        after = after_sections[section_name]
        lines.extend([f"### {section_name}", ""])
        if before == after:
            lines.extend(["No change.", ""])
            continue
        diff = difflib.unified_diff(
            before,
            after,
            fromfile="before",
            tofile="after",
            lineterm="",
        )
        lines.extend(f"    {line}" for line in diff)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
