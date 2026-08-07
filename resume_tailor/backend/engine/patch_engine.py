from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from resume_tailor.backend.providers.antigravity_writer import approved_edit_catalog
from resume_tailor.backend.engine.character_budget import (
    canonicalize_budget_text,
    compose_rendered_text,
    count_budget_characters,
    mutable_text_from_composite_proposal,
)
from resume_tailor.backend.engine.evidence import _NUMBER_RE, changed_content_ids
from resume_tailor.backend.engine.revision import approved_revision_targets
from resume_tailor.backend.utils.schemas import validate_payload, validate_resume_content_payload
from resume_tailor.backend.utils.utilities import (
    OllamaCanonicalSchemaError,
    OllamaCannotApplyError,
    OllamaEvidenceRejectionError,
    OllamaRevisionCannotApplyError,
    OllamaRevisionContractError,
    OllamaRevisionTechnicalFailureError,
    OllamaTailoringContractError,
    OllamaTechnicalFailureError,
    normalized_text,
)


def canonical_digest(payload: Any) -> str:
    """Compute deterministic SHA-256 over canonical JSON representation."""
    canonical_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def duplicate_catalog_target_ids(catalog: list[dict[str, Any]]) -> list[str]:
    """Return sorted target IDs repeated by more than one approved edit."""
    counts: dict[str, int] = {}
    for edit in catalog:
        target_id = edit.get("target_source_id")
        if isinstance(target_id, str):
            counts[target_id] = counts.get(target_id, 0) + 1
    return sorted(target_id for target_id, count in counts.items() if count > 1)


def _canonical_unicode(value: str) -> str:
    """Normalize canonically equivalent Unicode sequences for identity checks."""
    return canonicalize_budget_text(value)


def _normalized_claim_text(value: str) -> str:
    """Normalize compatibility forms, case, and whitespace for claim matching."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_forbidden_claim(text: str, claim: str) -> bool:
    normalized_text_value = _normalized_claim_text(text)
    normalized_claim = _normalized_claim_text(claim)
    if not normalized_claim:
        return False
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_claim)}(?!\w)",
            normalized_text_value,
        )
        is not None
    )


class TargetResolutionError(Exception):
    """Raised when a target source ID cannot be resolved or is not editable."""


@dataclass(frozen=True)
class TargetDescriptor:
    edit_id: str
    target_source_id: str
    operation: str  # "replace" or "append"
    kind: str  # "plain" or "composite_labelled"
    label: str | None
    current_mutable_text: str
    exact_rendered_existing_text: str
    maximum_rendered_characters: int
    proposed_text: str
    alignment_rationale: str
    evidence_source_ids: list[str]


@dataclass(frozen=True)
class CharacterBudgetViolation:
    """Sanitized details for one hard rendered-character budget violation."""

    edit_id: str
    target_source_id: str
    actual_characters: int
    maximum_characters: int


class PatchCharacterBudgetError(OllamaTailoringContractError):
    """One or more patches exceed authenticated hard character budgets."""

    validation_path = "character_budget"

    def __init__(self, violations: list[CharacterBudgetViolation]) -> None:
        if not violations:
            raise ValueError("At least one character-budget violation is required.")
        self.violations = tuple(violations)
        if len(violations) == 1:
            violation = violations[0]
            message = (
                f"Patch for {violation.edit_id} "
                f"({violation.actual_characters} chars) exceeds target budget "
                f"of {violation.maximum_characters}."
            )
        else:
            ids = [violation.edit_id for violation in violations]
            message = (
                "Patches exceed authenticated target character budgets: "
                f"{ids}."
            )
        super().__init__(message)


def parse_target_source_id(
    target_source_id: str,
    content: dict[str, Any],
) -> tuple[str, Any, str | int, str | None]:
    """Resolve target_source_id against content structure via strict allowlist.

    Returns:
        (kind, container, key_or_index, label_or_none)
    """
    if target_source_id == "professional_summary":
        if "professional_summary" not in content or not isinstance(content["professional_summary"], str):
            raise TargetResolutionError("professional_summary missing or invalid")
        return "plain", content, "professional_summary", None

    if target_source_id == "open_source.bullet":
        open_source = content.get("open_source")
        if not isinstance(open_source, dict) or "bullet" not in open_source or not isinstance(open_source["bullet"], str):
            raise TargetResolutionError("open_source.bullet missing or invalid")
        return "plain", open_source, "bullet", None

    if target_source_id == "education.coursework":
        education = content.get("education")
        if not isinstance(education, dict):
            raise TargetResolutionError("education missing")
        coursework = education.get("coursework")
        if not isinstance(coursework, dict):
            raise TargetResolutionError("education.coursework missing or invalid")
        label = coursework.get("label")
        text = coursework.get("text")
        if not isinstance(label, str) or not label.strip() or not isinstance(text, str):
            raise TargetResolutionError("education.coursework missing or invalid composite label/body")
        return "composite_labelled", coursework, "text", label

    if target_source_id == "education.certifications":
        education = content.get("education")
        if not isinstance(education, dict):
            raise TargetResolutionError("education missing")
        certs = education.get("certifications")
        if not isinstance(certs, dict):
            raise TargetResolutionError("education.certifications missing or invalid")
        label = certs.get("label")
        text = certs.get("text")
        if not isinstance(label, str) or not label.strip() or not isinstance(text, str):
            raise TargetResolutionError("education.certifications missing or invalid composite label/body")
        return "composite_labelled", certs, "text", label

    m_sg = re.fullmatch(r"skill_groups\.(\d+)", target_source_id)
    if m_sg:
        index = int(m_sg.group(1))
        groups = content.get("skill_groups")
        if not isinstance(groups, list) or index < 0 or index >= len(groups):
            raise TargetResolutionError(f"skill_groups index {index} out of bounds")
        group = groups[index]
        if not isinstance(group, dict):
            raise TargetResolutionError(f"skill_groups.{index} invalid")
        label = group.get("label")
        text = group.get("text")
        if not isinstance(label, str) or not label.strip() or not isinstance(text, str):
            raise TargetResolutionError(f"skill_groups.{index} missing or invalid composite label/body")
        return "composite_labelled", group, "text", label

    m_exp = re.fullmatch(r"experience\.bullets\.(\d+)", target_source_id)
    if m_exp:
        index = int(m_exp.group(1))
        exp = content.get("experience")
        if not isinstance(exp, dict):
            raise TargetResolutionError("experience missing")
        bullets = exp.get("bullets")
        if not isinstance(bullets, list) or index < 0 or index >= len(bullets):
            raise TargetResolutionError(f"experience.bullets index {index} out of bounds")
        return "plain", bullets, index, None

    m_proj = re.fullmatch(r"projects\.(\d+)\.bullets\.(\d+)", target_source_id)
    if m_proj:
        p_index = int(m_proj.group(1))
        b_index = int(m_proj.group(2))
        projects = content.get("projects")
        if not isinstance(projects, list) or p_index < 0 or p_index >= len(projects):
            raise TargetResolutionError(f"projects index {p_index} out of bounds")
        proj = projects[p_index]
        if not isinstance(proj, dict):
            raise TargetResolutionError(f"projects.{p_index} invalid")
        bullets = proj.get("bullets")
        if not isinstance(bullets, list) or b_index < 0 or b_index >= len(bullets):
            raise TargetResolutionError(f"projects.{p_index}.bullets index {b_index} out of bounds")
        return "plain", bullets, b_index, None

    raise TargetResolutionError(f"Target source ID {target_source_id!r} is not an editable target.")


def resolve_target_descriptor(
    edit: dict[str, Any],
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
) -> TargetDescriptor:
    edit_id = edit["edit_id"]
    target_id = edit["target_source_id"]
    operation = edit.get("operation", "replace")
    proposed_text = edit.get("proposed_text", "")
    rationale = edit.get("alignment_rationale", "")
    evidence_ids = edit.get("evidence_source_ids", [])

    kind, container, key, label = parse_target_source_id(target_id, master_content)
    current_mutable_text = canonicalize_budget_text(str(container[key]))
    exact_rendered_existing_text = compose_rendered_text(
        current_mutable_text,
        immutable_label=label if kind == "composite_labelled" else None,
    )

    budgets = {
        p["content_id"]: p["content_budget"]["maximum_characters"]
        for p in extracted_resume.get("paragraphs", [])
        if isinstance(p, dict) and "content_id" in p and "content_budget" in p
    }
    maximum_characters = budgets.get(target_id)
    if not isinstance(maximum_characters, int) or maximum_characters <= 0:
        raise TargetResolutionError(
            f"Target source ID {target_id!r} has no authenticated content budget."
        )

    return TargetDescriptor(
        edit_id=edit_id,
        target_source_id=target_id,
        operation=operation,
        kind=kind,
        label=label,
        current_mutable_text=current_mutable_text,
        exact_rendered_existing_text=exact_rendered_existing_text,
        maximum_rendered_characters=maximum_characters,
        proposed_text=proposed_text,
        alignment_rationale=rationale,
        evidence_source_ids=list(evidence_ids) if isinstance(evidence_ids, list) else [],
    )


def mutable_proposed_text(
    edit: dict[str, Any],
    descriptor: TargetDescriptor,
) -> str:
    """Return the approved proposed mutable body without an immutable label.

    For composite-labelled targets only, strip the leading prefix when it
    exactly matches the authenticated label (under canonical Unicode
    normalization) followed by ``:``.  Any other text—including colons in
    prose, URLs, version strings, or mismatched labels—is returned
    unchanged.
    """
    proposed = edit.get("proposed_text")
    if not isinstance(proposed, str) or not proposed.strip():
        return ""
    if descriptor.kind != "composite_labelled":
        return proposed

    if not isinstance(descriptor.label, str) or not descriptor.label.strip():
        raise TargetResolutionError(
            f"{descriptor.target_source_id!r} missing or invalid composite label/body"
        )

    try:
        body = mutable_text_from_composite_proposal(
            proposed,
            immutable_label=descriptor.label,
        )
    except (TypeError, ValueError) as exc:
        raise TargetResolutionError(
            f"{descriptor.target_source_id!r} missing or invalid composite label/body"
        ) from exc
    if body != proposed:
        if not body:
            raise TargetResolutionError(
                f"Approved edit {descriptor.edit_id} for"
                f" {descriptor.target_source_id!r} contains no mutable"
                f" text after its authenticated label prefix."
            )
        return body
    return proposed


def authorized_evidence_texts_for_edit(
    edit: dict[str, Any],
    descriptor: TargetDescriptor,
    extracted_resume: dict[str, Any],
) -> list[str]:
    """Resolve only the target and evidence blocks authenticated for one edit."""
    texts = [descriptor.exact_rendered_existing_text]
    source_index = {
        block.get("source_id"): block
        for block in extracted_resume.get("source_blocks", [])
        if isinstance(block, dict) and isinstance(block.get("source_id"), str)
    }
    source_ids: list[str] = [descriptor.target_source_id]
    evidence_ids = edit.get("evidence_source_ids")
    if isinstance(evidence_ids, list):
        source_ids.extend(item for item in evidence_ids if isinstance(item, str))
    for source_id in source_ids:
        block = source_index.get(source_id)
        exact_text = block.get("exact_text") if isinstance(block, dict) else None
        if isinstance(exact_text, str) and exact_text not in texts:
            texts.append(exact_text)
    return texts


def authenticated_metrics_for_edit(
    edit: dict[str, Any],
    descriptor: TargetDescriptor,
    extracted_resume: dict[str, Any],
) -> list[str]:
    return sorted(
        set(
            _NUMBER_RE.findall(
                " ".join(
                    authorized_evidence_texts_for_edit(
                        edit,
                        descriptor,
                        extracted_resume,
                    )
                )
            )
        )
    )


class StructuredItemGroundingError(OllamaTailoringContractError):
    """A structured-list item lacks exact authenticated source evidence."""


_STRUCTURED_LIST_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^skill_groups\.\d+$"),
    re.compile(r"^education\.coursework$"),
    re.compile(r"^education\.certifications$"),
)
_STRUCTURED_LIST_DELIMITERS = re.compile(r"\s*[,•]\s*")


def _validate_structured_list_items(
    *,
    edit_id: str,
    target_source_id: str,
    replacement_text: str,
    evidence_texts: list[str],
) -> None:
    """Require every structured-list item to occur in authenticated evidence.

    The check is deliberately lexical: each comma- or bullet-delimited item
    must occur as the same normalized phrase in the target's authorized source
    evidence.  It does not use fuzzy matching, aliases, semantic similarity,
    or provider judgment.
    """
    if not any(
        pattern.fullmatch(target_source_id)
        for pattern in _STRUCTURED_LIST_TARGET_PATTERNS
    ):
        return

    normalized_evidence = normalized_text(" ".join(evidence_texts))
    for item in _STRUCTURED_LIST_DELIMITERS.split(replacement_text):
        candidate = item.strip().rstrip(".")
        if candidate and normalized_text(candidate) not in normalized_evidence:
            item_kind = (
                "skill item"
                if re.fullmatch(r"skill_groups\.\d+", target_source_id)
                else "structured item"
            )
            raise StructuredItemGroundingError(
                f"Patch for {edit_id} adds {item_kind} {candidate!r} to "
                f"{target_source_id!r} without authenticated source evidence."
            )


def _validate_replacement_text(
    *,
    edit_id: str,
    descriptor: TargetDescriptor,
    replacement_text: Any,
    evidence_texts: list[str],
    forbidden_claims: list[Any],
    operation: str | None = None,
    enforce_character_budget: bool = True,
) -> str:
    if not isinstance(replacement_text, str) or not replacement_text.strip():
        raise OllamaTailoringContractError(
            f"Patch for {edit_id} has empty replacement_text."
        )
    canonical_replacement = canonicalize_budget_text(replacement_text)
    if canonical_replacement == _canonical_unicode(
        descriptor.current_mutable_text
    ):
        raise OllamaTailoringContractError(
            f"Patch for {edit_id} is a no-op replacement."
        )

    if descriptor.kind == "composite_labelled" and descriptor.label is not None:
        if normalized_text(canonical_replacement).startswith(
            normalized_text(descriptor.label)
        ):
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} illegally contains label {descriptor.label!r} "
                "inside replacement text."
            )

    effective_operation = operation or descriptor.operation
    if effective_operation == "append":
        canonical_current = canonicalize_budget_text(
            descriptor.current_mutable_text
        )
        if not canonical_replacement.startswith(canonical_current):
            raise OllamaTailoringContractError(
                f"Append patch for {edit_id} does not preserve the original prefix."
            )
        if count_budget_characters(canonical_replacement) <= count_budget_characters(
            canonical_current
        ):
            raise OllamaTailoringContractError(
                f"Append patch for {edit_id} did not add a nonempty suffix."
            )
    elif effective_operation != "replace":
        raise OllamaTailoringContractError(
            f"Patch for {edit_id} uses unsupported operation {effective_operation!r}."
        )

    immutable_label = (
        descriptor.label if descriptor.kind == "composite_labelled" else None
    )
    rendered_text = compose_rendered_text(
        canonical_replacement,
        immutable_label=immutable_label,
    )
    actual_characters = count_budget_characters(rendered_text)
    budget_violation = (
        CharacterBudgetViolation(
            edit_id=edit_id,
            target_source_id=descriptor.target_source_id,
            actual_characters=actual_characters,
            maximum_characters=descriptor.maximum_rendered_characters,
        )
        if actual_characters > descriptor.maximum_rendered_characters
        else None
    )
    if budget_violation is not None and enforce_character_budget:
        raise PatchCharacterBudgetError([budget_violation])

    if canonical_replacement != canonical_replacement.strip():
        raise OllamaTailoringContractError(
            f"Patch for {edit_id} contains leading or trailing whitespace. "
            "No content was silently stripped."
        )

    authorized_text = " ".join(evidence_texts)
    authenticated_metrics = set(_NUMBER_RE.findall(authorized_text))
    new_metrics = (
        set(_NUMBER_RE.findall(canonical_replacement)) - authenticated_metrics
    )
    if new_metrics:
        raise OllamaTailoringContractError(
            f"Patch for {edit_id} introduces unauthenticated numeric claims: "
            f"{sorted(new_metrics)}."
        )

    _validate_structured_list_items(
        edit_id=edit_id,
        target_source_id=descriptor.target_source_id,
        replacement_text=canonical_replacement,
        evidence_texts=evidence_texts,
    )

    for forbidden in forbidden_claims:
        if not isinstance(forbidden, str):
            continue
        if _contains_forbidden_claim(canonical_replacement, forbidden):
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} contains a forbidden claim."
            )
    return canonical_replacement


def apply_patch_to_target(
    target_source_id: str,
    replacement_text: str,
    content: dict[str, Any],
) -> None:
    """Apply replacement_text to content in-place at target_source_id."""
    kind, container, key, _label = parse_target_source_id(target_source_id, content)
    container[key] = replacement_text


def validate_and_apply_patches(
    *,
    payload: dict[str, Any],
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    approved_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Atomically validate a complete patch transaction and merge it locally."""
    try:
        validate_payload(
            payload,
            "ollama_tailoring_patch.schema.json",
            label="Gemma 4 12B patch payload",
        )
    except Exception as exc:
        raise OllamaCanonicalSchemaError(
            "Gemma 4 12B output failed canonical patch envelope validation."
        ) from exc

    catalog = approved_edit_catalog(approved_analysis)
    duplicate_targets = duplicate_catalog_target_ids(catalog)
    if duplicate_targets:
        raise OllamaTailoringContractError(
            "The approved edit catalog repeats target source IDs: "
            f"{duplicate_targets}."
        )
    expected_sha256 = canonical_digest(catalog)
    actual_sha256 = payload.get("catalog_sha256")
    if not isinstance(actual_sha256, str) or actual_sha256 != expected_sha256:
        raise OllamaTailoringContractError(
            "The returned catalog_sha256 digest does not match the current approved edit catalog."
        )

    status = payload.get("status")
    if status == "cannot_apply":
        detail = payload.get("cannot_apply")
        allowed_ids = {edit["edit_id"] for edit in catalog}
        if not isinstance(detail, dict) or detail.get("edit_id") not in allowed_ids:
            raise OllamaEvidenceRejectionError(
                "The local writer returned an unknown approved edit ID in cannot_apply."
            )
        raise OllamaCannotApplyError(
            f"The local writer could not apply approved {detail['edit_id']} "
            f"({detail['reason_code']})."
        )
    if status == "technical_failure":
        detail = payload.get("technical_failure")
        reason_code = detail.get("reason_code") if isinstance(detail, dict) else "unknown"
        raise OllamaTechnicalFailureError(
            f"The local writer reported technical failure {reason_code}. "
            "Provider prose was omitted."
        )
    if status != "complete":
        raise OllamaTailoringContractError(
            f"Unexpected status {status!r} in patch envelope."
        )

    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise OllamaTailoringContractError(
            "Patch envelope 'patches' field must be an array."
        )
    if len(patches) != len(catalog):
        raise OllamaTailoringContractError(
            f"Patch set size ({len(patches)}) does not match approved edit count "
            f"({len(catalog)})."
        )

    descriptors_by_edit_id: dict[str, TargetDescriptor] = {}
    edits_by_edit_id: dict[str, dict[str, Any]] = {}
    try:
        for edit in catalog:
            descriptor = resolve_target_descriptor(
                edit,
                master_content,
                extracted_resume,
            )
            descriptors_by_edit_id[edit["edit_id"]] = descriptor
            edits_by_edit_id[edit["edit_id"]] = edit
    except TargetResolutionError as exc:
        raise OllamaTailoringContractError(
            "The approved patch target catalog cannot be resolved safely."
        ) from exc

    seen_edit_ids: set[str] = set()
    seen_target_ids: set[str] = set()
    validated_patches: list[dict[str, Any]] = []
    budget_violations: list[CharacterBudgetViolation] = []
    forbidden_claims = approved_analysis.get("forbidden_claims", [])
    if not isinstance(forbidden_claims, list):
        forbidden_claims = []

    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise OllamaTailoringContractError(f"Patch {index} is not an object.")
        edit_id = patch.get("edit_id")
        target_id = patch.get("target_source_id")
        operation = patch.get("operation")

        if edit_id not in descriptors_by_edit_id:
            raise OllamaTailoringContractError(
                f"Patch {index} specifies unknown edit_id {edit_id!r}."
            )
        if edit_id in seen_edit_ids:
            raise OllamaTailoringContractError(
                f"Duplicate patch for edit_id {edit_id!r}."
            )
        seen_edit_ids.add(edit_id)
        descriptor = descriptors_by_edit_id[edit_id]
        edit = edits_by_edit_id[edit_id]

        if target_id != descriptor.target_source_id:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} target_source_id {target_id!r} does not "
                f"match approved target {descriptor.target_source_id!r}."
            )
        if target_id in seen_target_ids:
            raise OllamaTailoringContractError(
                f"Duplicate patch targeting {target_id!r}."
            )
        seen_target_ids.add(target_id)
        if operation != descriptor.operation:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} operation {operation!r} does not match "
                f"approved operation {descriptor.operation!r}."
            )

        try:
            canonical_replacement = _validate_replacement_text(
                edit_id=edit_id,
                descriptor=descriptor,
                replacement_text=patch.get("replacement_text"),
                evidence_texts=authorized_evidence_texts_for_edit(
                    edit,
                    descriptor,
                    extracted_resume,
                ),
                forbidden_claims=forbidden_claims,
            )
        except PatchCharacterBudgetError as exc:
            budget_violations.extend(exc.violations)
            continue
        canonical_patch = copy.deepcopy(patch)
        canonical_patch["replacement_text"] = canonical_replacement
        validated_patches.append(canonical_patch)

    if budget_violations:
        raise PatchCharacterBudgetError(budget_violations)

    if seen_edit_ids != set(descriptors_by_edit_id):
        raise OllamaTailoringContractError(
            "The returned patch set did not cover every approved edit exactly once."
        )

    original_digest = canonical_digest(master_content)
    tailored_content = copy.deepcopy(master_content)
    for patch in validated_patches:
        apply_patch_to_target(
            patch["target_source_id"],
            patch["replacement_text"],
            tailored_content,
        )
    if canonical_digest(master_content) != original_digest:
        raise OllamaTailoringContractError(
            "The authenticated master resume was mutated during patch application."
        )
    if original_digest == canonical_digest(tailored_content) and catalog:
        raise OllamaTailoringContractError(
            "Patch application failed to produce changes in tailored content."
        )

    validate_resume_content_payload(
        tailored_content,
        label="Deterministic tailored resume",
    )
    changed_ids = changed_content_ids(master_content, tailored_content)
    approved_target_ids = {edit["target_source_id"] for edit in catalog}
    if set(changed_ids) != approved_target_ids:
        unapproved = set(changed_ids) - approved_target_ids
        missing = approved_target_ids - set(changed_ids)
        details: list[str] = []
        if unapproved:
            details.append(f"unapproved changed: {sorted(unapproved)}")
        if missing:
            details.append(f"missing approved changes: {sorted(missing)}")
        raise OllamaTailoringContractError(
            f"Tailored content changed target mismatch ({'; '.join(details)})."
        )
    return tailored_content


def validate_and_apply_revision_patches(
    *,
    payload: dict[str, Any],
    current_tailored_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    approved_analysis: dict[str, Any],
    qa_result: dict[str, Any],
) -> dict[str, Any]:
    """Atomically validate and apply one patch for every authorized QA target."""
    try:
        validate_payload(
            payload,
            "ollama_revision_patch.schema.json",
            label="Gemma 4 12B revision patch payload",
        )
    except Exception as exc:
        raise OllamaRevisionContractError(
            "The local writer violated the one-shot revision response contract."
        ) from exc

    target_map = approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=approved_analysis,
    )
    if not target_map:
        raise OllamaRevisionContractError(
            "The supplied QA findings contain no authenticated revision target."
        )
    expected_sha256 = canonical_digest(target_map)
    actual_sha256 = payload.get("authorization_sha256")
    if not isinstance(actual_sha256, str) or actual_sha256 != expected_sha256:
        raise OllamaRevisionContractError(
            "The returned authorization_sha256 digest does not match current "
            "revision authorization."
        )

    authorized_issue_ids = {
        issue_id for issue_ids in target_map.values() for issue_id in issue_ids
    }
    status = payload.get("status")
    if status == "cannot_apply":
        detail = payload.get("cannot_apply")
        if (
            not isinstance(detail, dict)
            or detail.get("issue_id") not in authorized_issue_ids
        ):
            raise OllamaRevisionContractError(
                "The local writer returned an unknown QA issue ID in cannot_apply."
            )
        raise OllamaRevisionCannotApplyError(
            f"The local writer could not apply bounded correction for "
            f"{detail['issue_id']} ({detail['reason_code']})."
        )
    if status == "technical_failure":
        detail = payload.get("technical_failure")
        reason_code = detail.get("reason_code") if isinstance(detail, dict) else "unknown"
        raise OllamaRevisionTechnicalFailureError(
            f"The local writer revision reported technical failure {reason_code}."
        )
    if status != "complete":
        raise OllamaRevisionContractError(
            f"Unexpected status {status!r} in revision patch envelope."
        )

    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise OllamaRevisionContractError(
            "Revision patch envelope 'patches' field must be an array."
        )
    if len(patches) != len(target_map):
        raise OllamaRevisionContractError(
            "Revision patch count does not match the authenticated target count."
        )

    approved_catalog = approved_edit_catalog(approved_analysis)
    duplicate_targets = duplicate_catalog_target_ids(approved_catalog)
    if duplicate_targets:
        raise OllamaRevisionContractError(
            "The approved edit catalog repeats revision target source IDs: "
            f"{duplicate_targets}."
        )
    catalog_by_target = {
        edit["target_source_id"]: edit for edit in approved_catalog
    }
    descriptors_by_target: dict[str, TargetDescriptor] = {}
    try:
        for target_id in target_map:
            edit = catalog_by_target.get(target_id)
            if edit is None:
                raise TargetResolutionError(
                    f"Revision target {target_id!r} has no approved edit."
                )
            descriptors_by_target[target_id] = resolve_target_descriptor(
                edit,
                current_tailored_content,
                extracted_resume,
            )
    except TargetResolutionError as exc:
        raise OllamaRevisionContractError(
            "The revision target catalog cannot be resolved safely."
        ) from exc

    forbidden_claims = approved_analysis.get("forbidden_claims", [])
    if not isinstance(forbidden_claims, list):
        forbidden_claims = []
    seen_issue_ids: set[str] = set()
    seen_target_ids: set[str] = set()
    validated_patches: list[dict[str, Any]] = []

    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise OllamaRevisionContractError(
                f"Revision patch {index} is not an object."
            )
        issue_id = patch.get("issue_id")
        target_id = patch.get("target_source_id")
        if issue_id not in authorized_issue_ids:
            raise OllamaRevisionContractError(
                f"Revision patch {index} references unknown issue_id {issue_id!r}."
            )
        if issue_id in seen_issue_ids:
            raise OllamaRevisionContractError(
                f"Duplicate revision patch for issue_id {issue_id!r}."
            )
        seen_issue_ids.add(issue_id)
        if target_id in seen_target_ids:
            raise OllamaRevisionContractError(
                f"Duplicate revision patch targeting {target_id!r}."
            )
        seen_target_ids.add(target_id)
        if issue_id not in target_map.get(target_id, []):
            raise OllamaRevisionContractError(
                f"Target {target_id!r} is not authorized for QA issue {issue_id!r}."
            )

        descriptor = descriptors_by_target[target_id]
        edit = catalog_by_target[target_id]
        try:
            canonical_replacement = _validate_replacement_text(
                edit_id=issue_id,
                descriptor=descriptor,
                replacement_text=patch.get("replacement_text"),
                evidence_texts=authorized_evidence_texts_for_edit(
                    edit,
                    descriptor,
                    extracted_resume,
                ),
                forbidden_claims=forbidden_claims,
                operation="replace",
            )
        except OllamaTailoringContractError as exc:
            raise OllamaRevisionContractError(str(exc)) from exc
        canonical_patch = copy.deepcopy(patch)
        canonical_patch["replacement_text"] = canonical_replacement
        validated_patches.append(canonical_patch)

    if seen_target_ids != set(target_map):
        raise OllamaRevisionContractError(
            "The revision patch set did not cover every authenticated target."
        )

    current_digest = canonical_digest(current_tailored_content)
    revised_content = copy.deepcopy(current_tailored_content)
    for patch in validated_patches:
        apply_patch_to_target(
            patch["target_source_id"],
            patch["replacement_text"],
            revised_content,
        )
    if canonical_digest(current_tailored_content) != current_digest:
        raise OllamaRevisionContractError(
            "The current tailored resume was mutated during revision application."
        )
    validate_resume_content_payload(
        revised_content,
        label="Deterministic revised resume",
    )

    from resume_tailor.backend.engine.revision import validate_revision_scope
    validate_revision_scope(
        initial_content=current_tailored_content,
        revised_content=revised_content,
        qa_result=qa_result,
        approved_analysis=approved_analysis,
    )
    return revised_content
