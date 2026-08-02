from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .antigravity_writer import approved_edit_catalog
from .evidence import _NUMBER_RE, changed_content_ids
from .revision import approved_revision_targets
from .schemas import validate_payload, validate_resume_content_payload
from .utilities import (
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
        if not isinstance(coursework, dict) or "label" not in coursework or "text" not in coursework:
            raise TargetResolutionError("education.coursework missing or invalid")
        return "composite_labelled", coursework, "text", str(coursework["label"])

    if target_source_id == "education.certifications":
        education = content.get("education")
        if not isinstance(education, dict):
            raise TargetResolutionError("education missing")
        certs = education.get("certifications")
        if not isinstance(certs, dict) or "label" not in certs or "text" not in certs:
            raise TargetResolutionError("education.certifications missing or invalid")
        return "composite_labelled", certs, "text", str(certs["label"])

    m_sg = re.fullmatch(r"skill_groups\.(\d+)", target_source_id)
    if m_sg:
        index = int(m_sg.group(1))
        groups = content.get("skill_groups")
        if not isinstance(groups, list) or index < 0 or index >= len(groups):
            raise TargetResolutionError(f"skill_groups index {index} out of bounds")
        group = groups[index]
        if not isinstance(group, dict) or "label" not in group or "text" not in group:
            raise TargetResolutionError(f"skill_groups.{index} invalid")
        return "composite_labelled", group, "text", str(group["label"])

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
    current_mutable_text = str(container[key])

    if kind == "composite_labelled":
        exact_rendered_existing_text = f"{label}: {current_mutable_text}"
    else:
        exact_rendered_existing_text = current_mutable_text

    budgets = {
        p["content_id"]: p["content_budget"]["maximum_characters"]
        for p in extracted_resume.get("paragraphs", [])
        if isinstance(p, dict) and "content_id" in p and "content_budget" in p
    }
    maximum_characters = budgets.get(target_id, 1000)

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
    """Atomically validate returned patch payload and apply to master_content deep copy."""
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

    status = payload.get("status")
    if status == "cannot_apply":
        detail = payload.get("cannot_apply")
        allowed_ids = {
            edit["edit_id"] for edit in approved_edit_catalog(approved_analysis)
        }
        if not isinstance(detail, dict) or detail.get("edit_id") not in allowed_ids:
            raise OllamaEvidenceRejectionError(
                "The local writer returned an unknown approved edit ID in cannot_apply."
            )
        raise OllamaCannotApplyError(
            f"The local writer could not apply approved {detail['edit_id']} ({detail['reason_code']})."
        )

    if status == "technical_failure":
        detail = payload.get("technical_failure")
        reason_code = detail.get("reason_code") if isinstance(detail, dict) else "unknown"
        raise OllamaTechnicalFailureError(
            f"The local writer reported technical failure {reason_code}. Provider prose was omitted."
        )

    if status != "complete":
        raise OllamaTailoringContractError(f"Unexpected status {status!r} in patch envelope.")

    catalog = approved_edit_catalog(approved_analysis)
    expected_sha256 = canonical_digest(catalog)
    actual_sha256 = payload.get("catalog_sha256")
    if not isinstance(actual_sha256, str) or actual_sha256 != expected_sha256:
        raise OllamaTailoringContractError(
            "The returned catalog_sha256 digest does not match the current approved edit catalog."
        )

    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise OllamaTailoringContractError("Patch envelope 'patches' field must be an array.")

    if len(patches) != len(catalog):
        raise OllamaTailoringContractError(
            f"Patch set size ({len(patches)}) does not match approved edit count ({len(catalog)})."
        )

    descriptors_by_edit_id: dict[str, TargetDescriptor] = {}
    for edit in catalog:
        descriptor = resolve_target_descriptor(edit, master_content, extracted_resume)
        descriptors_by_edit_id[edit["edit_id"]] = descriptor

    seen_edit_ids: set[str] = set()
    seen_target_ids: set[str] = set()

    # Authenticated metrics calculation for numeric claim verification
    from .evidence import _resume_text
    authorized_evidence_texts = [_resume_text(master_content)]
    source_blocks = extracted_resume.get("source_blocks", [])
    if isinstance(source_blocks, list):
        for block in source_blocks:
            if isinstance(block, dict) and isinstance(block.get("exact_text"), str):
                authorized_evidence_texts.append(block["exact_text"])
    for edit in approved_analysis.get("recommended_edits", []):
        if isinstance(edit, dict):
            for block in edit.get("resolved_evidence", []):
                if isinstance(block, dict) and isinstance(block.get("exact_text"), str):
                    authorized_evidence_texts.append(block["exact_text"])

    authenticated_metrics = set(_NUMBER_RE.findall(" ".join(authorized_evidence_texts)))
    forbidden_claims = approved_analysis.get("forbidden_claims", [])

    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise OllamaTailoringContractError(f"Patch {idx} is not an object.")

        edit_id = patch.get("edit_id")
        target_id = patch.get("target_source_id")
        operation = patch.get("operation")
        replacement_text = patch.get("replacement_text")

        if edit_id not in descriptors_by_edit_id:
            raise OllamaTailoringContractError(f"Patch {idx} specifies unknown edit_id {edit_id!r}.")

        if edit_id in seen_edit_ids:
            raise OllamaTailoringContractError(f"Duplicate patch for edit_id {edit_id!r}.")
        seen_edit_ids.add(edit_id)

        desc = descriptors_by_edit_id[edit_id]

        if target_id != desc.target_source_id:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} target_source_id {target_id!r} does not match approved target {desc.target_source_id!r}."
            )

        if target_id in seen_target_ids:
            raise OllamaTailoringContractError(f"Duplicate patch targeting {target_id!r}.")
        seen_target_ids.add(target_id)

        if operation != desc.operation:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} operation {operation!r} does not match approved operation {desc.operation!r}."
            )

        if not isinstance(replacement_text, str) or not replacement_text.strip():
            raise OllamaTailoringContractError(f"Patch for {edit_id} has empty replacement_text.")

        # Check no-op replacement
        if replacement_text == desc.current_mutable_text:
            raise OllamaTailoringContractError(f"Patch for {edit_id} is a no-op replacement.")

        # Label embedding check for composite fields
        if desc.kind == "composite_labelled" and desc.label is not None:
            label_norm = normalized_text(desc.label)
            replacement_norm = normalized_text(replacement_text)
            if replacement_norm.startswith(label_norm):
                raise OllamaTailoringContractError(
                    f"Patch for {edit_id} illegally contains label {desc.label!r} inside replacement text."
                )

        # Operation semantics check
        if operation == "append":
            if not replacement_text.startswith(desc.current_mutable_text):
                raise OllamaTailoringContractError(
                    f"Append patch for {edit_id} does not preserve the original prefix."
                )
            if len(replacement_text) <= len(desc.current_mutable_text):
                raise OllamaTailoringContractError(
                    f"Append patch for {edit_id} did not add a nonempty suffix."
                )

        # Rendered budget check
        if desc.kind == "composite_labelled" and desc.label is not None:
            rendered_text = f"{desc.label}: {replacement_text}"
        else:
            rendered_text = replacement_text

        if len(rendered_text) > desc.maximum_rendered_characters:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} ({len(rendered_text)} chars) exceeds target budget of {desc.maximum_rendered_characters}."
            )

        # Numeric claims check
        patch_metrics = set(_NUMBER_RE.findall(replacement_text))
        new_metrics = patch_metrics - authenticated_metrics
        if new_metrics:
            raise OllamaTailoringContractError(
                f"Patch for {edit_id} introduces unauthenticated numeric claims: {sorted(new_metrics)}."
            )

        # Forbidden claims check
        norm_replacement = normalized_text(replacement_text)
        for forbidden in forbidden_claims:
            norm_forbidden = normalized_text(forbidden)
            if len(norm_forbidden) >= 8 and norm_forbidden in norm_replacement:
                raise OllamaTailoringContractError(
                    f"Patch for {edit_id} contains forbidden claim {forbidden!r}."
                )

    # Atomic application to deep copy of master_content
    tailored_content = copy.deepcopy(master_content)

    for patch in patches:
        apply_patch_to_target(
            patch["target_source_id"],
            patch["replacement_text"],
            tailored_content,
        )

    # Verify original master_content is byte-for-byte unchanged
    if canonical_digest(master_content) == canonical_digest(tailored_content) and len(catalog) > 0:
        raise OllamaTailoringContractError("Patch application failed to produce changes in tailored content.")

    # Validate final tailored content schema
    validate_resume_content_payload(
        tailored_content,
        label="Deterministic tailored resume",
    )

    # Check changed content IDs match approved targets exactly
    changed_ids = changed_content_ids(master_content, tailored_content)
    approved_target_ids = {edit["target_source_id"] for edit in catalog}
    if set(changed_ids) != approved_target_ids:
        unapproved_changed = set(changed_ids) - approved_target_ids
        missing_changed = approved_target_ids - set(changed_ids)
        details = []
        if unapproved_changed:
            details.append(f"unapproved changed: {sorted(unapproved_changed)}")
        if missing_changed:
            details.append(f"missing approved changes: {sorted(missing_changed)}")
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
    """Atomically validate returned revision patch payload and apply to current_tailored_content deep copy."""
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

    issue_ids = {
        issue["issue_id"]
        for issue in qa_result.get("issues", [])
        if isinstance(issue, dict) and isinstance(issue.get("issue_id"), str)
    }

    status = payload.get("status")
    if status == "cannot_apply":
        detail = payload.get("cannot_apply")
        if not isinstance(detail, dict) or detail.get("issue_id") not in issue_ids:
            raise OllamaRevisionContractError(
                "The local writer returned an unknown QA issue ID in cannot_apply."
            )
        raise OllamaRevisionCannotApplyError(
            f"The local writer could not apply bounded correction for {detail['issue_id']} ({detail['reason_code']})."
        )

    if status == "technical_failure":
        detail = payload.get("technical_failure")
        reason_code = detail.get("reason_code") if isinstance(detail, dict) else "unknown"
        raise OllamaRevisionTechnicalFailureError(
            f"The local writer revision reported technical failure {reason_code}."
        )

    if status != "complete":
        raise OllamaRevisionContractError(f"Unexpected status {status!r} in revision patch envelope.")

    target_map = approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=approved_analysis,
    )
    expected_sha256 = canonical_digest(target_map)
    actual_sha256 = payload.get("authorization_sha256")
    if not isinstance(actual_sha256, str) or actual_sha256 != expected_sha256:
        raise OllamaRevisionContractError(
            "The returned authorization_sha256 digest does not match current revision authorization."
        )

    patches = payload.get("patches")
    if not isinstance(patches, list):
        raise OllamaRevisionContractError("Revision patch envelope 'patches' field must be an array.")

    seen_issue_ids: set[str] = set()

    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            raise OllamaRevisionContractError(f"Revision patch {idx} is not an object.")

        issue_id = patch.get("issue_id")
        target_id = patch.get("target_source_id")
        replacement_text = patch.get("replacement_text")

        if issue_id not in issue_ids:
            raise OllamaRevisionContractError(f"Revision patch {idx} references unknown issue_id {issue_id!r}.")

        if issue_id in seen_issue_ids:
            raise OllamaRevisionContractError(f"Duplicate revision patch for issue_id {issue_id!r}.")
        seen_issue_ids.add(issue_id)

        authorized_issues_for_target = target_map.get(target_id, [])
        if issue_id not in authorized_issues_for_target:
            raise OllamaRevisionContractError(
                f"Target {target_id!r} is not authorized for QA issue {issue_id!r}."
            )

        if not isinstance(replacement_text, str) or not replacement_text.strip():
            raise OllamaRevisionContractError(f"Revision patch for {issue_id} has empty replacement_text.")

    # Apply to deep copy
    revised_content = copy.deepcopy(current_tailored_content)

    for patch in patches:
        apply_patch_to_target(
            patch["target_source_id"],
            patch["replacement_text"],
            revised_content,
        )

    # Validate against canonical tailored resume schema
    validate_resume_content_payload(
        revised_content,
        label="Deterministic revised resume",
    )

    from .revision import validate_revision_scope
    validate_revision_scope(
        current_tailored_content,
        revised_content,
        target_map,
    )

    return revised_content
