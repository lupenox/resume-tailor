"""Deterministic structured-field patch compiler.

Python exclusively owns:

- ``skill_groups.N``
- ``education.coursework``
- ``education.certifications``

Gemma must never receive, author, revise, echo, or return patches for these
targets.  This module partitions approved edit catalogs, compiles deterministic
patches from the authenticated mutable body, and combines them with model-
authored prose patches for unified validation through the existing authoritative
``validate_and_apply_patches()`` path.
"""

from __future__ import annotations

import re
from typing import Any

from resume_tailor.backend.engine.patch_engine import (
    StructuredItemGroundingError,
    TargetResolutionError,
    _validate_replacement_text,
    authorized_evidence_texts_for_edit,
    mutable_proposed_text,
    resolve_target_descriptor,
)
from resume_tailor.backend.utils.utilities import (
    OllamaTailoringContractError,
    TailoringPreflightError,
)


# ---------------------------------------------------------------------------
# Target classification
# ---------------------------------------------------------------------------

#: Regex patterns identifying the three deterministic structured target families.
_DETERMINISTIC_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^skill_groups\.\d+$"),
    re.compile(r"^education\.coursework$"),
    re.compile(r"^education\.certifications$"),
)


def is_deterministic_structured_target(target_source_id: str) -> bool:
    """Return ``True`` when *target_source_id* names a structured target.

    Deterministic structured targets are compiled locally by Python and are
    never supplied to Gemma.
    """
    return any(
        pattern.fullmatch(target_source_id)
        for pattern in _DETERMINISTIC_TARGET_PATTERNS
    )


# ---------------------------------------------------------------------------
# Catalog partition
# ---------------------------------------------------------------------------


def partition_edit_catalog(
    catalog: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *catalog* into deterministic structured edits and prose edits.

    Returns ``(deterministic_edits, prose_edits)`` preserving original order
    within each partition.  Edit identity (``edit_id``, ``target_source_id``)
    is preserved exactly.
    """
    deterministic: list[dict[str, Any]] = []
    prose: list[dict[str, Any]] = []
    for edit in catalog:
        target_id = edit.get("target_source_id")
        if isinstance(target_id, str) and is_deterministic_structured_target(target_id):
            deterministic.append(edit)
        else:
            prose.append(edit)
    return deterministic, prose


# ---------------------------------------------------------------------------
# Deterministic patch compilation
# ---------------------------------------------------------------------------


class DeterministicPatchError(TailoringPreflightError):
    """A deterministic structured patch failed pre-provider validation.

    Only sanitized, nonprivate fields are exposed.
    """

    def __init__(
        self,
        *,
        edit_id: str,
        target_source_id: str,
        reason_code: str,
    ) -> None:
        self.edit_id = edit_id
        self.target_source_id = target_source_id
        self.reason_code = reason_code
        super().__init__(
            f"Deterministic patch for {edit_id} targeting "
            f"{target_source_id!r} failed: {reason_code}. "
            "No writer request was launched."
        )


def compile_deterministic_structured_patches(
    *,
    deterministic_edits: list[dict[str, Any]],
    master_content: dict[str, Any],
    extracted_resume: dict[str, Any],
    approved_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compile exact patches for deterministic structured targets.

    For every edit:

    1. Resolve the authenticated target descriptor via the strict resolver.
    2. Obtain the mutable approved body through ``mutable_proposed_text()``.
    3. Preserve the authenticated Python-owned label.
    4. Construct the exact patch with the approved mutable body.
    5. Validate using the same authoritative rules used for model output.
    6. Return the list of validated patch dicts.

    Raises ``DeterministicPatchError`` (before any provider invocation) when
    a deterministic patch is invalid.  Error messages expose only
    ``edit_id``, ``target_source_id``, and a stable nonprivate reason code.
    """
    forbidden_claims = approved_analysis.get("forbidden_claims", [])
    if not isinstance(forbidden_claims, list):
        forbidden_claims = []

    patches: list[dict[str, Any]] = []
    for edit in deterministic_edits:
        edit_id = edit["edit_id"]
        target_id = edit["target_source_id"]
        operation = edit.get("operation", "replace")

        # 1. Resolve target descriptor.
        try:
            descriptor = resolve_target_descriptor(
                edit, master_content, extracted_resume,
            )
        except TargetResolutionError:
            raise DeterministicPatchError(
                edit_id=edit_id,
                target_source_id=target_id,
                reason_code="target_resolution_failed",
            )

        # 2. Obtain the mutable approved body.
        try:
            replacement_text = mutable_proposed_text(edit, descriptor)
        except TargetResolutionError:
            raise DeterministicPatchError(
                edit_id=edit_id,
                target_source_id=target_id,
                reason_code="mutable_body_extraction_failed",
            )

        if not isinstance(replacement_text, str) or not replacement_text.strip():
            raise DeterministicPatchError(
                edit_id=edit_id,
                target_source_id=target_id,
                reason_code="empty_mutable_body",
            )

        # 5. Validate through the same authoritative patch rules.
        evidence_texts = authorized_evidence_texts_for_edit(
            edit, descriptor, extracted_resume,
        )

        # Full authoritative replacement-text validation (same as model output).
        try:
            replacement_text = _validate_replacement_text(
                edit_id=edit_id,
                descriptor=descriptor,
                replacement_text=replacement_text,
                evidence_texts=evidence_texts,
                forbidden_claims=forbidden_claims,
                operation=operation,
            )
        except StructuredItemGroundingError:
            raise DeterministicPatchError(
                edit_id=edit_id,
                target_source_id=target_id,
                reason_code="item_grounding_failed",
            )
        except OllamaTailoringContractError:
            raise DeterministicPatchError(
                edit_id=edit_id,
                target_source_id=target_id,
                reason_code="replacement_validation_failed",
            )

        # 4. Construct exact patch.
        patches.append({
            "edit_id": edit_id,
            "target_source_id": target_id,
            "operation": operation,
            "replacement_text": replacement_text,
        })

    return patches


# ---------------------------------------------------------------------------
# Hybrid payload combination
# ---------------------------------------------------------------------------


def combine_hybrid_patch_payload(
    *,
    deterministic_patches: list[dict[str, Any]],
    prose_patches: list[dict[str, Any]],
    full_catalog: list[dict[str, Any]],
    full_catalog_sha256: str,
) -> dict[str, Any]:
    """Combine deterministic and prose patches into a single payload.

    The combined payload:

    - uses the full approved-catalog digest (not the writer-subset digest);
    - restores full approved-catalog order;
    - is ready for ``validate_and_apply_patches()``.
    """
    expected_ids: list[str] = []
    for edit in full_catalog:
        if not isinstance(edit, dict) or not isinstance(edit.get("edit_id"), str):
            raise OllamaTailoringContractError(
                "The approved catalog contains an invalid edit_id."
            )
        expected_ids.append(edit["edit_id"])
    if len(expected_ids) != len(set(expected_ids)):
        raise OllamaTailoringContractError(
            "The approved catalog contains duplicate edit IDs."
        )

    # Build an index from edit_id to patch while rejecting duplicates and
    # deterministic/prose collisions before restoring catalog order.
    patch_index: dict[str, dict[str, Any]] = {}
    deterministic_ids: set[str] = set()
    for patch in deterministic_patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("edit_id"), str):
            raise OllamaTailoringContractError(
                "A deterministic patch contains an invalid edit_id."
            )
        eid = patch["edit_id"]
        if eid in patch_index:
            raise OllamaTailoringContractError(
                f"Duplicate deterministic patch for edit_id {eid!r}."
            )
        patch_index[eid] = patch
        deterministic_ids.add(eid)

    prose_ids: set[str] = set()
    for patch in prose_patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("edit_id"), str):
            raise OllamaTailoringContractError(
                "A prose patch contains an invalid edit_id."
            )
        eid = patch["edit_id"]
        if eid in prose_ids:
            raise OllamaTailoringContractError(
                f"Duplicate prose patch for edit_id {eid!r}."
            )
        if eid in deterministic_ids:
            raise OllamaTailoringContractError(
                "Deterministic/prose patch collision for "
                f"edit_id {eid!r}."
            )
        patch_index[eid] = patch
        prose_ids.add(eid)

    expected_id_set = set(expected_ids)
    actual_id_set = set(patch_index)
    if actual_id_set != expected_id_set:
        missing_ids = sorted(expected_id_set - actual_id_set)
        extra_ids = sorted(actual_id_set - expected_id_set)
        details: list[str] = []
        if missing_ids:
            details.append(f"missing IDs: {missing_ids}")
        if extra_ids:
            details.append(f"extra IDs: {extra_ids}")
        raise OllamaTailoringContractError(
            "Hybrid patch edit IDs do not exactly match the approved catalog "
            f"({'; '.join(details)})."
        )

    # Restore full catalog order.
    ordered_patches = [patch_index[eid] for eid in expected_ids]

    return {
        "status": "complete",
        "message": "Hybrid compilation successful.",
        "cannot_apply": None,
        "technical_failure": None,
        "catalog_sha256": full_catalog_sha256,
        "patches": ordered_patches,
    }


# ---------------------------------------------------------------------------
# Hybrid metadata (sanitized, no private content)
# ---------------------------------------------------------------------------


def hybrid_execution_metadata(
    *,
    deterministic_patches: list[dict[str, Any]],
    prose_patches: list[dict[str, Any]],
    deterministic_edits: list[dict[str, Any]],
    prose_edits: list[dict[str, Any]],
    full_catalog_sha256: str,
    writer_subset_sha256: str | None,
    ollama_invoked: bool,
) -> dict[str, Any]:
    """Build sanitized execution metadata without any private content."""
    if deterministic_edits and prose_edits:
        execution_mode = "hybrid"
    elif prose_edits:
        execution_mode = "prose_only"
    else:
        execution_mode = "deterministic_only"
    return {
        "execution_mode": execution_mode,
        "deterministic_patch_count": len(deterministic_patches),
        "gemma_patch_count": len(prose_patches),
        "deterministic_target_ids": [
            edit["target_source_id"] for edit in deterministic_edits
        ],
        "prose_target_ids": [
            edit["target_source_id"] for edit in prose_edits
        ],
        "full_catalog_digest": full_catalog_sha256,
        "writer_subset_digest": writer_subset_sha256,
        "ollama_invoked": ollama_invoked,
    }


def deterministic_only_metadata(
    *,
    deterministic_patches: list[dict[str, Any]],
    deterministic_edits: list[dict[str, Any]],
    full_catalog_sha256: str,
) -> dict[str, Any]:
    """Build metadata for a deterministic-only run (no provider invoked)."""
    return {
        "execution_mode": "deterministic_only",
        "deterministic_patch_count": len(deterministic_patches),
        "gemma_patch_count": 0,
        "deterministic_target_ids": [
            edit["target_source_id"] for edit in deterministic_edits
        ],
        "prose_target_ids": [],
        "full_catalog_digest": full_catalog_sha256,
        "writer_subset_digest": None,
        "ollama_invoked": False,
        "writer_skipped": True,
        "writer_skipped_reason": (
            "all_targets_deterministic" if deterministic_edits else "empty_catalog"
        ),
    }
