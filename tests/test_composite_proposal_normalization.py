"""Tests for composite-target proposed-text normalization.

All data is synthetic.  No provider, network, or private content is used.
"""
from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from resume_tailor import ollama_writer as writer
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.patch_engine import (
    TargetDescriptor,
    TargetResolutionError,
    mutable_proposed_text,
    validate_and_apply_patches,
)
from resume_tailor.utilities import (
    OllamaTailoringContractError,
    TailoringPreflightError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _descriptor(
    *,
    kind: str = "composite_labelled",
    label: str | None = "Software & Data",
    edit_id: str = "edit.001",
    target_source_id: str = "skill_groups.2",
    operation: str = "replace",
    current_mutable_text: str = "Python, Docker",
) -> TargetDescriptor:
    if label is not None:
        rendered = f"{label}: {current_mutable_text}"
    else:
        rendered = current_mutable_text
    return TargetDescriptor(
        edit_id=edit_id,
        target_source_id=target_source_id,
        operation=operation,
        kind=kind,
        label=label,
        current_mutable_text=current_mutable_text,
        exact_rendered_existing_text=rendered,
        maximum_rendered_characters=500,
        proposed_text="",
        alignment_rationale="",
        evidence_source_ids=[],
    )


def _setup_synthetic_inputs(
    master_resume: Path,
    proposed_text: str,
    *,
    operation: str = "replace",
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    extracted, _ = extract_resume(master_resume)
    job_desc = "Synthetic job description requiring AI."
    reqs = build_job_requirement_catalog(job_desc)
    analysis = {
        "role_summary": "Synthetic AI Engineer Role",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": ["AI"],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": ["AI"],
        "missing_or_unsupported_requirements": [],
        "recommended_edits": [
            {
                "edit_id": "edit.001",
                "target_source_id": "skill_groups.2",
                "operation": operation,
                "proposed_text": proposed_text,
                "alignment_rationale": "Add AI keyword to skill group.",
                "evidence_source_ids": ["skill_groups.2", "projects.0.bullets.0"],
                "resolved_evidence": [
                    {
                        "source_id": "skill_groups.2",
                        "section_context": "Technical Skills",
                        "exact_text": (
                            f"{extracted['content']['skill_groups'][2]['label']}"
                            f": {extracted['content']['skill_groups'][2]['text']}"
                        ),
                    }
                ],
            }
        ],
        "immutable_facts": [],
        "forbidden_claims": [],
        "content_budget_guidance": [],
        "questions_for_user": [],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
    }
    resolved_analysis, issues = resolve_analysis_evidence(
        analysis, extracted, reqs
    )
    assert not issues
    return extracted, job_desc, reqs, resolved_analysis


# ---------------------------------------------------------------------------
# 1  Body-only composite text is unchanged.
# ---------------------------------------------------------------------------

def test_01_body_only_composite_text_unchanged():
    edit = {"proposed_text": "Python, AWS"}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "Python, AWS"


# ---------------------------------------------------------------------------
# 2  Exact authenticated label prefix is stripped.
# ---------------------------------------------------------------------------

def test_02_exact_authenticated_label_stripped():
    edit = {"proposed_text": "Software & Data: Python, AWS"}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "Python, AWS"


# ---------------------------------------------------------------------------
# 3  Canonically equivalent Unicode authenticated label is stripped.
# ---------------------------------------------------------------------------

def test_03_unicode_equivalent_label_stripped():
    # Build a label with a pre-composed ampersand-like character mix.
    # Use NFC vs NFD of the exact same label text.
    nfc_label = unicodedata.normalize("NFC", "Softwaré & Data")
    nfd_label = unicodedata.normalize("NFD", "Softwaré & Data")
    assert nfc_label != nfd_label  # byte-level differs
    desc = _descriptor(label=nfc_label)
    edit = {"proposed_text": f"{nfd_label}: Python, AWS"}
    result = mutable_proposed_text(edit, desc)
    assert result == "Python, AWS"


def test_03b_unicode_body_preserves_nfd():
    nfc_label = unicodedata.normalize("NFC", "Softwaré & Data")
    nfd_label = unicodedata.normalize("NFD", "Softwaré & Data")
    nfd_body = unicodedata.normalize("NFD", "Pythón, AWS")
    desc = _descriptor(label=nfc_label)
    edit = {"proposed_text": f"{nfd_label}: {nfd_body}"}
    result = mutable_proposed_text(edit, desc)
    assert result == nfd_body
    assert result != unicodedata.normalize("NFC", nfd_body)
    assert [ord(c) for c in result] == [ord(c) for c in nfd_body]


# ---------------------------------------------------------------------------
# 4  Mismatched label prefix remains completely unchanged.
# ---------------------------------------------------------------------------

def test_04_mismatched_label_unchanged():
    edit = {"proposed_text": "AI & Agentic Systems: Python, AWS"}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "AI & Agentic Systems: Python, AWS"


# ---------------------------------------------------------------------------
# 5–9  Specific prose/URL/version/time colon cases remain unchanged.
# ---------------------------------------------------------------------------

def test_05_prose_colon_unchanged():
    text = "Built agent with two stages: intake and validation"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


def test_06_url_colon_unchanged():
    text = "https://github.com/user/repo"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


def test_07_version_colon_unchanged():
    text = "Python (3.11): Docker, AWS"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


def test_08_time_colon_unchanged():
    text = "12:30 time format"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


def test_09_claim_bearing_colon_unchanged():
    text = "Managed 12 engineers: Built Python workflows"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


# ---------------------------------------------------------------------------
# 10  Multiple colons, first prefix is exact authenticated label.
# ---------------------------------------------------------------------------

def test_10_multiple_colons_exact_label_stripped():
    edit = {"proposed_text": "Software & Data: Python 3.11: Docker, AWS"}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "Python 3.11: Docker, AWS"


def test_10b_multiple_colons_nonmatching_prefix_unchanged():
    text = "Focus: Python 3.11: Docker, AWS"
    edit = {"proposed_text": text}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == text


# ---------------------------------------------------------------------------
# 11  Whitespace around exact authenticated prefix.
# ---------------------------------------------------------------------------

def test_11_outer_whitespace_handled():
    edit = {"proposed_text": "  Software & Data: Python, AWS  "}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "Python, AWS"


def test_11b_inner_whitespace_after_colon():
    edit = {"proposed_text": "Software & Data:   Python, AWS"}
    desc = _descriptor()
    result = mutable_proposed_text(edit, desc)
    assert result == "Python, AWS"


# ---------------------------------------------------------------------------
# 12  Empty text after exact authenticated label fails closed.
# ---------------------------------------------------------------------------

def test_12_empty_after_exact_label_fails_closed():
    edit = {"proposed_text": "Software & Data:"}
    desc = _descriptor()
    with pytest.raises(TargetResolutionError, match="skill_groups.2"):
        mutable_proposed_text(edit, desc)


# ---------------------------------------------------------------------------
# 13  Whitespace-only text after exact authenticated label fails closed.
# ---------------------------------------------------------------------------

def test_13_whitespace_only_after_exact_label_fails_closed():
    edit = {"proposed_text": "Software & Data:   "}
    desc = _descriptor()
    with pytest.raises(TargetResolutionError, match="skill_groups.2"):
        mutable_proposed_text(edit, desc)


# ---------------------------------------------------------------------------
# 14  Plain targets bypass normalization.
# ---------------------------------------------------------------------------

def test_14_plain_targets_bypass_normalization():
    text = "Not a label: just text"
    edit = {"proposed_text": text}
    desc = _descriptor(kind="plain", label=None)
    result = mutable_proposed_text(edit, desc)
    assert result == text


# ---------------------------------------------------------------------------
# 15  Missing authenticated labels fail closed.
# ---------------------------------------------------------------------------

from resume_tailor.patch_engine import parse_target_source_id

def test_15a_missing_label_key():
    content = {"skill_groups": [{"text": "Python"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15b_label_none():
    content = {"skill_groups": [{"label": None, "text": "Python"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15c_empty_label():
    content = {"skill_groups": [{"label": "", "text": "Python"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15d_whitespace_only_label():
    content = {"skill_groups": [{"label": "   ", "text": "Python"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15e_non_string_label():
    content = {"skill_groups": [{"label": 123, "text": "Python"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15f_missing_text():
    content = {"skill_groups": [{"label": "Skills"}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15g_text_none():
    content = {"skill_groups": [{"label": "Skills", "text": None}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15h_non_string_text():
    content = {"skill_groups": [{"label": "Skills", "text": []}]}
    with pytest.raises(TargetResolutionError, match="skill_groups.0 missing or invalid composite label/body"):
        parse_target_source_id("skill_groups.0", content)

def test_15i_valid_composite_remains_accepted():
    content = {"skill_groups": [{"label": "Skills", "text": "Python"}]}
    kind, container, key, label = parse_target_source_id("skill_groups.0", content)
    assert kind == "composite_labelled"
    assert label == "Skills"
    assert key == "text"
    assert container["text"] == "Python"


# ---------------------------------------------------------------------------
# 16  Missing budgets fail closed (via resolve_target_descriptor).
# ---------------------------------------------------------------------------

def test_16_missing_budget_fails_closed(master_resume: Path):
    extracted, _ = extract_resume(master_resume)
    # Remove all budgets
    for p in extracted.get("paragraphs", []):
        if p.get("content_id") == "skill_groups.2":
            p["content_budget"]["maximum_characters"] = 0
    edit = {
        "edit_id": "edit.001",
        "target_source_id": "skill_groups.2",
        "operation": "replace",
        "proposed_text": "Python",
        "alignment_rationale": "",
        "evidence_source_ids": [],
    }
    from resume_tailor.patch_engine import resolve_target_descriptor
    with pytest.raises(TargetResolutionError, match="no authenticated content budget"):
        resolve_target_descriptor(edit, extracted["content"], extracted)


# ---------------------------------------------------------------------------
# 17  Exact-label stripping does not bypass no-op rejection.
# ---------------------------------------------------------------------------

def test_17_exact_label_stripping_does_not_bypass_noop(master_resume: Path):
    from resume_tailor.utilities import OllamaTailoringContractError
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume, "dummy")
    current = extracted["content"]["skill_groups"][2]["text"]
    label = extracted["content"]["skill_groups"][2]["label"]

    analysis["recommended_edits"][0]["proposed_text"] = f"{label}: {current}"
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": catalog[0]["edit_id"],
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "replacement_text": current,
            }
        ]
    }
    master_copy = copy.deepcopy(extracted["content"])
    with pytest.raises(OllamaTailoringContractError, match="is a no-op replacement"):
        validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )
    assert extracted["content"] == master_copy


# ---------------------------------------------------------------------------
# 18  Exact-label stripping does not bypass append-prefix requirements.
# ---------------------------------------------------------------------------

def test_18_exact_label_stripping_with_append(master_resume: Path):
    from resume_tailor.utilities import OllamaTailoringContractError
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume, "dummy", operation="append")
    current = extracted["content"]["skill_groups"][2]["text"]

    extracted["source_blocks"].append({
        "source_id": "fake.18",
        "exact_text": f"AI, Different prefix",
    })
    analysis["recommended_edits"][0]["evidence_source_ids"].append("fake.18")

    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    def apply_append(replacement: str):
        payload = {
            "status": "complete",
            "message": "Complete",
            "catalog_sha256": valid_digest,
            "cannot_apply": None,
            "technical_failure": None,
            "patches": [
                {
                    "edit_id": catalog[0]["edit_id"],
                    "target_source_id": "skill_groups.2",
                    "operation": "append",
                    "replacement_text": replacement,
                }
            ]
        }
        return validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    master_copy = copy.deepcopy(extracted["content"])
    valid = apply_append(f"{current}, AI")
    assert valid["skill_groups"][2]["text"] == f"{current}, AI"

    extracted["content"] = copy.deepcopy(master_copy)
    with pytest.raises(OllamaTailoringContractError, match="does not preserve the original prefix"):
        apply_append("Different prefix, AI")

    extracted["content"] = copy.deepcopy(master_copy)
    with pytest.raises(OllamaTailoringContractError, match="is a no-op replacement"):
        apply_append(current)


# ---------------------------------------------------------------------------
# 19  Initial tailoring uses corrected behavior.
# ---------------------------------------------------------------------------

def test_19_initial_tailoring_exact_label_stripped(master_resume: Path):
    extracted, _ = extract_resume(master_resume)
    label = extracted["content"]["skill_groups"][2]["label"]
    proposed = f"{label}: FastAPI, pytest, Linux, AI"
    extracted2, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume, proposed
    )
    prompt = writer.build_ollama_tailoring_prompt(
        master_content=extracted2["content"],
        extracted_resume=extracted2,
        job_description=job_desc,
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic Corp",
        role="AI Engineer",
    )
    assert "mutable_proposed_body" in prompt
    assert "immutable_label" in prompt
    assert "FastAPI, pytest, Linux, AI" in prompt


def test_20_revision_uses_corrected_behavior(master_resume: Path):
    from resume_tailor.patch_engine import validate_and_apply_revision_patches
    from resume_tailor.revision import approved_revision_targets
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume, "dummy")
    current = extracted["content"]["skill_groups"][2]["text"]
    label = extracted["content"]["skill_groups"][2]["label"]

    extracted["source_blocks"].append({
        "source_id": "fake.20",
        "exact_text": f"AI, Wrong Label: {current}",
    })
    analysis["recommended_edits"][0]["evidence_source_ids"].append("fake.20")

    qa_result = {
        "status": "material_findings",
        "rationale": "Missing skills",
        "issues": [
            {
                "issue_id": "issue.001",
                "severity": "blocking",
                "description": "Add AI",
                "affected_content_id": "skill_groups.2"
            }
        ]
    }

    proposed_exact = f"{label}: {current}, AI"
    analysis["recommended_edits"][0]["proposed_text"] = proposed_exact
    prompt = writer.build_ollama_revision_prompt(
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa_result,
        company="Company",
        role="Role"
    )
    assert f'"mutable_proposed_body":"{current}, AI"' in prompt

    mismatched = f"Wrong Label: {current}, AI"
    analysis["recommended_edits"][0]["proposed_text"] = mismatched
    prompt_mismatched = writer.build_ollama_revision_prompt(
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa_result,
        company="Company",
        role="Role"
    )
    assert f'"mutable_proposed_body":"Wrong Label: {current}, AI"' in prompt_mismatched

    target_map = approved_revision_targets(qa_result=qa_result, approved_analysis=analysis)
    from resume_tailor.patch_engine import canonical_digest
    auth_sha256 = canonical_digest(target_map)
    payload = {
        "status": "complete",
        "message": "Complete",
        "cannot_apply": None,
        "technical_failure": None,
        "authorization_sha256": auth_sha256,
        "patches": [
            {
                "issue_id": "issue.001",
                "target_source_id": "skill_groups.2",
                "replacement_text": f"{current}, AI",
            }
        ]
    }
    master_copy = copy.deepcopy(extracted["content"])
    revised = validate_and_apply_revision_patches(
        payload=payload,
        current_tailored_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa_result,
    )
    assert revised["skill_groups"][2]["label"] == label
    assert revised["skill_groups"][2]["text"] == f"{current}, AI"
    assert master_copy == extracted["content"]


# ---------------------------------------------------------------------------
# 21  Authenticated labels unchanged in merged output.
# ---------------------------------------------------------------------------

def test_21_authenticated_labels_unchanged_in_merge(master_resume: Path):
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume,
        "FastAPI, pytest, Linux, AI",
    )
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)
    current_text = extracted["content"]["skill_groups"][2]["text"]
    original_label = extracted["content"]["skill_groups"][2]["label"]
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": catalog[0]["edit_id"],
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "replacement_text": "FastAPI, pytest, Linux, AI",
            },
        ],
    }
    tailored = validate_and_apply_patches(
        payload=payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
    )
    assert tailored["skill_groups"][2]["label"] == original_label
    assert tailored["skill_groups"][2]["text"] == "FastAPI, pytest, Linux, AI"


def test_23_master_content_not_mutated(master_resume: Path):
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume,
        "FastAPI, pytest, Linux, AI",
    )
    master_copy = copy.deepcopy(extracted["content"])
    _ = writer.build_ollama_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=job_desc,
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic Corp",
        role="AI Engineer",
    )
    assert extracted["content"] == master_copy


# ---------------------------------------------------------------------------
# 24  Evidence, metric, skill, forbidden-claim, budget checks authoritative.
# ---------------------------------------------------------------------------

def test_24_downstream_validation_unchanged(master_resume: Path):
    """validate_and_apply_patches still rejects unsupported skills even when
    the proposed_text contained a mismatched prefix that was preserved."""
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume,
        "FastAPI, pytest, Linux, AI",
    )
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": catalog[0]["edit_id"],
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "replacement_text": "FastAPI, pytest, Linux, AI, Go",
            },
        ],
    }
    from resume_tailor.utilities import OllamaTailoringContractError
    with pytest.raises(OllamaTailoringContractError, match="without authenticated source evidence"):
        validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


@pytest.mark.parametrize(
    ("target_source_id", "unsupported_item"),
    [
        ("skill_groups.2", "COBOL"),
        ("education.coursework", "Quantum Cryptography"),
        ("education.certifications", "CISSP"),
    ],
)
def test_24b_full_payload_rejects_unsupported_structured_item(
    master_resume: Path,
    target_source_id: str,
    unsupported_item: str,
) -> None:
    """The final applicator grounds every Python-owned list target itself."""
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(
        master_resume,
        unsupported_item,
    )
    analysis = copy.deepcopy(analysis)
    analysis["recommended_edits"] = [
        {
            "target_source_id": target_source_id,
            "operation": "replace",
            "proposed_text": unsupported_item,
            "alignment_rationale": "Synthetic unsupported structured item.",
            "evidence_source_ids": [target_source_id],
            "resolved_evidence": [],
        }
    ]
    catalog = writer.approved_edit_catalog(analysis)
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": writer.canonical_digest(catalog),
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": catalog[0]["edit_id"],
                "target_source_id": target_source_id,
                "operation": "replace",
                "replacement_text": unsupported_item,
            }
        ],
    }

    with pytest.raises(
        OllamaTailoringContractError,
        match=rf"{target_source_id!s}.*without authenticated source evidence",
    ):
        validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# ---------------------------------------------------------------------------
# 25  No provider call occurs during preflight failures.
# ---------------------------------------------------------------------------

def test_25_no_provider_call_during_preflight_failure(master_resume: Path, monkeypatch):
    extracted, _ = extract_resume(master_resume)
    label = extracted["content"]["skill_groups"][2]["label"]
    extracted2, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume,
        f"{label}:   ",
    )

    sentinel_called = False
    def mock_invoke(*args, **kwargs):
        nonlocal sentinel_called
        sentinel_called = True
        raise RuntimeError("Provider layer reached!")

    monkeypatch.setattr(writer, "_invoke_payload", mock_invoke)

    with pytest.raises(TailoringPreflightError, match="No writer request"):
        writer.invoke_ollama(
            master_content=extracted2["content"],
            extracted_resume=extracted2,
            job_description=job_desc,
            job_requirements=reqs,
            approved_analysis=analysis,
            company="Synthetic Corp",
            role="AI Engineer",
            run_directory=Path("/tmp"),
            timeout_seconds=30,
        )
    assert not sentinel_called


# ---------------------------------------------------------------------------
# 27–35  Composite no-op elimination and changed-label rejections
# ---------------------------------------------------------------------------


def test_27_full_labeled_proposal_identical_to_current_composite_value(
    master_resume: Path,
) -> None:
    """Identical full labeled proposal is discarded as a body-only no-op."""
    extracted, _ = extract_resume(master_resume)
    label = extracted["content"]["skill_groups"][2]["label"]
    text = extracted["content"]["skill_groups"][2]["text"]
    proposed = f"{label}: {text}"
    _extracted2, _job_desc, _reqs, analysis = _setup_synthetic_inputs(
        master_resume, proposed
    )
    assert not analysis.get("recommended_edits")
    assert "edit.001" in analysis.get("discarded_no_op_edit_ids", [])
    assert "skill_groups.2" not in analysis.get("discarded_no_op_edit_ids", [])
    for em in analysis.get("evidence_map", []):
        assert "edit.001" not in em.get("evidence_source_ids", [])


def test_28_body_only_proposal_identical_to_current_mutable_body(
    master_resume: Path,
) -> None:
    """Body-only proposal equal to the current mutable body is discarded."""
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][2]["text"]
    _extracted2, _job_desc, _reqs, analysis = _setup_synthetic_inputs(
        master_resume, text
    )
    assert not analysis.get("recommended_edits")
    assert "edit.001" in analysis.get("discarded_no_op_edit_ids", [])
    for em in analysis.get("evidence_map", []):
        assert "edit.001" not in em.get("evidence_source_ids", [])


def test_29_body_only_changed_proposal_remains_active(master_resume: Path) -> None:
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][2]["text"]
    proposed = f"{text[:-5]} NEW"
    _extracted2, _job_desc, _reqs, analysis = _setup_synthetic_inputs(
        master_resume, proposed
    )
    assert len(analysis.get("recommended_edits")) == 1
    assert analysis["recommended_edits"][0]["proposed_text"] == proposed
    assert "edit.001" not in analysis.get("discarded_no_op_edit_ids", [])


def test_30_full_exact_label_plus_changed_body_normalizes_and_remains_active(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    label = extracted["content"]["skill_groups"][2]["label"]
    text = extracted["content"]["skill_groups"][2]["text"]
    proposed = f"{label}: {text[:-5]} NEW"
    _extracted2, _job_desc, _reqs, analysis = _setup_synthetic_inputs(
        master_resume, proposed
    )
    assert len(analysis.get("recommended_edits")) == 1
    assert analysis["recommended_edits"][0]["proposed_text"] == f"{text[:-5]} NEW"
    assert "edit.001" not in analysis.get("discarded_no_op_edit_ids", [])
    assert "edit.001" in analysis.get("normalized_composite_edit_ids", [])
    assert "skill_groups.2" not in analysis.get("normalized_composite_edit_ids", [])


def test_31_changed_immutable_label_is_rejected(master_resume: Path) -> None:
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][2]["text"]
    proposed = f"Different Label: {text}"
    reqs = build_job_requirement_catalog(
        "Synthetic job description requiring AI."
    )
    analysis = {
        "recommended_edits": [
            {
                "edit_id": "edit.001",
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "proposed_text": proposed,
                "evidence_source_ids": ["skill_groups.2", "projects.0.bullets.0"],
                "resolved_evidence": [],
            }
        ],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
        "evidence_map": [],
    }
    resolved_analysis, issues = resolve_analysis_evidence(
        analysis, extracted, reqs
    )
    assert any(i.code == "invalid_composite_label" for i in issues)
    issue = next(i for i in issues if i.code == "invalid_composite_label")
    assert issue.location == "recommended_edits[0].proposed_text"
    assert not resolved_analysis.get("recommended_edits")
    assert "edit.001" in resolved_analysis.get("invalid_composite_edit_ids", [])
    assert "skill_groups.2" not in resolved_analysis.get(
        "invalid_composite_edit_ids", []
    )


@pytest.mark.parametrize(
    "proposed_builder",
    [
        lambda label, text: f"Unrelated Category: {text} NEW",
        lambda label, text: f"{label}: {label}: {text} NEW",
        lambda label, text: f": {text} NEW",
        lambda label, text: label,
        lambda label, text: "TECHNICAL SKILLS",
    ],
    ids=[
        "unrelated_label",
        "duplicated_label",
        "malformed_empty_label",
        "bare_immutable_label",
        "section_heading_text",
    ],
)
def test_32_invalid_composite_label_variants_are_rejected(
    master_resume: Path,
    proposed_builder,
) -> None:
    extracted, _ = extract_resume(master_resume)
    label = extracted["content"]["skill_groups"][2]["label"]
    text = extracted["content"]["skill_groups"][2]["text"]
    proposed = proposed_builder(label, text)
    reqs = build_job_requirement_catalog(
        "Synthetic job description requiring AI."
    )
    analysis = {
        "recommended_edits": [
            {
                "edit_id": "edit.001",
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "proposed_text": proposed,
                "evidence_source_ids": ["skill_groups.2", "projects.0.bullets.0"],
                "resolved_evidence": [],
            }
        ],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
        "evidence_map": [],
    }
    resolved_analysis, issues = resolve_analysis_evidence(
        analysis, extracted, reqs
    )
    assert any(i.code == "invalid_composite_label" for i in issues)
    assert not resolved_analysis.get("recommended_edits")
    assert "edit.001" in resolved_analysis.get("invalid_composite_edit_ids", [])


def test_33_evidence_map_scope_for_no_op(master_resume: Path) -> None:
    """Discarding a no-op edit removes only that edit's evidence-map association."""
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][0]["text"]
    # Start from a real catalog, then add a second synthetic requirement so the
    # test does not depend on extractor splitting of the job description text.
    reqs = build_job_requirement_catalog(
        "Synthetic job description requiring AI. Also requires Python."
    )
    reqs["requirements"].append(
        {
            "requirement_id": "text.002",
            "category": "unstructured_requirement",
            "exact_text": "Python experience",
        }
    )
    req_id_1 = reqs["requirements"][0]["requirement_id"]
    req_id_2 = reqs["requirements"][1]["requirement_id"]

    analysis = {
        "supported_requirement_mappings": [
            {
                "requirement_id": req_id_1,
                "requirement": "req 1",
                "evidence_source_ids": ["edit.002", "skill_groups.0"],
            },
            {
                "requirement_id": req_id_2,
                "requirement": "req 2",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ],
        "unsupported_requirement_ids": [],
        "recommended_edits": [
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.0",
                "operation": "replace",
                "proposed_text": text,
                "evidence_source_ids": ["skill_groups.0"],
                "resolved_evidence": [],
            },
            {
                "edit_id": "edit.003",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": "Changed text here",
                "evidence_source_ids": ["skill_groups.0"],
                "resolved_evidence": [],
            },
        ],
        "evidence_map": [
            {
                "requirement_id": req_id_1,
                "evidence_source_ids": ["edit.002", "skill_groups.0"],
            },
            {
                "requirement_id": req_id_2,
                "evidence_source_ids": ["skill_groups.0"],
            },
        ],
        "requirement_assessment": [
            {
                "requirement_id": req_id_1,
                "requirement": "req 1",
                "status": "supported",
                "strength": "strong",
                "support_provenance": "test",
                "evidence_source_ids": ["edit.002", "skill_groups.0"],
                "resolved_evidence": [],
            },
            {
                "requirement_id": req_id_2,
                "requirement": "req 2",
                "status": "supported",
                "strength": "strong",
                "support_provenance": "test",
                "evidence_source_ids": ["skill_groups.0"],
                "resolved_evidence": [],
            },
        ],
    }

    resolved, issues = resolve_analysis_evidence(analysis, extracted, reqs)
    assert not any(i.code == "invalid_composite_label" for i in issues)

    # No-op edit is discarded by edit_id, not by target_source_id.
    assert "edit.002" in resolved.get("discarded_no_op_edit_ids", [])
    assert "skill_groups.0" not in resolved.get("discarded_no_op_edit_ids", [])
    assert "edit.002" not in {
        e.get("edit_id") for e in resolved.get("recommended_edits", [])
    }

    em = {
        e["requirement_id"]: e["evidence_source_ids"]
        for e in resolved.get("evidence_map", [])
    }
    assert req_id_1 in em
    assert req_id_2 in em
    # Orphan association for the discarded edit is gone.
    assert "edit.002" not in em[req_id_1]
    # Unrelated valid evidence on the same source remains.
    assert "skill_groups.0" in em[req_id_1]
    assert "skill_groups.0" in em[req_id_2]

    # Another active edit may still cite the same source block.
    active_edits = {e["edit_id"]: e for e in resolved.get("recommended_edits", [])}
    assert "edit.003" in active_edits
    assert "skill_groups.0" in active_edits["edit.003"]["evidence_source_ids"]


def test_34_all_no_op_result_causes_zero_writer_calls(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][2]["text"]
    extracted2, job_desc, reqs, analysis = _setup_synthetic_inputs(
        master_resume, text
    )
    assert not analysis.get("recommended_edits")
    assert analysis.get("discarded_no_op_edit_ids")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Writer must not be invoked for an all-no-op catalog")

    monkeypatch.setattr(writer, "run_ollama_request", fail_if_called)
    monkeypatch.setattr(writer, "_invoke_payload", fail_if_called)

    tailored = writer.invoke_ollama(
        master_content=extracted2["content"],
        extracted_resume=extracted2,
        job_description=job_desc,
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic Corp",
        role="AI Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert tailored == extracted2["content"]


def test_35_historical_resolved_artifacts_remain_readable(
    master_resume: Path,
) -> None:
    """Historical target-ID metadata remains readable; new writes use edit IDs."""
    extracted, _ = extract_resume(master_resume)
    text = extracted["content"]["skill_groups"][2]["text"]
    reqs = build_job_requirement_catalog(
        "Synthetic job description requiring AI."
    )
    # Historical artifact shape: discarded list stored target_source_id values.
    analysis = {
        "discarded_no_op_edit_ids": ["skill_groups.1"],
        "normalized_composite_edit_ids": ["skill_groups.0"],
        "recommended_edits": [
            {
                "edit_id": "edit.001",
                "target_source_id": "skill_groups.2",
                "operation": "replace",
                "proposed_text": text,
                "evidence_source_ids": ["skill_groups.2", "projects.0.bullets.0"],
                "resolved_evidence": [],
            }
        ],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
        "evidence_map": [],
    }
    resolved, issues = resolve_analysis_evidence(analysis, extracted, reqs)
    assert not issues
    # Historical target IDs are preserved for compatibility.
    assert "skill_groups.1" in resolved.get("discarded_no_op_edit_ids", [])
    assert "skill_groups.0" in resolved.get("normalized_composite_edit_ids", [])
    # New no-op writes actual edit IDs, not target IDs.
    assert "edit.001" in resolved.get("discarded_no_op_edit_ids", [])
    assert "skill_groups.2" not in resolved.get("discarded_no_op_edit_ids", [])
    assert not resolved.get("recommended_edits")
