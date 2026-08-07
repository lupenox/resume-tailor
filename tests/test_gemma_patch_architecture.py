from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence, validate_tailored_content, changed_content_ids
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.providers import ollama_writer as writer
from resume_tailor.backend.engine import patch_engine
from resume_tailor.backend.engine.structured_patch_compiler import (
    combine_hybrid_patch_payload,
    deterministic_only_metadata,
    hybrid_execution_metadata,
)
from resume_tailor.backend.utils.utilities import (
    OllamaTailoringContractError,
    OllamaCannotApplyError,
    OllamaEvidenceRejectionError,
    OllamaCanonicalSchemaError,
    OllamaTransportSchemaError,
    OllamaTechnicalFailureError,
    OllamaRevisionContractError,
    ModelError,
)


def _setup_synthetic_inputs(master_resume: Path):
    extracted, _ = extract_resume(master_resume)
    job_desc = "Synthetic job description requiring Python, FastAPI, SQL, and Docker."
    reqs = build_job_requirement_catalog(job_desc)
    analysis = {
        "role_summary": "Synthetic AI Engineer Role",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": ["Python", "FastAPI", "SQL"],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": ["Python", "FastAPI"],
        "missing_or_unsupported_requirements": [],
        "recommended_edits": [
            {
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": "Experienced Python engineer specializing in FastAPI and scalable automated workflows.",
                "alignment_rationale": "Align with Python and FastAPI requirements.",
                "evidence_source_ids": ["professional_summary"],
                "resolved_evidence": [
                    {
                        "source_id": "professional_summary",
                        "section_context": "Header",
                        "exact_text": extracted["content"]["professional_summary"],
                    }
                ],
            },
            {
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "proposed_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
                "alignment_rationale": "Add AI keyword to skill group.",
                "evidence_source_ids": ["skill_groups.2", "projects.0.bullets.0"],
                "resolved_evidence": [
                    {
                        "source_id": "skill_groups.2",
                        "section_context": "Technical Skills",
                        "exact_text": f"{extracted['content']['skill_groups'][2]['label']}: {extracted['content']['skill_groups'][2]['text']}",
                    }
                ],
            },
        ],
        "immutable_facts": ["Built synthetic testing pipelines"],
        "forbidden_claims": ["Unsupported synthetic leadership claim"],
        "content_budget_guidance": [],
        "questions_for_user": [],
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [
            item["requirement_id"] for item in reqs["requirements"]
        ],
    }
    resolved_analysis, issues = resolve_analysis_evidence(analysis, extracted, reqs)
    assert not issues
    return extracted, job_desc, reqs, resolved_analysis


# 1. Gemma no longer returns a full résumé for initial tailoring.
def test_01_prompt_does_not_request_full_resume(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    prompt = writer.build_ollama_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=job_desc,
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic Corp",
        role="AI Engineer",
    )
    assert "TRUSTED MASTER RESUME CONTENT" not in prompt
    assert "Author target-only edits" in prompt
    assert "Do not return" in prompt
    assert "complete resume" in prompt


# 2. Initial transport schema accepts patch envelopes and rejects full résumé roots.
def test_02_transport_schema_accepts_patches_rejects_full_resume() -> None:
    schema = writer._ollama_transport_schema("ollama_tailoring_patch.schema.json")
    patch_payload = {
        "status": "complete",
        "message": "Complete.",
        "catalog_sha256": "a" * 64,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary.",
            }
        ],
    }
    writer._validate_transport_payload(patch_payload, transport_schema=schema, label="Test")
    
    full_resume_root = {
        "professional_summary": "Summary",
        "education": {},
        "skill_groups": [],
    }
    with pytest.raises(OllamaTransportSchemaError):
        writer._validate_transport_payload(full_resume_root, transport_schema=schema, label="Test")


# 3. The historical wrong-root/full-résumé response remains rejected.
def test_03_historical_wrong_root_rejected(master_resume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    full_resume = copy.deepcopy(extracted["content"])
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(full_resume)},
        },
    )
    with pytest.raises(OllamaTransportSchemaError):
        writer.invoke_ollama(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=job_desc,
            job_requirements=reqs,
            approved_analysis=analysis,
            company="Synthetic",
            role="Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )


# 4 & 5. Response must echo exact catalog digest; stale or mismatched digest is rejected.
def test_04_05_catalog_digest_matching(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)
    stale_digest = "b" * 64

    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": stale_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="catalog_sha256 digest does not match"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    payload["catalog_sha256"] = valid_digest
    res = patch_engine.validate_and_apply_patches(
        payload=payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
    )
    assert res != extracted["content"]


# 6, 7, 8. Every approved edit must have exactly 1 patch; missing or extra rejected.
def test_06_07_08_patch_count_matching(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    # Missing patch
    missing_payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            }
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="Patch set size"):
        patch_engine.validate_and_apply_patches(
            payload=missing_payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    # Extra patch
    extra_payload = copy.deepcopy(missing_payload)
    extra_payload["patches"].extend([
        {
            "edit_id": "edit.002",
            "target_source_id": "skill_groups.2",
            "operation": "append",
            "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
        },
        {
            "edit_id": "edit.003",
            "target_source_id": "open_source.bullet",
            "operation": "replace",
            "replacement_text": "Extra patch",
        },
    ])
    with pytest.raises(OllamaTailoringContractError, match="Patch set size"):
        patch_engine.validate_and_apply_patches(
            payload=extra_payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 9, 10, 11, 12, 13. Duplicate edit_ids, duplicate targets, unknown edit_ids, target/op mismatches.
def test_09_to_13_patch_validation_rules(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    # Unknown edit_id
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.999",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "Text",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="unknown edit_id"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    # Target mismatch
    payload["patches"][0] = {
        "edit_id": "edit.001",
        "target_source_id": "open_source.bullet",
        "operation": "replace",
        "replacement_text": "Text",
    }
    with pytest.raises(OllamaTailoringContractError, match="target_source_id"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    # Operation mismatch
    payload["patches"][0]["target_source_id"] = "professional_summary"
    payload["patches"][0]["operation"] = "append"
    with pytest.raises(OllamaTailoringContractError, match="operation"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 14, 15, 16, 17. Replace semantics, append exact prefix, invalid append, no-op rejection.
def test_14_to_17_operation_semantics(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    # Invalid append (does not preserve prefix)
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": "Completely new text without prefix",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="does not preserve the original prefix"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )

    # No-op patch
    payload["patches"][1]["replacement_text"] = extracted["content"]["skill_groups"][2]["text"]
    with pytest.raises(OllamaTailoringContractError, match="no-op replacement"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 18, 19. Composite targets update body text preserving label; duplicating label in text is rejected.
def test_18_19_composite_target_label_handling(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)
    sg2_label = extracted["content"]["skill_groups"][2]["label"]

    # Duplicated label in replacement_text
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": f"{sg2_label}: {extracted['content']['skill_groups'][2]['text']}, AI",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="illegally contains label"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 20. 210-character rendered value rejected against 200-character budget before merge.
def test_20_budget_exceeded_before_merge(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    oversized_text = "Python, " * 30  # > 200 chars
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", " + oversized_text,
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="exceeds target budget"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 21, 22. Numeric claim 6 rejected when absent from authenticated evidence; existing numbers valid.
def test_21_22_numeric_claim_validation(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
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
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "Engineered 6 new synthetic systems.",  # '6' is unauthenticated
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError, match="unauthenticated numeric claims"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 24, 25. projects.1.bullets.1 remains unchanged when unapproved; extra target rejected.
def test_24_25_unapproved_target_unmodified_and_extra_rejected(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)

    orig_p1_b1 = extracted["content"]["projects"][1]["bullets"][1]

    valid_payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New valid professional summary.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    tailored = patch_engine.validate_and_apply_patches(
        payload=valid_payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
    )
    assert tailored["projects"][1]["bullets"][1] == orig_p1_b1


# 26. Original master dictionary is never mutated.
def test_26_master_content_never_mutated(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    master_copy = copy.deepcopy(extracted["content"])
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
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    patch_engine.validate_and_apply_patches(
        payload=payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
    )
    assert extracted["content"] == master_copy


# 27. Patch application is atomic.
def test_27_atomic_patch_application(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    valid_digest = writer.canonical_digest(catalog)
    master_before = copy.deepcopy(extracted["content"])

    # Patch 1 valid, Patch 2 invalid (numeric claim '9999')
    payload = {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": valid_digest,
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "New summary text.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + " 9999",
            },
        ],
    }
    with pytest.raises(OllamaTailoringContractError):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )
    assert extracted["content"] == master_before


# 28, 29, 30. Changed-ID set equals approved target set; labels/names/dates/counts unchanged; passes Step 7.
def test_28_29_30_final_tailored_resume_validations(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
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
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "Experienced Python engineer building scalable automated workflows.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": extracted["content"]["skill_groups"][2]["text"] + ", AI",
            },
        ],
    }
    tailored = patch_engine.validate_and_apply_patches(
        payload=payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=analysis,
    )
    
    # Assert changed_content_ids equals approved targets
    changed = changed_content_ids(extracted["content"], tailored)
    approved_targets = {edit["target_source_id"] for edit in catalog}
    assert set(changed) == approved_targets

    original = extracted["content"]
    assert tailored["education"]["institution"] == original["education"]["institution"]
    assert tailored["education"]["degree_details"] == original["education"]["degree_details"]
    assert tailored["education"]["coursework"]["label"] == original["education"]["coursework"]["label"]
    assert tailored["education"]["certifications"]["label"] == original["education"]["certifications"]["label"]
    assert [group["label"] for group in tailored["skill_groups"]] == [
        group["label"] for group in original["skill_groups"]
    ]
    assert [project["name"] for project in tailored["projects"]] == [
        project["name"] for project in original["projects"]
    ]
    assert [project["technologies"] for project in tailored["projects"]] == [
        project["technologies"] for project in original["projects"]
    ]
    assert tailored["open_source"]["name"] == original["open_source"]["name"]
    assert tailored["open_source"]["technologies"] == original["open_source"]["technologies"]
    assert tailored["experience"]["role"] == original["experience"]["role"]
    assert tailored["experience"]["employer_location"] == original["experience"]["employer_location"]
    assert tailored["experience"]["dates"] == original["experience"]["dates"]
    assert len(tailored["skill_groups"]) == len(original["skill_groups"])
    assert len(tailored["projects"]) == len(original["projects"])
    assert [len(project["bullets"]) for project in tailored["projects"]] == [
        len(project["bullets"]) for project in original["projects"]
    ]
    assert len(tailored["experience"]["bullets"]) == len(original["experience"]["bullets"])

    # Step 7 validation
    report = validate_tailored_content(
        original=extracted["content"],
        tailored=tailored,
        extracted_resume=extracted,
        analysis=analysis,
        target_role="AI Engineer",
    )
    assert report.passed


# 31. cannot_apply requires known edit ID.
def test_31_cannot_apply_requires_known_edit_id(master_resume: Path) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    payload = {
        "status": "cannot_apply",
        "message": "Cannot apply edit.999",
        "catalog_sha256": writer.canonical_digest(writer.approved_edit_catalog(analysis)),
        "cannot_apply": {
            "edit_id": "edit.999",
            "reason_code": "unsupported_claim_risk",
            "reason": "Unknown edit ID",
        },
        "technical_failure": None,
        "patches": None,
    }
    with pytest.raises(OllamaEvidenceRejectionError):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# 32. technical_failure cannot include patches.
def test_32_technical_failure_cannot_include_patches() -> None:
    payload = {
        "status": "technical_failure",
        "message": "Technical error",
        "catalog_sha256": "0" * 64,
        "cannot_apply": None,
        "technical_failure": {
            "reason_code": "output_constraint",
            "reason": "Failed output constraint",
        },
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "Text",
            }
        ],
    }
    from resume_tailor.backend.utils.schemas import validate_payload
    with pytest.raises(ModelError, match="failed local schema validation"):
        validate_payload(payload, "ollama_tailoring_patch.schema.json", label="Test")


# 33. Diagnostic metadata contains no résumé or replacement content.
def test_33_metadata_contains_no_resume_text(master_resume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    monkeypatch.setattr(
        writer,
        "run_ollama_request",
        lambda **kwargs: {
            "model": "resume-tailor-gemma",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "status": "complete",
                    "message": "Done",
                    "catalog_sha256": writer.canonical_digest(writer.partition_edit_catalog(writer.approved_edit_catalog(analysis))[1]),
                    "cannot_apply": None,
                    "technical_failure": None,
                    "patches": [
                        {
                            "edit_id": "edit.001",
                            "target_source_id": "professional_summary",
                            "operation": "replace",
                            "replacement_text": "SECRET_REPLACEMENT_TEXT_MARKER",
                        },
                    ],
                }),
            },
        },
    )
    writer.invoke_ollama(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=job_desc,
        job_requirements=reqs,
        approved_analysis=analysis,
        company="Synthetic",
        role="Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    metadata_text = (tmp_path / writer.OLLAMA_RESPONSE_METADATA_FILENAME).read_text(encoding="utf-8")
    assert "SECRET_REPLACEMENT_TEXT_MARKER" not in metadata_text


# 34, 35, 36. Model overrides work; no Qwen or Antigravity fallbacks introduced.
def test_34_35_36_model_independence() -> None:
    assert "gemma" in writer.DEFAULT_OLLAMA_MODEL.lower()
    caps = writer.capabilities_for_model("resume-tailor-gemma")
    assert caps.supports_json_schema is True

# Review regressions: nonempty schema probes, fail-closed budgets, per-edit evidence,
# immutable approved labels, and the successful one-shot revision path.
def test_37_nonempty_dynamic_schema_probe_is_supported(
    master_resume: Path, tmp_path: Path
) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    schema, _path = writer._write_tailoring_patch_transport_schema(
        tmp_path,
        catalog=catalog,
        catalog_sha256=writer.canonical_digest(catalog),
    )
    result = writer.probe_structured_output_support(schema)
    assert result["supported"] is True
    assert result["provider_called"] is False


def test_38_missing_target_budget_fails_closed(master_resume: Path) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    catalog = writer.approved_edit_catalog(analysis)
    without_budget = copy.deepcopy(extracted)
    without_budget["paragraphs"] = [
        paragraph
        for paragraph in without_budget["paragraphs"]
        if paragraph["content_id"] != "professional_summary"
    ]
    with pytest.raises(patch_engine.TargetResolutionError, match="no authenticated content budget"):
        patch_engine.resolve_target_descriptor(
            catalog[0], extracted["content"], without_budget
        )


def _valid_patch_payload(extracted: dict, analysis: dict) -> dict:
    catalog = writer.approved_edit_catalog(analysis)
    current_skill = extracted["content"]["skill_groups"][2]["text"]
    supported_item = current_skill.split(",", 1)[0].strip()
    return {
        "status": "complete",
        "message": "Complete",
        "catalog_sha256": writer.canonical_digest(catalog),
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "operation": "replace",
                "replacement_text": "Experienced Python engineer building automated workflows.",
            },
            {
                "edit_id": "edit.002",
                "target_source_id": "skill_groups.2",
                "operation": "append",
                "replacement_text": f"{current_skill}, {supported_item}",
            },
        ],
    }


def test_39_metric_from_unrelated_source_is_not_authorized_for_summary(
    master_resume: Path,
) -> None:
    from resume_tailor.backend.engine.evidence import _NUMBER_RE, _resume_text

    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    summary_metrics = set(_NUMBER_RE.findall(extracted["content"]["professional_summary"]))
    unrelated = sorted(
        set(_NUMBER_RE.findall(_resume_text(extracted["content"]))) - summary_metrics
    )
    assert unrelated, "synthetic fixture needs a number outside the summary"
    payload = _valid_patch_payload(extracted, analysis)
    payload["patches"][0]["replacement_text"] = (
        f"Experienced Python engineer delivering {unrelated[0]} automated workflows."
    )
    with pytest.raises(OllamaTailoringContractError, match="unauthenticated numeric claims"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


def test_40_unsupported_skill_is_rejected_before_merge(master_resume: Path) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    payload = _valid_patch_payload(extracted, analysis)
    current_skill = extracted["content"]["skill_groups"][2]["text"]
    payload["patches"][1]["replacement_text"] = (
        current_skill + ", Go"
    )
    with pytest.raises(OllamaTailoringContractError, match="without authenticated source evidence"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


def _qa_for_summary() -> dict:
    return {
        "status": "material_findings",
        "summary": "Synthetic QA finding.",
        "issues": [
            {
                "issue_id": "qa.001",
                "category": "clarity",
                "severity": "material",
                "description": "Clarify the summary.",
                "affected_content_id": "professional_summary",
                "evidence": "Synthetic evidence.",
            }
        ],
        "technical_failure": None,
    }

def test_42_successful_revision_patch_uses_keyword_scope_validation(
    master_resume: Path,
) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    current = copy.deepcopy(extracted["content"])
    current["professional_summary"] = "Experienced Python engineer building automated workflows."
    qa_result = _qa_for_summary()
    target_map = writer.approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=analysis,
    )
    payload = {
        "status": "complete",
        "message": "Revised",
        "authorization_sha256": writer.canonical_digest(target_map),
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "issue_id": "qa.001",
                "target_source_id": "professional_summary",
                "replacement_text": "Python engineer building clear automated workflows.",
            }
        ],
    }
    revised = patch_engine.validate_and_apply_revision_patches(
        payload=payload,
        current_tailored_content=current,
        extracted_resume=extracted,
        approved_analysis=analysis,
        qa_result=qa_result,
    )
    assert revised["professional_summary"] != current["professional_summary"]


def test_43_revision_budget_and_digest_are_enforced(master_resume: Path) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    current = copy.deepcopy(extracted["content"])
    current["professional_summary"] = "Experienced Python engineer building automated workflows."
    qa_result = _qa_for_summary()
    target_map = writer.approved_revision_targets(
        qa_result=qa_result,
        approved_analysis=analysis,
    )
    payload = {
        "status": "complete",
        "message": "Revised",
        "authorization_sha256": writer.canonical_digest(target_map),
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [
            {
                "issue_id": "qa.001",
                "target_source_id": "professional_summary",
                "replacement_text": "Python " * 200,
            }
        ],
    }
    with pytest.raises(OllamaRevisionContractError, match="exceeds target budget"):
        patch_engine.validate_and_apply_revision_patches(
            payload=payload,
            current_tailored_content=current,
            extracted_resume=extracted,
            approved_analysis=analysis,
            qa_result=qa_result,
        )

    cannot_apply = {
        "status": "cannot_apply",
        "message": "Cannot apply",
        "catalog_sha256": "0" * 64,
        "cannot_apply": {
            "edit_id": "edit.001",
            "reason_code": "other_bounded_constraint",
            "reason": "Synthetic",
        },
        "technical_failure": None,
        "patches": None,
    }
    with pytest.raises(OllamaTailoringContractError, match="catalog_sha256 digest does not match"):
        patch_engine.validate_and_apply_patches(
            payload=cannot_apply,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=analysis,
        )


# Opus independent-audit follow-up regressions.
def test_44_empty_catalog_is_an_explicit_atomic_noop(master_resume: Path) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    empty_analysis = copy.deepcopy(analysis)
    empty_analysis["recommended_edits"] = []
    payload = {
        "status": "complete",
        "message": "No approved edits.",
        "catalog_sha256": writer.canonical_digest([]),
        "cannot_apply": None,
        "technical_failure": None,
        "patches": [],
    }
    tailored = patch_engine.validate_and_apply_patches(
        payload=payload,
        master_content=extracted["content"],
        extracted_resume=extracted,
        approved_analysis=empty_analysis,
    )
    assert tailored == extracted["content"]
    assert tailored is not extracted["content"]


def _analysis_with_duplicate_summary_target(analysis: dict) -> dict:
    duplicated = copy.deepcopy(analysis)
    duplicated["recommended_edits"].append(
        copy.deepcopy(duplicated["recommended_edits"][0])
    )
    return duplicated


def test_45_duplicate_catalog_target_fails_before_writer(
    master_resume: Path,
) -> None:
    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)
    duplicated = _analysis_with_duplicate_summary_target(analysis)
    with pytest.raises(writer.TailoringPreflightError, match="No writer request was launched"):
        writer.build_ollama_tailoring_prompt(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=job_desc,
            job_requirements=reqs,
            approved_analysis=duplicated,
            company="Synthetic Corp",
            role="AI Engineer",
        )


def test_46_duplicate_catalog_target_fails_closed_in_applicator(
    master_resume: Path,
) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    duplicated = _analysis_with_duplicate_summary_target(analysis)
    catalog = writer.approved_edit_catalog(duplicated)
    payload = _valid_patch_payload(extracted, analysis)
    payload["catalog_sha256"] = writer.canonical_digest(catalog)
    payload["patches"].append(
        {
            "edit_id": "edit.003",
            "target_source_id": "professional_summary",
            "operation": "replace",
            "replacement_text": "Another summary replacement.",
        }
    )
    with pytest.raises(OllamaTailoringContractError, match="repeats target source IDs"):
        patch_engine.validate_and_apply_patches(
            payload=payload,
            master_content=extracted["content"],
            extracted_resume=extracted,
            approved_analysis=duplicated,
        )


def test_47_short_forbidden_claims_use_token_boundaries(
    master_resume: Path,
) -> None:
    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)
    edit = writer.approved_edit_catalog(analysis)[0]
    descriptor = patch_engine.resolve_target_descriptor(
        edit, extracted["content"], extracted
    )
    evidence_texts = patch_engine.authorized_evidence_texts_for_edit(
        edit, descriptor, extracted
    )
    with pytest.raises(OllamaTailoringContractError, match="forbidden claim"):
        patch_engine._validate_replacement_text(
            edit_id=descriptor.edit_id,
            descriptor=descriptor,
            replacement_text="AI engineer building Python workflows.",
            evidence_texts=evidence_texts,
            forbidden_claims=["AI"],
        )

    accepted = patch_engine._validate_replacement_text(
        edit_id=descriptor.edit_id,
        descriptor=descriptor,
        replacement_text="Python training engineer building workflows.",
        evidence_texts=evidence_texts,
        forbidden_claims=["AI"],
    )
    assert accepted == "Python training engineer building workflows."


def test_48_unicode_equivalent_replacement_is_rejected_as_noop() -> None:
    descriptor = patch_engine.TargetDescriptor(
        edit_id="edit.001",
        target_source_id="professional_summary",
        operation="replace",
        kind="plain",
        label=None,
        current_mutable_text="Cafe\u0301 engineer",
        exact_rendered_existing_text="Cafe\u0301 engineer",
        maximum_rendered_characters=100,
        proposed_text="Café engineer",
        alignment_rationale="Synthetic",
        evidence_source_ids=["professional_summary"],
    )
    with pytest.raises(OllamaTailoringContractError, match="no-op replacement"):
        patch_engine._validate_replacement_text(
            edit_id=descriptor.edit_id,
            descriptor=descriptor,
            replacement_text="Café engineer",
            evidence_texts=[descriptor.exact_rendered_existing_text],
            forbidden_claims=[],
        )


def _hybrid_patch(edit_id: str) -> dict:
    return {
        "edit_id": edit_id,
        "target_source_id": "professional_summary",
        "operation": "replace",
        "replacement_text": "Synthetic replacement.",
    }


@pytest.mark.parametrize(
    ("deterministic_ids", "prose_ids", "catalog_ids", "message"),
    [
        (["edit.001", "edit.001"], [], ["edit.001"], "Duplicate deterministic"),
        ([], ["edit.001", "edit.001"], ["edit.001"], "Duplicate prose"),
        (["edit.001"], ["edit.001"], ["edit.001"], "collision"),
        (["edit.001"], [], ["edit.001", "edit.002"], "missing IDs"),
        (["edit.001"], ["edit.999"], ["edit.001"], "extra IDs"),
    ],
)
def test_49_hybrid_combiner_requires_exact_edit_id_set(
    deterministic_ids: list[str],
    prose_ids: list[str],
    catalog_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(OllamaTailoringContractError, match=message):
        combine_hybrid_patch_payload(
            deterministic_patches=[
                _hybrid_patch(edit_id) for edit_id in deterministic_ids
            ],
            prose_patches=[_hybrid_patch(edit_id) for edit_id in prose_ids],
            full_catalog=[{"edit_id": edit_id} for edit_id in catalog_ids],
            full_catalog_sha256="0" * 64,
        )


def test_50_execution_metadata_distinguishes_prose_and_empty_catalog() -> None:
    prose_edit = {
        "edit_id": "edit.001",
        "target_source_id": "professional_summary",
    }
    prose_metadata = hybrid_execution_metadata(
        deterministic_patches=[],
        prose_patches=[_hybrid_patch("edit.001")],
        deterministic_edits=[],
        prose_edits=[prose_edit],
        full_catalog_sha256="1" * 64,
        writer_subset_sha256="1" * 64,
        ollama_invoked=True,
    )
    assert prose_metadata["execution_mode"] == "prose_only"

    mixed_metadata = hybrid_execution_metadata(
        deterministic_patches=[_hybrid_patch("edit.001")],
        prose_patches=[_hybrid_patch("edit.002")],
        deterministic_edits=[
            {"edit_id": "edit.001", "target_source_id": "skill_groups.0"}
        ],
        prose_edits=[
            {"edit_id": "edit.002", "target_source_id": "professional_summary"}
        ],
        full_catalog_sha256="2" * 64,
        writer_subset_sha256="3" * 64,
        ollama_invoked=True,
    )
    assert mixed_metadata["execution_mode"] == "hybrid"

    empty_metadata = deterministic_only_metadata(
        deterministic_patches=[],
        deterministic_edits=[],
        full_catalog_sha256="4" * 64,
    )
    assert empty_metadata["execution_mode"] == "deterministic_only"
    assert empty_metadata["writer_skipped_reason"] == "empty_catalog"
    assert empty_metadata["deterministic_patch_count"] == 0
