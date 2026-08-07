"""Tests for the deterministic structured-field patch compiler.

All test data is synthetic.  No private résumé, job, or provider artifacts
are inspected, printed, copied, or committed.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from resume_tailor.backend.providers.antigravity_writer import approved_edit_catalog
from resume_tailor.backend.providers.ollama_writer import (
    _write_tailoring_patch_transport_schema,
    build_ollama_tailoring_prompt,
)
from resume_tailor.backend.engine.patch_engine import (
    TargetResolutionError,
    _validate_structured_list_items,
    canonical_digest,
    duplicate_catalog_target_ids,
    mutable_proposed_text,
    resolve_target_descriptor,
    validate_and_apply_patches,
)
from resume_tailor.backend.engine.structured_patch_compiler import (
    DeterministicPatchError,
    combine_hybrid_patch_payload,
    compile_deterministic_structured_patches,
    deterministic_only_metadata,
    hybrid_execution_metadata,
    is_deterministic_structured_target,
    partition_edit_catalog,
)
from resume_tailor.backend.utils.utilities import (
    OllamaCannotApplyError,
    OllamaRevisionContractError,
    OllamaTailoringContractError,
    OllamaTechnicalFailureError,
    OllamaTransportSchemaError,
    TailoringPreflightError,
    atomic_write_json,
)

import resume_tailor.backend.providers.ollama_writer as writer


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _synthetic_master_content() -> dict[str, Any]:
    """Minimal synthetic master résumé content."""
    return {
        "professional_summary": "Synthetic AI engineer with experience in Python.",
        "skill_groups": [
            {"label": "Software & Data", "text": "Python, FastAPI, PostgreSQL"},
            {"label": "AI & ML", "text": "OpenAI SDKs, Gemini API, Groq API, Deepgram STT"},
            {"label": "DevOps", "text": "Docker, GitHub Actions, Linux"},
        ],
        "experience": {
            "role": "Software Engineer",
            "employer_location": "Synthetic Corp, Remote",
            "dates": "2022–Present",
            "bullets": [
                "Built synthetic API serving 1000 requests per second.",
                "Integrated OpenAI SDKs into production pipeline.",
            ],
        },
        "education": {
            "institution": "Synthetic University",
            "degree_details": "B.S. Computer Science, 2022",
            "coursework": {
                "label": "Relevant Coursework",
                "text": "Machine Learning, Data Structures, Algorithms",
            },
            "certifications": {
                "label": "Certifications",
                "text": "AWS Solutions Architect, Google Cloud Associate",
            },
        },
        "projects": [
            {
                "name": "Synthetic Project Alpha",
                "technologies": "Python, FastAPI",
                "bullets": [
                    "Implemented feature extraction pipeline.",
                    "Deployed to production with 99.9% uptime.",
                ],
            },
            {
                "name": "Synthetic Project Beta",
                "technologies": "PostgreSQL, Redis",
                "bullets": [
                    "Optimized database queries.",
                    "Reduced latency by 50%.",
                ],
            },
            {
                "name": "Synthetic Project Gamma",
                "technologies": "Docker, Kubernetes",
                "bullets": [
                    "Containerized legacy application.",
                    "Automated deployment pipeline.",
                ],
            },
        ],
        "open_source": {
            "name": "SyntheticOSS",
            "technologies": "Python, Rust",
            "bullet": "Contributed synthetic improvements to the CLI.",
        },
    }


def _synthetic_extracted_resume(
    content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extracted résumé wrapper with source blocks and paragraphs."""
    c = content or _synthetic_master_content()
    source_blocks = [
        {
            "source_id": "professional_summary",
            "section_context": "summary",
            "exact_text": c["professional_summary"],
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "skill_groups.0",
            "section_context": "skills",
            "exact_text": f"{c['skill_groups'][0]['label']}: {c['skill_groups'][0]['text']}",
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "skill_groups.1",
            "section_context": "skills",
            "exact_text": f"{c['skill_groups'][1]['label']}: {c['skill_groups'][1]['text']}",
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "skill_groups.2",
            "section_context": "skills",
            "exact_text": f"{c['skill_groups'][2]['label']}: {c['skill_groups'][2]['text']}",
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "experience.bullets.0",
            "section_context": "experience",
            "exact_text": c["experience"]["bullets"][0],
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "experience.bullets.1",
            "section_context": "experience",
            "exact_text": c["experience"]["bullets"][1],
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "education.coursework",
            "section_context": "education",
            "exact_text": f"{c['education']['coursework']['label']}: {c['education']['coursework']['text']}",
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "education.certifications",
            "section_context": "education",
            "exact_text": f"{c['education']['certifications']['label']}: {c['education']['certifications']['text']}",
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "projects.0.bullets.0",
            "section_context": "projects",
            "exact_text": c["projects"][0]["bullets"][0],
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "projects.0.bullets.1",
            "section_context": "projects",
            "exact_text": c["projects"][0]["bullets"][1],
            "editable": True,
            "evidence_allowed": True,
        },
        {
            "source_id": "open_source.bullet",
            "section_context": "open_source",
            "exact_text": c["open_source"]["bullet"],
            "editable": True,
            "evidence_allowed": True,
        },
    ]
    paragraphs = [
        {"content_id": b["source_id"], "content_budget": {"maximum_characters": 500}}
        for b in source_blocks
    ]
    return {
        "content": copy.deepcopy(c),
        "source_blocks": source_blocks,
        "paragraphs": paragraphs,
    }


def _synthetic_approved_analysis(
    edits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Approved analysis with synthetic edits."""
    c = _synthetic_master_content()
    if edits is None:
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"{c['skill_groups'][0]['label']}: {c['skill_groups'][0]['text']}",
                "proposed_text": f"{c['skill_groups'][0]['label']}: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder for relevance.",
                "evidence_source_ids": ["skill_groups.0"],
            },
            {
                "target_source_id": "skill_groups.1",
                "existing_text": f"{c['skill_groups'][1]['label']}: {c['skill_groups'][1]['text']}",
                "proposed_text": f"{c['skill_groups'][1]['label']}: Gemini API, OpenAI SDKs, Groq API, Deepgram STT",
                "operation": "replace",
                "alignment_rationale": "Reorder for relevance.",
                "evidence_source_ids": ["skill_groups.1"],
            },
            {
                "target_source_id": "professional_summary",
                "existing_text": c["professional_summary"],
                "proposed_text": "Synthetic AI engineer skilled in Python and FastAPI.",
                "operation": "replace",
                "alignment_rationale": "Focus on target technologies.",
                "evidence_source_ids": ["professional_summary", "skill_groups.0"],
            },
            {
                "target_source_id": "experience.bullets.0",
                "existing_text": c["experience"]["bullets"][0],
                "proposed_text": "Engineered synthetic API handling 1000 requests per second.",
                "operation": "replace",
                "alignment_rationale": "Stronger action verb.",
                "evidence_source_ids": ["experience.bullets.0"],
            },
        ]
    return {
        "recommended_edits": edits,
        "immutable_facts": ["Synthetic University", "B.S. Computer Science"],
        "forbidden_claims": ["Kubernetes", "GraphQL"],
        "questions_for_user": None,
    }


def _mock_preflight(monkeypatch_or_patcher=None):
    """Return a mock for preflight_tailoring_inputs that returns the catalog."""
    def _fake_preflight(**kwargs):
        return approved_edit_catalog(kwargs["approved_analysis"])
    return _fake_preflight


# ---------------------------------------------------------------------------
# 1. Target classifier: recognizes all and only the three families
# ---------------------------------------------------------------------------


class TestTargetClassifier:
    @pytest.mark.parametrize("target_id", [
        "skill_groups.0", "skill_groups.1", "skill_groups.2",
        "skill_groups.99",
        "education.coursework",
        "education.certifications",
    ])
    def test_deterministic_targets_recognized(self, target_id: str) -> None:
        assert is_deterministic_structured_target(target_id) is True

    @pytest.mark.parametrize("target_id", [
        "professional_summary",
        "open_source.bullet",
        "experience.bullets.0",
        "experience.bullets.1",
        "projects.0.bullets.0",
        "projects.0.bullets.1",
        "projects.1.bullets.0",
    ])
    def test_prose_targets_not_classified_as_deterministic(self, target_id: str) -> None:
        assert is_deterministic_structured_target(target_id) is False

    def test_malformed_targets_not_classified(self) -> None:
        assert is_deterministic_structured_target("skill_groups") is False
        assert is_deterministic_structured_target("education.coursework.text") is False
        assert is_deterministic_structured_target("education.certifications.label") is False
        assert is_deterministic_structured_target("") is False


# ---------------------------------------------------------------------------
# 2. Catalog partition: preserves order and edit identity
# ---------------------------------------------------------------------------


class TestCatalogPartition:
    def test_partition_preserves_order_and_identity(self) -> None:
        catalog = approved_edit_catalog(_synthetic_approved_analysis())
        det, prose = partition_edit_catalog(catalog)
        assert len(det) == 2
        assert det[0]["target_source_id"] == "skill_groups.0"
        assert det[1]["target_source_id"] == "skill_groups.1"
        assert det[0]["edit_id"] == "edit.001"
        assert det[1]["edit_id"] == "edit.002"
        assert len(prose) == 2
        assert prose[0]["target_source_id"] == "professional_summary"
        assert prose[1]["target_source_id"] == "experience.bullets.0"
        assert prose[0]["edit_id"] == "edit.003"
        assert prose[1]["edit_id"] == "edit.004"

    def test_all_deterministic(self) -> None:
        c = _synthetic_master_content()
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {c['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        catalog = approved_edit_catalog(_synthetic_approved_analysis(edits))
        det, prose = partition_edit_catalog(catalog)
        assert len(det) == 1
        assert len(prose) == 0

    def test_all_prose(self) -> None:
        c = _synthetic_master_content()
        edits = [
            {
                "target_source_id": "professional_summary",
                "existing_text": c["professional_summary"],
                "proposed_text": "Updated summary.",
                "operation": "replace",
                "alignment_rationale": "Improve.",
                "evidence_source_ids": ["professional_summary"],
            },
        ]
        catalog = approved_edit_catalog(_synthetic_approved_analysis(edits))
        det, prose = partition_edit_catalog(catalog)
        assert len(det) == 0
        assert len(prose) == 1


# ---------------------------------------------------------------------------
# 3–6. Gemma prompt/schema exclusion tests (mocking preflight)
# ---------------------------------------------------------------------------


class TestGemmaExclusion:
    def test_skill_groups_never_in_gemma_prompt(self, monkeypatch) -> None:
        """Test 3: Skill groups never appear in a Gemma prompt."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        prompt = build_ollama_tailoring_prompt(
            master_content=mc, extracted_resume=er,
            job_description="Synthetic job.", job_requirements={},
            approved_analysis=aa, company="SyntheticCorp", role="AI Engineer",
            prose_edits=prose_edits, prose_catalog_sha256=prose_sha,
        )
        assert "skill_groups" not in prompt

    def test_coursework_never_in_gemma_prompt(self, monkeypatch) -> None:
        """Test 4: Coursework never appears in a Gemma prompt."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "education.coursework",
                "existing_text": f"Relevant Coursework: {mc['education']['coursework']['text']}",
                "proposed_text": "Relevant Coursework: Algorithms, Data Structures, Machine Learning",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["education.coursework"],
            },
            {
                "target_source_id": "professional_summary",
                "existing_text": mc["professional_summary"],
                "proposed_text": "Synthetic AI engineer skilled in Python and FastAPI.",
                "operation": "replace",
                "alignment_rationale": "Improve.",
                "evidence_source_ids": ["professional_summary"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        prompt = build_ollama_tailoring_prompt(
            master_content=mc, extracted_resume=er,
            job_description="Synthetic job.", job_requirements={},
            approved_analysis=aa, company="SyntheticCorp", role="AI Engineer",
            prose_edits=prose_edits, prose_catalog_sha256=prose_sha,
        )
        assert "education.coursework" not in prompt
        assert "Relevant Coursework" not in prompt

    def test_certifications_never_in_gemma_prompt(self, monkeypatch) -> None:
        """Test 5: Certifications never appear in a Gemma prompt."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "education.certifications",
                "existing_text": f"Certifications: {mc['education']['certifications']['text']}",
                "proposed_text": "Certifications: Google Cloud Associate, AWS Solutions Architect",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["education.certifications"],
            },
            {
                "target_source_id": "professional_summary",
                "existing_text": mc["professional_summary"],
                "proposed_text": "Synthetic AI engineer skilled in Python and FastAPI.",
                "operation": "replace",
                "alignment_rationale": "Improve.",
                "evidence_source_ids": ["professional_summary"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        prompt = build_ollama_tailoring_prompt(
            master_content=mc, extracted_resume=er,
            job_description="Synthetic job.", job_requirements={},
            approved_analysis=aa, company="SyntheticCorp", role="AI Engineer",
            prose_edits=prose_edits, prose_catalog_sha256=prose_sha,
        )
        assert "education.certifications" not in prompt

    def test_structured_targets_never_in_transport_schema(self, tmp_path) -> None:
        """Test 6: Structured targets never appear in the Ollama transport schema."""
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        schema, _ = _write_tailoring_patch_transport_schema(
            tmp_path, catalog=prose_edits, catalog_sha256=prose_sha,
        )
        schema_text = json.dumps(schema)
        assert "skill_groups" not in schema_text
        assert "education.coursework" not in schema_text
        assert "education.certifications" not in schema_text


# ---------------------------------------------------------------------------
# 7–9. Schema counts and digest tests
# ---------------------------------------------------------------------------


class TestSchemaDigests:
    def test_writer_schema_patch_count_equals_prose_count(self, tmp_path) -> None:
        """Test 7."""
        catalog = approved_edit_catalog(_synthetic_approved_analysis())
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        schema, _ = _write_tailoring_patch_transport_schema(
            tmp_path, catalog=prose_edits, catalog_sha256=prose_sha,
        )
        patches_schema = schema.get("properties", {}).get("patches", {})
        for branch in patches_schema.get("oneOf", []):
            if branch.get("type") == "array":
                assert branch.get("minItems") == len(prose_edits)
                assert branch.get("maxItems") == len(prose_edits)

    def test_writer_digest_based_on_prose_subset(self, tmp_path) -> None:
        """Test 8."""
        catalog = approved_edit_catalog(_synthetic_approved_analysis())
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        full_sha = canonical_digest(catalog)
        assert prose_sha != full_sha
        schema, _ = _write_tailoring_patch_transport_schema(
            tmp_path, catalog=prose_edits, catalog_sha256=prose_sha,
        )
        assert schema["properties"]["catalog_sha256"]["enum"] == [prose_sha]

    def test_final_combined_digest_based_on_full_catalog(self) -> None:
        """Test 9."""
        catalog = approved_edit_catalog(_synthetic_approved_analysis())
        full_sha = canonical_digest(catalog)
        combined = combine_hybrid_patch_payload(
            deterministic_patches=[
                {"edit_id": "edit.001", "target_source_id": "skill_groups.0",
                 "operation": "replace", "replacement_text": "FastAPI, Python, PostgreSQL"},
                {"edit_id": "edit.002", "target_source_id": "skill_groups.1",
                 "operation": "replace", "replacement_text": "Gemini API, OpenAI SDKs, Groq API, Deepgram STT"},
            ],
            prose_patches=[
                {"edit_id": "edit.003", "target_source_id": "professional_summary",
                 "operation": "replace", "replacement_text": "Synthetic AI engineer skilled in Python and FastAPI."},
                {"edit_id": "edit.004", "target_source_id": "experience.bullets.0",
                 "operation": "replace", "replacement_text": "Engineered synthetic API handling 1000 requests per second."},
            ],
            full_catalog=catalog,
            full_catalog_sha256=full_sha,
        )
        assert combined["catalog_sha256"] == full_sha


# ---------------------------------------------------------------------------
# 10–14. Deterministic patch content tests
# ---------------------------------------------------------------------------


class TestDeterministicPatchContent:
    def test_exact_approved_mutable_body(self) -> None:
        """Test 10."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert patches[0]["replacement_text"] == "FastAPI, Python, PostgreSQL"
        assert patches[1]["replacement_text"] == "Gemini API, OpenAI SDKs, Groq API, Deepgram STT"

    def test_labels_stripped_before_compilation(self) -> None:
        """Test 11."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        for p in patches:
            assert not p["replacement_text"].startswith("Software & Data:")
            assert not p["replacement_text"].startswith("AI & ML:")

    def test_mismatched_prefix_subject_to_evidence_validation(self) -> None:
        """Test 12."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Wrong Label: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        assert "item_grounding_failed" in str(exc_info.value)

    def test_authenticated_labels_remain_unchanged(self) -> None:
        """Test 13."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert mc["skill_groups"][0]["label"] == "Software & Data"
        assert mc["skill_groups"][1]["label"] == "AI & ML"

    def test_structured_list_item_order_preserved(self) -> None:
        """Test 14."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert patches[0]["replacement_text"] == "FastAPI, Python, PostgreSQL"
        assert "Gemini API" in patches[1]["replacement_text"]
        assert "OpenAI SDKs" in patches[1]["replacement_text"]


# ---------------------------------------------------------------------------
# 15–17. Technology evidence grounding tests
# ---------------------------------------------------------------------------


class TestEvidenceGrounding:
    def test_supported_specific_technology_list_succeeds(self) -> None:
        """Test 15."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert len(patches) == 2

    def test_llm_provider_apis_fails_when_not_authenticated(self) -> None:
        """Test 16."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.1",
                "existing_text": f"AI & ML: {mc['skill_groups'][1]['text']}",
                "proposed_text": "AI & ML: LLM provider APIs, OpenAI SDKs",
                "operation": "replace",
                "alignment_rationale": "Generalize.",
                "evidence_source_ids": ["skill_groups.1"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        assert "item_grounding_failed" in str(exc_info.value)

    def test_llm_provider_apis_succeeds_when_in_evidence(self) -> None:
        """Test 17."""
        mc = _synthetic_master_content()
        mc["skill_groups"][1]["text"] = "LLM provider APIs, OpenAI SDKs, Gemini API"
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.1",
                "existing_text": f"AI & ML: {mc['skill_groups'][1]['text']}",
                "proposed_text": "AI & ML: OpenAI SDKs, LLM provider APIs",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.1"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert len(patches) == 1
        assert "LLM provider APIs" in patches[0]["replacement_text"]


# ---------------------------------------------------------------------------
# 18–19. Coursework and certification item failures
# ---------------------------------------------------------------------------


class TestCourseworkCertificationGrounding:
    def test_unsupported_coursework_items_fail(self) -> None:
        """Test 18."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "education.coursework",
                "existing_text": f"Relevant Coursework: {mc['education']['coursework']['text']}",
                "proposed_text": "Relevant Coursework: Quantum Computing, Data Structures",
                "operation": "replace",
                "alignment_rationale": "Focus.",
                "evidence_source_ids": ["education.coursework"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        assert "item_grounding_failed" in str(exc_info.value)

    def test_unsupported_certification_items_fail(self) -> None:
        """Test 19."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "education.certifications",
                "existing_text": f"Certifications: {mc['education']['certifications']['text']}",
                "proposed_text": "Certifications: CISSP, AWS Solutions Architect",
                "operation": "replace",
                "alignment_rationale": "Focus.",
                "evidence_source_ids": ["education.certifications"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        assert "item_grounding_failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 20–24. Metrics, forbidden claims, char budgets, no-op, append
# ---------------------------------------------------------------------------


class TestValidationRulesPreserved:
    def test_metrics_remain_evidence_bound(self) -> None:
        """Test 20."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        # "95%" is not in the evidence for skill_groups.0.
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: Python, FastAPI, 95% PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Add metric.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError):
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )

    def test_forbidden_claims_remain_rejected(self) -> None:
        """Test 21."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: Python, Kubernetes, FastAPI",
                "operation": "replace",
                "alignment_rationale": "Add forbidden.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError):
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )

    def test_character_budgets_enforced(self) -> None:
        """Test 22."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        for p in er["paragraphs"]:
            if p["content_id"] == "skill_groups.0":
                p["content_budget"]["maximum_characters"] = 30
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError):
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )

    def test_unicode_noop_rejected(self) -> None:
        """Test 23."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "operation": "replace",
                "alignment_rationale": "No change.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError):
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )

    def test_append_semantics_enforced(self) -> None:
        """Test 24."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        current_text = mc["skill_groups"][0]["text"]
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {current_text}",
                "proposed_text": f"Software & Data: {current_text}, FastAPI",
                "operation": "append",
                "alignment_rationale": "Add skill.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        assert patches[0]["replacement_text"].startswith(current_text)


# ---------------------------------------------------------------------------
# 25–26. Duplicate and malformed failures
# ---------------------------------------------------------------------------


class TestDuplicateAndMalformed:
    def test_duplicate_structured_targets_fail(self) -> None:
        """Test 25."""
        mc = _synthetic_master_content()
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: FastAPI, Python",
                "operation": "replace",
                "alignment_rationale": "A.",
                "evidence_source_ids": ["skill_groups.0"],
            },
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: Python, FastAPI",
                "operation": "replace",
                "alignment_rationale": "B.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        dups = duplicate_catalog_target_ids(catalog)
        assert "skill_groups.0" in dups

    def test_missing_label_fails_closed(self) -> None:
        """Test 26."""
        mc = _synthetic_master_content()
        mc["skill_groups"][0]["label"] = ""
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f": {mc['skill_groups'][0]['text']}",
                "proposed_text": ": FastAPI, Python",
                "operation": "replace",
                "alignment_rationale": "Fix.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        assert "target_resolution_failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 27–32. Hybrid flow integration tests (mock preflight and Ollama)
# ---------------------------------------------------------------------------


def _make_mock_response(edits, sha):
    """Build a synthetic response payload."""
    return {
        "status": "complete",
        "message": "Synthetic success message.",
        "cannot_apply": None,
        "technical_failure": None,
        "catalog_sha256": sha,
        "patches": [
            {"edit_id": e["edit_id"], "target_source_id": e["target_source_id"],
             "operation": e.get("operation", "replace"),
             "replacement_text": e.get("proposed_text", "Synthetic replacement.")}
            for e in edits
        ],
    }


class TestHybridFlow:
    def test_mixed_catalog_invokes_provider_once_for_prose(self, tmp_path, monkeypatch) -> None:
        """Test 27."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        call_count = {"n": 0}

        def mock_request(**kwargs):
            call_count["n"] += 1
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps(_make_mock_response(prose_edits, prose_sha))},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        tailored = writer.invoke_ollama(
            master_content=mc, extracted_resume=er, job_description="Synthetic job.",
            job_requirements={}, approved_analysis=aa, company="SC", role="AE",
            run_directory=tmp_path, timeout_seconds=30, model="test-model",
        )
        assert call_count["n"] == 1
        assert tailored["skill_groups"][0]["text"] != mc["skill_groups"][0]["text"]

    def test_provider_response_with_structured_target_rejected(self, tmp_path, monkeypatch) -> None:
        """Test 28."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)

        def mock_request(**kwargs):
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps({
                    "status": "complete",
                    "message": "Success",
                    "cannot_apply": None,
                    "technical_failure": None,
                    "catalog_sha256": prose_sha,
                    "patches": [
                        {"edit_id": "edit.003", "target_source_id": "professional_summary",
                         "operation": "replace", "replacement_text": "Updated."},
                        {"edit_id": "edit.004", "target_source_id": "skill_groups.0",
                         "operation": "replace", "replacement_text": "Injected."},
                    ],
                })},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        with pytest.raises(OllamaTransportSchemaError, match="transport validation at patches"):
            writer.invoke_ollama(
                master_content=mc, extracted_resume=er, job_description="Synthetic job.",
                job_requirements={}, approved_analysis=aa, company="SC", role="AE",
                run_directory=tmp_path, timeout_seconds=30, model="test-model",
            )

    def test_provider_prose_failure_applies_no_deterministic_patches(self, tmp_path, monkeypatch) -> None:
        """Test 29."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)
        original = copy.deepcopy(mc)

        def mock_request(**kwargs):
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps({
                    "status": "technical_failure",
                    "message": "Failed.",
                    "cannot_apply": None,
                    "catalog_sha256": prose_sha,
                    "patches": None,
                    "technical_failure": {"reason_code": "other_technical_failure", "reason": "Test failure"},
                })},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        with pytest.raises(OllamaTechnicalFailureError):
            writer.invoke_ollama(
                master_content=mc, extracted_resume=er, job_description="Synthetic job.",
                job_requirements={}, approved_analysis=aa, company="SC", role="AE",
                run_directory=tmp_path, timeout_seconds=30, model="test-model",
            )
        assert mc == original

    def test_provider_cannot_apply_applies_nothing(self, tmp_path, monkeypatch) -> None:
        """Test 30."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        original = copy.deepcopy(mc)
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)

        def mock_request(**kwargs):
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps({
                    "status": "cannot_apply",
                    "message": "Cannot apply.",
                    "technical_failure": None,
                    "patches": None,
                    "catalog_sha256": prose_sha,
                    "cannot_apply": {"edit_id": "edit.003", "reason_code": "unsupported_claim_risk", "reason": "Risk"},
                })},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        with pytest.raises(OllamaCannotApplyError):
            writer.invoke_ollama(
                master_content=mc, extracted_resume=er, job_description="Synthetic job.",
                job_requirements={}, approved_analysis=aa, company="SC", role="AE",
                run_directory=tmp_path, timeout_seconds=30, model="test-model",
            )
        assert mc == original

    def test_original_master_unchanged_on_failure(self) -> None:
        """Test 31."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        original = copy.deepcopy(mc)
        edits = [
            {
                "target_source_id": "skill_groups.1",
                "existing_text": f"AI & ML: {mc['skill_groups'][1]['text']}",
                "proposed_text": "AI & ML: LLM provider APIs",
                "operation": "replace",
                "alignment_rationale": "Generalize.",
                "evidence_source_ids": ["skill_groups.1"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        with pytest.raises(DeterministicPatchError):
            compile_deterministic_structured_patches(
                deterministic_edits=approved_edit_catalog(aa),
                master_content=mc, extracted_resume=er, approved_analysis=aa,
            )
        assert mc == original

    def test_successful_hybrid_preserves_full_catalog_order(self, tmp_path, monkeypatch) -> None:
        """Test 32."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_sha = canonical_digest(prose_edits)

        def mock_request(**kwargs):
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps(_make_mock_response(prose_edits, prose_sha))},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        tailored = writer.invoke_ollama(
            master_content=mc, extracted_resume=er, job_description="Synthetic job.",
            job_requirements={}, approved_analysis=aa, company="SC", role="AE",
            run_directory=tmp_path, timeout_seconds=30, model="test-model",
        )
        assert tailored["skill_groups"][0]["text"] == "FastAPI, Python, PostgreSQL"
        assert tailored["skill_groups"][1]["text"] == "Gemini API, OpenAI SDKs, Groq API, Deepgram STT"
        assert tailored["professional_summary"] != mc["professional_summary"]


# ---------------------------------------------------------------------------
# 33–34. Deterministic-only runs
# ---------------------------------------------------------------------------


class TestDeterministicOnly:
    def test_deterministic_only_makes_zero_provider_calls(self, tmp_path, monkeypatch) -> None:
        """Test 33."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        call_count = {"n": 0}

        def mock_request(**kwargs):
            call_count["n"] += 1
            raise RuntimeError("Should not be called.")

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        tailored = writer.invoke_ollama(
            master_content=mc, extracted_resume=er, job_description="Synthetic job.",
            job_requirements={}, approved_analysis=aa, company="SC", role="AE",
            run_directory=tmp_path, timeout_seconds=30, model="test-model",
        )
        assert call_count["n"] == 0
        assert tailored["skill_groups"][0]["text"] == "FastAPI, Python, PostgreSQL"

    def test_deterministic_only_metadata_does_not_claim_gemma(self, tmp_path, monkeypatch) -> None:
        """Test 34."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.0",
                "existing_text": f"Software & Data: {mc['skill_groups'][0]['text']}",
                "proposed_text": "Software & Data: FastAPI, Python, PostgreSQL",
                "operation": "replace",
                "alignment_rationale": "Reorder.",
                "evidence_source_ids": ["skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", lambda **kw: (_ for _ in ()).throw(RuntimeError))
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        writer.invoke_ollama(
            master_content=mc, extracted_resume=er, job_description="Synthetic job.",
            job_requirements={}, approved_analysis=aa, company="SC", role="AE",
            run_directory=tmp_path, timeout_seconds=30, model="test-model",
        )
        metadata = json.loads((tmp_path / "ollama-response-envelope.json").read_text())
        assert metadata["provider"] == "deterministic"
        assert metadata["execution"]["ollama_invoked"] is False
        assert metadata["execution"]["gemma_patch_count"] == 0
        assert metadata["execution"]["writer_skipped"] is True


# ---------------------------------------------------------------------------
# 35. Prose-only catalogs
# ---------------------------------------------------------------------------


class TestProseOnly:
    def test_prose_only_catalogs_preserve_current_behavior(self, tmp_path, monkeypatch) -> None:
        """Test 35."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "professional_summary",
                "existing_text": mc["professional_summary"],
                "proposed_text": "Synthetic AI engineer skilled in Python and FastAPI.",
                "operation": "replace",
                "alignment_rationale": "Improve.",
                "evidence_source_ids": ["professional_summary", "skill_groups.0"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        prose_sha = canonical_digest(catalog)

        def mock_request(**kwargs):
            return {
                "done": True, "done_reason": "stop", "model": "test-model",
                "message": {"role": "assistant", "content": json.dumps({
                    "status": "complete",
                    "message": "Success",
                    "cannot_apply": None,
                    "technical_failure": None,
                    "catalog_sha256": prose_sha,
                    "patches": [{"edit_id": "edit.001", "target_source_id": "professional_summary",
                                 "operation": "replace",
                                 "replacement_text": "Synthetic AI engineer skilled in Python and FastAPI."}],
                })},
                "prompt_eval_count": 100, "eval_count": 50,
            }

        monkeypatch.setattr(writer, "preflight_tailoring_inputs", _mock_preflight())
        monkeypatch.setattr(writer, "run_ollama_request", mock_request)
        monkeypatch.setattr(writer, "probe_structured_output_support", lambda s: {})
        tailored = writer.invoke_ollama(
            master_content=mc, extracted_resume=er, job_description="Synthetic job.",
            job_requirements={}, approved_analysis=aa, company="SC", role="AE",
            run_directory=tmp_path, timeout_seconds=30, model="test-model",
        )
        assert tailored["professional_summary"] == "Synthetic AI engineer skilled in Python and FastAPI."


# ---------------------------------------------------------------------------
# 36–37. Revision policy tests
# ---------------------------------------------------------------------------


class TestRevisionPolicy:
    def test_structured_revision_target_fails_before_provider(self) -> None:
        """Test 36."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        qa_result = {
            "status": "material_findings",
            "issues": [{"issue_id": "qa.001", "affected_content_id": "skill_groups.0",
                         "description": "Synthetic QA issue."}],
        }
        with pytest.raises(OllamaRevisionContractError, match="structured_target_requires_new_analysis"):
            writer.invoke_ollama_revision(
                current_tailored_content=mc, extracted_resume=er,
                approved_analysis=aa, qa_result=qa_result,
                company="SC", role="AE", run_directory=None,
                timeout_seconds=30, attempt_number=1, model="test-model",
            )

# ---------------------------------------------------------------------------
# 38. Existing Step 7 evidence validation remains authoritative
# ---------------------------------------------------------------------------


class TestExistingEvidenceValidation:
    def test_step7_evidence_validation_remains_authoritative(self) -> None:
        """Test 38."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        full_sha = canonical_digest(catalog)
        payload = {
            "status": "complete",
            "message": "Success",
            "cannot_apply": None,
            "technical_failure": None,
            "catalog_sha256": full_sha,
            "patches": [
                {"edit_id": "edit.001", "target_source_id": "skill_groups.0",
                 "operation": "replace", "replacement_text": "Python, FastAPI, LLM provider APIs"},
                {"edit_id": "edit.002", "target_source_id": "skill_groups.1",
                 "operation": "replace", "replacement_text": "Gemini API, OpenAI SDKs, Groq API, Deepgram STT"},
                {"edit_id": "edit.003", "target_source_id": "professional_summary",
                 "operation": "replace", "replacement_text": "Synthetic AI engineer skilled in Python and FastAPI."},
                {"edit_id": "edit.004", "target_source_id": "experience.bullets.0",
                 "operation": "replace", "replacement_text": "Engineered synthetic API handling 1000 requests per second."},
            ],
        }
        with pytest.raises(OllamaTailoringContractError, match="skill item"):
            validate_and_apply_patches(
                payload=payload, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )


# ---------------------------------------------------------------------------
# 39. No private content in sanitized diagnostics
# ---------------------------------------------------------------------------


class TestSanitizedDiagnostics:
    def test_no_private_content_in_diagnostics(self) -> None:
        """Test 39."""
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        edits = [
            {
                "target_source_id": "skill_groups.1",
                "existing_text": f"AI & ML: {mc['skill_groups'][1]['text']}",
                "proposed_text": "AI & ML: LLM provider APIs",
                "operation": "replace",
                "alignment_rationale": "Generalize.",
                "evidence_source_ids": ["skill_groups.1"],
            },
        ]
        aa = _synthetic_approved_analysis(edits)
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        with pytest.raises(DeterministicPatchError) as exc_info:
            compile_deterministic_structured_patches(
                deterministic_edits=det_edits, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )
        error_msg = str(exc_info.value)
        assert "edit.001" in error_msg
        assert "skill_groups.1" in error_msg
        assert "LLM provider APIs" not in error_msg
        assert "OpenAI SDKs" not in error_msg
        assert "AI & ML" not in error_msg


# ---------------------------------------------------------------------------
# Synthetic regression: live failure class reproduction
# ---------------------------------------------------------------------------


class TestLiveFailureRegression:
    def test_structured_edit_never_in_gemma_authority(self) -> None:
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        _, prose_edits = partition_edit_catalog(catalog)
        prose_target_ids = {e["target_source_id"] for e in prose_edits}
        assert "skill_groups.1" not in prose_target_ids
        assert "skill_groups.0" not in prose_target_ids

    def test_invalid_provider_patch_rejected_if_injected(self) -> None:
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        full_sha = canonical_digest(catalog)
        payload = {
            "status": "complete",
            "message": "Success",
            "cannot_apply": None,
            "technical_failure": None,
            "catalog_sha256": full_sha,
            "patches": [
                {"edit_id": "edit.001", "target_source_id": "skill_groups.0",
                 "operation": "replace", "replacement_text": "FastAPI, Python, PostgreSQL"},
                {"edit_id": "edit.002", "target_source_id": "skill_groups.1",
                 "operation": "replace", "replacement_text": "LLM provider APIs"},
                {"edit_id": "edit.003", "target_source_id": "professional_summary",
                 "operation": "replace", "replacement_text": "Synthetic AI engineer skilled in Python and FastAPI."},
                {"edit_id": "edit.004", "target_source_id": "experience.bullets.0",
                 "operation": "replace", "replacement_text": "Engineered synthetic API handling 1000 requests per second."},
            ],
        }
        with pytest.raises(OllamaTailoringContractError, match="skill item"):
            validate_and_apply_patches(
                payload=payload, master_content=mc,
                extracted_resume=er, approved_analysis=aa,
            )

    def test_deterministic_list_preserves_exact_technologies(self) -> None:
        mc = _synthetic_master_content()
        er = _synthetic_extracted_resume(mc)
        aa = _synthetic_approved_analysis()
        catalog = approved_edit_catalog(aa)
        det_edits, _ = partition_edit_catalog(catalog)
        patches = compile_deterministic_structured_patches(
            deterministic_edits=det_edits, master_content=mc,
            extracted_resume=er, approved_analysis=aa,
        )
        ai_ml_patch = next(p for p in patches if p["target_source_id"] == "skill_groups.1")
        assert ai_ml_patch["replacement_text"] == "Gemini API, OpenAI SDKs, Groq API, Deepgram STT"
        assert "LLM provider APIs" not in ai_ml_patch["replacement_text"]
