from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS

import resume_tailor.codex_analysis as codex_analysis_module
from resume_tailor.codex_analysis import invoke_codex_analysis
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.schemas import (
    audit_codex_transport_schema,
    build_codex_analysis_transport_schema,
    codex_transport_schema_path,
    derive_codex_transport_schema,
    load_schema,
    normalize_unique_arrays,
    prepare_codex_analysis_transport_schema,
    validate_codex_analysis_transport_artifact,
    validate_payload,
)
from resume_tailor.utilities import CodexSchemaCompatibilityError


def _walk_keywords(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keywords(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keywords(child)


def _valid_analysis() -> dict[str, Any]:
    return {
        "role_summary": "Evidence-backed AI engineering role.",
        "fit_assessment": {
            "overall": "Supported fit.",
            "strengths": ["Python"],
            "gaps": ["No unsupported scale evidence"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": "skill.001",
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": ["skill.002"],
        "recommended_edits": [],
        "immutable_facts": ["Expected Dec 2026"],
        "forbidden_claims": ["GraphQL"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


def _job_catalog() -> dict[str, Any]:
    return build_job_requirement_catalog(
        "Skills: Python and RAG.",
        structured_job={"technologies_and_skills": ["Python", "RAG"]},
    )


@pytest.mark.parametrize(
    "canonical_name",
    [
        "codex_analysis.schema.json",
        "final_qa_provider.schema.json",
    ],
)
def test_codex_transport_schemas_contain_no_unique_items(
    canonical_name: str,
) -> None:
    transport = json.loads(
        codex_transport_schema_path(canonical_name).read_text(encoding="utf-8")
    )
    keywords = set(_walk_keywords(transport))
    assert "uniqueItems" not in keywords
    assert "minLength" not in keywords
    assert "maxLength" not in keywords
    audit_codex_transport_schema(transport, label=canonical_name)


def test_nested_unsupported_transport_keyword_is_detected() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["values"],
        "properties": {
            "values": {
                "type": "array",
                "items": {
                    "type": "string",
                    "allOf": [{"type": "string"}],
                },
            }
        },
    }
    with pytest.raises(
        CodexSchemaCompatibilityError,
        match=r"/properties/values/items/allOf.*unsupported keyword",
    ):
        audit_codex_transport_schema(schema, label="nested-test")


def test_canonical_schema_retains_local_uniqueness_constraints() -> None:
    canonical = load_schema("codex_analysis.schema.json")
    final_qa = load_schema("final_qa.schema.json")
    linkedin_job = load_schema("linkedin_job.schema.json")
    keywords = set(_walk_keywords(canonical))
    assert "uniqueItems" in keywords
    assert "minLength" in keywords
    assert (
        canonical["properties"]["fit_assessment"]["properties"]["strengths"][
            "uniqueItems"
        ]
        is True
    )
    assert final_qa["properties"]["issues"]["uniqueItems"] is True
    assert final_qa["$defs"]["resolved_issue"]["properties"]["issue_id"][
        "pattern"
    ] == r"^qa\.[0-9]{3}$"
    assert linkedin_job["properties"]["responsibilities"]["uniqueItems"] is True
    assert linkedin_job["properties"]["normalized_job_description"][
        "maxLength"
    ] == MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS


def test_transport_preserves_supported_bounds_and_references() -> None:
    canonical = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["values"],
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"$ref": "#/$defs/value"},
            }
        },
        "$defs": {
            "value": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            }
        },
    }
    transport = derive_codex_transport_schema(canonical, label="bounds-test")
    values = transport["properties"]["values"]
    assert values["minItems"] == 1
    assert values["maxItems"] == 3
    assert values["items"] == {"$ref": "#/$defs/value"}
    assert transport["$defs"]["value"]["minimum"] == 1
    assert transport["$defs"]["value"]["maximum"] == 10


def test_exact_duplicates_are_removed_and_warning_is_recorded() -> None:
    payload = _valid_analysis()
    payload["fit_assessment"]["strengths"] = ["Python", "Python", "Testing"]
    payload["supported_requirement_mappings"].append(
        dict(payload["supported_requirement_mappings"][0])
    )

    normalized, warnings = normalize_unique_arrays(
        payload,
        "codex_analysis.schema.json",
    )

    assert normalized["fit_assessment"]["strengths"] == ["Python", "Testing"]
    assert len(normalized["supported_requirement_mappings"]) == 2
    assert len(warnings) == 1
    assert all("exact duplicate" in warning for warning in warnings)
    validate_payload(
        normalized,
        "codex_analysis.schema.json",
        label="normalized analysis",
    )


def test_valid_codex_output_passes_canonical_validation() -> None:
    validate_payload(
        _valid_analysis(),
        "codex_analysis.schema.json",
        label="valid analysis",
    )


def test_schema_preflight_fails_before_codex_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def incompatible_schema(*args: Any, **kwargs: Any) -> Any:
        raise CodexSchemaCompatibilityError("nested unsupported keyword")

    def forbidden_process_launch(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Codex process launched before schema preflight completed")

    monkeypatch.setattr(
        codex_analysis_module,
        "prepare_codex_analysis_transport_schema",
        incompatible_schema,
    )
    monkeypatch.setattr(
        codex_analysis_module,
        "run_command",
        forbidden_process_launch,
    )

    with pytest.raises(
        CodexSchemaCompatibilityError,
        match="nested unsupported keyword",
    ):
        invoke_codex_analysis(
            extracted_resume={"paragraphs": []},
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable="/definitely/not/executed/codex",
        )


def test_codex_adapter_uses_transport_and_persists_normalization_warning(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from resume_tailor.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    schema_log = tmp_path / "schema-paths.txt"
    monkeypatch.setenv("STUB_CODEX_MODE", "duplicates")
    monkeypatch.setenv("STUB_CODEX_SCHEMA_LOG", str(schema_log))

    payload = invoke_codex_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=_job_catalog(),
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "codex"),
    )

    assert payload["supported_requirement_mappings"][0]["requirement_id"] == (
        "skill.001"
    )
    used_schema = Path(schema_log.read_text(encoding="utf-8").strip())
    assert used_schema.name == "codex-analysis-transport.schema.json"
    assert (
        tmp_path / "codex-analysis-normalization-warnings.json"
    ).is_file()
    persisted = json.loads(
        (tmp_path / "codex-analysis.json").read_text(encoding="utf-8")
    )
    assert persisted == payload


def test_run_schema_uses_separate_evidence_and_editable_enums(
    master_resume: Path,
) -> None:
    from resume_tailor.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    transport, evidence_ids, editable_ids, requirement_ids = (
        build_codex_analysis_transport_schema(
            extracted,
            _job_catalog(),
        )
    )
    properties = transport["properties"]
    evidence_arrays = (
        properties["supported_requirement_mappings"]["items"]["properties"][
            "evidence_source_ids"
        ],
        properties["recommended_edits"]["items"]["properties"][
            "evidence_source_ids"
        ],
    )
    for array_schema in evidence_arrays:
        assert array_schema["minItems"] == 1
        assert array_schema["items"]["enum"] == evidence_ids
        assert "" not in array_schema["items"]["enum"]
        assert "section.technical_skills" not in array_schema["items"]["enum"]
    requirement_field = properties["supported_requirement_mappings"]["items"][
        "properties"
    ]["requirement_id"]
    assert requirement_field["enum"] == requirement_ids
    assert properties["unsupported_requirement_ids"]["items"]["enum"] == (
        requirement_ids
    )

    for name in ("recommended_edits", "content_budget_guidance"):
        target = properties[name]["items"]["properties"]["target_source_id"]
        assert target["enum"] == editable_ids
        assert "projects.0.heading" not in target["enum"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence", ""),
        ("evidence", "source.unknown"),
        ("evidence", "section.technical_skills"),
        ("evidence_array", []),
        ("requirement", ""),
        ("requirement", "requirement.unknown"),
        ("target", ""),
        ("target", "source.unknown"),
        ("target", "projects.0.heading"),
    ],
)
def test_run_schema_rejects_malformed_or_inappropriate_ids(
    field: str,
    value: Any,
    master_resume: Path,
) -> None:
    import jsonschema
    from resume_tailor.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    transport, _, _, _ = build_codex_analysis_transport_schema(
        extracted,
        _job_catalog(),
    )
    payload = _valid_analysis()
    if field == "evidence":
        payload["supported_requirement_mappings"][0]["evidence_source_ids"] = [value]
    elif field == "evidence_array":
        payload["supported_requirement_mappings"][0]["evidence_source_ids"] = value
    elif field == "requirement":
        payload["supported_requirement_mappings"][0]["requirement_id"] = value
    else:
        payload["recommended_edits"] = [
            {
                "target_source_id": value,
                "operation": "replace",
                "proposed_text": "Synthetic edit.",
                "alignment_rationale": "Synthetic test.",
                "evidence_source_ids": ["professional_summary"],
            }
        ]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, transport)


def test_generated_schema_artifact_is_hashed_and_revalidated(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    from resume_tailor.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    artifact = prepare_codex_analysis_transport_schema(
        extracted,
        _job_catalog(),
        tmp_path,
    )

    assert artifact.path.name == "codex-analysis-transport.schema.json"
    assert len(artifact.sha256) == 64
    assert artifact.size_bytes == artifact.path.stat().st_size
    validate_codex_analysis_transport_artifact(
        artifact,
        extracted,
        _job_catalog(),
        tmp_path,
    )

    artifact.path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        CodexSchemaCompatibilityError,
        match="changed before Codex launch",
    ):
        validate_codex_analysis_transport_artifact(
            artifact,
            extracted,
            _job_catalog(),
            tmp_path,
        )


def test_generated_schema_rejects_requirement_catalog_drift(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    from resume_tailor.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    artifact = prepare_codex_analysis_transport_schema(
        extracted,
        catalog,
        tmp_path,
    )
    changed = _job_catalog()
    changed["requirements"][0]["requirement_id"] = "skill.003"

    with pytest.raises(
        CodexSchemaCompatibilityError,
        match="source or job-requirement catalog",
    ):
        validate_codex_analysis_transport_artifact(
            artifact,
            extracted,
            changed,
            tmp_path,
        )


def test_generated_schema_complexity_fails_closed_without_fallback() -> None:
    extracted = {
        "source_blocks": [
            {
                "source_id": "x" * 513,
                "evidence_allowed": True,
                "editable": True,
                "exact_text": "Synthetic source.",
            }
        ]
    }

    with pytest.raises(
        CodexSchemaCompatibilityError,
        match="enum string exceeds.*512-byte safety limit",
    ):
        build_codex_analysis_transport_schema(extracted, _job_catalog())
