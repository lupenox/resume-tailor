from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import resume_tailor.codex_analysis as codex_analysis_module
from resume_tailor.codex_analysis import invoke_codex_analysis
from resume_tailor.schemas import (
    audit_codex_transport_schema,
    codex_transport_schema_path,
    derive_codex_transport_schema,
    load_schema,
    normalize_unique_arrays,
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
        "matched_requirements": ["Python"],
        "evidence_map": [
            {
                "requirement": "Python",
                "resume_evidence": ["Python"],
                "strength": "strong",
            }
        ],
        "supported_ats_keywords": ["Python"],
        "missing_or_unsupported_requirements": ["RAG"],
        "recommended_edits": [],
        "immutable_facts": ["Expected Dec 2026"],
        "forbidden_claims": ["GraphQL"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


@pytest.mark.parametrize(
    "canonical_name",
    ["codex_analysis.schema.json", "final_qa.schema.json"],
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
    keywords = set(_walk_keywords(canonical))
    assert "uniqueItems" in keywords
    assert "minLength" in keywords
    assert (
        canonical["properties"]["fit_assessment"]["properties"]["strengths"][
            "uniqueItems"
        ]
        is True
    )
    assert final_qa["properties"]["material_issues"]["uniqueItems"] is True


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
    payload["supported_ats_keywords"] = ["Python", "Python"]

    normalized, warnings = normalize_unique_arrays(
        payload,
        "codex_analysis.schema.json",
    )

    assert normalized["fit_assessment"]["strengths"] == ["Python", "Testing"]
    assert normalized["supported_ats_keywords"] == ["Python"]
    assert len(warnings) == 2
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
    def incompatible_schema(_: str) -> Path:
        raise CodexSchemaCompatibilityError("nested unsupported keyword")

    def forbidden_process_launch(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Codex process launched before schema preflight completed")

    monkeypatch.setattr(
        codex_analysis_module,
        "codex_transport_schema_path",
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
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "codex"),
    )

    assert payload["matched_requirements"] == ["Python"]
    assert payload["supported_ats_keywords"] == ["Python"]
    used_schema = Path(schema_log.read_text(encoding="utf-8").strip())
    assert used_schema.name == "codex_analysis.openai.schema.json"
    assert (
        tmp_path / "codex-analysis-normalization-warnings.json"
    ).is_file()
    persisted = json.loads(
        (tmp_path / "codex-analysis.json").read_text(encoding="utf-8")
    )
    assert persisted == payload
