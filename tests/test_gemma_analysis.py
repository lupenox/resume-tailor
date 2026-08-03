"""Gemma Local Ollama analysis-provider tests (mocked Ollama only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.analysis import (
    ANALYSIS_RESOLVED_FILENAME,
    CODEX_ANALYSIS_RESOLVED_FILENAME,
    DEFAULT_ANALYSIS_PROVIDER,
    analysis_provider_label,
    analysis_workflow_label,
    invoke_analysis,
    normalize_analysis_provider,
    write_resolved_analysis_artifact,
    workflow_stages_for_provider,
)
from resume_tailor.cli import build_parser
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.gemma_analysis import (
    GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME,
    GEMMA_ANALYSIS_PROMPT_FILENAME,
    GEMMA_ANALYSIS_RESPONSE_FILENAME,
    GEMMA_ANALYSIS_SCHEMA_FILENAME,
    gemma_analysis_chat_request_for_tests,
    invoke_gemma_analysis,
    parse_exact_analysis_json,
    prepare_gemma_analysis_schema,
    resolve_gemma_analysis_model,
)
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.utilities import (
    GemmaAnalysisTimeoutError,
    GemmaInnerAnalysisError,
    GemmaModelUnavailableError,
    GemmaOllamaUnavailableError,
    GemmaTransportEnvelopeError,
    OllamaConnectionError,
    SourceEvidenceError,
)


def _job_catalog() -> dict:
    return build_job_requirement_catalog(
        "Skills: Python and RAG.",
        structured_job={"technologies_and_skills": ["Python", "RAG"]},
    )


def _valid_analysis(requirements: dict) -> dict[str, Any]:
    requirement_ids = [item["requirement_id"] for item in requirements["requirements"]]
    supported = requirement_ids[0]
    return {
        "role_summary": "Stubbed Gemma target role analysis.",
        "fit_assessment": {
            "overall": "Supported fit based on the supplied master resume.",
            "strengths": ["Python evidence"],
            "gaps": ["Unsupported technologies remain unsupported"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": supported,
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": [
            item for item in requirement_ids if item != supported
        ],
        "recommended_edits": [],
        "immutable_facts": ["Example Institute"],
        "forbidden_claims": ["RAG", "GraphQL"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


def _ollama_body(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = content
    return {
        "model": "resume-tailor-gemma",
        "created_at": "2026-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {"bodies": [], "requests": [], "calls": 0}

    def _set_bodies(*bodies: dict[str, Any]) -> None:
        state["bodies"] = list(bodies)

    def fake_run_ollama_request(**kwargs: Any) -> dict[str, Any]:
        state["calls"] += 1
        state["requests"].append(kwargs)
        if not state["bodies"]:
            raise OllamaConnectionError(
                "The localhost Ollama API request failed. Provider output was "
                "omitted; confirm that Ollama is running on 127.0.0.1:11434."
            )
        return state["bodies"].pop(0)

    monkeypatch.setattr(
        "resume_tailor.gemma_analysis.run_ollama_request",
        fake_run_ollama_request,
    )
    state["set_bodies"] = _set_bodies
    return state


def test_gemma_local_is_default_and_labels() -> None:
    assert DEFAULT_ANALYSIS_PROVIDER == "gemma_local"
    assert normalize_analysis_provider(None) == "gemma_local"
    assert analysis_provider_label("gemma_local") == "Gemma Local"
    assert analysis_workflow_label("gemma_local") == "Gemma Local analysis"
    assert analysis_workflow_label("codex") == "Codex analysis"
    assert analysis_workflow_label("grok_cli") == "Grok CLI analysis"
    assert analysis_workflow_label("grok") == "Grok CLI analysis"
    stages = dict(workflow_stages_for_provider("grok_cli"))
    assert stages["codex_analysis"] == "Grok CLI analysis"
    stages_gemma = dict(workflow_stages_for_provider("gemma_local"))
    assert stages_gemma["codex_analysis"] == "Gemma Local analysis"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "template/sample_resume.docx",
            "--company",
            "Example",
            "--role",
            "Dev",
            "--job-file",
            "job.txt",
        ]
    )
    assert args.analysis_provider == "gemma_local"


def test_model_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMMA_ANALYSIS_MODEL", raising=False)
    monkeypatch.delenv("GEMMA_WRITER_MODEL", raising=False)
    assert resolve_gemma_analysis_model(None) == "resume-tailor-gemma"
    monkeypatch.setenv("GEMMA_WRITER_MODEL", "writer-model")
    assert resolve_gemma_analysis_model(None) == "writer-model"
    monkeypatch.setenv("GEMMA_ANALYSIS_MODEL", "analysis-model")
    assert resolve_gemma_analysis_model(None) == "analysis-model"
    assert resolve_gemma_analysis_model("explicit-model") == "explicit-model"
    # Empty environment values must not become an empty model name.
    monkeypatch.setenv("GEMMA_ANALYSIS_MODEL", "   ")
    monkeypatch.setenv("GEMMA_WRITER_MODEL", "")
    assert resolve_gemma_analysis_model(None) == "resume-tailor-gemma"
    assert resolve_gemma_analysis_model("") == "resume-tailor-gemma"


def test_chat_request_shape_stream_temperature_think_omitted(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    schema_info = prepare_gemma_analysis_schema(
        extracted,
        _job_catalog(),
        tmp_path,
    )
    request = gemma_analysis_chat_request_for_tests(
        model="resume-tailor-gemma",
        prompt="synthetic",
        format_schema=schema_info["schema"],
    )
    assert request["stream"] is False
    assert request["options"]["temperature"] == 0
    assert "think" not in request
    assert request["format"] == schema_info["schema"]
    format_schema = request["format"]
    assert "$schema" not in format_schema
    assert "title" not in format_schema
    assert "allOf" not in format_schema
    # Safety-bearing constraints retained in the constrained transport.
    assert format_schema["type"] == "object"
    assert "role_summary" in format_schema["required"]
    assert "enum" in format_schema["properties"]["unsupported_requirement_ids"][
        "items"
    ]
    assert format_schema["additionalProperties"] is False


def test_successful_schema_constrained_gemma_analysis(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    analysis = _valid_analysis(requirements)
    mock_ollama["set_bodies"](_ollama_body(analysis))
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert payload["role_summary"]
    request = mock_ollama["requests"][0]
    assert request["path"] == "/api/chat"
    body = request["body"]
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert "think" not in body
    assert "format" in body
    assert request["heartbeat_handler"] is None or callable(
        request.get("heartbeat_handler")
    )
    assert (tmp_path / GEMMA_ANALYSIS_SCHEMA_FILENAME).is_file()
    assert (tmp_path / GEMMA_ANALYSIS_PROMPT_FILENAME).is_file()
    assert (tmp_path / GEMMA_ANALYSIS_RESPONSE_FILENAME).is_file()
    assert (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).is_file()
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "success"
    assert diagnostic["think_property_omitted"] is True
    assert diagnostic["hidden_reasoning_excluded"] is True
    meta = write_resolved_analysis_artifact(
        tmp_path,
        payload,
        provider="gemma_local",
    )
    assert meta["provider"] == "gemma_local"
    assert meta["legacy_codex_alias_written"] is False
    document = json.loads(
        (tmp_path / ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert document["provider"] == "gemma_local"
    assert not (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).exists()


def test_invoke_analysis_dispatches_to_gemma(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    mock_ollama["set_bodies"](_ollama_body(_valid_analysis(requirements)))
    payload = invoke_analysis(
        provider="gemma_local",
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert "role_summary" in payload


def test_ollama_unavailable(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    with pytest.raises(GemmaOllamaUnavailableError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "ollama_unavailable"


def test_timeout_classification(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)

    def raise_timeout(**_kwargs: Any) -> dict[str, Any]:
        raise OllamaConnectionError(
            "The localhost Ollama request exceeded its bounded timeout. The full "
            "worker process group was stopped and provider content was omitted."
        )

    monkeypatch.setattr(
        "resume_tailor.gemma_analysis.run_ollama_request",
        raise_timeout,
    )
    with pytest.raises(GemmaAnalysisTimeoutError, match="timed out"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=12,
        )


def test_malformed_transport_and_inner_failures(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    # First attempt incomplete envelope; second attempt also bad — one repair only.
    mock_ollama["set_bodies"](
        {"done": False, "message": {"content": "{}"}},
        {"done": False, "message": {"content": "{}"}},
    )
    with pytest.raises(GemmaTransportEnvelopeError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=requirements,
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    assert mock_ollama["calls"] == 2  # one repair attempt


def test_fenced_and_trailing_json_rejected() -> None:
    with pytest.raises(GemmaInnerAnalysisError, match="Markdown"):
        parse_exact_analysis_json("```json\n{}\n```")
    with pytest.raises(GemmaInnerAnalysisError, match="trailing"):
        parse_exact_analysis_json('{"a":1}\ncommentary')
    with pytest.raises(GemmaInnerAnalysisError, match="multiple"):
        parse_exact_analysis_json('{"a":1}\n{"b":2}')


def test_schema_invalid_triggers_one_repair_then_stops(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    invalid = {"role_summary": "incomplete"}
    mock_ollama["set_bodies"](_ollama_body(invalid), _ollama_body(invalid))
    with pytest.raises(SourceEvidenceError, match="canonical evidence contract"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=requirements,
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    assert mock_ollama["calls"] == 2
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "schema_failure"


def test_repair_then_success(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    valid = _valid_analysis(requirements)
    mock_ollama["set_bodies"](
        _ollama_body("```json\n" + json.dumps(valid) + "\n```"),
        _ollama_body(valid),
    )
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert payload["role_summary"]
    assert mock_ollama["calls"] == 2
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "success"
    assert diagnostic["repair_used"] is True


def test_invalid_source_ids_not_retried_inside_provider(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    """Schema-valid but evidence-invalid payloads leave invoke once; evidence fails later."""
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    analysis = _valid_analysis(requirements)
    analysis["recommended_edits"] = [
        {
            "target_source_id": "professional_summary",
            "operation": "replace",
            "proposed_text": "Synthetic proposed text.",
            "alignment_rationale": "Synthetic regression case.",
            "evidence_source_ids": ["paragraph 3 describes the summary"],
        }
    ]
    mock_ollama["set_bodies"](_ollama_body(analysis))
    raw = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    assert mock_ollama["calls"] == 1  # no repair for evidence failures
    _, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert issues


def test_character_budget_violation_after_analysis(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    analysis = _valid_analysis(requirements)
    analysis["recommended_edits"] = [
        {
            "target_source_id": "skill_groups.0",
            "operation": "replace",
            "proposed_text": "X" * 5000,
            "alignment_rationale": "Synthetic over-budget structured proposal.",
            "evidence_source_ids": ["skill_groups.0"],
        }
    ]
    mock_ollama["set_bodies"](_ollama_body(analysis))
    raw = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    _, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert any(issue.code == "structured_proposal_over_budget" for issue in issues)
    assert mock_ollama["calls"] == 1


def test_repair_prompt_uses_only_sanitized_failure_class(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    valid = _valid_analysis(requirements)
    mock_ollama["set_bodies"](
        _ollama_body("```json\n" + json.dumps(valid) + "\n```"),
        _ollama_body(valid),
    )
    invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role with PRIVATE_DIAG_MARKER",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    # Second request is the repair; failure class is a stable token only.
    repair_prompt = mock_ollama["requests"][1]["body"]["messages"][1]["content"]
    assert "PREVIOUS ATTEMPT FAILED LOCAL VALIDATION" in repair_prompt
    assert "Failure class: malformed_inner_analysis" in repair_prompt
    assert "malformed_inner_analysis:" not in repair_prompt
    assert "validation at" not in repair_prompt
    assert "PRIVATE_DIAG_MARKER" in repair_prompt  # original immutable job input only


def test_no_credential_or_reasoning_leakage(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-should-not-appear")
    mock_ollama["set_bodies"](_ollama_body(_valid_analysis(_job_catalog())))
    invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=_job_catalog(),
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    for path in tmp_path.glob("gemma-analysis-*"):
        text = path.read_text(encoding="utf-8")
        assert "secret-should-not-appear" not in text
        assert "OLLAMA_API_KEY" not in text


def test_model_unavailable_classification(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)

    def raise_reject(**_kwargs: Any) -> dict[str, Any]:
        raise OllamaConnectionError(
            "The localhost Ollama API rejected the request. Response content was "
            "omitted; confirm the configured model name and server status."
        )

    monkeypatch.setattr(
        "resume_tailor.gemma_analysis.run_ollama_request",
        raise_reject,
    )
    with pytest.raises((GemmaModelUnavailableError, GemmaOllamaUnavailableError)):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            model="missing-model",
        )
