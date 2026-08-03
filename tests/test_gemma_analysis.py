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
from resume_tailor.codex_analysis import build_analysis_prompt
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.gemma_analysis import (
    DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS,
    GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME,
    GEMMA_ANALYSIS_PROMPT_FILENAME,
    GEMMA_ANALYSIS_RESPONSE_FILENAME,
    GEMMA_ANALYSIS_SCHEMA_FILENAME,
    build_gemma_analysis_prompt,
    estimate_prompt_tokens,
    gemma_analysis_chat_request_for_tests,
    invoke_gemma_analysis,
    parse_exact_analysis_json,
    prepare_gemma_analysis_schema,
    resolve_gemma_analysis_max_output_tokens,
    resolve_gemma_analysis_model,
)
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.utilities import (
    GemmaAnalysisTimeoutError,
    GemmaInnerAnalysisError,
    GemmaModelUnavailableError,
    GemmaOllamaInternalError,
    GemmaOllamaUnavailableError,
    GemmaOutputLimitError,
    OllamaRequestError,
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


def _ollama_body(
    content: str | dict[str, Any],
    *,
    done_reason: str = "stop",
    eval_count: int | None = 20,
) -> dict[str, Any]:
    if isinstance(content, dict):
        text = json.dumps(content, ensure_ascii=False)
    else:
        text = content
    body: dict[str, Any] = {
        "model": "resume-tailor-gemma",
        "created_at": "2026-01-01T00:00:00Z",
        "done": True,
        "done_reason": done_reason,
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": 10,
    }
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


@pytest.fixture
def mock_ollama(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {"bodies": [], "requests": [], "calls": 0, "errors": []}

    def _set_bodies(*bodies: dict[str, Any]) -> None:
        state["bodies"] = list(bodies)
        state["errors"] = []

    def _set_errors(*errors: BaseException) -> None:
        state["errors"] = list(errors)
        state["bodies"] = []

    def fake_run_ollama_request(**kwargs: Any) -> dict[str, Any]:
        state["calls"] += 1
        state["requests"].append(kwargs)
        if state["errors"]:
            raise state["errors"].pop(0)
        if not state["bodies"]:
            raise OllamaRequestError(
                "The localhost Ollama server refused the connection. Confirm Ollama "
                "is running on 127.0.0.1:11434.",
                classification="connection_refused",
            )
        return state["bodies"].pop(0)

    monkeypatch.setattr(
        "resume_tailor.gemma_analysis.run_ollama_request",
        fake_run_ollama_request,
    )
    state["set_bodies"] = _set_bodies
    state["set_errors"] = _set_errors
    return state


def test_gemma_local_is_default_and_labels() -> None:
    assert DEFAULT_ANALYSIS_PROVIDER == "gemma_local"
    assert normalize_analysis_provider(None) == "gemma_local"
    assert analysis_provider_label("gemma_local") == "Gemma Local"
    assert analysis_workflow_label("gemma_local") == "Gemma Local analysis"
    assert analysis_workflow_label("codex") == "Codex analysis"
    assert analysis_workflow_label("grok_cli") == "Grok CLI analysis"
    stages = dict(workflow_stages_for_provider("grok_cli"))
    assert stages["codex_analysis"] == "Grok CLI analysis"
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


def test_model_and_output_token_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMMA_ANALYSIS_MODEL", raising=False)
    monkeypatch.delenv("GEMMA_WRITER_MODEL", raising=False)
    monkeypatch.delenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", raising=False)
    assert resolve_gemma_analysis_model(None) == "resume-tailor-gemma"
    assert (
        resolve_gemma_analysis_max_output_tokens(None)
        == DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS
    )
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "4096")
    assert resolve_gemma_analysis_max_output_tokens(None) == 4096
    monkeypatch.setenv("GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS", "   ")
    assert (
        resolve_gemma_analysis_max_output_tokens(None)
        == DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS
    )
    monkeypatch.setenv("GEMMA_ANALYSIS_MODEL", "   ")
    monkeypatch.setenv("GEMMA_WRITER_MODEL", "")
    assert resolve_gemma_analysis_model(None) == "resume-tailor-gemma"


def test_chat_request_includes_num_predict_and_omits_think(
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
        max_output_tokens=2048,
    )
    assert request["stream"] is False
    assert request["options"]["temperature"] == 0
    assert request["options"]["num_predict"] == 2048
    assert "think" not in request
    assert "$schema" not in request["format"]
    assert request["format"]["additionalProperties"] is False


def test_prompt_compaction_smaller_than_legacy_shared_prompt(
    master_resume: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    catalog = _job_catalog()
    job = "Skills: Python and RAG. Build agentic systems with evidence."
    legacy = build_analysis_prompt(
        extracted,
        job,
        catalog,
        company="Example",
        role="Developer",
    )
    compact = build_gemma_analysis_prompt(
        extracted,
        job,
        catalog,
        company="Example",
        role="Developer",
    )
    legacy_bytes = len(legacy.encode("utf-8"))
    compact_bytes = len(compact.encode("utf-8"))
    assert compact_bytes < legacy_bytes
    # Authority markers preserved.
    assert "SOURCE_CATALOG" in compact
    assert "JOB_REQUIREMENTS" in compact
    assert "evidence_allowed" in compact
    assert "unsupported_requirement_ids" in compact
    assert "skill_groups.N" in compact
    assert "character_counting_contract" in compact
    assert "BEGIN_UNTRUSTED_JOB_DESCRIPTION_" in compact
    assert estimate_prompt_tokens(compact) < estimate_prompt_tokens(legacy)


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
    body = mock_ollama["requests"][0]["body"]
    assert body["options"]["num_predict"] == DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS
    assert "think" not in body
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "success"
    assert diagnostic["max_output_tokens"] == DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS
    assert diagnostic["configured_timeout_seconds"] == 30
    meta = write_resolved_analysis_artifact(
        tmp_path,
        payload,
        provider="gemma_local",
    )
    assert meta["legacy_codex_alias_written"] is False
    assert not (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).exists()
    document = json.loads(
        (tmp_path / ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert document["provider"] == "gemma_local"


def test_connection_refused_is_ollama_unavailable(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_errors"](
        OllamaRequestError(
            "refused",
            classification="connection_refused",
        )
    )
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
    assert mock_ollama["calls"] == 1


def test_model_missing_is_model_unavailable(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_errors"](
        OllamaRequestError(
            "not found",
            classification="http_error",
            http_status=404,
        )
    )
    with pytest.raises(GemmaModelUnavailableError):
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
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "model_unavailable"


def test_active_generation_deadline_is_analysis_timeout(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_errors"](
        OllamaRequestError(
            "The localhost Ollama request exceeded its bounded timeout.",
            classification="timeout",
        )
    )
    with pytest.raises(GemmaAnalysisTimeoutError, match="generation time limit"):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=15,
        )
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "analysis_timeout"
    assert diagnostic["configured_timeout_seconds"] == 15
    assert diagnostic["generation_active"] is True
    assert mock_ollama["calls"] == 1  # no repair on timeout


def test_http_500_is_ollama_internal_error_not_unavailable(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_errors"](
        OllamaRequestError(
            "internal server error",
            classification="http_error",
            http_status=500,
        )
    )
    with pytest.raises(GemmaOllamaInternalError):
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
    assert diagnostic["classification"] == "ollama_internal_error"
    assert diagnostic["http_status"] == 500
    assert "unavailable" not in diagnostic["classification"]


def test_valid_json_at_exact_num_predict_with_normal_stop_succeeds(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    """eval_count == num_predict must not reject complete valid JSON with stop."""
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    analysis = _valid_analysis(requirements)
    mock_ollama["set_bodies"](
        _ollama_body(analysis, done_reason="stop", eval_count=3072)
    )
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        max_output_tokens=3072,
    )
    assert payload["role_summary"]
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "success"
    assert mock_ollama["calls"] == 1


def test_explicit_length_stop_is_output_limit_reached(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    mock_ollama["set_bodies"](
        _ollama_body('{"role_summary":', done_reason="length", eval_count=3072)
    )
    with pytest.raises(GemmaOutputLimitError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            max_output_tokens=3072,
        )
    assert mock_ollama["calls"] == 1
    diagnostic = json.loads(
        (tmp_path / GEMMA_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "output_limit_reached"
    assert diagnostic["output_ceiling_reached"] is True


def test_incomplete_json_at_num_predict_without_stop_reason(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    """Missing/ambiguous stop + ceiling + incomplete body → output_limit_reached."""
    extracted, _ = extract_resume(master_resume)
    body = _ollama_body('{"role_summary":', done_reason="stop", eval_count=3072)
    body.pop("done_reason")
    mock_ollama["set_bodies"](body)
    with pytest.raises(GemmaOutputLimitError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            max_output_tokens=3072,
        )
    assert mock_ollama["calls"] == 1  # not repaired


def test_incomplete_json_at_num_predict_with_normal_stop_may_repair(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    """Normal stop + incomplete JSON is repairable once (not auto output_limit)."""
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    valid = _valid_analysis(requirements)
    mock_ollama["set_bodies"](
        _ollama_body('{"role_summary":', done_reason="stop", eval_count=3072),
        _ollama_body(valid, done_reason="stop", eval_count=100),
    )
    payload = invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        max_output_tokens=3072,
    )
    assert payload["role_summary"]
    assert mock_ollama["calls"] == 2
    for request in mock_ollama["requests"]:
        assert request["body"]["options"]["num_predict"] == 3072


def test_truncated_json_never_accepted_without_length_reason(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    # Incomplete JSON with stop reason still fails parse; one repair only.
    mock_ollama["set_bodies"](
        _ollama_body('{"role_summary": "x"', done_reason="stop"),
        _ollama_body('{"role_summary": "x"', done_reason="stop"),
    )
    with pytest.raises(GemmaInnerAnalysisError):
        invoke_gemma_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
        )
    assert mock_ollama["calls"] == 2


def test_schema_invalid_may_receive_one_repair(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    valid = _valid_analysis(requirements)
    mock_ollama["set_bodies"](
        _ollama_body({"role_summary": "incomplete"}),
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
    repair_prompt = mock_ollama["requests"][1]["body"]["messages"][1]["content"]
    assert "failure_class=schema_failure" in repair_prompt
    # Full malformed multi-k response is not re-injected.
    assert "incomplete" not in repair_prompt or "failure_class" in repair_prompt
    assert mock_ollama["requests"][1]["body"]["options"]["num_predict"] == (
        DEFAULT_GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS
    )


def test_repair_does_not_include_full_malformed_response(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    valid = _valid_analysis(requirements)
    huge = "X" * 5000
    mock_ollama["set_bodies"](
        _ollama_body("```json\n" + huge + "\n```"),
        _ollama_body(valid),
    )
    invoke_gemma_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
    )
    repair_prompt = mock_ollama["requests"][1]["body"]["messages"][1]["content"]
    assert "failure_class=malformed_inner_analysis" in repair_prompt
    assert huge not in repair_prompt


def test_fenced_and_trailing_json_rejected() -> None:
    with pytest.raises(GemmaInnerAnalysisError, match="Markdown"):
        parse_exact_analysis_json("```json\n{}\n```")
    with pytest.raises(GemmaInnerAnalysisError, match="trailing"):
        parse_exact_analysis_json('{"a":1}\ncommentary')


def test_invalid_source_ids_not_retried_inside_provider(
    master_resume: Path,
    tmp_path: Path,
    mock_ollama: dict[str, Any],
) -> None:
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
    assert mock_ollama["calls"] == 1
    _, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert issues


def test_ui_timeout_message_not_unavailable() -> None:
    from resume_tailor.ui import _safe_error_message

    message = _safe_error_message(GemmaAnalysisTimeoutError(900))
    assert "generation time limit" in message.casefold()
    assert "unavailable" not in message.casefold()
