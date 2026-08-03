"""Grok Build analysis-provider tests (mocked subprocess only)."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from resume_tailor.analysis import (
    ANALYSIS_RESOLVED_FILENAME,
    CODEX_ANALYSIS_RESOLVED_FILENAME,
    DEFAULT_ANALYSIS_PROVIDER,
    analysis_provider_label,
    invoke_analysis,
    normalize_analysis_provider,
    unwrap_resolved_analysis_document,
    write_resolved_analysis_artifact,
)
from resume_tailor.cli import build_parser
from resume_tailor.codex_analysis import invoke_codex_analysis
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.grok_analysis import (
    GROK_ANALYSIS_DIAGNOSTIC_FILENAME,
    GROK_ANALYSIS_PROMPT_FILENAME,
    GROK_ANALYSIS_RESPONSE_FILENAME,
    GROK_ANALYSIS_SCHEMA_FILENAME,
    GROK_ANALYSIS_TRANSPORT_FILENAME,
    grok_analysis_args,
    invoke_grok_analysis,
    parse_grok_inner_analysis,
    parse_grok_transport_envelope,
    resolve_grok_executable,
)
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.utilities import (
    CodexUsageLimitError,
    GrokAuthenticationError,
    GrokExecutableError,
    GrokInnerAnalysisError,
    GrokProcessError,
    GrokPromptTooLargeError,
    GrokTimeoutError,
    GrokTransportEnvelopeError,
    GrokUsageLimitError,
    ModelError,
    SourceEvidenceError,
)


def _job_catalog() -> dict:
    return build_job_requirement_catalog(
        "Skills: Python and RAG.",
        structured_job={"technologies_and_skills": ["Python", "RAG"]},
    )


def _valid_envelope(text: str, *, stop_reason: str = "end_turn") -> str:
    return json.dumps(
        {
            "text": text,
            "stopReason": stop_reason,
            "thought": "must-never-appear-in-artifacts",
            "sessionId": "s",
            "requestId": "r",
            "usage": {"x": 1},
            "modelUsage": {"grok-4.5-build": {}},
        }
    )


def test_codex_remains_default_analysis_provider() -> None:
    assert DEFAULT_ANALYSIS_PROVIDER == "codex"
    assert normalize_analysis_provider(None) == "codex"
    assert normalize_analysis_provider("") == "codex"
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
    assert args.analysis_provider == "codex"


def test_explicit_grok_provider_selection() -> None:
    assert normalize_analysis_provider("grok") == "grok"
    assert analysis_provider_label("grok") == "Grok"
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
            "--analysis-provider",
            "grok",
        ]
    )
    assert args.analysis_provider == "grok"


def test_resolve_grok_executable_from_path(
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = str(stubs_on_path / "grok")
    resolved = resolve_grok_executable(explicit)
    assert Path(resolved).name == "grok"
    assert os.access(resolved, os.X_OK)

    # Prefer the verified home path only when present; otherwise PATH resolution.
    monkeypatch.setattr(
        "resume_tailor.grok_analysis.DEFAULT_GROK_EXECUTABLE",
        tmp_path / "no-home-grok",
    )
    monkeypatch.setenv("PATH", str(stubs_on_path))
    resolved_path = resolve_grok_executable()
    assert Path(resolved_path).resolve() == (stubs_on_path / "grok").resolve()


def test_resolve_grok_executable_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(
        "resume_tailor.grok_analysis.DEFAULT_GROK_EXECUTABLE",
        tmp_path / "missing-grok",
    )
    with pytest.raises(GrokExecutableError, match="not found"):
        resolve_grok_executable()


def test_grok_command_shape(stubs_on_path: Path) -> None:
    args = grok_analysis_args(
        executable=str(stubs_on_path / "grok"),
        prompt="synthetic prompt",
    )
    assert args[0].endswith("grok") or Path(args[0]).name == "grok"
    assert args[1:] == [
        "--no-auto-update",
        "-p",
        "synthetic prompt",
        "--output-format",
        "json",
    ]
    assert "shell" not in " ".join(args).casefold()


def test_successful_grok_transport_and_valid_inner_analysis(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    liveness: list[tuple[float, bool]] = []
    payload = invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=_job_catalog(),
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
        progress_handler=lambda elapsed, alive: liveness.append((elapsed, alive)),
    )
    assert payload["role_summary"]
    assert payload["supported_requirement_mappings"]
    assert (tmp_path / GROK_ANALYSIS_SCHEMA_FILENAME).is_file()
    assert (tmp_path / GROK_ANALYSIS_PROMPT_FILENAME).is_file()
    assert (tmp_path / GROK_ANALYSIS_TRANSPORT_FILENAME).is_file()
    assert (tmp_path / GROK_ANALYSIS_RESPONSE_FILENAME).is_file()
    assert (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).is_file()
    transport = json.loads(
        (tmp_path / GROK_ANALYSIS_TRANSPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert transport["thought_excluded"] is True
    assert "thought" not in transport
    assert "must-never-appear" not in (
        tmp_path / GROK_ANALYSIS_TRANSPORT_FILENAME
    ).read_text(encoding="utf-8")
    prompt_diag = (tmp_path / GROK_ANALYSIS_PROMPT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "body_omitted" not in prompt_diag or "omitted" in prompt_diag.casefold()
    assert "STUB_THOUGHT" not in prompt_diag
    diagnostic = json.loads(
        (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "success"
    assert diagnostic["thought_excluded"] is True
    assert diagnostic["credentials_excluded"] is True
    assert liveness[0] == (0.0, True)
    assert liveness[-1][1] is False


def test_invoke_analysis_dispatches_to_grok(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    payload = invoke_analysis(
        provider="grok",
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=_job_catalog(),
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    assert "role_summary" in payload


def test_malformed_transport_json() -> None:
    with pytest.raises(GrokTransportEnvelopeError):
        parse_grok_transport_envelope("not-json")


def test_transport_not_object() -> None:
    with pytest.raises(GrokTransportEnvelopeError, match="not an object"):
        parse_grok_transport_envelope("[1, 2]")


def test_missing_or_nonstring_text() -> None:
    with pytest.raises(GrokTransportEnvelopeError, match="text"):
        parse_grok_transport_envelope(
            json.dumps({"stopReason": "end_turn", "text": 12})
        )
    with pytest.raises(GrokTransportEnvelopeError, match="text"):
        parse_grok_transport_envelope(json.dumps({"stopReason": "end_turn"}))


def test_unacceptable_stop_reason() -> None:
    with pytest.raises(GrokTransportEnvelopeError, match="stopReason"):
        parse_grok_transport_envelope(
            json.dumps({"text": "{}", "stopReason": "length"})
        )


def test_markdown_fenced_inner_response() -> None:
    with pytest.raises(GrokInnerAnalysisError, match="Markdown"):
        parse_grok_inner_analysis("```json\n{}\n```")


def test_multiple_inner_json_documents() -> None:
    with pytest.raises(GrokInnerAnalysisError, match="multiple"):
        parse_grok_inner_analysis('{"a":1}\n{"b":2}')


def test_trailing_inner_commentary() -> None:
    with pytest.raises(GrokInnerAnalysisError, match="trailing"):
        parse_grok_inner_analysis('{"a": 1}\ncommentary')


def test_malformed_inner_json() -> None:
    with pytest.raises(GrokInnerAnalysisError, match="malformed"):
        parse_grok_inner_analysis("{not-json")


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed_transport", GrokTransportEnvelopeError),
        ("transport_not_object", GrokTransportEnvelopeError),
        ("missing_text", GrokTransportEnvelopeError),
        ("nonstring_text", GrokTransportEnvelopeError),
        ("bad_stop_reason", GrokTransportEnvelopeError),
        ("markdown_fence", GrokInnerAnalysisError),
        ("multiple_json", GrokInnerAnalysisError),
        ("trailing_text", GrokInnerAnalysisError),
        ("malformed_inner", GrokInnerAnalysisError),
        ("auth_failure", GrokAuthenticationError),
        ("usage_limit", GrokUsageLimitError),
        ("nonzero", GrokProcessError),
    ],
)
def test_grok_stub_error_modes(
    mode: str,
    error_type: type[Exception],
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("STUB_GROK_MODE", mode)
    with pytest.raises(error_type):
        invoke_grok_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "grok"),
        )
    diagnostic = tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME
    assert diagnostic.is_file()
    body = diagnostic.read_text(encoding="utf-8")
    assert "STUB_THOUGHT" not in body
    assert "apify_api_" not in body.casefold()
    assert "token=" not in body.casefold()


def test_valid_transport_schema_invalid_analysis(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("STUB_GROK_MODE", "schema_invalid")
    with pytest.raises(SourceEvidenceError, match="canonical evidence contract"):
        invoke_grok_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "grok"),
        )
    diagnostic = json.loads(
        (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "schema_failure"


def test_invalid_source_ids_rejected_after_grok(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    monkeypatch.setenv("STUB_GROK_MODE", "invalid_source_ids")
    raw = invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    resolved, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert issues
    assert any("source" in issue.code or "unknown" in issue.code for issue in issues)
    assert resolved is not None


def test_unsupported_requirement_partition_failure(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    monkeypatch.setenv("STUB_GROK_MODE", "unsupported_claim")
    raw = invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    _, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert issues


def test_character_budget_violation_for_structured_field(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    monkeypatch.setenv("STUB_GROK_MODE", "character_budget")
    raw = invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    _, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert any(
        issue.code == "structured_proposal_over_budget" for issue in issues
    )


def test_timeout_and_clean_termination(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("STUB_GROK_MODE", "timeout")
    with pytest.raises(GrokTimeoutError, match="timed out"):
        invoke_grok_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=1,
            executable=str(stubs_on_path / "grok"),
        )
    diagnostic = json.loads(
        (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "timeout"


def test_codex_quota_specific_classification(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)

    def fake_run_command(args, **kwargs):  # type: ignore[no-untyped-def]
        from resume_tailor.utilities import CommandResult

        return CommandResult(
            tuple(str(a) for a in args),
            "",
            "Error: You've hit your usage limit. Try again later.",
            1,
        )

    monkeypatch.setattr(
        "resume_tailor.codex_analysis.run_command",
        fake_run_command,
    )
    with pytest.raises(CodexUsageLimitError, match="usage limit"):
        invoke_codex_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "codex"),
        )


def test_generic_codex_failure_is_not_quota(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("STUB_CODEX_MODE", "process_error")
    with pytest.raises(ModelError) as caught:
        invoke_codex_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "codex"),
        )
    assert not isinstance(caught.value, CodexUsageLimitError)
    assert "usage limit" not in str(caught.value).casefold()


def test_no_automatic_codex_to_grok_invocation(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Codex failure must not invoke the Grok executable."""
    extracted, _ = extract_resume(master_resume)
    grok_log = tmp_path / "grok-invocations.jsonl"
    monkeypatch.setenv("STUB_GROK_INVOCATION_LOG", str(grok_log))
    monkeypatch.setenv("STUB_CODEX_MODE", "process_error")
    with pytest.raises(ModelError):
        invoke_analysis(
            provider="codex",
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "codex"),
        )
    assert not grok_log.exists()


def test_thought_field_excluded_from_artifacts(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=_job_catalog(),
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    for name in (
        GROK_ANALYSIS_PROMPT_FILENAME,
        GROK_ANALYSIS_TRANSPORT_FILENAME,
        GROK_ANALYSIS_RESPONSE_FILENAME,
        GROK_ANALYSIS_DIAGNOSTIC_FILENAME,
    ):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "STUB_THOUGHT_MUST_NEVER_BE_TREATED_AS_ANALYSIS" not in text
        assert '"thought"' not in text or "thought_excluded" in text


def test_no_credential_leakage_in_diagnostics(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    monkeypatch.setenv("STUB_GROK_MODE", "auth_failure")
    monkeypatch.setenv("GROK_API_KEY", "super-secret-value")
    monkeypatch.setenv("XAI_API_TOKEN", "another-secret")
    with pytest.raises(GrokAuthenticationError):
        invoke_grok_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "grok"),
        )
    for path in tmp_path.glob("grok-analysis-*"):
        content = path.read_text(encoding="utf-8")
        assert "super-secret-value" not in content
        assert "another-secret" not in content
        assert "GROK_API_KEY" not in content
        assert "XAI_API_TOKEN" not in content


def test_existing_codex_behavior_unchanged(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
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
    assert payload["supported_requirement_mappings"]
    assert (tmp_path / "codex-analysis.json").is_file()
    # Grok artifacts must not appear for a pure Codex run.
    assert not (tmp_path / GROK_ANALYSIS_RESPONSE_FILENAME).exists()


def test_resolved_artifact_is_provider_neutral_for_grok(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _job_catalog()
    raw = invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Python role",
        job_requirements=requirements,
        company="Example",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable=str(stubs_on_path / "grok"),
    )
    resolved, issues = resolve_analysis_evidence(raw, extracted, requirements)
    assert issues == []
    meta = write_resolved_analysis_artifact(
        tmp_path,
        resolved,
        provider="grok",
    )
    assert meta["filename"] == ANALYSIS_RESOLVED_FILENAME
    assert meta["provider"] == "grok"
    assert meta["legacy_codex_alias_written"] is False
    assert (tmp_path / ANALYSIS_RESOLVED_FILENAME).is_file()
    assert not (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).exists()
    document = json.loads(
        (tmp_path / ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert document["provider"] == "grok"
    assert document["provider_label"] == "Grok"
    assert unwrap_resolved_analysis_document(document)["role_summary"]


def test_resolved_artifact_codex_alias_only_for_codex(tmp_path: Path) -> None:
    analysis = {
        "role_summary": "x",
        "fit_assessment": {"overall": "y", "strengths": [], "gaps": []},
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": [],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }
    meta = write_resolved_analysis_artifact(
        tmp_path,
        analysis,
        provider="codex",
    )
    assert meta["legacy_codex_alias_written"] is True
    assert (tmp_path / ANALYSIS_RESOLVED_FILENAME).is_file()
    assert (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).is_file()
    document = json.loads(
        (tmp_path / ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert document["provider"] == "codex"
    bare = json.loads(
        (tmp_path / CODEX_ANALYSIS_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert bare["role_summary"] == "x"
    assert "provider" not in bare


def test_e2big_is_prompt_too_large_not_executable_unavailable(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)

    def raise_e2big(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        from resume_tailor.utilities import DependencyError

        cause = OSError(errno.E2BIG, "Argument list too long")
        raise DependencyError("Could not run grok: Argument list too long") from cause

    monkeypatch.setattr("resume_tailor.grok_analysis.run_command", raise_e2big)
    with pytest.raises(GrokPromptTooLargeError, match="argument-size limit"):
        invoke_grok_analysis(
            extracted_resume=extracted,
            job_description="Python role",
            job_requirements=_job_catalog(),
            company="Example",
            role="Developer",
            run_directory=tmp_path,
            timeout_seconds=30,
            executable=str(stubs_on_path / "grok"),
        )
    diagnostic = json.loads(
        (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["classification"] == "prompt_too_large"
    assert diagnostic.get("prompt_contents_omitted") is True
    text = (tmp_path / GROK_ANALYSIS_DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    assert "Python role" not in text
    assert "skill_groups" not in text


def test_transport_parser_ignores_thought_for_correctness() -> None:
    analysis = {
        "role_summary": "x",
        "fit_assessment": {"overall": "y", "strengths": [], "gaps": []},
        "supported_requirement_mappings": [],
        "unsupported_requirement_ids": [],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": [],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }
    envelope = parse_grok_transport_envelope(
        _valid_envelope(json.dumps(analysis))
    )
    assert envelope["text"]
    # Correctness path uses only text + stopReason.
    inner = parse_grok_inner_analysis(envelope["text"])
    assert inner["role_summary"] == "x"
