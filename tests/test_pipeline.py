from __future__ import annotations

import json
from pathlib import Path

import pytest

import resume_tailor.cli as cli_module
from resume_tailor.analytics import ANALYTICS_DATABASE_FILENAME, AnalyticsStore
from resume_tailor.cli import build_parser, main, run_pipeline
from resume_tailor.orchestration import ApprovalResponse, PipelineHooks
from resume_tailor.utilities import (
    ApifyLinkedInRetrievalError,
    ApprovalError,
    InputError,
    TailoringPreflightError,
)
from resume_tailor.utilities import ExitCode, sha256_file


LINKEDIN_JOB_URL = (
    "https://www.linkedin.com/jobs/view/general-ai-role-4123456789/"
)


def _analytics_store(tmp_path: Path) -> AnalyticsStore:
    return AnalyticsStore(
        tmp_path / "application-data" / ANALYTICS_DATABASE_FILENAME
    )


def _description_with_length(length: int) -> str:
    fragment = "Build safe Python services and automated tests. "
    value = (fragment * ((length // len(fragment)) + 1))[:length]
    if value[-1].isspace():
        value = value[:-1] + "x"
    assert len(value) == length
    return value


def _canonical_apify_posting(
    *,
    description: str | None = None,
) -> dict[str, object]:
    return {
        "fetch_status": "success",
        "requested_url": LINKEDIN_JOB_URL,
        "final_resolved_url": LINKEDIN_JOB_URL,
        "linkedin_job_id": "4123456789",
        "job_title": "Machine Learning Engineer",
        "company": "Example AI Systems",
        "location": "Remote",
        "workplace_type": "remote",
        "employment_type": "Full-time",
        "salary": None,
        "seniority_level": None,
        "date_posted": None,
        "applicant_count": None,
        "retrieval_source": "apify",
        "normalized_job_description": description or (
            "Build safe Python services for synthetic evidence workflows. "
            "Collaborate with engineers, validate structured inputs, write tests, "
            "document decisions, maintain APIs, review failures, and improve "
            "reliable automation across a privacy-conscious local application. "
            "This additional synthetic detail ensures the posting is substantive "
            "without containing personal, company-confidential, or résumé data."
        ),
        "responsibilities": [],
        "required_qualifications": [],
        "preferred_qualifications": [],
        "technologies_and_skills": [],
        "ai_focus_areas": [],
        "warnings": ["Synthetic Apify fixture."],
    }


def _stub_apify_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: list[str] | None = None,
    description: str | None = None,
) -> None:
    posting = _canonical_apify_posting(description=description)

    def retrieve(**kwargs: object) -> dict[str, object]:
        if calls is not None:
            calls.append("apify_retrieval")
        run_directory = kwargs["run_directory"]
        assert isinstance(run_directory, Path)
        (run_directory / "job-source.json").write_text(
            json.dumps(posting, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_directory / "apify-linkedin-retrieval-diagnostic.json").write_text(
            json.dumps(
                {
                    "provider": "apify",
                    "classification": "success",
                    "provider_output_omitted": True,
                    "api_token_omitted": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return dict(posting)

    monkeypatch.setattr(cli_module, "invoke_apify_linkedin_retrieval", retrieve)


def _arguments(
    master_resume: Path,
    job_file: Path,
    output_dir: Path,
    *,
    yes: bool = True,
) -> list[str]:
    arguments = [
        "--resume",
        str(master_resume),
        "--job-file",
        str(job_file),
        "--company",
        "Example Talent",
        "--role",
        "Agentic AI Developer",
        "--output-dir",
        str(output_dir),
        "--timeout",
        "30s",
        "--writer-provider",
        "antigravity",
        # Pipeline fixtures use the Codex CLI stub; select it explicitly so the
        # free Gemma Local default does not change existing Codex coverage.
        "--analysis-provider",
        "codex",
    ]
    if yes:
        arguments.append("--yes")
    return arguments


def _url_arguments(
    master_resume: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "--resume",
        str(master_resume),
        "--job-url",
        LINKEDIN_JOB_URL,
        "--output-dir",
        str(output_dir),
        "--timeout",
        "30s",
        "--writer-provider",
        "antigravity",
        "--analysis-provider",
        "codex",
    ]


def test_simulated_pipeline_artifact_tree_and_source_immutability(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_hash = sha256_file(master_resume)
    schema_log = tmp_path / "codex-schema-paths.txt"
    monkeypatch.setenv("STUB_CODEX_SCHEMA_LOG", str(schema_log))
    output_dir = tmp_path / "Tailored Resumes"
    code = main(_arguments(master_resume, job_file, output_dir))
    assert code == ExitCode.OK
    run_directories = list(output_dir.iterdir())
    assert len(run_directories) == 1
    run = run_directories[0]
    basename = "Sample-Candidate-Example-Talent-Agentic-AI-Developer"
    expected = {
        "job-description.txt",
        "job-requirements.json",
        "extracted-master-resume.json",
        "codex-analysis-transport.schema.json",
        "codex-analysis.json",
        "analysis-resolved.json",
        "codex-analysis-resolved.json",
        "codex-analysis-approval.json",
        "antigravity-response.json",
        "antigravity-response-envelope.json",
        "tailored-content.json",
        "content-diff.md",
        f"{basename}.docx",
        f"{basename}.pdf",
        "preview.png",
        "final-qa.md",
        "run-metadata.json",
    }
    assert expected <= {path.name for path in run.iterdir()}
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["source_resume"]["unchanged"] is True
    assert metadata["analysis_inputs"]["version"] == 2
    assert metadata["analytics"]["database_filename"] == ANALYTICS_DATABASE_FILENAME
    assert metadata["analytics"]["resume_version_id"] is not None
    assert "database_path" not in metadata["analytics"]
    analytics = _analytics_store(tmp_path)
    assert analytics.summary()["totals"]["unique_jobs_viewed"] == 1
    assert analytics.summary()["totals"]["applications_submitted"] == 0
    assert len(analytics.sanitized_export()["resume_versions"]) == 1
    assert len(metadata["analysis_inputs"]["job_description_sha256"]) == 64
    assert len(metadata["analysis_inputs"]["extracted_resume_sha256"]) == 64
    assert len(metadata["analysis_inputs"]["job_requirements_sha256"]) == 64
    schema_metadata = metadata["codex_analysis_transport_schema"]
    assert schema_metadata["filename"] == "codex-analysis-transport.schema.json"
    assert len(schema_metadata["sha256"]) == 64
    assert schema_metadata["generated_from_source_and_requirement_catalogs"] is True
    assert schema_metadata["job_requirement_id_count"] > 0
    assert sha256_file(run / schema_metadata["filename"]) == schema_metadata["sha256"]
    approval_metadata = metadata["codex_analysis_approval"]
    assert approval_metadata["filename"] == "codex-analysis-approval.json"
    assert approval_metadata["decision"] == "approved"
    assert sha256_file(run / approval_metadata["filename"]) == approval_metadata[
        "sha256"
    ]
    response_metadata = metadata["antigravity_response"]
    assert response_metadata["execution_mode"] == "print"
    assert response_metadata["output_format"] == "stream-json"
    assert response_metadata["response_envelope_type"] == (
        "stream-json-event-result:json-wrapper-structured_output"
    )
    assert response_metadata["validation_result"] == "PASS"
    assert response_metadata["cli_version"] == "1.1.8-stub"
    assert sha256_file(run / response_metadata["response"]["filename"]) == (
        response_metadata["response"]["sha256"]
    )
    assert "work" not in {path.name for path in run.iterdir()}
    assert sha256_file(master_resume) == before_hash
    used_schemas = [
        Path(line).name
        for line in schema_log.read_text(encoding="utf-8").splitlines()
    ]
    assert used_schemas == [
        "codex-analysis-transport.schema.json",
        "final_qa_provider.openai.schema.json",
    ]


def test_default_pipeline_routes_approved_writing_to_local_gemma(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_gemma(**kwargs: object) -> dict:
        calls.append("gemma")
        return kwargs["extracted_resume"]["content"]  # type: ignore[index]

    fake_metadata = {
        "response": {"filename": "ollama-response.json", "sha256": "0" * 64},
        "response_envelope_type": "ollama-chat-message-content-json",
        "output_format": "json-schema",
        "validation_result": "PASS",
    }
    monkeypatch.setattr(cli_module, "invoke_ollama", fake_gemma)
    monkeypatch.setattr(
        cli_module,
        "invoke_antigravity",
        lambda **kwargs: pytest.fail("Antigravity was invoked for a default run"),
    )
    monkeypatch.setattr(
        cli_module,
        "load_ollama_response_metadata",
        lambda *args, **kwargs: fake_metadata,
    )
    monkeypatch.setattr(
        cli_module,
        "_tailoring_dependency_versions",
        lambda *args, **kwargs: {
            "ollama": "synthetic",
            "ollama_model": "resume-tailor-gemma",
            "libreoffice": "synthetic",
        },
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            str(master_resume),
            "--job-file",
            str(job_file),
            "--company",
            "Example Talent",
            "--role",
            "Agentic AI Developer",
            "--output-dir",
            str(tmp_path / "output"),
            "--timeout",
            "30s",
            "--analysis-provider",
            "codex",
        ]
    )

    def approval(request: object) -> ApprovalResponse:
        kind = request.kind  # type: ignore[attr-defined]
        return ApprovalResponse("approve" if kind == "codex_analysis" else "reject")

    with pytest.raises(ApprovalError):
        run_pipeline(args, hooks=PipelineHooks(approval_handler=approval))

    assert calls == ["gemma"]
    run = next((tmp_path / "output").iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["writer"] == {
        "provider": "ollama",
        "name": "Gemma 4 12B",
        "model": "resume-tailor-gemma",
        "document_format": "headless",
    }
    assert metadata["revision_cycle"]["initial"]["provider"] == "ollama"


def test_deterministic_only_pipeline_reaches_evidence_and_rendering_without_provider(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real writer entrypoint and every post-writer pipeline gate."""
    import resume_tailor.docx_render as docx_render_module
    import resume_tailor.headless_render as headless_render_module
    import resume_tailor.ollama_writer as ollama_writer_module

    progress: list[tuple[str, str]] = []
    provider_calls: list[str] = []

    monkeypatch.setattr(
        cli_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {"resume_tailor": "synthetic", "codex": "mocked"},
    )
    monkeypatch.setattr(
        cli_module,
        "_tailoring_dependency_versions",
        lambda *_args, **_kwargs: {
            "ollama": "not-invoked",
            "ollama_model": "not-loaded",
            "libreoffice": "mocked",
        },
    )

    def fake_analysis(**kwargs: object) -> dict[str, object]:
        extracted = kwargs["extracted_resume"]
        requirements = kwargs["job_requirements"]
        run_directory = kwargs["run_directory"]
        assert isinstance(extracted, dict)
        assert isinstance(requirements, dict)
        assert isinstance(run_directory, Path)
        engineering = extracted["content"]["skill_groups"][2]
        raw: dict[str, object] = {
            "role_summary": "Synthetic deterministic tailoring run.",
            "fit_assessment": {
                "overall": "Synthetic evidence-backed fit.",
                "strengths": ["Locally authenticated structured skills"],
                "gaps": ["No unsupported claims are added"],
            },
            "supported_requirement_mappings": [],
            "unsupported_requirement_ids": [
                item["requirement_id"] for item in requirements["requirements"]
            ],
            "recommended_edits": [
                {
                    "target_source_id": "skill_groups.2",
                    "operation": "replace",
                    "proposed_text": (
                        f"{engineering['label']}: FastAPI, JSON Schema, pytest, SQL"
                    ),
                    "alignment_rationale": (
                        "Surface an already authenticated language in the engineering group."
                    ),
                    "evidence_source_ids": ["skill_groups.2", "skill_groups.0"],
                }
            ],
            "immutable_facts": [],
            "forbidden_claims": ["GraphQL"],
            "content_budget_guidance": [],
            "questions_for_user": [],
        }
        (run_directory / "codex-analysis.json").write_text(
            json.dumps(raw),
            encoding="utf-8",
        )
        return raw

    monkeypatch.setattr(cli_module, "invoke_analysis", fake_analysis)
    monkeypatch.setattr(cli_module, "invoke_codex_analysis", fake_analysis)

    def reject_provider(**_kwargs: object) -> object:
        provider_calls.append("ollama")
        pytest.fail("The deterministic-only pipeline invoked Ollama")

    monkeypatch.setattr(
        ollama_writer_module,
        "run_ollama_request",
        reject_provider,
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_antigravity",
        lambda **_kwargs: pytest.fail("The deterministic pipeline invoked Antigravity"),
    )

    def fake_render(**kwargs: object) -> None:
        destination = kwargs["destination_path"]
        assert isinstance(destination, Path)
        destination.write_bytes(b"synthetic deterministic docx")

    def fake_export(**kwargs: object) -> str:
        pdf_path = kwargs["pdf_path"]
        preview_path = kwargs["preview_path"]
        assert isinstance(pdf_path, Path)
        assert isinstance(preview_path, Path)
        pdf_path.write_bytes(b"%PDF-1.4 synthetic deterministic")
        preview_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
        return "Synthetic deterministic resume text"

    def fake_final_qa(**kwargs: object) -> dict[str, object]:
        run_directory = kwargs["run_directory"]
        generation = kwargs["generation"]
        assert isinstance(run_directory, Path)
        assert generation == "initial"
        result: dict[str, object] = {
            "status": "pass",
            "summary": "Synthetic read-only QA passed.",
            "issues": [],
            "technical_failure": None,
        }
        (run_directory / "final-qa.initial.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        (run_directory / "final-qa.initial.md").write_text(
            "# Synthetic read-only QA\n\nPASS\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(headless_render_module, "render_headless_docx", fake_render)
    monkeypatch.setattr(docx_render_module, "export_and_validate_pdf", fake_export)
    monkeypatch.setattr(cli_module, "invoke_final_qa", fake_final_qa)
    monkeypatch.setattr(cli_module, "run_initial_qa", fake_final_qa)

    parser = build_parser()
    output_dir = tmp_path / "deterministic-output"
    args = parser.parse_args(
        [
            "--resume",
            str(master_resume),
            "--job-file",
            str(job_file),
            "--company",
            "Synthetic Systems",
            "--role",
            "Evidence Engineer",
            "--output-dir",
            str(output_dir),
            "--analytics-db",
            str(tmp_path / "deterministic-analytics.sqlite3"),
            "--timeout",
            "30s",
            "--yes",
            "--analysis-provider",
            "codex",
        ]
    )
    hooks = PipelineHooks(
        progress_handler=lambda stage, message, _payload: progress.append(
            (stage, message)
        )
    )

    run = run_pipeline(args, hooks=hooks)

    assert provider_calls == []
    assert {stage for stage, _message in progress} >= {
        "evidence_validation",
        "rendering",
        "final_qa",
        "complete",
    }
    assert not any("Gemma 4 12B is writing" in message for _, message in progress)
    assert not (run / "ollama-response.json").exists()

    response_metadata = json.loads(
        (run / "ollama-response-envelope.json").read_text(encoding="utf-8")
    )
    assert response_metadata["provider"] == "deterministic"
    assert response_metadata["runtime"] == "local"
    assert response_metadata["model"] is None
    assert response_metadata["response_envelope_type"] == "deterministic-local-patches"
    assert response_metadata["output_format"] == "deterministic-json"
    assert response_metadata["execution"]["execution_mode"] == "deterministic_only"
    assert response_metadata["execution"]["ollama_invoked"] is False
    assert response_metadata.get("response") is None

    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["writer"] == {
        "provider": "deterministic",
        "name": "Deterministic local compiler",
        "model": None,
        "document_format": "headless",
        "runtime": "local",
        "ollama_invoked": False,
    }
    initial = metadata["revision_cycle"]["initial"]
    assert initial["provider"] == "deterministic"
    assert initial["runtime"] == "local"
    assert initial["model"] is None
    assert initial["ollama_invoked"] is False
    assert initial["execution_mode"] == "deterministic-local"
    assert initial["execution"] == response_metadata["execution"]
    assert "response" not in initial
    tailored_reference = initial["tailored_content"]
    assert tailored_reference["filename"] == "tailored-content.initial.json"
    assert sha256_file(run / tailored_reference["filename"]) == tailored_reference[
        "sha256"
    ]
    tailored = json.loads(
        (run / tailored_reference["filename"]).read_text(encoding="utf-8")
    )
    assert tailored["skill_groups"][2]["text"] == (
        "FastAPI, JSON Schema, pytest, SQL"
    )


def test_default_gemma_artifact_preflight_failure_is_provider_specific(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_artifact_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise InputError("Synthetic authenticated artifact mismatch.")

    monkeypatch.setattr(
        cli_module,
        "_tailoring_dependency_versions",
        lambda *args, **kwargs: {
            "ollama": "synthetic",
            "ollama_model": "resume-tailor-gemma",
            "libreoffice": "synthetic",
        },
    )
    monkeypatch.setattr(
        cli_module,
        "verify_tailoring_run_artifacts",
        fail_artifact_preflight,
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_ollama",
        lambda **kwargs: pytest.fail("Local Ollama ran after a failed local preflight"),
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            str(master_resume),
            "--job-file",
            str(job_file),
            "--company",
            "Example Talent",
            "--role",
            "Agentic AI Developer",
            "--output-dir",
            str(tmp_path / "output"),
            "--timeout",
            "30s",
            "--yes",
            "--analysis-provider",
            "codex",
        ]
    )

    with pytest.raises(TailoringPreflightError):
        run_pipeline(args)

    run = next((tmp_path / "output").iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "ollama-tailoring-preflight"
    assert metadata["failure_class"] == "ollama-tailoring-preflight"
    assert metadata["error"]["type"] == "TailoringPreflightError"
    assert not (run / "ollama-response.json").exists()


def test_failure_preserves_useful_artifacts(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_MODE", "questions")
    output_dir = tmp_path / "output"
    code = main(_arguments(master_resume, job_file, output_dir))
    assert code == ExitCode.WAITING
    run = next(output_dir.iterdir())
    assert (run / "job-description.txt").is_file()
    assert (run / "extracted-master-resume.json").is_file()
    assert (run / "codex-analysis.json").is_file()
    assert (run / "analysis-resolved.json").is_file()
    assert (run / "codex-analysis-resolved.json").is_file()
    resolved_doc = json.loads(
        (run / "analysis-resolved.json").read_text(encoding="utf-8")
    )
    assert resolved_doc["provider"] == "codex"
    assert "analysis" in resolved_doc
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "FAILED"
    assert metadata["source_resume"]["unchanged"] is True


def test_analytics_failure_warns_but_does_not_corrupt_resume_pipeline(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stubs_on_path
    monkeypatch.setenv("STUB_CODEX_MODE", "questions")
    monkeypatch.setattr(
        cli_module.AnalyticsStore,
        "record_job_viewed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic analytics outage with private detail")
        ),
    )
    output_dir = tmp_path / "analytics-warning-output"

    code = main(_arguments(master_resume, job_file, output_dir))

    assert code == ExitCode.WAITING
    run = next(output_dir.iterdir())
    assert (run / "analysis-resolved.json").is_file()
    assert (run / "codex-analysis-resolved.json").is_file()
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["analytics"]["warnings"] == [
        {
            "operation": "the validated local viewed posting",
            "error_type": "RuntimeError",
            "retryable_from_preserved_artifacts": True,
        }
    ]
    assert "private detail" not in json.dumps(metadata["analytics"])


@pytest.mark.parametrize("source_mode", ["job-file", "clipboard"])
def test_local_text_modes_never_enable_apify_retrieval(
    source_mode: str,
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation_log = tmp_path / f"{source_mode}-codex-invocations.jsonl"
    monkeypatch.setenv("STUB_CODEX_INVOCATION_LOG", str(invocation_log))
    monkeypatch.setenv("STUB_CODEX_MODE", "questions")
    monkeypatch.setattr(
        cli_module,
        "invoke_apify_linkedin_retrieval",
        lambda **_: pytest.fail("Local text input enabled Apify retrieval"),
    )
    output_dir = tmp_path / f"{source_mode}-output"
    if source_mode == "job-file":
        arguments = _arguments(master_resume, job_file, output_dir)
    else:
        monkeypatch.setattr(
            cli_module,
            "read_clipboard",
            lambda: (
                "Synthetic complete job description supplied by the local "
                "clipboard for a Python role with testing and documentation.",
                "clipboard-stub",
            ),
        )
        arguments = [
            "--resume",
            str(master_resume),
            "--clipboard",
            "--company",
            "Example Talent",
            "--role",
            "Agentic AI Developer",
            "--output-dir",
            str(output_dir),
            "--timeout",
            "30s",
            "--yes",
            "--analysis-provider",
            "codex",
        ]

    code = main(arguments)

    assert code == ExitCode.WAITING
    roles = [
        json.loads(line)["role"]
        for line in invocation_log.read_text(encoding="utf-8").splitlines()
    ]
    assert roles == ["resume_analysis"]
    run = next(output_dir.iterdir())
    assert not (run / "job-source.json").exists()
    assert not (run / "apify-linkedin-retrieval-diagnostic.json").exists()
    summary = _analytics_store(tmp_path).summary()
    assert summary["totals"]["unique_jobs_viewed"] == 1
    assert summary["recently_viewed_jobs"][0]["current_status"] == "viewed"
    assert _analytics_store(tmp_path).sanitized_export()["jobs"][0]["source"] == (
        "file" if source_mode == "job-file" else "clipboard"
    )


def test_unstructured_print_wrapper_is_classified_without_prose_extraction(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_AGY_MODE", "unstructured_print_wrapper")
    output_dir = tmp_path / "output"

    code = main(_arguments(master_resume, job_file, output_dir))

    assert code == ExitCode.MODEL
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["failure_class"] == "antigravity-response-envelope"
    assert metadata["error"]["type"] == "AntigravityResponseEnvelopeError"
    response_metadata = metadata["antigravity_response"]
    assert response_metadata["response_envelope_type"] == (
        "json-wrapper-unstructured-response"
    )
    assert response_metadata["validation_result"] == "REJECTED"
    assert not (run / "tailored-content.json").exists()
    assert not list(run.glob("*.docx"))


def test_human_analysis_refusal_stops_before_antigravity(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "refuse")
    output_dir = tmp_path / "output"
    code = main(_arguments(master_resume, job_file, output_dir, yes=False))
    assert code == ExitCode.APPROVAL
    run = next(output_dir.iterdir())
    assert (run / "codex-analysis.json").is_file()
    assert not (run / "antigravity-response.json").exists()
    assert not (run / "codex-analysis-approval.json").exists()


@pytest.mark.parametrize("failure_mode", ["source_id_failure", "empty_source_ids"])
def test_source_evidence_failure_precedes_approval_and_antigravity(
    failure_mode: str,
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_MODE", failure_mode)
    monkeypatch.setattr(
        "builtins.input",
        lambda _: pytest.fail("approval was requested before source validation"),
    )
    output_dir = tmp_path / "source-evidence-output"

    code = main(_arguments(master_resume, job_file, output_dir, yes=False))

    assert code == ExitCode.TRUTHFULNESS
    run = next(output_dir.iterdir())
    assert (run / "codex-analysis.json").is_file()
    assert not (run / "codex-analysis-resolved.json").exists()
    assert not (run / "codex-analysis-approval.json").exists()
    assert not (run / "antigravity-response.json").exists()
    assert not list(run.glob("*.docx"))
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "codex-analysis"
    assert metadata["failure_class"] == "source-evidence-analysis"
    assert metadata["error"]["type"] == "SourceEvidenceError"


def test_user_rejecting_linkedin_confirmation_stops_before_resume_analysis(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def reject(prompt: str) -> str:
        prompts.append(prompt)
        return "reject"

    _stub_apify_retrieval(monkeypatch)
    monkeypatch.setattr("builtins.input", reject)
    output_dir = tmp_path / "url-output"
    code = main(_url_arguments(master_resume, output_dir))
    assert code == ExitCode.APPROVAL
    run = next(output_dir.iterdir())
    assert run.name.startswith("apify-linkedin-retrieval-")
    assert (run / "job-source.json").is_file()
    assert (run / "apify-linkedin-retrieval-diagnostic.json").is_file()
    assert (run / "job-description.txt").is_file()
    assert not (run / "codex-analysis.json").exists()
    assert not (run / "extracted-master-resume.json").exists()
    assert not (run / "antigravity-response.json").exists()
    assert not list(run.glob("*.docx"))
    assert prompts == ['LinkedIn posting: type "approve" to continue: ']
    summary = _analytics_store(tmp_path).summary()
    assert summary["totals"]["unique_jobs_viewed"] == 1
    assert summary["totals"]["jobs_approved_for_tailoring"] == 0


def test_6318_character_canonical_posting_reaches_review_gate(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    description = _description_with_length(6_318)
    _stub_apify_retrieval(monkeypatch, description=description)
    monkeypatch.setattr("builtins.input", lambda _: "reject")
    output_dir = tmp_path / "url-output"

    code = main(_url_arguments(master_resume, output_dir))

    assert code == ExitCode.APPROVAL
    run = next(output_dir.iterdir())
    preserved = (run / "job-description.txt").read_text(encoding="utf-8").rstrip()
    assert len(preserved) == 6_318
    assert not (run / "extracted-master-resume.json").exists()
    assert not (run / "codex-analysis.json").exists()
    assert _analytics_store(tmp_path).summary()["totals"]["unique_jobs_viewed"] == 1


def test_apify_is_the_only_step_2_retrieval_provider(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _stub_apify_retrieval(monkeypatch, calls=calls)
    monkeypatch.setattr(
        cli_module,
        "invoke_analysis",
        lambda **_: pytest.fail("Analysis ran before posting approval"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_codex_analysis",
        lambda **_: pytest.fail("Codex analysis ran before posting approval"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_antigravity",
        lambda **_: pytest.fail("Antigravity was invoked during Step 2"),
    )
    monkeypatch.setattr(
        cli_module,
        "_tailoring_dependency_versions",
        lambda *_args, **_kwargs: pytest.fail(
            "Writer dependencies were invoked during Step 2"
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {
            "resume_tailor": "0.1.0",
            "codex": "stub",
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "reject")
    output_dir = tmp_path / "url-output"

    code = main(_url_arguments(master_resume, output_dir))

    assert code == ExitCode.APPROVAL
    assert calls == ["apify_retrieval"]
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["apify_linkedin_retrieval"] == {
        "provider": "apify",
        "interface": "Apify API v2",
        "actor_configuration": "APIFY_ACTOR_ID",
        "actor_input_format": "searchUrls",
        "authentication_transport": "bearer-header",
        "retrieval_only": True,
        "automatic_fallback": False,
    }
    assert not (run / "codex-analysis.json").exists()
    assert not (run / "antigravity-response.json").exists()


def test_malformed_apify_result_is_stage_specific_and_stops_before_analysis(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stubs_on_path

    def fail(**kwargs: object) -> dict[str, object]:
        run_directory = kwargs["run_directory"]
        assert isinstance(run_directory, Path)
        (run_directory / "apify-linkedin-retrieval-diagnostic.json").write_text(
            json.dumps(
                {
                    "provider": "apify",
                    "classification": "malformed_output",
                    "provider_output_omitted": True,
                    "api_token_omitted": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise ApifyLinkedInRetrievalError("malformed_output")

    monkeypatch.setattr(cli_module, "invoke_apify_linkedin_retrieval", fail)
    monkeypatch.setattr(
        cli_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {
            "resume_tailor": "0.1.0",
            "codex": "stub",
        },
    )
    output_dir = tmp_path / "url-output"

    code = main(_url_arguments(master_resume, output_dir))

    assert code == ExitCode.MODEL
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "apify-linkedin-retrieval"
    assert metadata["failure_class"] == "apify-linkedin-retrieval"
    assert metadata["retrieval_classification"] == "malformed_output"
    assert metadata["error"]["type"] == "ApifyLinkedInRetrievalError"
    assert (run / "apify-linkedin-retrieval-diagnostic.json").is_file()
    assert not (run / "codex-analysis.json").exists()
    assert not (run / "extracted-master-resume.json").exists()
    assert not (run / "antigravity-response.json").exists()
    assert not list(run.glob("*.docx"))
    assert _analytics_store(tmp_path).summary()["totals"]["unique_jobs_viewed"] == 0


def test_url_pipeline_continues_after_explicit_confirmation(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    invocation_log = tmp_path / "codex-invocations.jsonl"
    antigravity_log = tmp_path / "antigravity-transport.json"
    monkeypatch.setenv("STUB_CODEX_INVOCATION_LOG", str(invocation_log))
    monkeypatch.setenv("STUB_AGY_TRANSPORT_LOG", str(antigravity_log))
    _stub_apify_retrieval(monkeypatch)

    def approve(prompt: str) -> str:
        prompts.append(prompt)
        if "Initial QA provider" in prompt or "provider number" in prompt:
            return "codex"
        if "complete" in prompt.casefold() and "revise" in prompt.casefold():
            return "complete"
        if "Optional one-shot revision" in prompt:
            return "complete"
        return "approve"

    monkeypatch.setattr("builtins.input", approve)
    output_dir = tmp_path / "url-output"
    code = main(_url_arguments(master_resume, output_dir))
    assert code == ExitCode.OK
    run = next(output_dir.iterdir())
    assert run.name.startswith("example-ai-systems-machine-learning-engineer-")
    basename = "Sample-Candidate-Example-AI-Systems-Machine-Learning-Engineer"
    assert (run / "job-source.json").is_file()
    assert (run / "job-description.txt").is_file()
    assert (run / f"{basename}.docx").is_file()
    assert (run / f"{basename}.pdf").is_file()
    assert (run / "preview.png").is_file()
    assert (run / "final-qa.md").is_file()
    codex_roles = [
        json.loads(line)["role"]
        for line in invocation_log.read_text(encoding="utf-8").splitlines()
    ]
    assert codex_roles == ["resume_analysis", "final_qa"]
    assert json.loads(antigravity_log.read_text(encoding="utf-8"))["role"] == (
        "tailored_resume_writer"
    )
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["initial_qa_provider"] == "codex"
    assert metadata["company"] == "Example AI Systems"
    assert metadata["role"] == "Machine Learning Engineer"
    assert metadata["source_resume"]["unchanged"] is True
    analytics = _analytics_store(tmp_path)
    assert analytics.summary()["totals"]["jobs_approved_for_tailoring"] == 1
    assert analytics.summary()["totals"]["applications_submitted"] == 0
    assert len(analytics.sanitized_export()["resume_versions"]) == 1
    assert prompts[0] == 'LinkedIn posting: type "approve" to continue: '
    assert prompts[1] == 'Codex analysis: type "approve" to continue: '
    assert prompts[2] == 'Tailored content diff: type "approve" to continue: '
    assert any(
        "provider number or id" in prompt or "Initial QA provider" in prompt
        for prompt in prompts
    )


def test_initial_generation_metadata_preserves_sanitized_budget_repair(
    tmp_path: Path,
) -> None:
    tailored_path = tmp_path / "tailored-content.initial.json"
    tailored_path.write_text('{"synthetic":true}', encoding="utf-8")
    repair = {
        "attempted": True,
        "maximum_attempts": 1,
        "attempt_count": 1,
        "outcome": "PASS",
        "validation_path": "pass",
        "violations": [
            {
                "edit_id": "edit.001",
                "target_source_id": "professional_summary",
                "actual_characters": 194,
                "maximum_characters": 193,
            }
        ],
    }

    initial, deterministic_only = cli_module._initial_generation_metadata(
        response_metadata={
            "provider": "gemma",
            "runtime": "ollama",
            "model": "resume-tailor-gemma",
            "ollama_invoked": True,
            "response_envelope_type": "ollama-chat-message-content-json",
            "output_format": "json-schema",
            "budget_repair": repair,
        },
        writer_provider="ollama",
        ollama_model="resume-tailor-gemma",
        tailored_content_path=tailored_path,
    )

    assert deterministic_only is False
    assert initial["budget_repair"] == repair
    assert initial["tailored_content"]["sha256"] == sha256_file(tailored_path)
