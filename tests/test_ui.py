from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

import resume_tailor.ui.cli as cli_module
from resume_tailor.backend.utils.analytics import observation_from_local_job
from resume_tailor.ui.cli import run_pipeline
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.jobs.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
from resume_tailor.backend.engine.orchestration import ApprovalResponse, PipelineHooks
from resume_tailor.backend.engine.retry import analysis_input_manifest, antigravity_retry_failure_kind
from resume_tailor.backend.utils.schemas import load_schema
import resume_tailor.ui.ui as ui
from resume_tailor.ui.ui import COOKIE_NAME, DEFAULT_HOST, create_app
from resume_tailor.ui.ui import _failure_kind_from_metadata
from resume_tailor.ui.ui_cli import build_parser as build_ui_parser
from resume_tailor.backend.utils.utilities import (
    ApifyConfigurationError,
    ApifyLinkedInRetrievalError,
    CancellationError,
    CodexSchemaCompatibilityError,
    AntigravityLaunchSizeError,
    ExitCode,
    InputError,
    OllamaBudgetError,
    OllamaConnectionError,
    atomic_write_json,
    sha256_file,
)


JOB_URL = "https://www.linkedin.com/jobs/view/platform-engineer-4123456789/"


def _description_with_length(length: int) -> str:
    fragment = "Build safe Python services and automated tests. "
    value = (fragment * ((length // len(fragment)) + 1))[:length]
    if value[-1].isspace():
        value = value[:-1] + "x"
    assert len(value) == length
    return value


def _posting(*, injection: bool = False) -> dict[str, Any]:
    hostile = "<script>alert('page')</script>" if injection else ""
    return {
        "fetch_status": "success",
        "requested_url": JOB_URL,
        "final_resolved_url": JOB_URL,
        "linkedin_job_id": "4123456789",
        "job_title": f"LLM Platform Engineer {hostile}".strip(),
        "company": f"Northwind AI {hostile}".strip(),
        "location": "Chicago, Illinois",
        "workplace_type": "hybrid",
        "employment_type": "Full-time",
        "salary": None,
        "seniority_level": None,
        "date_posted": None,
        "applicant_count": None,
        "retrieval_source": "apify",
        "normalized_job_description": (
            "Build safe Python AI orchestration systems and evaluate language "
            "model behavior with a collaborative engineering team. "
            + hostile
        ),
        "responsibilities": ["Develop maintainable Python services.", hostile or "Test AI systems."],
        "required_qualifications": ["Python software engineering"],
        "preferred_qualifications": ["Containers"],
        "technologies_and_skills": ["Python", hostile or "Docker"],
        "ai_focus_areas": ["LLM evaluation"],
        "warnings": [hostile] if injection else [],
    }


def _analysis() -> dict[str, Any]:
    return {
        "role_summary": "A Python and AI orchestration engineering role.",
        "fit_assessment": {
            "overall": "Strong evidence-backed fit.",
            "strengths": ["Python project evidence"],
            "gaps": ["No unsupported production-scale claim"],
        },
        "matched_requirements": ["Python"],
        "evidence_map": [],
        "requirement_assessment": [
            {
                "requirement_id": "skill.001",
                "requirement": "Python",
                "category": "technology_and_skill",
                "status": "present_verbatim",
                "support_provenance": "local_exact_phrase",
                "strength": "strong",
                "evidence_source_ids": ["skill_groups.0"],
                "resolved_evidence": [
                    {
                        "source_id": "skill_groups.0",
                        "section_context": "TECHNICAL SKILLS",
                        "exact_text": "Languages: Python, JavaScript, SQL",
                    }
                ],
            }
        ],
        "ats_keywords": [
            {"keyword": "Python", "evidence_source_ids": ["skill_groups.0"]}
        ],
        "ats_keyword_assessment": [
            {
                "keyword": "Python",
                "status": "present_verbatim",
                "evidence_source_ids": ["skill_groups.0"],
                "resolved_evidence": [
                    {
                        "source_id": "skill_groups.0",
                        "section_context": "TECHNICAL SKILLS",
                        "exact_text": "Languages: Python, JavaScript, SQL",
                    }
                ],
            }
        ],
        "supported_ats_keywords": ["Python"],
        "unsupported_ats_keywords": [],
        "missing_or_unsupported_requirements": ["RAG is not supported"],
        "recommended_edits": [
            {
                "target_source_id": "professional_summary",
                "operation": "replace",
                "proposed_text": "Evidence-backed summary",
                "evidence_source_ids": ["skill_groups.0"],
                "resume_section": "Professional Summary",
                "existing_text": "Existing summary",
                "alignment_rationale": "Surfaces supported Python work.",
                "resolved_evidence": [
                    {
                        "source_id": "skill_groups.0",
                        "section_context": "TECHNICAL SKILLS",
                        "exact_text": "Languages: Python, JavaScript, SQL",
                    }
                ],
            }
        ],
        "immutable_facts": ["Expected Dec 2026"],
        "forbidden_claims": ["RAG", "GraphQL"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


class StubbedUIPipeline:
    """In-process UI orchestration stub: no model, network, or renderer calls."""

    def __init__(
        self,
        *,
        fetch_failure: str | None = None,
        injection: bool = False,
        block_initial: bool = False,
    ) -> None:
        self.fetch_failure = fetch_failure
        self.injection = injection
        self.block_initial = block_initial
        self.calls: list[str] = []
        self.namespaces: list[argparse.Namespace] = []
        self.job_texts: list[str] = []

    def __call__(self, args: argparse.Namespace, *, hooks: PipelineHooks) -> Path:
        self.namespaces.append(args)
        if args.job_file is not None:
            self.job_texts.append(args.job_file.read_text(encoding="utf-8"))
        run_directory = args.output_dir / f"stub-ui-{uuid.uuid4().hex}"
        run_directory.mkdir(mode=0o700)
        self.calls.append("validating_input")
        hooks.progress(
            "validating_input",
            "Stub validated the local input.",
            run_directory=str(run_directory),
        )
        if self.block_initial:
            self.calls.append("blocked")
            while not hooks.cancel_event.wait(0.02):
                pass
            raise CancellationError("Stub run cancelled.")

        company = args.company
        role = args.role
        if args.job_url is not None:
            self.calls.append("fetching_job")
            hooks.progress("fetching_job", "Stub is retrieving the posting.")
            if self.fetch_failure is not None:
                raise InputError(self.fetch_failure)
            posting = _posting(injection=self.injection)
            (run_directory / "job-source.json").write_text(
                json.dumps(posting),
                encoding="utf-8",
            )
            (run_directory / "job-description.txt").write_text(
                posting["normalized_job_description"],
                encoding="utf-8",
            )
            hooks.progress("confirming_posting", "Stub posting needs confirmation.")
            self.calls.append("linkedin_approval")
            response = hooks.approve(
                kind="linkedin_posting",
                title="LinkedIn posting",
                payload=posting,
                assume_yes=False,
            )
            if response.action == "use_pasted":
                (run_directory / "job-description.txt").write_text(
                    str(response.data["job_description"]),
                    encoding="utf-8",
                )
            company = posting["company"]
            role = posting["job_title"]

        self.calls.append("codex_analysis")
        hooks.progress("codex_analysis", "Stub Codex analysis is complete.")
        hooks.progress("reviewing_changes", "Stub analysis needs approval.")
        hooks.approve(
            kind="codex_analysis",
            title="Codex analysis",
            payload=_analysis(),
            assume_yes=False,
        )
        self.calls.append("antigravity_tailoring")
        hooks.progress("antigravity_tailoring", "Stub tailoring is complete.")
        hooks.progress("evidence_validation", "Stub evidence checks passed.")
        hooks.approve(
            kind="tailored_content",
            title="Tailored content diff",
            payload={
                "content_diff": "# Diff\n\nNo unsupported claims.",
                "tailored_content": {"professional_summary": "Safe"},
                "evidence": {
                    "passed": True,
                    "issues": [],
                    "introduced_technologies": [],
                    "introduced_metrics": [],
                    "introduced_role_labels": [],
                    "introduced_availability": [],
                },
            },
            assume_yes=False,
        )
        self.calls.append("rendering")
        hooks.progress("rendering", "Stub produced local DOCX and PDF files.")
        basename = "Sample-Candidate-Northwind-AI-LLM-Platform-Engineer"
        (run_directory / f"{basename}.docx").write_bytes(b"stub-docx")
        (run_directory / f"{basename}.pdf").write_bytes(b"%PDF-1.4 stub")
        (run_directory / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
        (run_directory / "codex-analysis.json").write_text(
            json.dumps(_analysis()),
            encoding="utf-8",
        )
        (run_directory / "tailored-content.json").write_text(
            json.dumps({"professional_summary": "Safe"}),
            encoding="utf-8",
        )
        (run_directory / "content-diff.md").write_text(
            "# Diff\n\nNo unsupported claims.",
            encoding="utf-8",
        )
        (run_directory / "final-qa.md").write_text(
            "# Final Codex QA\n\nPASS",
            encoding="utf-8",
        )
        hooks.progress("final_qa", "Stub final QA passed.")
        metadata = {
            "application": "resume-tailor",
            "status": "COMPLETE",
            "stage": "complete",
            "created_at": "2026-07-29T12:00:00+00:00",
            "company": company,
            "role": role,
            "factual_integrity": {"status": "PASS", "issues": []},
            "layout_validation": {"status": "PASS", "pages": 1},
            "final_qa": {
                "status": "PASS",
                "summary": "Stubbed read-only QA passed.",
                "material_issues": [],
                "improvement_assessment": "Alignment improved safely.",
            },
        }
        (run_directory / "run-metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        hooks.progress(
            "complete",
            "Stub run completed.",
            run_directory=str(run_directory),
            company=str(company),
            role=str(role),
        )
        return run_directory


@asynccontextmanager
async def _client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        yield client
    app.state.manager.shutdown()


async def _session(client: httpx.AsyncClient) -> str:
    response = await client.get("/")
    assert response.status_code == 200
    token = client.cookies.get(COOKIE_NAME)
    assert token
    return token


async def _start(
    client: httpx.AsyncClient,
    token: str,
    *,
    mode: str = "url",
    files: dict[str, tuple[str, bytes, str]] | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    data = {
        "csrf_token": token,
        "resume_mode": "master",
        "job_mode": mode,
        "job_url": JOB_URL if mode == "url" else "",
        "company": "Example Company" if mode != "url" else "",
        "role": "AI Engineer" if mode != "url" else "",
        "pasted_description": (
            "Build safe Python language-model systems and automated tests."
            if mode == "pasted"
            else ""
        ),
        # UI pipeline fixtures still exercise the Codex CLI stub unless a test
        # explicitly selects another analysis provider.
        "analysis_provider": "codex",
    }
    if extra:
        data.update(extra)
    response = await client.post("/runs", data=data, files=files)
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


async def _wait(app: Any, run_id: str, status: str, kind: str | None = None) -> dict[str, Any]:
    return app.state.manager.wait_for(
        run_id,
        lambda item: item["status"] == status
        and (kind is None or (item["approval"] or {}).get("kind") == kind),
        timeout=5,
    )


async def _approve(
    client: httpx.AsyncClient,
    token: str,
    run_id: str,
    action: str = "approve",
    fallback: str = "",
) -> httpx.Response:
    return await client.post(
        f"/runs/{run_id}/approval",
        data={
            "csrf_token": token,
            "action": action,
            "fallback_description": fallback,
        },
    )


def _write_synthetic_source_failure(
    output_directory: Path,
    synthetic_resume: Path,
    *,
    name: str = "synthetic-source-evidence-failure",
) -> Path:
    run = output_directory / name
    run.mkdir(parents=True, mode=0o700)
    job_description = "Build a synthetic Python evidence-validation service.\n"
    (run / "job-description.txt").write_text(job_description, encoding="utf-8")
    atomic_write_json(
        run / "job-requirements.json",
        build_job_requirement_catalog(job_description),
    )
    extracted, _ = extract_resume(synthetic_resume)
    atomic_write_json(run / "extracted-master-resume.json", extracted)
    source_hash = sha256_file(synthetic_resume)
    metadata = {
        "application": "resume-tailor",
        "application_version": "0.1.0",
        "status": "FAILED",
        "stage": "codex-analysis",
        "failure_class": "source-evidence-analysis",
        "created_at": "2026-07-30T12:00:00+00:00",
        "company": "Synthetic Systems",
        "role": "Evidence Engineer",
        "job_source": "linkedin-url",
        "source_resume": {
            "filename": synthetic_resume.name,
            "sha256_before": source_hash,
            "sha256_after": source_hash,
            "unchanged": True,
        },
        "analysis_inputs": analysis_input_manifest(
            run,
            source_resume_sha256=source_hash,
        ),
        "analysis": {
            "provider": "codex",
            "name": "Codex",
            "automatic_fallback": False,
        },
        "error": {
            "type": "SourceEvidenceError",
            "message": (
                "Codex analysis failed local source-evidence validation:\n"
                "- PRIVATE SYNTHETIC SOURCE QUOTATION MUST NOT RENDER"
            ),
            "exit_code": 13,
        },
        "artifacts": [],
    }
    atomic_write_json(run / "run-metadata.json", metadata)
    return run


def _write_synthetic_antigravity_launch_failure(
    output_directory: Path,
    synthetic_resume: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    job = output_directory.parent / "synthetic-antigravity-job.txt"
    job.write_text(
        "Build synthetic Python evidence validation workflows.\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        resume=synthetic_resume,
        clipboard=False,
        job_file=job,
        job_url=None,
        company="Synthetic Systems",
        role="Evidence Engineer",
        output_dir=output_directory,
        yes=False,
        keep_workdir=False,
        timeout=(30, "30s"),
        analysis_provider="codex",
        writer_provider="antigravity",
        ollama_model="resume-tailor-gemma",
    )
    original = cli_module.invoke_antigravity

    def fail_before_provider(**_kwargs: Any) -> dict[str, Any]:
        raise AntigravityLaunchSizeError(
            "Antigravity could not start because the request exceeded the "
            "operating system's command-line size."
        )

    monkeypatch.setattr(cli_module, "invoke_antigravity", fail_before_provider)
    try:
        with pytest.raises(AntigravityLaunchSizeError):
            run_pipeline(
                args,
                hooks=PipelineHooks(
                    approval_handler=lambda request: (
                        ApprovalResponse("approve")
                        if request.kind == "codex_analysis"
                        else pytest.fail(
                            "Initial synthetic failure passed the analysis gate"
                        )
                    )
                ),
            )
    finally:
        monkeypatch.setattr(cli_module, "invoke_antigravity", original)
    run = next(
        child
        for child in output_directory.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )
    assert (run / "codex-analysis-approval.json").is_file()
    return run


def _write_synthetic_antigravity_waiting_failure(
    output_directory: Path,
    synthetic_resume: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    run = _write_synthetic_antigravity_launch_failure(
        output_directory,
        synthetic_resume,
        monkeypatch,
    )
    atomic_write_json(
        run / "antigravity-response.json",
        {
            "status": "SUCCESS",
            "structured_output": {
                "status": "WAITING",
                "message": (
                    "Plan mode activated. Ready after synthetic requirements "
                    "are provided."
                ),
                "questions_for_user": [
                    "What synthetic task would you like to plan?"
                ],
                "tailored_resume": None,
            },
        },
    )
    metadata_path = run / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("failure_class", None)
    metadata["error"] = {
        "type": "WaitingError",
        "message": (
            "Antigravity needs more information; review the synthetic questions."
        ),
        "exit_code": 3,
    }
    atomic_write_json(metadata_path, metadata)
    return run


def _write_synthetic_antigravity_envelope_failure(
    output_directory: Path,
    synthetic_resume: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    valid_response: bool,
) -> Path:
    run = _write_synthetic_antigravity_launch_failure(
        output_directory,
        synthetic_resume,
        monkeypatch,
    )
    extracted = json.loads(
        (run / "extracted-master-resume.json").read_text(encoding="utf-8")
    )
    complete = {
        "status": "complete",
        "message": "Applied the approved synthetic edit plan.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": extracted["content"],
    }
    atomic_write_json(
        run / "antigravity-response.json",
        {
            "conversation_id": "00000000-0000-4000-8000-000000000000",
            "duration_seconds": 1.25,
            "json_schema": load_schema("tailored_resume.schema.json"),
            "num_turns": 1,
            "response": (
                json.dumps(complete, ensure_ascii=False)
                if valid_response
                else "Synthetic prose with {\"status\":\"complete\"}."
            ),
            "status": "SUCCESS",
            "usage": {
                "cache_read_tokens": 10,
                "input_tokens": 20,
                "output_tokens": 5,
                "thinking_tokens": 2,
                "total_tokens": 25,
            },
        },
    )
    metadata_path = run / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["failure_class"] = "antigravity-response-envelope"
    metadata["error"] = {
        "type": "AntigravityResponseEnvelopeError",
        "message": "Antigravity returned JSON in an unsupported response format.",
        "exit_code": 10,
    }
    metadata["tools"]["antigravity"] = "1.1.8-stub"
    metadata["artifacts"] = sorted(
        path.name for path in run.iterdir() if path.is_file()
    )
    atomic_write_json(metadata_path, metadata)
    return run


def test_ui_startup_health_and_localhost_binding(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(),
    )
    assert app.state.settings.host == DEFAULT_HOST == "127.0.0.1"

    async def scenario() -> None:
        async with _client(app) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["bind_host"] == "127.0.0.1"
            dashboard = await client.get("/")
            assert "New Tailoring Run" in dashboard.text
            assert "default-src &#x27;self&#x27;" not in dashboard.text
            assert dashboard.headers["content-security-policy"].startswith(
                "default-src 'self'"
            )

    asyncio.run(scenario())


def test_ui_defaults_to_the_shared_cli_pipeline(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
    )
    assert app.state.manager.pipeline_runner is run_pipeline


def test_workflow_sidebar_shows_recorded_analysis_provider(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    """Regression: analysis stage label must follow metadata.analysis.provider."""
    from resume_tailor.backend.engine.analysis import workflow_stages_for_provider

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    for provider, expected_label in (
        ("gemma_local", "Gemma Local analysis"),
        ("codex", "Codex analysis"),
        ("grok_cli", "Grok CLI analysis"),
        ("grok", "Grok CLI analysis"),
    ):
        run = output_directory / f"run-{provider}"
        run.mkdir()
        atomic_write_json(
            run / "run-metadata.json",
            {
                "application": "resume-tailor",
                "status": "FAILED",
                "stage": "codex-analysis",
                "company": "Example",
                "role": "Developer",
                "analysis": {
                    "provider": provider,
                    "name": provider,
                    "automatic_fallback": False,
                },
                "error": {
                    "type": "ModelError",
                    "message": "synthetic stop",
                    "exit_code": 10,
                },
            },
        )
        app = create_app(
            output_directory=output_directory,
            master_resume=master_resume,
            pipeline_runner=StubbedUIPipeline(),
        )
        snapshot = app.state.manager.snapshot(f"history-{run.name}")
        stages = dict(snapshot["workflow_stages"])
        assert stages["codex_analysis"] == expected_label
        assert dict(workflow_stages_for_provider(provider))[
            "codex_analysis"
        ] == expected_label

        async def scenario() -> None:
            async with _client(app) as client:
                await _session(client)
                page = await client.get(f"/runs/history-{run.name}")
                assert page.status_code == 200
                assert expected_label in page.text

        asyncio.run(scenario())


def test_ui_cli_does_not_offer_remote_binding() -> None:
    parser = build_ui_parser()
    args = parser.parse_args(["--no-browser"])
    assert args.port == 8765
    assert args.analytics_db.name == "job-search-analytics.sqlite3"
    with pytest.raises(SystemExit):
        parser.parse_args(["--host", "0.0.0.0"])


@pytest.mark.parametrize(
    "url",
    [
        "http://www.linkedin.com/jobs/view/4123456789/",
        "https://example.com/jobs/view/4123456789/",
        "not-a-url",
    ],
)
def test_url_mode_form_validation(
    url: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(),
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            response = await client.post(
                "/runs",
                data={
                    "csrf_token": token,
                    "resume_mode": "master",
                    "job_mode": "url",
                    "job_url": url,
                },
            )
            assert response.status_code == 422
            assert "Check the run details" in response.text

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["pasted", "file"])
def test_file_and_clipboard_text_input_modes(
    mode: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            files = (
                {
                    "job_file": (
                        "job.txt",
                        b"Build reliable Python AI evaluation systems.",
                        "text/plain",
                    )
                }
                if mode == "file"
                else None
            )
            run_id = await _start(client, token, mode=mode, files=files)
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            assert pipeline.namespaces[0].job_url is None
            assert pipeline.namespaces[0].writer_provider == "ollama"
            assert pipeline.namespaces[0].ollama_model == "resume-tailor-gemma"
            assert pipeline.namespaces[0].job_source_override == mode
            assert pipeline.namespaces[0].analytics_db == (
                app.state.settings.analytics_database
            )
            assert pipeline.job_texts
            assert "Python" in pipeline.job_texts[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["pasted", "file"])
def test_6318_character_local_text_modes_reach_the_pipeline(
    mode: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )
    description = _description_with_length(6_318)

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            files = (
                {"job_file": ("job.txt", description.encode(), "text/plain")}
                if mode == "file"
                else None
            )
            run_id = await _start(
                client,
                token,
                mode=mode,
                files=files,
                extra={"pasted_description": description},
            )
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            assert len(pipeline.job_texts[0].strip()) == 6_318

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["pasted", "file"])
def test_over_limit_local_text_modes_report_actual_and_permitted(
    mode: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )
    actual = MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS + 1
    description = _description_with_length(actual)

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            files = (
                {"job_file": ("job.txt", description.encode(), "text/plain")}
                if mode == "file"
                else None
            )
            response = await client.post(
                "/runs",
                data={
                    "csrf_token": token,
                    "resume_mode": "master",
                    "job_mode": mode,
                    "company": "Example Company",
                    "role": "AI Engineer",
                    "pasted_description": description if mode == "pasted" else "",
                },
                files=files,
            )
            assert response.status_code == 422
            assert f"{actual:,}" in response.text
            assert f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,}" in response.text
            assert pipeline.namespaces == []

    asyncio.run(scenario())


def test_successful_job_confirmation_and_continuation(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            page = await client.get(f"/runs/{run_id}")
            assert "Northwind AI" in page.text
            assert "Develop maintainable Python services." in page.text
            assert "Python" in page.text
            response = await _approve(client, token, run_id)
            assert response.status_code == 303
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            assert pipeline.calls.index("linkedin_approval") < pipeline.calls.index(
                "codex_analysis"
            )

    asyncio.run(scenario())


def test_url_mode_has_apify_only_retrieval_surface(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            assert not hasattr(pipeline.namespaces[0], "linkedin_provider")
            page = await client.get("/")
            assert "Apify sends this exact public URL" in page.text
            assert "Retrieval provider" not in page.text
            assert "linkedin_provider" not in page.text
            assert "Codex uses live search" not in page.text

    asyncio.run(scenario())


def test_rejected_job_confirmation_stops_before_resume_analysis(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            await _approve(client, token, run_id, action="cancel")
            await _wait(app, run_id, "CANCELLED")
            assert "fetching_job" in pipeline.calls
            assert "codex_analysis" not in pipeline.calls
            assert not list((tmp_path / "output").rglob("*.docx"))

    asyncio.run(scenario())


def test_real_pipeline_records_view_only_after_ui_posting_gate_is_presented(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posting = _posting()

    def retrieve(**kwargs: object) -> dict[str, Any]:
        run_directory = kwargs["run_directory"]
        assert isinstance(run_directory, Path)
        (run_directory / "job-source.json").write_text(
            json.dumps(posting),
            encoding="utf-8",
        )
        return dict(posting)

    monkeypatch.setattr(cli_module, "invoke_apify_linkedin_retrieval", retrieve)
    monkeypatch.setattr(
        cli_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {
            "resume_tailor": "synthetic",
            "codex": "synthetic",
        },
    )
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        analytics_database_path=tmp_path / "private-data" / "analytics.sqlite3",
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            snapshot = app.state.manager.snapshot(run_id)
            assert snapshot["approval"]["kind"] == "linkedin_posting"
            summary = app.state.analytics_store.summary()
            assert summary["totals"]["unique_jobs_viewed"] == 1
            assert summary["totals"]["jobs_approved_for_tailoring"] == 0
            await _approve(client, token, run_id, action="cancel")
            await _wait(app, run_id, "CANCELLED")

    asyncio.run(scenario())


def test_real_pipeline_failed_retrieval_is_not_counted_as_viewed(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "invoke_apify_linkedin_retrieval",
        lambda **_kwargs: (_ for _ in ()).throw(
            ApifyLinkedInRetrievalError("malformed_output")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {
            "resume_tailor": "synthetic",
            "codex": "synthetic",
        },
    )
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        analytics_database_path=tmp_path / "private-data" / "analytics.sqlite3",
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "FAILED")
            assert app.state.analytics_store.summary()["totals"]["unique_jobs_viewed"] == 0

    asyncio.run(scenario())


def test_real_ui_pasted_input_keeps_pasted_analytics_source(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_MODE", "questions")
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        analytics_database_path=tmp_path / "private-data" / "analytics.sqlite3",
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            await _wait(app, run_id, "FAILED")
            exported = app.state.analytics_store.sanitized_export()
            assert exported["jobs"][0]["source"] == "pasted_text"
            assert exported["aggregate_statistics"]["totals"]["unique_jobs_viewed"] == 1

    asyncio.run(scenario())


def test_use_pasted_description_from_confirmation_gate(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            response = await _approve(
                client,
                token,
                run_id,
                action="use_pasted",
                fallback="Complete manually copied Python AI engineering posting.",
            )
            assert response.status_code == 303
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            run = app.state.manager.snapshot(run_id)
            assert (
                Path(run["artifact_directory"]) / "job-description.txt"
            ).read_text(encoding="utf-8").startswith("Complete manually")

    asyncio.run(scenario())


def test_over_limit_confirmation_fallback_reports_actual_and_permitted(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )
    actual = MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS + 1

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            response = await _approve(
                client,
                token,
                run_id,
                action="use_pasted",
                fallback="x" * actual,
            )
            assert response.status_code == 409
            detail = response.json()["detail"]
            assert f"{actual:,}" in detail
            assert f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,}" in detail
            assert app.state.manager.snapshot(run_id)["status"] == "AWAITING_APPROVAL"

    asyncio.run(scenario())


def test_resume_change_approval_and_rejection(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            page = await client.get(f"/runs/{run_id}")
            assert "Review evidence-backed recommendations" in page.text
            assert "Requirement-to-source review" in page.text
            assert "model-assessed semantic match" not in page.text.casefold()
            assert "Languages: Python, JavaScript, SQL" in page.text
            assert "skill.001" in page.text
            assert "Evidence-backed summary" in page.text
            assert "RAG is not supported" in page.text
            await _approve(client, token, run_id)
            await _wait(app, run_id, "AWAITING_APPROVAL", "tailored_content")
            assert "antigravity_tailoring" in pipeline.calls

            await _approve(client, token, run_id, action="cancel")
            await _wait(app, run_id, "CANCELLED")
            assert "rendering" not in pipeline.calls

    asyncio.run(scenario())


def test_successful_completed_run_results(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            await _approve(client, token, run_id)
            await _wait(app, run_id, "AWAITING_APPROVAL", "tailored_content")
            await _approve(client, token, run_id)
            completed = await _wait(app, run_id, "COMPLETE")
            assert completed["metadata"]["layout_validation"]["pages"] == 1
            page = await client.get(f"/runs/{run_id}")
            assert "Your tailored résumé is ready" in page.text
            assert "Factual integrity" in page.text
            assert "PDF preview" in page.text
            assert "<iframe" in page.text

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        "Apify authentication failed. Retry with --job-file.",
        "The configured Apify Actor was unavailable.",
    ],
)
def test_failed_retrieval_messages_are_safe(
    failure: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(fetch_failure=failure),
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "FAILED")
            page = await client.get(f"/runs/{run_id}")
            assert "pipeline stopped safely" in page.text
            assert "text file or pasted description" in page.text

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("missing_token", "Apify token configuration required"),
        ("missing_actor_id", "Apify Actor configuration required"),
        ("authentication_failure", "Apify authentication failed"),
        ("actor_not_found", "Apify Actor not found"),
        ("actor_timeout", "Apify Actor timed out"),
        ("actor_failure", "Apify Actor run failed"),
        ("empty_dataset", "Empty Apify dataset"),
        ("no_matching_result", "LinkedIn result mismatch"),
        ("insufficient_content", "Incomplete LinkedIn content"),
        ("network_error", "Apify network failure"),
        ("rate_limited", "Apify rate limit reached"),
        ("provider_failure", "Apify retrieval provider failure"),
        ("malformed_output", "Apify result format failure"),
    ],
)
def test_apify_retrieval_failures_are_classified_and_sanitized(
    classification: str,
    expected: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    def failing_pipeline(
        args: argparse.Namespace,
        *,
        hooks: PipelineHooks,
    ) -> Path:
        run_directory = args.output_dir / "apify-retrieval-failure"
        run_directory.mkdir(mode=0o700)
        hooks.progress(
            "fetching_job",
            "Retrieving the posting with Apify.",
            run_directory=str(run_directory),
        )
        if classification in {
            "missing_token",
            "missing_actor_id",
            "invalid_token",
            "invalid_actor_id",
        }:
            raise ApifyConfigurationError(classification)
        raise ApifyLinkedInRetrievalError(classification)

    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=failing_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            failed = await _wait(app, run_id, "FAILED")
            assert "synthetic-secret-token" not in failed["message"]
            assert "PRIVATE POSTING" not in failed["message"]
            assert failed["retrieval_classification"] == classification
            page = await client.get(f"/runs/{run_id}")
            assert expected in page.text
            assert "does not invent missing content" in page.text
            assert "No résumé content was sent" in page.text
            assert "synthetic-secret-token" not in page.text
            assert "PRIVATE POSTING" not in page.text

    asyncio.run(scenario())


def test_codex_schema_failure_is_concise_with_collapsed_sanitized_details(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    secret_prompt = "PRIVATE MASTER RESUME PROMPT"

    def incompatible_pipeline(
        args: argparse.Namespace,
        *,
        hooks: PipelineHooks,
    ) -> Path:
        run_directory = args.output_dir / "schema-failure"
        run_directory.mkdir(mode=0o700)
        hooks.progress(
            "codex_analysis",
            "Preparing the Codex analysis schema.",
            run_directory=str(run_directory),
        )
        raise CodexSchemaCompatibilityError(
            "codex_analysis.openai.schema.json is incompatible at "
            "/properties/strengths/uniqueItems: unsupported keyword.\n"
            "BEGIN_TRUSTED_MASTER_RESUME_JSON\n"
            f"{secret_prompt}\n"
            "END_TRUSTED_MASTER_RESUME_JSON"
        )

    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=incompatible_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            failed = await _wait(app, run_id, "FAILED")
            assert failed["message"] == (
                "Codex could not start because its output schema was incompatible."
            )
            assert "uniqueItems" in failed["error"]
            assert secret_prompt not in failed["error"]
            page = await client.get(f"/runs/{run_id}")
            assert (
                "Codex could not start because its output schema was incompatible."
                in page.text
            )
            assert "Sanitized technical details" in page.text
            assert '<details class="diff-details technical-details">' in page.text
            assert secret_prompt not in page.text

    asyncio.run(scenario())


def test_source_evidence_failure_guidance_and_safe_retry(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_source_failure(
        output_directory,
        master_resume,
    )
    monkeypatch.setattr(
        "resume_tailor.ui.cli._tailoring_dependency_versions",
        lambda *_args, **_kwargs: pytest.fail(
            "Antigravity dependencies checked before approval"
        ),
    )
    monkeypatch.setattr(
        "resume_tailor.ui.cli._analysis_dependency_versions",
        lambda *_args, **_kwargs: {
            "resume_tailor": "synthetic",
            "codex": "synthetic",
            "analysis_provider": "codex",
        },
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "Model evidence-contract failure" in page.text
            assert "violated the authoritative local evidence contract" in page.text
            assert "Retry analysis" in page.text
            assert "LinkedIn fallback" not in page.text
            assert "job-file" not in page.text
            assert "PRIVATE SYNTHETIC SOURCE QUOTATION" not in page.text
            assert '<details class="diff-details technical-details">' in page.text

            retry = await client.post(
                f"/runs/{source_id}/retry-codex-analysis",
                data={"csrf_token": token},
            )
            assert retry.status_code == 303
            retry_id = retry.headers["location"].rsplit("/", 1)[-1]
            renewed = await _wait(
                app,
                retry_id,
                "AWAITING_APPROVAL",
                "codex_analysis",
            )
            assert renewed["source_mode"] == "retry"
            retry_directory = Path(renewed["artifact_directory"])
            assert retry_directory != source_run
            assert not (retry_directory / "job-source.json").exists()
            assert (retry_directory / "job-requirements.json").is_file()
            assert not (retry_directory / "antigravity-response.json").exists()
            assert (
                (retry_directory / "analysis-resolved.json").is_file()
                or (retry_directory / "codex-analysis-resolved.json").is_file()
            )
            metadata = json.loads(
                (retry_directory / "run-metadata.json").read_text(encoding="utf-8")
            )
            assert metadata["job_source"] == "source-evidence-retry"
            assert metadata["retry_of"] == source_run.name

    asyncio.run(scenario())


def test_antigravity_launch_failure_guidance_and_authenticated_recovery(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_antigravity_launch_failure(
        output_directory,
        master_resume,
        monkeypatch,
    )
    before = {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        cli_module,
        "invoke_analysis",
        lambda **_kwargs: pytest.fail("Analysis ran during Antigravity recovery"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_codex_analysis",
        lambda **_kwargs: pytest.fail("Codex ran during Antigravity recovery"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_apify_linkedin_retrieval",
        lambda **_kwargs: pytest.fail(
            "Apify retrieval ran during Antigravity recovery"
        ),
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert (
                "Antigravity could not start because the request exceeded the "
                "operating system’s command-line size."
            ) in page.text
            assert "confirmed posting and approved Codex analysis were preserved" in (
                page.text
            )
            assert "Retry Antigravity tailoring" in page.text
            assert "Retry analysis" not in page.text
            assert "LinkedIn retrieval fallback" not in page.text
            assert "codex-analysis-approval.json" not in page.text
            assert '<details class="diff-details technical-details">' in page.text

            retry = await client.post(
                f"/runs/{source_id}/retry-antigravity-tailoring",
                data={"csrf_token": token},
            )
            assert retry.status_code == 303
            retry_id = retry.headers["location"].rsplit("/", 1)[-1]
            recovered = await _wait(
                app,
                retry_id,
                "AWAITING_APPROVAL",
                "tailored_content",
            )
            assert recovered["source_mode"] == "antigravity-retry"
            recovery_directory = Path(recovered["artifact_directory"])
            assert recovery_directory != source_run
            assert not (recovery_directory / "codex-analysis.json").exists()
            assert (
                (recovery_directory / "analysis-resolved.json").is_file()
                or (recovery_directory / "codex-analysis-resolved.json").is_file()
            )
            assert (
                recovery_directory / "codex-analysis-approval.json"
            ).is_file()
            assert (recovery_directory / "antigravity-response.json").is_file()
            assert (recovery_directory / "tailored-content.json").is_file()
            assert (recovery_directory / "content-diff.md").is_file()
            assert not list(recovery_directory.glob("*.docx"))
            metadata = json.loads(
                (recovery_directory / "run-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            assert metadata["retry_kind"] == "antigravity-tailoring"
            assert metadata["retry_of"] == source_run.name
            assert metadata["approved_analysis_reused"] is True
            assert metadata["recovery_inputs"]["approval_record_sha256"]
            cancel = await _approve(
                client,
                token,
                retry_id,
                action="cancel",
            )
            assert cancel.status_code == 303
            await _wait(app, retry_id, "CANCELLED")

    asyncio.run(scenario())
    assert before == {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }


def test_antigravity_waiting_guidance_and_authenticated_step_six_recovery(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_antigravity_waiting_failure(
        output_directory,
        master_resume,
        monkeypatch,
    )
    before = {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        cli_module,
        "invoke_analysis",
        lambda **_kwargs: pytest.fail("Analysis ran during Antigravity recovery"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_codex_analysis",
        lambda **_kwargs: pytest.fail("Codex ran during Antigravity recovery"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_apify_linkedin_retrieval",
        lambda **_kwargs: pytest.fail(
            "Apify retrieval ran during Antigravity recovery"
        ),
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "Post-approval tailoring contract failure" in page.text
            assert "did not apply the approved tailoring plan" in page.text
            assert "not a request for missing experience" in page.text
            assert "Retry Antigravity tailoring" in page.text
            assert "Retry analysis" not in page.text
            assert "LinkedIn retrieval fallback" not in page.text
            assert "Plan mode activated" not in page.text
            assert "What synthetic task" not in page.text
            assert '<details class="diff-details technical-details">' in page.text

            retry = await client.post(
                f"/runs/{source_id}/retry-antigravity-tailoring",
                data={"csrf_token": token},
            )
            assert retry.status_code == 303
            retry_id = retry.headers["location"].rsplit("/", 1)[-1]
            recovered = await _wait(
                app,
                retry_id,
                "AWAITING_APPROVAL",
                "tailored_content",
            )
            recovery_directory = Path(recovered["artifact_directory"])
            assert recovered["source_mode"] == "antigravity-retry"
            assert recovery_directory != source_run
            assert not (recovery_directory / "codex-analysis.json").exists()
            assert (recovery_directory / "antigravity-response.json").is_file()
            assert (recovery_directory / "tailored-content.json").is_file()
            assert (recovery_directory / "content-diff.md").is_file()
            assert not list(recovery_directory.glob("*.docx"))
            assert not list(recovery_directory.glob("*.pdf"))
            metadata = json.loads(
                (recovery_directory / "run-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            assert metadata["retry_kind"] == "antigravity-tailoring"
            assert metadata["retry_of"] == source_run.name
            assert metadata["approved_analysis_reused"] is True
            cancel = await _approve(
                client,
                token,
                retry_id,
                action="cancel",
            )
            assert cancel.status_code == 303
            await _wait(app, retry_id, "CANCELLED")

    asyncio.run(scenario())
    assert before == {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }


def test_apify_result_format_failure_shows_safe_input_fallback(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"

    def fail(**kwargs: object) -> dict[str, object]:
        run_directory = kwargs["run_directory"]
        assert isinstance(run_directory, Path)
        atomic_write_json(
            run_directory / "apify-linkedin-retrieval-diagnostic.json",
            {
                "provider": "apify",
                "classification": "malformed_output",
                "provider_output_omitted": True,
                "api_token_omitted": True,
            },
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
    code = cli_module.main(
        [
            "--resume",
            str(master_resume),
            "--job-url",
            JOB_URL,
            "--output-dir",
            str(output_directory),
            "--timeout",
            "30s",
            "--analysis-provider",
            "codex",
        ]
    )
    assert code == ExitCode.MODEL
    source_run = next(output_directory.iterdir())
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            await _session(client)
            page = await client.get(f"/runs/history-{source_run.name}")
            assert page.status_code == 200
            assert "Apify result format failure" in page.text
            assert "Use another job input" in page.text
            assert "No résumé content was sent for analysis or tailoring" in page.text
            assert "apify-linkedin-retrieval-diagnostic.json" in page.text
            assert "Retry Antigravity tailoring" not in page.text
            assert "Reprocess preserved Antigravity response" not in page.text
            assert "Offline reprocessing unavailable" not in page.text
            assert "New run required" not in page.text

    asyncio.run(scenario())


def test_valid_preserved_response_reprocesses_offline_to_content_diff_gate(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_antigravity_envelope_failure(
        output_directory,
        master_resume,
        monkeypatch,
        valid_response=True,
    )
    before = {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        cli_module,
        "invoke_analysis",
        lambda **_kwargs: pytest.fail("Analysis ran during offline reprocessing"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_codex_analysis",
        lambda **_kwargs: pytest.fail("Codex ran during offline reprocessing"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_apify_linkedin_retrieval",
        lambda **_kwargs: pytest.fail("LinkedIn ran during offline reprocessing"),
    )
    monkeypatch.setattr(
        cli_module,
        "invoke_antigravity",
        lambda **_kwargs: pytest.fail("Antigravity ran during offline reprocessing"),
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "Antigravity response-format failure" in page.text
            assert "Reprocess preserved Antigravity response" in page.text
            assert "Retry Antigravity tailoring" in page.text
            assert "Synthetic prose" not in page.text

            response = await client.post(
                f"/runs/{source_id}/reprocess-antigravity-response",
                data={"csrf_token": token},
            )
            assert response.status_code == 303
            run_id = response.headers["location"].rsplit("/", 1)[-1]
            reprocessed = await _wait(
                app,
                run_id,
                "AWAITING_APPROVAL",
                "tailored_content",
            )
            assert reprocessed["source_mode"] == "antigravity-response-reprocess"
            directory = Path(reprocessed["artifact_directory"])
            assert directory != source_run
            assert (directory / "antigravity-response.json").is_file()
            assert (directory / "tailored-content.json").is_file()
            assert (directory / "content-diff.md").is_file()
            assert not (directory / "codex-analysis.json").exists()
            assert not list(directory.glob("*.docx"))
            assert not list(directory.glob("*.pdf"))
            metadata = json.loads(
                (directory / "run-metadata.json").read_text(encoding="utf-8")
            )
            assert metadata["retry_kind"] == "antigravity-response-reprocess"
            assert metadata["provider_calls_reused"] == {
                "apify_linkedin_retrieval": False,
                "codex_analysis": False,
                "antigravity_tailoring": False,
            }
            assert metadata["antigravity_response"]["reprocessed_offline"] is True
            cancel = await _approve(client, token, run_id, action="cancel")
            assert cancel.status_code == 303
            await _wait(app, run_id, "CANCELLED")

    asyncio.run(scenario())
    assert before == {
        path.name: sha256_file(path)
        for path in source_run.iterdir()
        if path.is_file()
    }


def test_unstructured_preserved_response_offers_retry_but_not_reprocessing(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_antigravity_envelope_failure(
        output_directory,
        master_resume,
        monkeypatch,
        valid_response=False,
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "Offline reprocessing unavailable" in page.text
            assert "Reprocess preserved Antigravity response" not in page.text
            assert "Retry Antigravity tailoring" in page.text
            response = await client.post(
                f"/runs/{source_id}/reprocess-antigravity-response",
                data={"csrf_token": token},
            )
            assert response.status_code == 409
            assert app.state.manager._records == {}

    asyncio.run(scenario())


def test_antigravity_recovery_is_not_offered_without_approval_record(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_antigravity_launch_failure(
        output_directory,
        master_resume,
        monkeypatch,
    )
    metadata_path = source_run / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("codex_analysis_approval")
    atomic_write_json(metadata_path, metadata)
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "New run required" in page.text
            assert "predates the authenticated Codex approval record" in page.text
            assert "Retry Antigravity tailoring" not in page.text
            retry = await client.post(
                f"/runs/{source_id}/retry-antigravity-tailoring",
                data={"csrf_token": token},
            )
            assert retry.status_code == 409
            assert app.state.manager._records == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("changed_input", ["job", "extraction", "requirements"])
def test_source_evidence_retry_refuses_changed_input_hash(
    changed_input: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output"
    source_run = _write_synthetic_source_failure(
        output_directory,
        master_resume,
        name=f"changed-{changed_input}",
    )
    changed = {
        "job": source_run / "job-description.txt",
        "extraction": source_run / "extracted-master-resume.json",
        "requirements": source_run / "job-requirements.json",
    }[changed_input]
    changed.write_bytes(changed.read_bytes() + b" ")
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        pipeline_runner=run_pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            source_id = f"history-{source_run.name}"
            page = await client.get(f"/runs/{source_id}")
            assert page.status_code == 200
            assert "New run required" in page.text
            assert "Retry analysis" not in page.text
            retry = await client.post(
                f"/runs/{source_id}/retry-codex-analysis",
                data={"csrf_token": token},
            )
            assert retry.status_code == 409
            assert app.state.manager._records == {}

    asyncio.run(scenario())


def test_cancellation_and_double_submission_prevention(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline(block_initial=True)
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            await _wait(app, run_id, "RUNNING")
            duplicate = await client.post(
                "/runs",
                data={
                    "csrf_token": token,
                    "resume_mode": "master",
                    "job_mode": "pasted",
                    "company": "Other",
                    "role": "Other",
                    "pasted_description": "Another complete Python job description.",
                },
            )
            assert duplicate.status_code == 409
            cancel = await client.post(
                f"/runs/{run_id}/cancel",
                data={"csrf_token": token},
            )
            assert cancel.status_code == 303
            await _wait(app, run_id, "CANCELLED")
            assert pipeline.namespaces
            thread = app.state.manager._records[run_id].thread
            assert thread is not None
            thread.join(timeout=1)
            assert not thread.is_alive()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("resume.pdf", b"%PDF"),
        ("resume.docx", b"not-a-zip"),
    ],
)
def test_uploaded_resume_validation(
    filename: str,
    content: bytes,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(),
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            response = await client.post(
                "/runs",
                data={
                    "csrf_token": token,
                    "resume_mode": "upload",
                    "job_mode": "pasted",
                    "company": "Example",
                    "role": "AI Engineer",
                    "pasted_description": "Build safe Python AI systems.",
                },
                files={
                    "resume_upload": (
                        filename,
                        content,
                        "application/octet-stream",
                    )
                },
            )
            assert response.status_code == 422
            assert "uploaded" in response.text.casefold()

    asyncio.run(scenario())


def test_valid_uploaded_resume_reaches_shared_pipeline(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    pipeline = StubbedUIPipeline()
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=pipeline,
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            response = await client.post(
                "/runs",
                data={
                    "csrf_token": token,
                    "resume_mode": "upload",
                    "job_mode": "pasted",
                    "company": "Example",
                    "role": "AI Engineer",
                    "pasted_description": "Build safe Python AI systems.",
                },
                files={
                    "resume_upload": (
                        "compatible resume.docx",
                        master_resume.read_bytes(),
                        (
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                    )
                },
            )
            assert response.status_code == 303
            run_id = response.headers["location"].rsplit("/", 1)[-1]
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            assert pipeline.namespaces[0].resume.name == "uploaded-resume.docx"
            assert pipeline.namespaces[0].resume.is_file()

    asyncio.run(scenario())


def test_webpage_content_is_html_escaped(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(injection=True),
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token)
            await _wait(app, run_id, "AWAITING_APPROVAL", "linkedin_posting")
            page = await client.get(f"/runs/{run_id}")
            assert "<script>alert" not in page.text
            assert "&lt;script&gt;alert" in page.text

    asyncio.run(scenario())


def test_csrf_and_session_validation(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(),
    )

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as anonymous:
            denied = await anonymous.post(
                "/runs",
                data={
                    "csrf_token": app.state.settings.launch_token,
                    "resume_mode": "master",
                    "job_mode": "url",
                    "job_url": JOB_URL,
                },
            )
            assert denied.status_code == 403
        async with _client(app) as client:
            await _session(client)
            denied = await client.post(
                "/runs",
                data={
                    "csrf_token": "wrong-token",
                    "resume_mode": "master",
                    "job_mode": "url",
                    "job_url": JOB_URL,
                },
            )
            assert denied.status_code == 403

    asyncio.run(scenario())


def test_artifact_download_and_path_traversal(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        pipeline_runner=StubbedUIPipeline(),
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            run_id = await _start(client, token, mode="pasted")
            await _wait(app, run_id, "AWAITING_APPROVAL", "codex_analysis")
            await _approve(client, token, run_id)
            await _wait(app, run_id, "AWAITING_APPROVAL", "tailored_content")
            await _approve(client, token, run_id)
            completed = await _wait(app, run_id, "COMPLETE")
            docx_name = next(
                item["name"] for item in completed["artifacts"] if item["kind"] == "DOCX"
            )
            download = await client.get(
                f"/runs/{run_id}/artifacts/{docx_name}"
            )
            assert download.status_code == 200
            assert download.content == b"stub-docx"
            pdf_name = next(
                item["name"] for item in completed["artifacts"] if item["kind"] == "PDF"
            )
            inline_pdf = await client.get(
                f"/runs/{run_id}/artifacts/{pdf_name}?inline=true"
            )
            assert inline_pdf.status_code == 200
            assert inline_pdf.headers["x-frame-options"] == "SAMEORIGIN"
            assert "frame-ancestors 'self'" in inline_pdf.headers[
                "content-security-policy"
            ]
            assert inline_pdf.headers["content-disposition"].startswith("inline")

            secret = tmp_path / "output" / "secret.txt"
            secret.write_text("must not escape", encoding="utf-8")
            traversal = await client.get(
                f"/runs/{run_id}/artifacts/%2e%2e%2fsecret.txt"
            )
            assert traversal.status_code in {404, 405}
            with pytest.raises(InputError):
                app.state.manager.resolve_artifact(run_id, "../secret.txt")

            unrelated = tmp_path / "output" / "unrelated"
            unrelated.mkdir()
            (unrelated / "private.pdf").write_bytes(b"not-a-run")
            rejected_directory = await client.get(
                "/runs/history-unrelated/artifacts/private.pdf"
            )
            assert rejected_directory.status_code == 404

    asyncio.run(scenario())


def test_local_analytics_dashboard_and_explicit_tracking_actions(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    analytics_path = tmp_path / "private-data" / "job-search-analytics.sqlite3"
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        analytics_database_path=analytics_path,
        pipeline_runner=StubbedUIPipeline(),
    )
    tracked = app.state.analytics_store.record_job_viewed(
        observation_from_local_job(
            company="Example Company",
            title="Senior Platform Engineer",
            description="Build safe PostgreSQL services with continuous integration.",
            source="pasted_text",
        )
    )

    async def scenario() -> None:
        async with _client(app) as client:
            token = await _session(client)
            page = await client.get("/analytics")
            assert page.status_code == 200
            assert "Job-search analytics" in page.text
            assert "Unique jobs viewed" in page.text
            assert "Applications submitted" in page.text
            assert "Not enough data" in page.text
            assert "Save job" in page.text
            assert "Mark applied" in page.text
            assert "Record screening" in page.text
            assert "Record interview" in page.text
            assert "Record rejection" in page.text
            assert "Record offer" in page.text
            assert "Withdraw application" in page.text
            assert "Correct status" in page.text
            assert "Add note" in page.text

            saved = await client.post(
                f"/analytics/applications/{tracked.application_id}/status",
                data={"csrf_token": token, "action": "save"},
            )
            assert saved.status_code == 303
            applied = await client.post(
                f"/analytics/applications/{tracked.application_id}/status",
                data={"csrf_token": token, "action": "apply"},
            )
            assert applied.status_code == 303

            unconfirmed_correction = await client.post(
                f"/analytics/applications/{tracked.application_id}/correction",
                data={
                    "csrf_token": token,
                    "new_status": "saved",
                    "note": "Correct the prior click",
                },
            )
            assert unconfirmed_correction.status_code == 422
            corrected = await client.post(
                f"/analytics/applications/{tracked.application_id}/correction",
                data={
                    "csrf_token": token,
                    "new_status": "saved",
                    "note": "Correct the prior click",
                    "confirm_correction": "yes",
                },
            )
            assert corrected.status_code == 303

            unconfirmed_interview = await client.post(
                f"/analytics/applications/{tracked.application_id}/interviews",
                data={
                    "csrf_token": token,
                    "interview_type": "technical_interview",
                },
            )
            assert unconfirmed_interview.status_code == 422
            interview = await client.post(
                f"/analytics/applications/{tracked.application_id}/interviews",
                data={
                    "csrf_token": token,
                    "interview_type": "technical_interview",
                    "scheduled_at": "2026-08-10T09:30",
                    "contact_label": "Engineering panel",
                    "confirm_interview": "yes",
                },
            )
            assert interview.status_code == 303
            note = await client.post(
                f"/analytics/applications/{tracked.application_id}/notes",
                data={"csrf_token": token, "note": "Prepare system design examples"},
            )
            assert note.status_code == 303

            history = app.state.analytics_store.application_history(
                tracked.application_id
            )
            assert [item["event_kind"] for item in history].count("correction") == 1
            assert history[-1]["new_status"] == "technical_interview"
            refreshed = await client.get("/analytics")
            assert refreshed.status_code == 200
            assert "Active interviews" in refreshed.text
            assert "technical interview" in refreshed.text.casefold()

    asyncio.run(scenario())


def test_analytics_routes_require_local_session_and_csrf(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    app = create_app(
        output_directory=tmp_path / "output",
        master_resume=master_resume,
        analytics_database_path=tmp_path / "private-data" / "analytics.sqlite3",
        pipeline_runner=StubbedUIPipeline(),
    )
    tracked = app.state.analytics_store.record_job_viewed(
        observation_from_local_job(
            company="Example Company",
            title="Platform Engineer",
            description="Build safe local systems.",
            source="file",
        )
    )

    async def scenario() -> None:
        async with _client(app) as anonymous:
            denied = await anonymous.get("/analytics")
            assert denied.status_code == 403
        async with _client(app) as client:
            await _session(client)
            denied = await client.post(
                f"/analytics/applications/{tracked.application_id}/status",
                data={"csrf_token": "wrong", "action": "apply"},
            )
            assert denied.status_code == 403

    asyncio.run(scenario())


def test_ollama_budget_refusal_is_not_reported_as_a_connection_failure() -> None:
    """OllamaBudgetError subclasses OllamaConnectionError.

    It must be matched before the connection branch in both classification
    ladders, otherwise a local budget refusal (no request launched) would tell
    the operator to check that Ollama is running.
    """
    error = OllamaBudgetError(
        "The approved tailoring prompt needs 9000 tokens but the configured "
        "context window leaves 4096."
    )
    kind = ui._failure_kind_for_error(error, stage="ollama-tailoring")
    assert kind == "ollama_preflight"
    assert kind != "ollama_connection"

    message = ui._safe_error_message(error)
    assert "context window" in message
    assert "Ollama is running" not in message
    # Sanity check that a genuine connection failure still maps correctly.
    assert (
        ui._failure_kind_for_error(
            OllamaConnectionError("connection refused"), stage="ollama-tailoring"
        )
        == "ollama_connection"
    )


def test_new_ollama_failure_classes_keep_contract_guidance_and_no_recovery() -> None:
    """Phase 1 sub-classifications must reach the existing contract branch.

    They narrow diagnosis in run metadata. They must not silently fall through
    to a blank failure card, and they must not become retry/reprocess eligible.
    """
    for failure_class in (
        "ollama-malformed-json",
        "ollama-response-envelope",
        "ollama-transport-schema",
        "ollama-canonical-schema",
        "ollama-output-truncation",
        "ollama-downstream-evidence",
    ):
        metadata = {
            "status": "failed",
            "stage": "ollama-tailoring",
            "failure_class": failure_class,
            "error": {"type": "OllamaTransportSchemaError", "message": "rejected"},
        }
        assert _failure_kind_from_metadata(metadata) == "ollama_contract", failure_class
        # Authenticated Antigravity recovery must stay unavailable: the retry
        # classifier only recognises antigravity-* failure classes.
        assert antigravity_retry_failure_kind(metadata) is None, failure_class
        assert _failure_kind_from_metadata(metadata) != "antigravity_response_envelope"

    assert (
        _failure_kind_from_metadata(
            {
                "status": "failed",
                "stage": "ollama-tailoring",
                "failure_class": "ollama-budget-preflight",
                "error": {"type": "OllamaBudgetError", "message": "no room"},
            }
        )
        == "ollama_preflight"
    )


def test_deterministic_only_history_exposes_envelope_without_fake_response(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    tailoring_label = dict(ui.WORKFLOW_STAGES)["antigravity_tailoring"]
    assert tailoring_label == "Local résumé tailoring"
    assert "gemma" not in tailoring_label.casefold()
    assert "writing" not in tailoring_label.casefold()

    output_directory = tmp_path / "deterministic-history"
    run_directory = output_directory / "synthetic-deterministic-run"
    run_directory.mkdir(parents=True)
    envelope = {
        "version": 2,
        "provider": "deterministic",
        "runtime": "local",
        "model": None,
        "ollama_invoked": False,
        "execution_mode": "deterministic-local",
        "output_format": "deterministic-json",
        "response_envelope_type": "deterministic-local-patches",
        "validation_result": "PASS",
        "validation_path": "pass",
        "response": None,
        "execution": {
            "execution_mode": "deterministic_only",
            "deterministic_patch_count": 1,
            "gemma_patch_count": 0,
            "deterministic_target_ids": ["skill_groups.0"],
            "prose_target_ids": [],
            "full_catalog_digest": "a" * 64,
            "writer_subset_digest": None,
            "ollama_invoked": False,
        },
    }
    atomic_write_json(run_directory / "ollama-response-envelope.json", envelope)
    atomic_write_json(
        run_directory / "tailored-content.initial.json",
        {"synthetic": "deterministic content reference"},
    )
    atomic_write_json(
        run_directory / "run-metadata.json",
        {
            "application": "resume-tailor",
            "status": "COMPLETE",
            "stage": "complete",
            "created_at": "2026-08-03T12:00:00+00:00",
            "company": "Synthetic Systems",
            "role": "Evidence Engineer",
            "job_source": "file",
            "writer": {
                "provider": "deterministic",
                "name": "Deterministic local compiler",
                "model": None,
                "runtime": "local",
                "ollama_invoked": False,
            },
            "revision_cycle": {
                "final_generation": "initial",
                "initial": {
                    "provider": "deterministic",
                    "model": None,
                    "ollama_invoked": False,
                    "response_envelope_type": "deterministic-local-patches",
                    "output_format": "deterministic-json",
                    "tailored_content": {
                        "filename": "tailored-content.initial.json",
                        "sha256": sha256_file(
                            run_directory / "tailored-content.initial.json"
                        ),
                    },
                },
            },
        },
    )
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        analytics_database_path=tmp_path / "deterministic-history.sqlite3",
        pipeline_runner=StubbedUIPipeline(),
    )
    try:
        history = app.state.manager.history()
        entry = next(
            item
            for item in history
            if item["run_id"] == "history-synthetic-deterministic-run"
        )
        artifact_names = {artifact["name"] for artifact in entry["artifacts"]}
        assert "ollama-response-envelope.json" in artifact_names
        assert "ollama-response.json" not in artifact_names

        snapshot = app.state.manager.snapshot(entry["run_id"])
        assert snapshot["status"] == "COMPLETE"
        assert snapshot["failure_kind"] is None
        assert snapshot["retry_eligible"] is False
        assert snapshot["antigravity_retry_eligible"] is False
        assert snapshot["antigravity_reprocess_eligible"] is False
        assert snapshot["metadata"]["writer"]["provider"] == "deterministic"
        assert app.state.manager.resolve_artifact(
            entry["run_id"],
            "ollama-response-envelope.json",
        ) == (run_directory / "ollama-response-envelope.json").resolve()
        with pytest.raises(InputError, match="not available"):
            app.state.manager.resolve_artifact(
                entry["run_id"],
                "ollama-response.json",
            )
    finally:
        app.state.manager.shutdown()


def test_budget_repair_diagnostics_are_conditional_downloads_without_recovery(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "budget-repair-history"
    run_directory = output_directory / "synthetic-budget-repair-run"
    run_directory.mkdir(parents=True)
    atomic_write_json(
        run_directory / "run-metadata.json",
        {
            "application": "resume-tailor",
            "status": "FAILED",
            "stage": "ollama-tailoring",
            "failure_class": "ollama-tailoring-contract",
            "created_at": "2026-08-03T12:00:00+00:00",
            "company": "Synthetic Systems",
            "role": "Evidence Engineer",
            "job_source": "file",
            "error": {
                "type": "OllamaTailoringContractError",
                "message": "Synthetic repair remained over budget.",
            },
        },
    )
    repair_artifacts = {
        "ollama-budget-repair-response.json",
        "ollama-budget-repair-response-envelope.json",
        "ollama-budget-repair-transport.schema.json",
    }
    app = create_app(
        output_directory=output_directory,
        master_resume=master_resume,
        analytics_database_path=tmp_path / "budget-repair-history.sqlite3",
        pipeline_runner=StubbedUIPipeline(),
    )
    try:
        run_id = "history-synthetic-budget-repair-run"
        initial = app.state.manager.snapshot(run_id)
        assert repair_artifacts.isdisjoint(
            {artifact["name"] for artifact in initial["artifacts"]}
        )
        for artifact_name in repair_artifacts:
            with pytest.raises(InputError, match="not available"):
                app.state.manager.resolve_artifact(run_id, artifact_name)

        for artifact_name in repair_artifacts:
            atomic_write_json(
                run_directory / artifact_name,
                {"synthetic": True, "artifact": artifact_name},
            )

        snapshot = app.state.manager.snapshot(run_id)
        assert repair_artifacts <= {
            artifact["name"] for artifact in snapshot["artifacts"]
        }
        for artifact_name in repair_artifacts:
            assert app.state.manager.resolve_artifact(
                run_id,
                artifact_name,
            ) == (run_directory / artifact_name).resolve()

        assert snapshot["failure_kind"] == "ollama_contract"
        assert snapshot["retry_eligible"] is False
        assert snapshot["antigravity_retry_eligible"] is False
        assert snapshot["antigravity_reprocess_eligible"] is False
    finally:
        app.state.manager.shutdown()
