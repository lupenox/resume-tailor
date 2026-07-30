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

from resume_tailor.cli import run_pipeline
from resume_tailor.orchestration import PipelineHooks
from resume_tailor.ui import COOKIE_NAME, DEFAULT_HOST, create_app
from resume_tailor.ui_cli import build_parser as build_ui_parser
from resume_tailor.utilities import (
    CancellationError,
    CodexSchemaCompatibilityError,
    InputError,
)


JOB_URL = "https://www.linkedin.com/jobs/view/platform-engineer-4123456789/"


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
        "supported_ats_keywords": ["Python"],
        "missing_or_unsupported_requirements": ["RAG is not supported"],
        "recommended_edits": [
            {
                "resume_section": "Professional Summary",
                "existing_claim": "Existing summary",
                "proposed_replacement": "Evidence-backed summary",
                "alignment_rationale": "Surfaces supported Python work.",
                "exact_supporting_evidence": "Python",
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
            hooks.progress("fetching_job", "Stub is extracting the posting.")
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


def test_ui_cli_does_not_offer_remote_binding() -> None:
    parser = build_ui_parser()
    args = parser.parse_args(["--no-browser"])
    assert args.port == 8765
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
            assert pipeline.job_texts
            assert "Python" in pipeline.job_texts[0]

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


def test_rejected_job_confirmation_stops_before_codex(
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
            assert "codex_analysis" not in pipeline.calls
            assert not list((tmp_path / "output").rglob("*.docx"))

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
        "The LinkedIn posting requires login. Retry with --job-file.",
        "Antigravity read_url(linkedin.com) permission was denied.",
    ],
)
def test_failed_fetch_and_permission_denial(
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
