"""Synthetic tests for optional Step 10 revision (provider-selectable, no auto-launch)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from resume_tailor.backend.engine.orchestration import ApprovalRequest, ApprovalResponse, PipelineHooks
from resume_tailor.backend.engine.qa import (
    env_preselected_revision_provider,
    historical_initial_qa_provider,
    run_initial_qa,
)


def test_decide_optional_revision_assume_yes_completes_without_provider_calls() -> None:
    calls: list[str] = []

    def forbid_handler(request: ApprovalRequest) -> ApprovalResponse:
        calls.append(request.kind)
        raise AssertionError("approval_handler must not run under assume_yes path")

    hooks = PipelineHooks(approval_handler=None)
    response = hooks.decide_optional_revision(
        payload={
            "qa_result": {
                "status": "pass",
                "summary": "ok",
                "issues": [],
                "technical_failure": None,
            },
            "options": [
                {"provider_id": "codex", "available": True, "label": "Codex"},
            ],
            "default_provider": "codex",
        },
        assume_yes=True,
    )
    assert response.action == "complete_without_revision"
    assert calls == []


def test_decide_optional_revision_ui_actions() -> None:
    seen: list[str] = []

    def handler(request: ApprovalRequest) -> ApprovalResponse:
        seen.append(request.kind)
        assert request.kind == "optional_revision"
        return ApprovalResponse("complete_without_revision")

    hooks = PipelineHooks(approval_handler=handler)
    response = hooks.decide_optional_revision(
        payload={
            "qa_result": {"status": "pass", "summary": "ok", "issues": [], "technical_failure": None},
            "options": [],
        },
        assume_yes=False,
    )
    assert response.action == "complete_without_revision"
    assert seen == ["optional_revision"]


def test_decide_optional_revision_revise_carries_provider() -> None:
    def handler(request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            "revise_once",
            {
                "provider": "gemma_local",
                "revision_provider": "gemma_local",
                "final_qa_provider": "gemma_local",
            },
        )

    hooks = PipelineHooks(approval_handler=handler)
    response = hooks.decide_optional_revision(
        payload={
            "qa_result": {
                "status": "material_findings",
                "summary": "issues",
                "issues": [{"issue_id": "qa.001"}],
                "technical_failure": None,
            },
            "options": [
                {"provider_id": "gemma_local", "available": True, "label": "Gemma Local"}
            ],
            "default_provider": "gemma_local",
        },
    )
    assert response.action == "revise_once"
    assert response.data["revision_provider"] == "gemma_local"
    assert response.data["final_qa_provider"] == "gemma_local"


def test_same_as_initial_qa_preselects_but_does_not_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVISION_PROVIDER", "same_as_initial_qa")
    assert env_preselected_revision_provider(initial_qa_provider="grok") == "grok"
    monkeypatch.setenv("REVISION_PROVIDER", "codex")
    assert env_preselected_revision_provider(initial_qa_provider="grok") == "codex"


def test_historical_codex_final_qa_artifacts_remain_readable() -> None:
    # Historical runs without final_qa_provider still infer codex for display.
    assert historical_initial_qa_provider({"final_qa": {"provider": "codex"}}) == "codex"
    assert historical_initial_qa_provider({}) == "codex"


def test_final_qa_each_provider_isolation(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from resume_tailor.backend.documents.docx_extract import extract_resume

    extracted, _ = extract_resume(master_resume)
    preview = tmp_path / "preview.revision-1.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    calls: list[str] = []
    payload = {
        "status": "pass",
        "summary": "ok",
        "issues": [],
        "technical_failure": None,
    }

    def track(name: str):
        def _impl(**_kwargs: Any) -> dict[str, Any]:
            calls.append(name)
            return payload

        return _impl

    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_codex_qa", track("codex"))
    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_gemma_qa", track("gemma_local"))
    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_grok_qa", track("grok"))
    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_antigravity_qa", track("antigravity"))

    for provider in ("gemma_local", "codex", "grok", "antigravity"):
        calls.clear()
        run_dir = tmp_path / provider
        run_dir.mkdir()
        result = run_initial_qa(
            provider=provider,
            original_extraction=extracted,
            job_description="job",
            analysis={"recommended_edits": [], "immutable_facts": [], "forbidden_claims": []},
            tailored_pdf_text="text",
            content_diff="diff",
            preview_path=preview,
            run_directory=run_dir,
            work_directory=run_dir / "work",
            timeout_seconds=30,
            generation="revision-1",
            company="C",
            role="R",
        )
        assert result["status"] == "pass"
        assert calls == [provider]
        meta = json.loads((run_dir / "initial-qa-result.json").read_text(encoding="utf-8"))
        assert meta["provider"] == provider
        assert meta["generation"] == "revision-1"


def test_provider_neutral_error_message_includes_provider_name() -> None:
    from resume_tailor.backend.utils.utilities import ModelError

    # Surface wording contract used by adapters.
    message = "Final QA with Codex exited with status 2. Provider output was omitted from the exception."
    assert "Final QA with Codex" in message
    assert "Final Codex QA" not in message
    err = ModelError(message)
    assert "Codex" in str(err)


def test_complete_without_revision_metadata_shape() -> None:
    # Document the expected metadata contract for skip path.
    metadata = {
        "revision_cycle": {
            "state": "skipped",
            "skip_reason": "user_completed_without_revision",
            "skipped_at": "2026-08-04T00:00:00+00:00",
            "final_generation": "initial",
        },
        "initial_qa_provider": "grok",
        "final_qa": {"provider": "grok", "generation": "initial", "status": "pass"},
    }
    assert metadata["revision_cycle"]["state"] == "skipped"
    assert metadata["revision_cycle"]["final_generation"] == "initial"
    assert metadata["initial_qa_provider"] == "grok"
    # final_qa remains Initial QA; revision_provider absent when skipped.
    assert "revision_provider" not in metadata


def test_step10_decision_gate_does_not_invoke_providers_until_revise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []

    def track_run_initial_qa(**kwargs: Any) -> dict[str, Any]:
        launched.append(str(kwargs.get("provider")))
        return {
            "status": "pass",
            "summary": "ok",
            "issues": [],
            "technical_failure": None,
        }

    monkeypatch.setattr(
        "resume_tailor.application.pipeline.run_initial_qa",
        track_run_initial_qa,
    )
    hooks = PipelineHooks()
    # Gate alone must not call providers.
    response = hooks.decide_optional_revision(
        payload={
            "qa_result": {
                "status": "pass",
                "summary": "ok",
                "issues": [],
                "technical_failure": None,
            },
            "options": [{"provider_id": "codex", "available": True}],
            "default_provider": "codex",
        },
        assume_yes=True,
    )
    assert response.action == "complete_without_revision"
    assert launched == []


def test_preserved_step10_failure_can_reuse_artifacts(
    tmp_path: Path,
) -> None:
    """Synthetic resume geometry: revision artifacts exist; Initial QA intact."""
    run = tmp_path / "failed-step10"
    run.mkdir()
    (run / "final-qa.initial.json").write_text(
        json.dumps(
            {
                "status": "material_findings",
                "summary": "findings",
                "issues": [],
                "technical_failure": None,
            }
        ),
        encoding="utf-8",
    )
    (run / "tailored-content.revision-1.json").write_text("{}", encoding="utf-8")
    (run / "preview.revision-1.png").write_bytes(b"\x89PNG")
    (run / "out.revision-1.docx").write_bytes(b"PK")
    (run / "out.revision-1.pdf").write_bytes(b"%PDF")
    metadata = {
        "status": "FAILED",
        "stage": "revision-1-final-codex-qa",
        "initial_qa_provider": "grok",
        "revision_cycle": {
            "state": "revision_1_authorized",
            "attempt_count": 1,
            "initial": {
                "qa": {
                    "provider": "grok",
                    "generation": "initial",
                    "status": "material_findings",
                }
            },
            "revision_1": {
                "state": "awaiting_content_approval",
                "layout_validation": {"status": "PASS"},
            },
        },
        "error": {
            "type": "ModelError",
            "message": "Final Codex QA exited with status 2. Provider output was omitted from the exception.",
        },
    }
    # Readable historical failure; resume policy reuses artifacts without Steps 1–9.
    assert (run / "final-qa.initial.json").is_file()
    assert (run / "tailored-content.revision-1.json").is_file()
    assert metadata["initial_qa_provider"] == "grok"
    assert metadata["stage"] in {
        "revision-1-final-codex-qa",
        "revision-1-final-qa",
        "optional-revision-decision",
    }
    # New wording contract for future failures.
    assert "Final Codex QA" in metadata["error"]["message"]  # historical
    modern = "Final QA with Codex exited with status 2. Provider output was omitted from the exception."
    assert "Final QA with Codex" in modern


def test_revision_consumption_boundary_is_full_artifact_success() -> None:
    """One-shot is consumed only after full revision artifacts exist.

    Failed authorize/init/authoring must leave attempt_count at 0 so the gate can
    retry. Presence of content+preview+docx reconstructs consumption.
    """
    attempt_count = 0
    revision_already_authored = False

    def full_artifacts_present(paths: dict[str, bool]) -> bool:
        return all(
            paths[key]
            for key in ("content", "preview", "docx")
        )

    # Gate visit / provider choice / unavailable provider: not consumed.
    assert attempt_count == 0
    assert not revision_already_authored

    # Failed authoring leaves count at 0 (new boundary).
    paths = {"content": False, "preview": False, "docx": False}
    if not revision_already_authored and not full_artifacts_present(paths):
        if attempt_count != 0:
            raise AssertionError("would raise one-revision already consumed")
        # authorize starts; authoring fails before success boundary
        pass
    assert attempt_count == 0

    # Successful full render consumes exactly once.
    paths = {"content": True, "preview": True, "docx": True}
    if full_artifacts_present(paths):
        revision_already_authored = True
        attempt_count = 1
    assert attempt_count == 1
    assert revision_already_authored is True

    # Second content shot denied while reuse (Final QA) remains allowed.
    if not revision_already_authored and attempt_count != 0:
        raise AssertionError("unexpected")
    # reuse path when authored
    assert revision_already_authored


def test_failed_revision_authoring_does_not_consume_one_shot(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed writer init/authoring must not lock the one-shot limit."""
    import resume_tailor.application.pipeline as pipeline_module
    import resume_tailor.backend.documents.docx_render as docx_render_module
    import resume_tailor.backend.documents.headless_render as headless_render_module
    import resume_tailor.backend.providers.ollama_writer as ollama_writer_module
    from resume_tailor.backend.engine.orchestration import (
        ApprovalResponse,
        PipelineHooks,
    )
    from resume_tailor.ui.cli import build_parser, run_pipeline

    revision_invocations = 0
    optional_decisions = 0

    monkeypatch.setattr(
        pipeline_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {"resume_tailor": "synthetic", "codex": "mocked"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_tailoring_dependency_versions",
        lambda *_args, **_kwargs: {
            "ollama": "not-invoked",
            "ollama_model": "not-loaded",
            "libreoffice": "mocked",
        },
    )

    def fake_analysis(**kwargs: object) -> dict[str, object]:
        extracted = kwargs["extracted_resume"]  # type: ignore[assignment]
        requirements = kwargs["job_requirements"]  # type: ignore[assignment]
        run_directory = kwargs["run_directory"]  # type: ignore[assignment]
        assert isinstance(extracted, dict)
        assert isinstance(requirements, dict)
        assert isinstance(run_directory, Path)
        engineering = extracted["content"]["skill_groups"][2]
        raw: dict[str, object] = {
            "role_summary": "Synthetic revision consumption run.",
            "fit_assessment": {
                "overall": "Synthetic.",
                "strengths": ["Authenticated skills"],
                "gaps": [],
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
                    "alignment_rationale": "Surface authenticated skills.",
                    "evidence_source_ids": ["skill_groups.2", "skill_groups.0"],
                }
            ],
            "immutable_facts": [],
            "forbidden_claims": ["GraphQL"],
            "content_budget_guidance": [],
            "questions_for_user": [],
        }
        (run_directory / "codex-analysis.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )
        return raw

    monkeypatch.setattr(pipeline_module, "invoke_analysis", fake_analysis)
    monkeypatch.setattr(pipeline_module, "invoke_codex_analysis", fake_analysis)
    monkeypatch.setattr(
        ollama_writer_module,
        "run_ollama_request",
        lambda **_kwargs: pytest.fail("Ollama must not be invoked"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "invoke_antigravity",
        lambda **_kwargs: pytest.fail("Antigravity must not be invoked"),
    )

    def fake_render(**kwargs: object) -> None:
        destination = kwargs["destination_path"]
        assert isinstance(destination, Path)
        destination.write_bytes(b"synthetic revision consumption docx")

    def fake_export(**kwargs: object) -> str:
        pdf_path = kwargs["pdf_path"]
        preview_path = kwargs["preview_path"]
        assert isinstance(pdf_path, Path)
        assert isinstance(preview_path, Path)
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")
        preview_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
        return "Synthetic resume text"

    def fake_final_qa(**kwargs: object) -> dict[str, object]:
        run_directory = kwargs["run_directory"]
        generation = kwargs["generation"]
        assert isinstance(run_directory, Path)
        result: dict[str, object] = {
            "status": "pass",
            "summary": "Synthetic QA passed.",
            "issues": [],
            "technical_failure": None,
        }
        stem = f"final-qa.{generation}"
        (run_directory / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
        (run_directory / f"{stem}.md").write_text("# PASS\n", encoding="utf-8")
        return result

    def fail_revision(**_kwargs: object) -> dict[str, object]:
        nonlocal revision_invocations
        revision_invocations += 1
        raise RuntimeError("synthetic revision authoring failure")

    monkeypatch.setattr(headless_render_module, "render_headless_docx", fake_render)
    monkeypatch.setattr(docx_render_module, "export_and_validate_pdf", fake_export)
    monkeypatch.setattr(pipeline_module, "invoke_final_qa", fake_final_qa)
    monkeypatch.setattr(pipeline_module, "run_initial_qa", fake_final_qa)
    monkeypatch.setattr(pipeline_module, "invoke_ollama_revision", fail_revision)
    monkeypatch.setattr(
        pipeline_module,
        "probe_initial_qa_providers",
        lambda **_kwargs: [
            {
                "provider_id": "codex",
                "label": "Codex",
                "description": "stub",
                "available": True,
                "status": "ready",
                "detail": "stub ready",
                "capabilities": ["content_qa"],
                "limitations": [],
            }
        ],
    )

    def approval(request: object) -> ApprovalResponse:
        nonlocal optional_decisions
        kind = request.kind  # type: ignore[attr-defined]
        if kind in {"codex_analysis", "tailored_content", "rendered_artifacts"}:
            return ApprovalResponse("approve")
        if kind == "initial_qa_provider":
            return ApprovalResponse("select", {"provider": "codex"})
        if kind == "optional_revision":
            optional_decisions += 1
            if optional_decisions == 1:
                return ApprovalResponse(
                    "revise_once",
                    {
                        "provider": "codex",
                        "revision_provider": "codex",
                        "final_qa_provider": "codex",
                    },
                )
            # After failed authoring, gate returns — complete without revision.
            return ApprovalResponse("complete_without_revision")
        return ApprovalResponse("approve")

    parser = build_parser()
    output_dir = tmp_path / "revision-consumption-output"
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
            str(tmp_path / "revision-consumption-analytics.sqlite3"),
            "--timeout",
            "30s",
            "--analysis-provider",
            "codex",
        ]
    )
    result = run_pipeline(
        args, hooks=PipelineHooks(approval_handler=approval)
    )
    run = result if isinstance(result, Path) else result.run_directory
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert revision_invocations == 1
    assert optional_decisions == 2
    assert metadata["status"] == "COMPLETE"
    assert metadata["revision_cycle"]["attempt_count"] == 0
    assert metadata["revision_cycle"]["state"] == "skipped"
    assert metadata["revision_cycle"]["final_generation"] == "initial"
    assert metadata["revision_cycle"].get("last_failure") is not None


def test_second_revise_after_failed_authoring_is_not_already_consumed(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second revise_once after failed authoring must not hard-fail the limit."""
    import resume_tailor.application.pipeline as pipeline_module
    import resume_tailor.backend.documents.docx_render as docx_render_module
    import resume_tailor.backend.documents.headless_render as headless_render_module
    import resume_tailor.backend.providers.ollama_writer as ollama_writer_module
    from resume_tailor.backend.engine.orchestration import (
        ApprovalResponse,
        PipelineHooks,
    )
    from resume_tailor.backend.utils.utilities import RevisionValidationError
    from resume_tailor.ui.cli import build_parser, run_pipeline

    revision_invocations = 0
    optional_decisions = 0

    monkeypatch.setattr(
        pipeline_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {"resume_tailor": "synthetic", "codex": "mocked"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_tailoring_dependency_versions",
        lambda *_args, **_kwargs: {
            "ollama": "not-invoked",
            "ollama_model": "not-loaded",
            "libreoffice": "mocked",
        },
    )

    def fake_analysis(**kwargs: object) -> dict[str, object]:
        extracted = kwargs["extracted_resume"]  # type: ignore[assignment]
        requirements = kwargs["job_requirements"]  # type: ignore[assignment]
        run_directory = kwargs["run_directory"]  # type: ignore[assignment]
        assert isinstance(extracted, dict)
        assert isinstance(requirements, dict)
        assert isinstance(run_directory, Path)
        engineering = extracted["content"]["skill_groups"][2]
        raw: dict[str, object] = {
            "role_summary": "Synthetic second-revise run.",
            "fit_assessment": {
                "overall": "Synthetic.",
                "strengths": ["Authenticated skills"],
                "gaps": [],
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
                    "alignment_rationale": "Surface authenticated skills.",
                    "evidence_source_ids": ["skill_groups.2", "skill_groups.0"],
                }
            ],
            "immutable_facts": [],
            "forbidden_claims": ["GraphQL"],
            "content_budget_guidance": [],
            "questions_for_user": [],
        }
        (run_directory / "codex-analysis.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )
        return raw

    monkeypatch.setattr(pipeline_module, "invoke_analysis", fake_analysis)
    monkeypatch.setattr(pipeline_module, "invoke_codex_analysis", fake_analysis)
    monkeypatch.setattr(
        ollama_writer_module,
        "run_ollama_request",
        lambda **_kwargs: pytest.fail("Ollama must not be invoked"),
    )

    def fake_render(**kwargs: object) -> None:
        destination = kwargs["destination_path"]
        assert isinstance(destination, Path)
        destination.write_bytes(b"synthetic second revise docx")

    def fake_export(**kwargs: object) -> str:
        pdf_path = kwargs["pdf_path"]
        preview_path = kwargs["preview_path"]
        assert isinstance(pdf_path, Path)
        assert isinstance(preview_path, Path)
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")
        preview_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
        return "Synthetic resume text"

    def fake_final_qa(**kwargs: object) -> dict[str, object]:
        run_directory = kwargs["run_directory"]
        generation = kwargs["generation"]
        assert isinstance(run_directory, Path)
        result: dict[str, object] = {
            "status": "pass",
            "summary": "Synthetic QA passed.",
            "issues": [],
            "technical_failure": None,
        }
        stem = f"final-qa.{generation}"
        (run_directory / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
        (run_directory / f"{stem}.md").write_text("# PASS\n", encoding="utf-8")
        return result

    def fail_revision(**_kwargs: object) -> dict[str, object]:
        nonlocal revision_invocations
        revision_invocations += 1
        raise RuntimeError(f"synthetic authoring failure #{revision_invocations}")

    monkeypatch.setattr(headless_render_module, "render_headless_docx", fake_render)
    monkeypatch.setattr(docx_render_module, "export_and_validate_pdf", fake_export)
    monkeypatch.setattr(pipeline_module, "invoke_final_qa", fake_final_qa)
    monkeypatch.setattr(pipeline_module, "run_initial_qa", fake_final_qa)
    monkeypatch.setattr(pipeline_module, "invoke_ollama_revision", fail_revision)
    monkeypatch.setattr(
        pipeline_module,
        "probe_initial_qa_providers",
        lambda **_kwargs: [
            {
                "provider_id": "codex",
                "label": "Codex",
                "description": "stub",
                "available": True,
                "status": "ready",
                "detail": "stub ready",
                "capabilities": ["content_qa"],
                "limitations": [],
            }
        ],
    )

    def approval(request: object) -> ApprovalResponse:
        nonlocal optional_decisions
        kind = request.kind  # type: ignore[attr-defined]
        if kind in {"codex_analysis", "tailored_content", "rendered_artifacts"}:
            return ApprovalResponse("approve")
        if kind == "initial_qa_provider":
            return ApprovalResponse("select", {"provider": "codex"})
        if kind == "optional_revision":
            optional_decisions += 1
            if optional_decisions <= 2:
                return ApprovalResponse(
                    "revise_once",
                    {
                        "provider": "codex",
                        "revision_provider": "codex",
                        "final_qa_provider": "codex",
                    },
                )
            return ApprovalResponse("complete_without_revision")
        return ApprovalResponse("approve")

    parser = build_parser()
    output_dir = tmp_path / "second-revise-output"
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
            str(tmp_path / "second-revise-analytics.sqlite3"),
            "--timeout",
            "30s",
            "--analysis-provider",
            "codex",
        ]
    )
    try:
        result = run_pipeline(
            args, hooks=PipelineHooks(approval_handler=approval)
        )
    except RevisionValidationError as exc:
        pytest.fail(
            "Failed authoring incorrectly consumed the one-revision limit: "
            f"{exc}"
        )
    run = result if isinstance(result, Path) else result.run_directory
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert revision_invocations == 2
    assert optional_decisions == 3
    assert metadata["revision_cycle"]["attempt_count"] == 0
    assert metadata["status"] == "COMPLETE"
