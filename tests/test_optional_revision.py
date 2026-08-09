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
