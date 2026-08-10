"""Synthetic tests for selectable Initial QA providers (Step 9)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.orchestration import ApprovalRequest, ApprovalResponse, PipelineHooks
from resume_tailor.backend.engine.qa import (
    historical_initial_qa_provider,
    invoke_final_qa,
    normalize_initial_qa_provider,
    probe_initial_qa_providers,
    resolve_qa_payload,
    run_initial_qa,
)
from resume_tailor.backend.utils.utilities import InputError, ModelError


def _pass_payload() -> dict[str, Any]:
    return {
        "status": "pass",
        "summary": "Synthetic Initial QA passed.",
        "issues": [],
        "technical_failure": None,
    }


def _material_payload() -> dict[str, Any]:
    return {
        "status": "material_findings",
        "summary": "Synthetic material findings.",
        "issues": [
            {
                "category": "clarity",
                "severity": "medium",
                "description": "Summary could be clearer.",
                "affected_content_id": "professional_summary",
                "evidence_source_ids": ["professional_summary"],
                "correction_action": "improve_clarity",
                "correction_objective": "Improve clarity without inventing claims.",
            }
        ],
        "technical_failure": None,
    }


def _qa_kwargs(master_resume: Path, tmp_path: Path) -> dict[str, Any]:
    extracted, _ = extract_resume(master_resume)
    preview = tmp_path / "preview.initial.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    return {
        "original_extraction": extracted,
        "job_description": "Synthetic Python validation role.",
        "analysis": {
            "recommended_edits": [],
            "immutable_facts": ["Synthetic degree"],
            "forbidden_claims": [],
        },
        "tailored_pdf_text": "Synthetic rendered text.",
        "content_diff": "# Synthetic diff\n",
        "preview_path": preview,
        "run_directory": tmp_path,
        "work_directory": tmp_path / "work" / "initial-qa",
        "timeout_seconds": 30,
        "generation": "initial",
        "company": "Synthetic Corp",
        "role": "AI Solutions Engineer",
    }


def test_normalize_provider_ids() -> None:
    assert normalize_initial_qa_provider("gemma_local") == "gemma_local"
    assert normalize_initial_qa_provider("grok_cli") == "grok"
    assert normalize_initial_qa_provider("agy") == "antigravity"
    with pytest.raises(InputError):
        normalize_initial_qa_provider("openai")


def test_historical_codex_only_runs_remain_readable() -> None:
    assert historical_initial_qa_provider({}) == "codex"
    assert historical_initial_qa_provider({"final_qa": {"provider": "codex"}}) == "codex"
    assert (
        historical_initial_qa_provider({"initial_qa_provider": "gemma_local"})
        == "gemma_local"
    )


def test_probe_marks_unavailable_providers_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._probe_gemma_local",
        lambda: type(
            "O",
            (),
            {
                "as_dict": lambda self: {
                    "provider_id": "gemma_local",
                    "label": "Gemma Local",
                    "description": "x",
                    "available": False,
                    "status": "ollama_unavailable",
                    "detail": "Ollama down",
                    "capabilities": [],
                    "limitations": ["content_and_structure_only"],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._probe_codex",
        lambda: type(
            "O",
            (),
            {
                "as_dict": lambda self: {
                    "provider_id": "codex",
                    "label": "Codex",
                    "description": "x",
                    "available": True,
                    "status": "ready",
                    "detail": "ok",
                    "capabilities": [],
                    "limitations": [],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._probe_grok",
        lambda: type(
            "O",
            (),
            {
                "as_dict": lambda self: {
                    "provider_id": "grok",
                    "label": "Grok",
                    "description": "x",
                    "available": False,
                    "status": "cli_unavailable",
                    "detail": "missing",
                    "capabilities": [],
                    "limitations": ["content_and_structure_only"],
                }
            },
        )(),
    )
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._probe_antigravity",
        lambda: type(
            "O",
            (),
            {
                "as_dict": lambda self: {
                    "provider_id": "antigravity",
                    "label": "Antigravity",
                    "description": "x",
                    "available": False,
                    "status": "cli_unavailable",
                    "detail": "missing",
                    "capabilities": [],
                    "limitations": ["content_and_structure_only"],
                }
            },
        )(),
    )
    options = probe_initial_qa_providers(include_expensive=True)
    by_id = {item["provider_id"]: item for item in options}
    assert by_id["gemma_local"]["available"] is False
    assert by_id["gemma_local"]["status"] == "ollama_unavailable"
    assert by_id["codex"]["available"] is True
    assert by_id["grok"]["available"] is False


@pytest.mark.parametrize(
    "provider",
    ["gemma_local", "codex", "grok", "antigravity"],
)
def test_each_provider_selection_invokes_only_that_adapter(
    provider: str,
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def track(name: str, payload: dict[str, Any]):
        def _impl(**_kwargs: Any) -> dict[str, Any]:
            calls.append(name)
            return payload

        return _impl

    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_codex_qa", track("codex", _pass_payload()))
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_gemma_qa", track("gemma_local", _pass_payload())
    )
    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_grok_qa", track("grok", _pass_payload()))
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_antigravity_qa",
        track("antigravity", _pass_payload()),
    )

    result = run_initial_qa(provider=provider, **_qa_kwargs(master_resume, tmp_path))
    assert result["status"] == "pass"
    assert calls == [provider]
    meta = json.loads(
        (tmp_path / "initial-qa-result.json").read_text(encoding="utf-8")
    )
    assert meta["provider"] == provider
    if provider != "codex":
        assert "content_and_structure_only" in meta["limitations"]
    else:
        assert meta["limitations"] == []


def test_malformed_provider_output_rejected_locally(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_codex_qa",
        lambda **_k: {"status": "pass", "summary": "x", "issues": [{"bad": True}]},
    )
    with pytest.raises(ModelError):
        run_initial_qa(provider="codex", **_qa_kwargs(master_resume, tmp_path))


def test_canonical_schema_shared_across_providers(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    extracted, _ = extract_resume(master_resume)
    resolved = resolve_qa_payload(
        _material_payload(),
        original_extraction=extracted,
    )
    assert resolved["issues"][0]["issue_id"] == "qa.001"
    assert resolved["status"] == "material_findings"


def test_selection_hook_does_not_launch_provider_before_confirm() -> None:
    launched: list[str] = []

    def handler(request: ApprovalRequest) -> ApprovalResponse:
        assert request.kind == "initial_qa_provider"
        assert launched == []
        return ApprovalResponse("select", {"provider": "codex"})

    hooks = PipelineHooks(approval_handler=handler)
    response = hooks.select_initial_qa_provider(
        options=[
            {
                "provider_id": "codex",
                "available": True,
                "label": "Codex",
                "status": "ready",
            }
        ],
        default_provider="gemma_local",
        assume_yes=False,
    )
    assert response.action == "select"
    assert response.data["provider"] == "codex"
    assert launched == []


def test_assume_yes_confirms_preselection_without_fallback() -> None:
    hooks = PipelineHooks()
    response = hooks.select_initial_qa_provider(
        options=[
            {"provider_id": "gemma_local", "available": True},
            {"provider_id": "codex", "available": True},
        ],
        default_provider="gemma_local",
        assume_yes=True,
    )
    assert response.action == "select"
    assert response.data["provider"] == "gemma_local"


def test_no_silent_fallback_on_provider_exception(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_gemma_qa",
        lambda **_k: (_ for _ in ()).throw(ModelError("gemma failed")),
    )
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_codex_qa",
        lambda **_k: (_ for _ in ()).throw(AssertionError("codex must not run")),
    )
    with pytest.raises(ModelError, match="gemma failed"):
        run_initial_qa(provider="gemma_local", **_qa_kwargs(master_resume, tmp_path))


def test_packet_excludes_secrets(master_resume: Path, tmp_path: Path) -> None:
    from resume_tailor.backend.engine.qa import build_initial_qa_packet

    extracted, _ = extract_resume(master_resume)
    packet = build_initial_qa_packet(
        original_extraction=extracted,
        job_description="Job text",
        analysis={"recommended_edits": [], "immutable_facts": [], "forbidden_claims": []},
        tailored_content=extracted["content"],
        tailored_pdf_text="text",
        content_diff="diff",
        company="Corp",
        role="Role",
        job_requirements={"requirements": []},
        evidence_report={"passed": True},
        docx_path=tmp_path / "out.docx",
        pdf_path=tmp_path / "out.pdf",
        preview_path=tmp_path / "out.png",
        provider="gemma_local",
        generation="initial",
    )
    serialized = json.dumps(packet)
    for banned in ("API_TOKEN", "OPENAI_API_KEY", "password", "Bearer "):
        assert banned not in serialized
    assert "prompt" in packet
    assert packet["limitations"] == ["content_and_structure_only"]


def test_step10_receives_provider_neutral_result(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_gemma_qa",
        lambda **_k: _material_payload(),
    )
    result = run_initial_qa(provider="gemma_local", **_qa_kwargs(master_resume, tmp_path))
    # Same shape revision authorization and revision targets already consume.
    assert set(result) >= {"status", "summary", "issues", "technical_failure"}
    assert result["status"] == "material_findings"
    assert result["issues"][0]["issue_id"].startswith("qa.")


def test_gemma_records_visual_limitation(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_gemma_qa",
        lambda **_k: _pass_payload(),
    )
    run_initial_qa(provider="gemma_local", **_qa_kwargs(master_resume, tmp_path))
    diagnostic = json.loads(
        (tmp_path / "initial-qa-diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["capabilities"]["visual_layout_qa"] is False
    assert "content_and_structure_only" in diagnostic["limitations"]


def test_invoke_final_qa_compat_defaults_to_codex(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_codex(**_k: Any) -> dict[str, Any]:
        calls.append("codex")
        return _pass_payload()

    monkeypatch.setattr("resume_tailor.backend.engine.qa._invoke_codex_qa", fake_codex)
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._invoke_gemma_qa",
        lambda **_k: (_ for _ in ()).throw(AssertionError("gemma")),
    )
    kwargs = _qa_kwargs(master_resume, tmp_path)
    # Drop keys not accepted by invoke_final_qa's public kwargs for this call.
    result = invoke_final_qa(
        original_extraction=kwargs["original_extraction"],
        job_description=kwargs["job_description"],
        analysis=kwargs["analysis"],
        tailored_pdf_text=kwargs["tailored_pdf_text"],
        content_diff=kwargs["content_diff"],
        preview_path=kwargs["preview_path"],
        run_directory=kwargs["run_directory"],
        work_directory=kwargs["work_directory"],
        timeout_seconds=30,
        generation="initial",
    )
    assert result["status"] == "pass"
    assert calls == ["codex"]


def test_external_cli_probe_does_not_claim_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa.shutil.which",
        lambda name: f"/synthetic/bin/{name}" if name in {"codex", "agy"} else None,
    )
    monkeypatch.setattr(
        "resume_tailor.backend.providers.grok_analysis.resolve_grok_executable",
        lambda _value: "/synthetic/bin/grok",
    )
    monkeypatch.setattr(
        "resume_tailor.backend.engine.qa._probe_gemma_local",
        lambda: __import__(
            "resume_tailor.backend.engine.qa", fromlist=["InitialQAProviderOption"]
        ).InitialQAProviderOption(
            provider_id="gemma_local",
            label="Gemma Local",
            description="local",
            available=True,
            status="local_model_present",
            detail="present",
            capabilities=("content_qa",),
            limitations=("content_and_structure_only",),
            verification="local_only",
            auth_status="not_applicable",
        ),
    )
    options = probe_initial_qa_providers(include_expensive=True)
    by_id = {item["provider_id"]: item for item in options}
    assert by_id["codex"]["available"] is True
    assert by_id["codex"]["status"] == "cli_found"
    assert by_id["codex"]["auth_status"] == "not_checked"
    assert "Ready" not in by_id["codex"]["ui_status_label"]
    assert "not verified" in by_id["codex"]["ui_status_label"].casefold()
    assert by_id["grok"]["status"] == "cli_found"
    assert "not verified" in by_id["grok"]["ui_status_label"].casefold()
    assert by_id["gemma_local"]["status"] == "local_model_present"
    assert "Local model present" in by_id["gemma_local"]["ui_status_label"]


def test_grok_qa_classifies_e2big_as_prompt_too_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno

    from resume_tailor.backend.engine import qa as qa_module
    from resume_tailor.backend.utils.utilities import (
        DependencyError,
        GrokPromptTooLargeError,
    )

    monkeypatch.setattr(
        "resume_tailor.backend.providers.grok_analysis.resolve_grok_executable",
        lambda _value: "/synthetic/bin/grok",
    )
    monkeypatch.setattr(
        "resume_tailor.backend.providers.grok_analysis.grok_analysis_args",
        lambda **_kwargs: ["/synthetic/bin/grok", "-p", "x"],
    )

    def boom(*_args: object, **_kwargs: object) -> object:
        cause = OSError(errno.E2BIG, "Argument list too long")
        raise DependencyError("Could not run grok: Argument list too long") from cause

    monkeypatch.setattr(qa_module, "run_command", boom)
    monkeypatch.setattr(
        qa_module,
        "_load_provider_schema",
        lambda: {"type": "object"},
    )
    with pytest.raises(GrokPromptTooLargeError):
        qa_module._invoke_grok_qa(
            prompt="synthetic",
            run_directory=tmp_path,
            timeout_seconds=5,
        )
