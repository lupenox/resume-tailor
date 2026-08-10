from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

import resume_tailor.backend.engine.qa as qa_module
import resume_tailor.backend.engine.analysis as analysis_module
import resume_tailor.backend.engine.revision as revision_module
import resume_tailor.backend.providers.antigravity_analysis as agy_analysis_module
import resume_tailor.backend.providers.antigravity_transport as agy_transport_module
import resume_tailor.backend.providers.antigravity_writer as agy_writer_module
import resume_tailor.backend.providers.codex_analysis as codex_module
import resume_tailor.backend.providers.grok_analysis as grok_module
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.providers.antigravity_response import (
    AntigravityResponseCandidate,
)
from resume_tailor.backend.utils.schemas import schema_path
from resume_tailor.backend.utils.utilities import CommandResult, ModelError


def _analysis_payload(requirements: dict[str, Any]) -> dict[str, Any]:
    requirement_ids = [
        item["requirement_id"] for item in requirements["requirements"]
    ]
    return {
        "role_summary": "Synthetic evidence-backed role.",
        "fit_assessment": {
            "overall": "Supported fit.",
            "strengths": ["Python"],
            "gaps": [],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": requirement_ids[0],
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": requirement_ids[1:],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": [],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


def _requirements() -> dict[str, Any]:
    return build_job_requirement_catalog(
        "Skills: Python and synthetic orbital telemetry.",
        structured_job={
            "technologies_and_skills": ["Python", "Synthetic orbital telemetry"]
        },
    )


def _assert_private_workspace(workspace: Path, run_directory: Path) -> None:
    assert workspace.is_dir()
    assert workspace.resolve() != run_directory.resolve()
    assert run_directory.resolve() not in workspace.resolve().parents
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700


def _success_result(args: list[str], *, stdout: str = "") -> CommandResult:
    return CommandResult(tuple(args), stdout, "", 0)


def test_restricted_analysis_rejects_tool_capable_adapter_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = False

    def forbidden_launch(**_kwargs: Any) -> dict[str, Any]:
        nonlocal launched
        launched = True
        return {}

    monkeypatch.setattr(analysis_module, "invoke_codex_analysis", forbidden_launch)
    with pytest.raises(ModelError, match="cannot hard-disable shell and network tools"):
        analysis_module.invoke_analysis(
            provider="codex",
            extracted_resume={},
            job_description="Synthetic job.",
            job_requirements={},
            company="Synthetic",
            role="Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            restrict_external_tools=True,
        )
    assert launched is False


def test_restricted_qa_rejects_tool_capable_adapter_before_packet_write(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelError, match="cannot hard-disable shell and network tools"):
        qa_module.run_initial_qa(
            provider="antigravity",
            original_extraction={},
            job_description="Synthetic job.",
            analysis={},
            tailored_pdf_text="Synthetic PDF.",
            content_diff="Synthetic diff.",
            preview_path=tmp_path / "missing-preview.png",
            run_directory=tmp_path,
            work_directory=tmp_path / "work",
            timeout_seconds=30,
            restrict_external_tools=True,
        )
    assert not (tmp_path / qa_module.INITIAL_QA_REQUEST_FILENAME).exists()


def test_restricted_writing_rejects_tool_capable_adapters_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agy_writer_module,
        "require_executable",
        lambda _name: pytest.fail("writer adapter was launched"),
    )
    with pytest.raises(ModelError, match="cannot hard-disable shell and network tools"):
        agy_writer_module.invoke_antigravity(
            master_content={},
            extracted_resume={},
            job_description="Synthetic job.",
            job_requirements={},
            approved_analysis={},
            company="Synthetic",
            role="Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            antigravity_duration="30s",
            restrict_external_tools=True,
        )
    with pytest.raises(ModelError, match="cannot hard-disable shell and network tools"):
        revision_module.invoke_antigravity_revision(
            current_tailored_content={},
            extracted_resume={},
            approved_analysis={},
            qa_result={},
            company="Synthetic",
            role="Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            antigravity_duration="30s",
            attempt_number=1,
            restrict_external_tools=True,
        )


def test_codex_analysis_feature_off_preserves_legacy_workspace_and_output(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _requirements()
    payload = _analysis_payload(requirements)
    raw_output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    def fake_run(args: list[str], **kwargs: Any) -> CommandResult:
        assert Path(kwargs["cwd"]) == tmp_path
        assert args[args.index("--cd") + 1] == str(tmp_path)
        assert "--ignore-user-config" not in args
        schema = Path(args[args.index("--output-schema") + 1])
        output = Path(args[args.index("--output-last-message") + 1])
        assert schema.parent == tmp_path and schema.is_file()
        assert output == tmp_path / "codex-analysis.json"
        assert "env" not in kwargs
        output.write_text(raw_output, encoding="utf-8")
        return _success_result(args)

    monkeypatch.setattr(codex_module, "run_command", fake_run)
    result = codex_module.invoke_codex_analysis(
        extracted_resume=extracted,
        job_description="Synthetic Python role.",
        job_requirements=requirements,
        company="Synthetic",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable="codex",
    )

    assert result == payload
    assert (tmp_path / "codex-analysis.json").read_text(encoding="utf-8") == raw_output


def test_grok_analysis_is_one_turn_deny_all_in_private_workspace(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _requirements()
    payload = _analysis_payload(requirements)
    observed_workspace: Path | None = None
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_SYNTHETIC_NOT_REAL")
    monkeypatch.setattr(
        grok_module,
        "resolve_grok_executable",
        lambda _value=None: "/synthetic/grok",
    )

    def fake_run(args: list[str], **kwargs: Any) -> CommandResult:
        nonlocal observed_workspace
        workspace = Path(kwargs["cwd"])
        observed_workspace = workspace
        _assert_private_workspace(workspace, tmp_path)
        assert args[args.index("--cwd") + 1] == str(workspace)
        assert json.loads(args[args.index("--json-schema") + 1])["type"] == "object"
        assert args[args.index("--max-turns") + 1] == "1"
        assert args[args.index("--deny") + 1] == "*"
        assert args[args.index("--sandbox") + 1] == "strict"
        assert args[args.index("--reasoning-effort") + 1] == "high"
        assert "--model-strength" not in args
        assert {
            "--disable-web-search",
            "--no-subagents",
            "--no-memory",
            "--no-plan",
        }.issubset(args)
        assert kwargs["env"].get("GITHUB_TOKEN") is None
        envelope = {
            "text": json.dumps(payload),
            "stopReason": "end_turn",
        }
        return _success_result(args, stdout=json.dumps(envelope))

    monkeypatch.setattr(grok_module, "run_command", fake_run)
    result = grok_module.invoke_grok_analysis(
        extracted_resume=extracted,
        job_description="Synthetic Python role.",
        job_requirements=requirements,
        company="Synthetic",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable="/synthetic/grok",
        model="synthetic-model",
        model_strength="high",
        restricted=True,
    )

    assert result == payload
    # The locked Grok path uses its supported reasoning-effort spelling.
    # The historical public grok_analysis_args helper remains unchanged.
    assert observed_workspace is not None and not observed_workspace.exists()


def test_antigravity_analysis_feature_off_preserves_legacy_workspace(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    requirements = _requirements()
    payload = _analysis_payload(requirements)
    def fake_run(**kwargs: Any) -> CommandResult:
        workspace = Path(kwargs["cwd"])
        assert workspace == tmp_path
        schema = Path(kwargs["schema"])
        assert schema.parent == tmp_path and schema.is_file()
        return _success_result(["agy"], stdout="synthetic raw stream\n")

    monkeypatch.setattr(agy_analysis_module, "run_antigravity_prompt", fake_run)
    monkeypatch.setattr(agy_analysis_module, "parse_stream_json_events", lambda _v: [])
    monkeypatch.setattr(
        agy_analysis_module,
        "locate_stream_json_terminal",
        lambda _v: ({"synthetic": True}, "result"),
    )
    monkeypatch.setattr(
        agy_analysis_module,
        "locate_json_tailoring_candidate",
        lambda *_a, **_k: AntigravityResponseCandidate(payload, "json"),
    )

    result = agy_analysis_module.invoke_antigravity_analysis(
        extracted_resume=extracted,
        job_description="Synthetic Python role.",
        job_requirements=requirements,
        company="Synthetic",
        role="Developer",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable="agy",
    )

    assert result == payload
    assert (tmp_path / "antigravity-analysis-raw.txt").read_text() == (
        "synthetic raw stream\n"
    )


def test_antigravity_writer_feature_off_preserves_legacy_workspace_and_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"synthetic": True}
    def fake_run(**kwargs: Any) -> CommandResult:
        workspace = Path(kwargs["cwd"])
        assert workspace == tmp_path
        schema = Path(kwargs["schema"])
        assert schema == schema_path("tailored_resume.schema.json")
        return _success_result(["agy"], stdout="synthetic writer stream\n")

    monkeypatch.setattr(agy_writer_module, "run_antigravity_prompt", fake_run)
    monkeypatch.setattr(agy_writer_module, "parse_stream_json_events", lambda _v: [])
    monkeypatch.setattr(
        agy_writer_module,
        "locate_stream_json_terminal",
        lambda _v: ({"synthetic": True}, "result"),
    )
    monkeypatch.setattr(
        agy_writer_module,
        "locate_json_tailoring_candidate",
        lambda *_a, **_k: AntigravityResponseCandidate(payload, "json"),
    )

    candidate, response_path = agy_writer_module._invoke_antigravity_candidate(
        executable="agy",
        prompt="synthetic prompt",
        prompt_label="synthetic writer prompt",
        schema_name="tailored_resume.schema.json",
        response_filename="antigravity-response.json",
        metadata_filename="antigravity-response-envelope.json",
        run_directory=tmp_path,
        timeout_seconds=30,
        antigravity_duration="30s",
    )

    assert candidate.payload == payload
    assert response_path.read_text() == "synthetic writer stream\n"


def test_codex_qa_feature_off_preserves_legacy_paths_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"synthetic preview")
    work = tmp_path / "work"
    payload = {
        "status": "pass",
        "summary": "Synthetic QA passed.",
        "issues": [],
        "technical_failure": None,
    }
    raw_output = json.dumps(payload) + "\n"
    def fake_run(args: list[str], **kwargs: Any) -> CommandResult:
        assert Path(kwargs["cwd"]) == tmp_path
        assert args[args.index("--cd") + 1] == str(tmp_path)
        assert Path(args[args.index("--image") + 1]) == preview
        assert Path(args[args.index("--output-schema") + 1]).is_file()
        output = Path(args[args.index("--output-last-message") + 1])
        assert output == work / "final-qa.initial.provider.json"
        assert "env" not in kwargs
        output.write_text(raw_output, encoding="utf-8")
        return _success_result(args)

    monkeypatch.setattr(qa_module, "run_command", fake_run)
    result = qa_module._invoke_codex_qa(
        prompt="Synthetic final QA.",
        preview_path=preview,
        run_directory=tmp_path,
        work_directory=work,
        timeout_seconds=30,
        generation="initial",
        executable="codex",
    )

    assert result == payload
    assert (work / "final-qa.initial.provider.json").read_text() == raw_output


def test_grok_qa_is_one_turn_deny_all_in_private_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "pass",
        "summary": "Synthetic QA passed.",
        "issues": [],
        "technical_failure": None,
    }
    observed_workspace: Path | None = None
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_SYNTHETIC_NOT_REAL")

    def fake_run(args: list[str], **kwargs: Any) -> CommandResult:
        nonlocal observed_workspace
        workspace = Path(kwargs["cwd"])
        observed_workspace = workspace
        _assert_private_workspace(workspace, tmp_path)
        assert args[args.index("--cwd") + 1] == str(workspace)
        assert args[args.index("--max-turns") + 1] == "1"
        assert args[args.index("--deny") + 1] == "*"
        assert json.loads(args[args.index("--json-schema") + 1])["type"] == "object"
        assert kwargs["env"].get("GITHUB_TOKEN") is None
        envelope = {"text": json.dumps(payload), "stopReason": "end_turn"}
        return _success_result(args, stdout=json.dumps(envelope))

    monkeypatch.setattr(qa_module, "run_command", fake_run)
    monkeypatch.setattr(grok_module, "resolve_grok_executable", lambda _v=None: "grok")
    result = qa_module._invoke_grok_qa(
        prompt="Synthetic final QA.",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable="grok",
        restricted=True,
    )

    assert result == payload
    assert observed_workspace is not None and not observed_workspace.exists()


def test_antigravity_qa_feature_off_preserves_legacy_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "pass",
        "summary": "Synthetic QA passed.",
        "issues": [],
        "technical_failure": None,
    }
    def fake_run(args: list[str], **kwargs: Any) -> CommandResult:
        workspace = Path(kwargs["cwd"])
        assert workspace == tmp_path
        schema = Path(args[args.index("--json-schema") + 1])
        assert schema.is_file()
        assert "env" not in kwargs
        return _success_result(args, stdout=json.dumps(payload))

    monkeypatch.setattr(agy_transport_module, "run_command", fake_run)
    result = qa_module._invoke_antigravity_qa(
        prompt="Synthetic final QA.",
        run_directory=tmp_path,
        timeout_seconds=30,
        executable="agy",
    )

    assert result == payload
