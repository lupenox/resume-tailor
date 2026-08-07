from __future__ import annotations

import json
from pathlib import Path

import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.qa import invoke_final_qa
from resume_tailor.backend.utils.utilities import ModelError


def _invoke(
    *,
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    generation: str = "initial",
) -> dict:
    extracted, _ = extract_resume(master_resume)
    preview = tmp_path / f"preview.{generation}.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    return invoke_final_qa(
        original_extraction=extracted,
        job_description="Synthetic Python validation role.",
        analysis={"recommended_edits": []},
        tailored_pdf_text="Synthetic rendered text.",
        content_diff="# Synthetic diff\n",
        preview_path=preview,
        run_directory=tmp_path,
        work_directory=tmp_path / "work" / generation,
        timeout_seconds=30,
        generation=generation,
        executable=str(stubs_on_path / "codex"),
    )


def test_final_qa_pass_uses_fresh_read_only_session(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "codex-invocations.jsonl"
    monkeypatch.setenv("STUB_CODEX_INVOCATION_LOG", str(log))

    result = _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
    )

    assert result == {
        "status": "pass",
        "summary": "Stubbed read-only QA passed.",
        "issues": [],
        "technical_failure": None,
    }
    invocation = json.loads(log.read_text(encoding="utf-8"))
    assert invocation["role"] == "final_qa"
    assert invocation["generation"] == "initial"
    assert invocation["ephemeral"] is True
    assert invocation["sandbox"] == "read-only"
    assert (tmp_path / "final-qa.initial.json").is_file()


def test_material_findings_receive_local_issue_ids(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_QA_INITIAL_MODE", "qa_fail")

    result = _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
    )

    assert result["status"] == "material_findings"
    assert [issue["issue_id"] for issue in result["issues"]] == ["qa.001"]
    assert result["issues"][0]["affected_content_id"] == "professional_summary"
    assert "replacement" not in result["issues"][0]


def test_technical_failure_is_a_bounded_structured_outcome(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "STUB_CODEX_QA_INITIAL_MODE",
        "qa_technical_failure",
    )

    result = _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
    )

    assert result["status"] == "technical_failure"
    assert result["issues"] == []
    assert result["technical_failure"]["reason_code"] == "image_unavailable"


@pytest.mark.parametrize(
    "mode",
    [
        "qa_bad_json",
        "qa_unknown_category",
        "qa_unknown_content_id",
        "qa_replacement_text",
    ],
)
def test_malformed_unsupported_or_rewrite_qa_is_rejected(
    mode: str,
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_QA_INITIAL_MODE", mode)

    with pytest.raises(ModelError):
        _invoke(
            master_resume=master_resume,
            tmp_path=tmp_path,
            stubs_on_path=stubs_on_path,
        )


def test_two_qa_invocations_are_distinct_ephemeral_sessions(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "codex-invocations.jsonl"
    monkeypatch.setenv("STUB_CODEX_INVOCATION_LOG", str(log))

    _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
        generation="initial",
    )
    _invoke(
        master_resume=master_resume,
        tmp_path=tmp_path,
        stubs_on_path=stubs_on_path,
        generation="revision-1",
    )

    invocations = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["generation"] for item in invocations] == [
        "initial",
        "revision-1",
    ]
    assert all(item["ephemeral"] for item in invocations)
    assert all(item["sandbox"] == "read-only" for item in invocations)
