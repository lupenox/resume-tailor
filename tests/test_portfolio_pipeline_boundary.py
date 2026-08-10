from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import resume_tailor.application.pipeline as pipeline_module
import resume_tailor.application.portfolio as portfolio_module
from resume_tailor.application.models import PipelineRequest
from resume_tailor.application.pipeline import (
    _bind_portfolio_evidence_targets,
    _quarantine_unapproved_portfolio_artifacts,
    _restore_quarantined_portfolio_artifacts,
)
from resume_tailor.backend.engine.retry import (
    analysis_input_manifest,
    build_retry_context,
)
from resume_tailor.backend.utils.utilities import InputError


def _write_analysis_inputs(directory: Path, *, selection: bool) -> None:
    directory.mkdir()
    (directory / "extracted-master-resume.json").write_text(
        '{"source":"synthetic"}\n', encoding="utf-8"
    )
    (directory / "job-description.txt").write_text(
        "Synthetic job\n", encoding="utf-8"
    )
    (directory / "job-requirements.json").write_text(
        '{"requirements":[]}\n', encoding="utf-8"
    )
    if selection:
        (directory / "github-repository-selection.json").write_text(
            '{"decision":"approved"}\n', encoding="utf-8"
        )


def test_analysis_manifest_binds_optional_portfolio_selection(tmp_path: Path) -> None:
    without_selection = tmp_path / "without"
    _write_analysis_inputs(without_selection, selection=False)
    historical = analysis_input_manifest(
        without_selection,
        source_resume_sha256="a" * 64,
    )
    assert historical["version"] == 2
    assert "github_portfolio_selection_sha256" not in historical

    with_selection = tmp_path / "with"
    _write_analysis_inputs(with_selection, selection=True)
    bound = analysis_input_manifest(
        with_selection,
        source_resume_sha256="a" * 64,
    )
    assert bound["version"] == 3
    assert bound["github_portfolio_selection_sha256"] == hashlib.sha256(
        (with_selection / "github-repository-selection.json").read_bytes()
    ).hexdigest()


def test_portfolio_evidence_can_target_only_summary_skills_and_matching_project() -> None:
    extracted = {
        "content": {
            "projects": [
                {"name": "Resume Tailor"},
                {"name": "Different Project"},
            ]
        },
        "source_blocks": [
            {
                "source_id": "professional_summary",
                "editable": True,
            },
            {"source_id": "skill_groups.0", "editable": True},
            {"source_id": "projects.0.bullets.0", "editable": True},
            {"source_id": "projects.1.bullets.0", "editable": True},
            {"source_id": "experience.bullets.0", "editable": True},
        ],
    }
    supplied = [
        {
            "source_id": "github.101.readme",
            "source_kind": "github_repository",
            "repository_full_name": "synthetic/resume-tailor",
            "exact_text": "Evidence",
        },
        {
            "source_id": "github.202.readme",
            "source_kind": "github_repository",
            "repository_full_name": "synthetic/unlisted-project",
            "exact_text": "Other evidence",
        },
    ]

    bound = _bind_portfolio_evidence_targets(extracted, supplied)

    assert bound[0]["allowed_target_source_ids"] == [
        "professional_summary",
        "projects.0.bullets.0",
        "skill_groups.0",
    ]
    assert bound[1]["allowed_target_source_ids"] == [
        "professional_summary",
        "skill_groups.0",
    ]
    assert all(item["editable"] is False for item in bound)
    assert all(item["evidence_allowed"] is True for item in bound)


def test_authenticated_recovery_fails_closed_for_portfolio_run(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "github-repository-selection.json").write_text(
        '{"decision":"approved"}\n', encoding="utf-8"
    )

    with pytest.raises(InputError, match="portfolio run"):
        build_retry_context(
            run_directory,
            current_resume=tmp_path / "unused.docx",
        )


@pytest.mark.parametrize(
    ("analysis_provider", "writer_provider", "message"),
    [
        ("codex", "ollama", "tool-free résumé analysis provider"),
        ("antigravity", "ollama", "tool-free résumé analysis provider"),
        ("gemma_local", "antigravity", "local Ollama writer"),
    ],
)
def test_pipeline_rejects_tool_capable_portfolio_providers_before_cataloging(
    analysis_provider: str,
    writer_provider: str,
    message: str,
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portfolio_module,
        "run_portfolio_selection",
        lambda **_kwargs: pytest.fail("portfolio cataloging must not start"),
    )
    request = PipelineRequest(
        resume=master_resume,
        job_file=job_file,
        company="Synthetic Systems",
        role="Evidence Engineer",
        output_dir=tmp_path / "output",
        analytics_db=tmp_path / "analytics.sqlite3",
        analysis_provider=analysis_provider,
        writer_provider=writer_provider,
        github_portfolio=True,
        github_username="synthetic",
    )

    with pytest.raises(InputError, match=message):
        pipeline_module.run_pipeline(request)

    assert not request.output_dir.exists()


def test_unapproved_portfolio_artifacts_are_hidden_then_restored_exactly(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    catalog = b'{"repositories":[{"evidence":"unapproved"}]}\n'
    ranking = b'{"ranked_repositories":[{"rationale":"unapproved"}]}\n'
    selection = b'{"decision":"approved","repositories":[]}\n'
    (run_directory / "github-repository-catalog.json").write_bytes(catalog)
    (run_directory / "github-repository-ranking.json").write_bytes(ranking)
    (run_directory / "github-repository-selection.json").write_bytes(selection)

    held = _quarantine_unapproved_portfolio_artifacts(run_directory)

    assert not (run_directory / "github-repository-catalog.json").exists()
    assert not (run_directory / "github-repository-ranking.json").exists()
    assert (run_directory / "github-repository-selection.json").read_bytes() == selection

    _restore_quarantined_portfolio_artifacts(run_directory, held)

    assert (run_directory / "github-repository-catalog.json").read_bytes() == catalog
    assert (run_directory / "github-repository-ranking.json").read_bytes() == ranking
    assert held == {}


@pytest.mark.parametrize(
    ("decision", "selected_visibility", "expected_message"),
    [
        ("approved", "public", "synthetic stop after portfolio handoff"),
        ("skipped", None, "synthetic stop after portfolio handoff"),
        ("approved", "private", "approved private GitHub selection is local-only"),
    ],
)
def test_pipeline_handoff_uses_only_approved_evidence_and_preserves_skip_privacy(
    decision: str,
    selected_visibility: str | None,
    expected_message: str,
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / f"output-{decision}-{selected_visibility}"
    selection_repository = (
        SimpleNamespace(
            repository_id="101",
            visibility=selected_visibility,
        )
        if selected_visibility is not None
        else None
    )
    selection = SimpleNamespace(
        decision=decision,
        repositories=(selection_repository,) if selection_repository else (),
    )
    # The mixed catalog intentionally includes unselected private content. It
    # must not force local providers when the user skips or approves only public
    # repositories, and it must never remain visible during downstream calls.
    catalog = SimpleNamespace(
        repositories=(
            SimpleNamespace(repository_id="101", visibility="public"),
            SimpleNamespace(repository_id="202", visibility="private"),
        )
    )
    artifacts = {
        "github-repository-catalog.json": {
            "repositories": [
                {"repository_id": "101", "evidence": "public candidate"},
                {"repository_id": "202", "evidence": "PRIVATE_SENTINEL"},
            ]
        },
        "github-repository-ranking.json": {
            "ranked_repositories": [{"repository_id": "101"}]
        },
        "github-repository-selection.json": {
            "decision": decision,
            "selected_repository_ids": (
                ["101"] if selection_repository is not None else []
            ),
        },
    }

    def fake_portfolio_service(**kwargs: object) -> SimpleNamespace:
        run_directory = kwargs["run_directory"]
        assert isinstance(run_directory, Path)
        assert (run_directory / "job-requirements.json").is_file()
        for filename, document in artifacts.items():
            (run_directory / filename).write_text(
                json.dumps(document) + "\n",
                encoding="utf-8",
            )
        return SimpleNamespace(
            skipped=decision == "skipped",
            catalog=catalog,
            ranking=SimpleNamespace(),
            selection=selection,
        )

    monkeypatch.setattr(
        portfolio_module,
        "run_portfolio_selection",
        fake_portfolio_service,
    )
    monkeypatch.setattr(
        portfolio_module,
        "approved_portfolio_source_blocks",
        lambda _selection: [
            {
                "source_id": "github.101.readme.synthetic",
                "section_context": "Approved GitHub repository: synthetic/resume-tailor",
                "block_kind": "repository_evidence",
                "exact_text": "FastAPI and pytest are present in this repository.",
                "evidence_allowed": True,
                "editable": False,
                "source_kind": "github_repository",
                "repository_id": "101",
                "repository_full_name": "synthetic/resume-tailor",
                "head_sha": "1" * 40,
                "source_path": "README.md",
                "source_url": (
                    "https://github.com/synthetic/resume-tailor/blob/"
                    + "1" * 40
                    + "/README.md"
                ),
                "content_sha256": "2" * 64,
            }
        ],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {"resume_tailor": "synthetic", "codex": "synthetic"},
    )

    analysis_called = False

    def stop_at_analysis(**kwargs: object) -> dict[str, object]:
        nonlocal analysis_called
        analysis_called = True
        run_directory = kwargs["run_directory"]
        extracted = kwargs["extracted_resume"]
        assert isinstance(run_directory, Path)
        assert isinstance(extracted, dict)
        assert kwargs["restrict_external_tools"] is True
        assert not (run_directory / "github-repository-catalog.json").exists()
        assert not (run_directory / "github-repository-ranking.json").exists()
        assert (run_directory / "github-repository-selection.json").is_file()
        github_blocks = [
            block
            for block in extracted["source_blocks"]
            if block.get("source_kind") == "github_repository"
        ]
        if decision == "approved":
            assert len(github_blocks) == 1
            assert github_blocks[0]["source_id"] == "github.101.readme.synthetic"
            assert github_blocks[0]["allowed_target_source_ids"]
        else:
            assert github_blocks == []
        raise InputError("synthetic stop after portfolio handoff")

    monkeypatch.setattr(pipeline_module, "invoke_analysis", stop_at_analysis)

    request = PipelineRequest(
        resume=master_resume,
        job_file=job_file,
        company="Synthetic Systems",
        role="Evidence Engineer",
        output_dir=output_directory,
        analytics_db=tmp_path / "analytics.sqlite3",
        yes=True,
        timeout=(30, "30s"),
        writer_provider="ollama",
        analysis_provider="grok_cli",
        github_portfolio=True,
        github_username="synthetic",
    )

    with pytest.raises(InputError, match=expected_message):
        pipeline_module.run_pipeline(request)

    run_directory = next(output_directory.iterdir())
    assert (run_directory / "github-repository-catalog.json").is_file()
    assert (run_directory / "github-repository-ranking.json").is_file()
    if selected_visibility == "private":
        assert analysis_called is False
        assert not (run_directory / "extracted-master-resume.json").exists()
    else:
        assert analysis_called is True
        extracted = json.loads(
            (run_directory / "extracted-master-resume.json").read_text(
                encoding="utf-8"
            )
        )
        github_blocks = [
            block
            for block in extracted["source_blocks"]
            if block.get("source_kind") == "github_repository"
        ]
        assert (len(github_blocks) == 1) is (decision == "approved")
        metadata = json.loads(
            (run_directory / "run-metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["analysis_inputs"]["version"] == 3


def test_pipeline_metadata_redacts_github_credentials_from_failures(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "github_pat_SYNTHETIC_NON_PATTERN_CONTEXT_123456"

    def fail_portfolio(**_kwargs: object) -> None:
        raise InputError(
            f"GITHUB_TOKEN={secret} Authorization: Bearer {secret}"
        )

    monkeypatch.setattr(
        portfolio_module,
        "run_portfolio_selection",
        fail_portfolio,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_analysis_dependency_versions",
        lambda *_args, **_kwargs: {"resume_tailor": "synthetic", "codex": "synthetic"},
    )
    output_directory = tmp_path / "redacted-output"
    request = PipelineRequest(
        resume=master_resume,
        job_file=job_file,
        company="Synthetic Systems",
        role="Evidence Engineer",
        output_dir=output_directory,
        analytics_db=tmp_path / "redacted-analytics.sqlite3",
        yes=True,
        timeout=(30, "30s"),
        analysis_provider="gemma_local",
        writer_provider="ollama",
        github_portfolio=True,
        github_username="synthetic",
    )

    with pytest.raises(InputError):
        pipeline_module.run_pipeline(request)

    run_directory = next(output_directory.iterdir())
    serialized = (run_directory / "run-metadata.json").read_text(encoding="utf-8")
    assert secret not in serialized
    assert "[credential omitted]" in serialized
