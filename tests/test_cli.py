from __future__ import annotations

import argparse
import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

import resume_tailor.application.pipeline as application_pipeline
from resume_tailor.application.models import PipelineRequest, PipelineResult
from resume_tailor.backend.engine.orchestration import PipelineHooks
from resume_tailor.ui.cli import (
    _validate_mode_arguments,
    build_parser,
    main,
    pipeline_request_from_namespace,
    run_pipeline,
)
from resume_tailor.backend.jobs.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
from resume_tailor.backend.utils.utilities import (
    ApprovalError,
    ExitCode,
    ask_for_approval,
    filename_component,
    slugify,
)


def test_cli_requires_exactly_one_job_source() -> None:
    parser = build_parser()
    base = [
        "--resume",
        "resume.docx",
        "--company",
        "Example",
        "--role",
        "Developer",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--clipboard", "--job-file", "job.txt"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            base
            + [
                "--job-file",
                "job.txt",
                "--job-url",
                "https://www.linkedin.com/jobs/view/4123456789/",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            base
            + [
                "--clipboard",
                "--job-url",
                "https://www.linkedin.com/jobs/view/4123456789/",
            ]
        )


def test_url_mode_derives_company_and_role() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-url",
            "https://www.linkedin.com/jobs/view/4123456789/",
        ]
    )
    _validate_mode_arguments(parser, args)
    assert args.company is None
    assert args.role is None
    assert not hasattr(args, "linkedin_provider")


def test_local_gemma_is_the_default_writer() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
            "--company",
            "Example",
            "--role",
            "Developer",
        ]
    )

    assert args.writer_provider == "ollama"
    assert args.ollama_model == "resume-tailor-gemma"
    assert args.analytics_db.name == "job-search-analytics.sqlite3"


def test_pipeline_request_conversion_preserves_all_cli_and_recovery_values(
    tmp_path: Path,
) -> None:
    retry_context = object()
    antigravity_retry_context = object()
    antigravity_reprocess_context = object()
    namespace = argparse.Namespace(
        resume=tmp_path / "master.docx",
        clipboard=True,
        job_file=tmp_path / "job.txt",
        job_url="https://www.linkedin.com/jobs/view/4123456789/",
        company="Example",
        role="Developer",
        output_dir=tmp_path / "output",
        analytics_db=tmp_path / "analytics.sqlite3",
        yes=True,
        keep_workdir=True,
        timeout=(321, "321s"),
        writer_provider="antigravity",
        analysis_provider="grok_cli",
        ollama_model="synthetic-model",
        antigravity_model="pro",
        antigravity_strength="high",
        grok_model="grok-synthetic",
        grok_strength="medium",
        codex_model="codex-synthetic",
        codex_strength="low",
        initial_qa_provider="grok",
        github_portfolio=True,
        github_username="synthetic-user",
        github_include_private=True,
        github_allow_private_provider=True,
        github_analysis_provider="grok_cli",
        github_project_ids=["repo-101", "repo-202"],
        job_source_override="pasted",
        retry_context=retry_context,
        antigravity_retry_context=antigravity_retry_context,
        antigravity_reprocess_context=antigravity_reprocess_context,
    )

    request = pipeline_request_from_namespace(namespace)

    assert request == PipelineRequest(
        resume=namespace.resume,
        clipboard=namespace.clipboard,
        job_file=namespace.job_file,
        job_url=namespace.job_url,
        company=namespace.company,
        role=namespace.role,
        output_dir=namespace.output_dir,
        analytics_db=namespace.analytics_db,
        yes=namespace.yes,
        keep_workdir=namespace.keep_workdir,
        timeout=namespace.timeout,
        writer_provider=namespace.writer_provider,
        analysis_provider=namespace.analysis_provider,
        ollama_model=namespace.ollama_model,
        antigravity_model=namespace.antigravity_model,
        antigravity_strength=namespace.antigravity_strength,
        grok_model=namespace.grok_model,
        grok_strength=namespace.grok_strength,
        codex_model=namespace.codex_model,
        codex_strength=namespace.codex_strength,
        initial_qa_provider=namespace.initial_qa_provider,
        github_portfolio=True,
        github_username="synthetic-user",
        github_include_private=True,
        github_allow_private_provider=True,
        github_analysis_provider="grok_cli",
        github_project_ids=("repo-101", "repo-202"),
        job_source_override=namespace.job_source_override,
        retry_context=retry_context,
        antigravity_retry_context=antigravity_retry_context,
        antigravity_reprocess_context=antigravity_reprocess_context,
    )


def test_github_portfolio_cli_defaults_are_disabled() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
            "--company",
            "Example",
            "--role",
            "Developer",
        ]
    )
    _validate_mode_arguments(parser, args)
    request = pipeline_request_from_namespace(args)

    assert request.github_portfolio is False
    assert request.github_username is None
    assert request.github_include_private is False
    assert request.github_allow_private_provider is False
    assert request.github_analysis_provider is None
    assert request.github_project_ids == ()


def test_github_portfolio_cli_flags_and_explicit_projects() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
            "--company",
            "Example",
            "--role",
            "Developer",
            "--yes",
            "--github-portfolio",
            "--github-username",
            "synthetic-user",
            "--github-include-private",
            "--github-allow-private-provider",
            "--github-analysis-provider",
            "grok_cli",
            "--github-project",
            "synthetic-user/alpha",
            "--github-project",
            "synthetic-user/beta",
        ]
    )
    _validate_mode_arguments(parser, args)
    request = pipeline_request_from_namespace(args)

    assert request.github_portfolio is True
    assert request.github_username == "synthetic-user"
    assert request.github_include_private is True
    assert request.github_allow_private_provider is True
    assert request.github_analysis_provider == "grok_cli"
    assert request.github_project_ids == (
        "synthetic-user/alpha",
        "synthetic-user/beta",
    )


def test_github_portfolio_cli_defaults_to_safe_local_ranker() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
            "--company",
            "Example",
            "--role",
            "Developer",
            "--analysis-provider",
            "grok_cli",
            "--github-portfolio",
        ]
    )
    _validate_mode_arguments(parser, args)

    assert pipeline_request_from_namespace(args).github_analysis_provider == (
        "gemma_local"
    )


def test_github_portfolio_cli_rejects_coding_agent_rankers() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--resume",
                "resume.docx",
                "--job-file",
                "job.txt",
                "--company",
                "Example",
                "--role",
                "Developer",
                "--github-portfolio",
                "--github-analysis-provider",
                "codex",
            ]
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--github-username", "synthetic-user"],
        ["--github-portfolio", "--github-allow-private-provider"],
        ["--github-portfolio", "--github-project", "repo-1"],
        [
            "--github-portfolio",
            "--github-project",
            "repo-1",
            "--github-project",
            "repo-1",
        ],
    ],
)
def test_invalid_github_portfolio_cli_combinations_are_rejected(
    extra: list[str],
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
            "--company",
            "Example",
            "--role",
            "Developer",
            *extra,
        ]
    )
    with pytest.raises(SystemExit):
        _validate_mode_arguments(parser, args)


def test_cli_has_no_github_token_argument() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--resume",
                "resume.docx",
                "--job-file",
                "job.txt",
                "--company",
                "Example",
                "--role",
                "Developer",
                "--github-token",
                "synthetic-secret",
            ]
        )


def test_compatibility_run_pipeline_delegates_and_returns_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_parser().parse_args(
        [
            "--resume",
            str(tmp_path / "master.docx"),
            "--job-file",
            str(tmp_path / "job.txt"),
            "--company",
            "Example",
            "--role",
            "Developer",
        ]
    )
    hooks = PipelineHooks()
    expected = tmp_path / "completed-run"
    captured: dict[str, object] = {}

    def fake_service(
        request: PipelineRequest,
        *,
        hooks: PipelineHooks,
    ) -> PipelineResult:
        captured["request"] = request
        captured["hooks"] = hooks
        return PipelineResult(run_directory=expected)

    monkeypatch.setattr(application_pipeline, "run_pipeline", fake_service)

    assert run_pipeline(args, hooks=hooks) == expected
    assert isinstance(captured["request"], PipelineRequest)
    assert captured["hooks"] is hooks


def test_removed_linkedin_provider_flag_is_rejected() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--resume",
                "resume.docx",
                "--job-url",
                "https://www.linkedin.com/jobs/view/4123456789/",
                "--linkedin-provider",
                "apify",
            ]
        )


def test_file_mode_still_requires_company_and_role() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--resume",
            "resume.docx",
            "--job-file",
            "job.txt",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_mode_arguments(parser, args)


def test_missing_resume_returns_actionable_code(job_file: Path) -> None:
    code = main(
        [
            "--resume",
            str(job_file.parent / "missing.docx"),
            "--job-file",
            str(job_file),
            "--company",
            "Example",
            "--role",
            "Developer",
        ]
    )
    assert code == ExitCode.INPUT


def test_unsupported_resume_extension(job_file: Path, tmp_path: Path) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF")
    code = main(
        [
            "--resume",
            str(resume),
            "--job-file",
            str(job_file),
            "--company",
            "Example",
            "--role",
            "Developer",
        ]
    )
    assert code == ExitCode.INPUT


def test_cli_job_file_over_limit_reports_actual_and_permitted(
    master_resume: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual = MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS + 1
    job_file = tmp_path / "oversized-job.txt"
    job_file.write_text("x" * actual, encoding="utf-8")

    code = main(
        [
            "--resume",
            str(master_resume),
            "--job-file",
            str(job_file),
            "--company",
            "Example",
            "--role",
            "Developer",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert code == ExitCode.INPUT
    error = capsys.readouterr().err
    assert f"{actual:,}" in error
    assert f"{MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS:,}" in error


def test_human_approval_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "no")
    with pytest.raises(ApprovalError):
        ask_for_approval("test gate", assume_yes=False)


def test_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _: pytest.fail("input must not be called"),
    )
    ask_for_approval("test gate", assume_yes=True)


def test_output_naming_and_path_safety() -> None:
    assert slugify("../../Example Talent") == "example-talent"
    assert filename_component("SAMPLE CANDIDATE") == "Sample-Candidate"
    assert filename_component("Example Talent") == "Example-Talent"
    assert filename_component("Agentic AI Developer") == "Agentic-AI-Developer"


def test_existing_installation_refuses_overwrite(
    installer_source: Path,
    tmp_path: Path,
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    environment = os.environ.copy()
    environment["HOME"] = str(fake_home)
    first = subprocess.run(
        ["bash", str(installer_source / "install.sh")],
        cwd=installer_source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    installed = fake_home / ".local" / "share" / "resume-tailor"
    assert (installed / "resume_tailor" / "backend" / "jobs" / "linkedin_job.py").is_file()
    assert (installed / "resume_tailor" / "backend" / "jobs" / "apify_job.py").is_file()
    assert (installed / "resume_tailor" / "backend" / "providers" / "ollama_transport.py").is_file()
    assert (installed / "resume_tailor" / "backend" / "providers" / "ollama_writer.py").is_file()
    assert (installed / "resume_tailor" / "backend" / "documents" / "headless_render.py").is_file()
    assert not (installed / "resume_tailor" / "codex_linkedin.py").exists()
    assert (installed / "resume_tailor" / "backend" / "utils" / "smoke.py").is_file()
    assert (installed / "resume_tailor" / "ui" / "ui.py").is_file()
    assert (installed / "resume_tailor" / "templates" / "dashboard.html").is_file()
    assert (installed / "resume_tailor" / "static" / "app.css").is_file()
    installed_schemas = installed / "resume_tailor" / "schemas"
    assert (installed_schemas / "linkedin_job.schema.json").is_file()
    assert not (installed_schemas / "linkedin_job.openai.schema.json").exists()
    if (installer_source / ".venv" / "bin" / "python").is_file():
        assert (installed / ".venv").is_symlink()
        assert (installed / ".venv").resolve() == (installer_source / ".venv").resolve()
    else:
        assert not (installed / ".venv").exists()
    assert (fake_home / ".local" / "bin" / "tailor-resume-ui").is_file()
    launcher_environment = environment.copy()
    launcher_environment["PYTHON_BIN"] = sys.executable
    launcher_environment["PYTHONPATH"] = ""
    launcher_check = subprocess.run(
        [
            str(fake_home / ".local" / "bin" / "tailor-resume-ui"),
            "--version",
        ],
        cwd=tmp_path,
        env=launcher_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launcher_check.returncode == 0, launcher_check.stderr
    assert "tailor-resume-ui" in launcher_check.stdout
    assert not (
        fake_home / ".local" / "share" / "applications" / "resume-tailor.desktop"
    ).exists()
    second = subprocess.run(
        ["bash", str(installer_source / "install.sh")],
        cwd=installer_source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_desktop_launcher_requires_explicit_installer_option(
    installer_source: Path,
    tmp_path: Path,
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    environment = os.environ.copy()
    environment["HOME"] = str(fake_home)
    result = subprocess.run(
        ["bash", str(installer_source / "install.sh"), "--desktop"],
        cwd=installer_source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    desktop = (
        fake_home / ".local" / "share" / "applications" / "resume-tailor.desktop"
    )
    assert desktop.is_file()
    text = desktop.read_text(encoding="utf-8")
    expected_ui = fake_home / ".local" / "bin" / "tailor-resume-ui"
    assert f'Exec="{expected_ui}"' in text
    expected_icon = (
        fake_home
        / ".local"
        / "share"
        / "resume-tailor"
        / "resume_tailor"
        / "static"
        / "favicon.svg"
    )
    assert f"Icon={expected_icon}" in text
    shortcut = fake_home / "Desktop" / "Resume Tailor.desktop"
    assert shortcut.is_file()
    assert shortcut.stat().st_mode & 0o111
    assert shortcut.read_text(encoding="utf-8") == text

    uninstall = subprocess.run(
        [str(fake_home / ".local" / "share" / "resume-tailor" / "uninstall.sh")],
        cwd=tmp_path,
        env=environment,
        input="remove\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert uninstall.returncode == 0, uninstall.stderr
    assert not desktop.exists()
    assert not shortcut.exists()


def test_desktop_installer_refuses_unrelated_shortcut_even_with_force(
    installer_source: Path,
    tmp_path: Path,
) -> None:
    fake_home = tmp_path / "home"
    desktop_directory = fake_home / "Desktop"
    desktop_directory.mkdir(parents=True)
    shortcut = desktop_directory / "Resume Tailor.desktop"
    original = "[Desktop Entry]\nName=Unrelated App\n"
    shortcut.write_text(original, encoding="utf-8")
    environment = os.environ.copy()
    environment["HOME"] = str(fake_home)

    result = subprocess.run(
        [
            "bash",
            str(installer_source / "install.sh"),
            "--force",
            "--desktop",
        ],
        cwd=installer_source,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to overwrite unrelated file" in result.stderr
    assert shortcut.read_text(encoding="utf-8") == original
    assert not (fake_home / ".local" / "share" / "resume-tailor").exists()
