from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from resume_tailor.cli import _validate_mode_arguments, build_parser, main
from resume_tailor.job_text import MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
from resume_tailor.utilities import (
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


def test_local_qwen_is_the_default_writer() -> None:
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
    assert args.ollama_model == "resume-tailor-qwen"
    assert args.analytics_db.name == "job-search-analytics.sqlite3"


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
    assert (installed / "resume_tailor" / "linkedin_job.py").is_file()
    assert (installed / "resume_tailor" / "apify_job.py").is_file()
    assert (installed / "resume_tailor" / "ollama_transport.py").is_file()
    assert (installed / "resume_tailor" / "ollama_writer.py").is_file()
    assert (installed / "resume_tailor" / "headless_render.py").is_file()
    assert not (installed / "resume_tailor" / "codex_linkedin.py").exists()
    assert (installed / "resume_tailor" / "smoke.py").is_file()
    assert (installed / "resume_tailor" / "ui.py").is_file()
    assert (installed / "resume_tailor" / "templates" / "dashboard.html").is_file()
    assert (installed / "resume_tailor" / "static" / "app.css").is_file()
    assert (installed / "schemas" / "linkedin_job.schema.json").is_file()
    assert not (installed / "schemas" / "linkedin_job.openai.schema.json").exists()
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
