from __future__ import annotations

import json
from pathlib import Path

import pytest

from resume_tailor.cli import main
from resume_tailor.utilities import ExitCode, sha256_file


def _arguments(
    master_resume: Path,
    job_file: Path,
    output_dir: Path,
    *,
    yes: bool = True,
) -> list[str]:
    arguments = [
        "--resume",
        str(master_resume),
        "--job-file",
        str(job_file),
        "--company",
        "RG Talent",
        "--role",
        "Agentic AI Developer",
        "--output-dir",
        str(output_dir),
        "--timeout",
        "30s",
    ]
    if yes:
        arguments.append("--yes")
    return arguments


def _url_arguments(
    master_resume: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "--resume",
        str(master_resume),
        "--job-url",
        "https://www.linkedin.com/jobs/view/general-ai-role-4123456789/",
        "--output-dir",
        str(output_dir),
        "--timeout",
        "30s",
    ]


def test_simulated_pipeline_artifact_tree_and_source_immutability(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_hash = sha256_file(master_resume)
    schema_log = tmp_path / "codex-schema-paths.txt"
    monkeypatch.setenv("STUB_CODEX_SCHEMA_LOG", str(schema_log))
    output_dir = tmp_path / "Tailored Resumes"
    code = main(_arguments(master_resume, job_file, output_dir))
    assert code == ExitCode.OK
    run_directories = list(output_dir.iterdir())
    assert len(run_directories) == 1
    run = run_directories[0]
    basename = "Sample-Candidate-RG-Talent-Agentic-AI-Developer"
    expected = {
        "job-description.txt",
        "extracted-master-resume.json",
        "codex-analysis.json",
        "antigravity-response.json",
        "tailored-content.json",
        "content-diff.md",
        f"{basename}.docx",
        f"{basename}.pdf",
        "preview.png",
        "final-qa.md",
        "run-metadata.json",
    }
    assert expected <= {path.name for path in run.iterdir()}
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["source_resume"]["unchanged"] is True
    assert "work" not in {path.name for path in run.iterdir()}
    assert sha256_file(master_resume) == before_hash
    used_schemas = [
        Path(line).name
        for line in schema_log.read_text(encoding="utf-8").splitlines()
    ]
    assert used_schemas == [
        "codex_analysis.openai.schema.json",
        "final_qa.openai.schema.json",
    ]


def test_failure_preserves_useful_artifacts(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_MODE", "questions")
    output_dir = tmp_path / "output"
    code = main(_arguments(master_resume, job_file, output_dir))
    assert code == ExitCode.WAITING
    run = next(output_dir.iterdir())
    assert (run / "job-description.txt").is_file()
    assert (run / "extracted-master-resume.json").is_file()
    assert (run / "codex-analysis.json").is_file()
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "FAILED"
    assert metadata["source_resume"]["unchanged"] is True


def test_human_analysis_refusal_stops_before_antigravity(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "refuse")
    output_dir = tmp_path / "output"
    code = main(_arguments(master_resume, job_file, output_dir, yes=False))
    assert code == ExitCode.APPROVAL
    run = next(output_dir.iterdir())
    assert (run / "codex-analysis.json").is_file()
    assert not (run / "antigravity-response.json").exists()


def test_user_rejecting_linkedin_confirmation_stops_before_codex(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def reject(prompt: str) -> str:
        prompts.append(prompt)
        return "reject"

    monkeypatch.setattr("builtins.input", reject)
    output_dir = tmp_path / "url-output"
    code = main(_url_arguments(master_resume, output_dir))
    assert code == ExitCode.APPROVAL
    run = next(output_dir.iterdir())
    assert run.name.startswith("linkedin-job-fetch-")
    assert (run / "job-source.json").is_file()
    assert (run / "job-description.txt").is_file()
    assert not (run / "codex-analysis.json").exists()
    assert not list(run.glob("*.docx"))
    assert prompts == ['LinkedIn posting: type "approve" to continue: ']


def test_url_pipeline_continues_after_explicit_confirmation(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    def approve(prompt: str) -> str:
        prompts.append(prompt)
        return "approve"

    monkeypatch.setattr("builtins.input", approve)
    output_dir = tmp_path / "url-output"
    code = main(_url_arguments(master_resume, output_dir))
    assert code == ExitCode.OK
    run = next(output_dir.iterdir())
    assert run.name.startswith("example-ai-systems-machine-learning-engineer-")
    basename = "Sample-Candidate-Example-AI-Systems-Machine-Learning-Engineer"
    assert (run / "job-source.json").is_file()
    assert (run / "job-description.txt").is_file()
    assert (run / f"{basename}.docx").is_file()
    assert (run / f"{basename}.pdf").is_file()
    assert (run / "final-qa.md").is_file()
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["company"] == "Example AI Systems"
    assert metadata["role"] == "Machine Learning Engineer"
    assert metadata["source_resume"]["unchanged"] is True
    assert prompts == [
        'LinkedIn posting: type "approve" to continue: ',
        'Codex analysis: type "approve" to continue: ',
        'Tailored content diff: type "approve" to continue: ',
    ]
