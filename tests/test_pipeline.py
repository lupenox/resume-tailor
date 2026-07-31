from __future__ import annotations

import json
from pathlib import Path

import pytest

import resume_tailor.cli as cli_module
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
        "Example Talent",
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
    basename = "Sample-Candidate-Example-Talent-Agentic-AI-Developer"
    expected = {
        "job-description.txt",
        "job-requirements.json",
        "extracted-master-resume.json",
        "codex-analysis-transport.schema.json",
        "codex-analysis.json",
        "codex-analysis-resolved.json",
        "codex-analysis-approval.json",
        "antigravity-response.json",
        "antigravity-response-envelope.json",
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
    assert metadata["analysis_inputs"]["version"] == 2
    assert len(metadata["analysis_inputs"]["job_description_sha256"]) == 64
    assert len(metadata["analysis_inputs"]["extracted_resume_sha256"]) == 64
    assert len(metadata["analysis_inputs"]["job_requirements_sha256"]) == 64
    schema_metadata = metadata["codex_analysis_transport_schema"]
    assert schema_metadata["filename"] == "codex-analysis-transport.schema.json"
    assert len(schema_metadata["sha256"]) == 64
    assert schema_metadata["generated_from_source_and_requirement_catalogs"] is True
    assert schema_metadata["job_requirement_id_count"] > 0
    assert sha256_file(run / schema_metadata["filename"]) == schema_metadata["sha256"]
    approval_metadata = metadata["codex_analysis_approval"]
    assert approval_metadata["filename"] == "codex-analysis-approval.json"
    assert approval_metadata["decision"] == "approved"
    assert sha256_file(run / approval_metadata["filename"]) == approval_metadata[
        "sha256"
    ]
    response_metadata = metadata["antigravity_response"]
    assert response_metadata["execution_mode"] == "print"
    assert response_metadata["output_format"] == "json"
    assert response_metadata["response_envelope_type"] == (
        "json-wrapper-structured_output"
    )
    assert response_metadata["validation_result"] == "PASS"
    assert response_metadata["cli_version"] == "1.1.8-stub"
    assert sha256_file(run / response_metadata["response"]["filename"]) == (
        response_metadata["response"]["sha256"]
    )
    assert "work" not in {path.name for path in run.iterdir()}
    assert sha256_file(master_resume) == before_hash
    used_schemas = [
        Path(line).name
        for line in schema_log.read_text(encoding="utf-8").splitlines()
    ]
    assert used_schemas == [
        "codex-analysis-transport.schema.json",
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
    assert (run / "codex-analysis-resolved.json").is_file()
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "FAILED"
    assert metadata["source_resume"]["unchanged"] is True


def test_unstructured_print_wrapper_is_classified_without_prose_extraction(
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_AGY_MODE", "unstructured_print_wrapper")
    output_dir = tmp_path / "output"

    code = main(_arguments(master_resume, job_file, output_dir))

    assert code == ExitCode.MODEL
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["failure_class"] == "antigravity-response-envelope"
    assert metadata["error"]["type"] == "AntigravityResponseEnvelopeError"
    response_metadata = metadata["antigravity_response"]
    assert response_metadata["response_envelope_type"] == (
        "json-wrapper-unstructured-response"
    )
    assert response_metadata["validation_result"] == "REJECTED"
    assert not (run / "tailored-content.json").exists()
    assert not list(run.glob("*.docx"))


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
    assert not (run / "codex-analysis-approval.json").exists()


@pytest.mark.parametrize("failure_mode", ["source_id_failure", "empty_source_ids"])
def test_source_evidence_failure_precedes_approval_and_antigravity(
    failure_mode: str,
    master_resume: Path,
    job_file: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_CODEX_MODE", failure_mode)
    monkeypatch.setattr(
        "builtins.input",
        lambda _: pytest.fail("approval was requested before source validation"),
    )
    output_dir = tmp_path / "source-evidence-output"

    code = main(_arguments(master_resume, job_file, output_dir, yes=False))

    assert code == ExitCode.TRUTHFULNESS
    run = next(output_dir.iterdir())
    assert (run / "codex-analysis.json").is_file()
    assert not (run / "codex-analysis-resolved.json").exists()
    assert not (run / "codex-analysis-approval.json").exists()
    assert not (run / "antigravity-response.json").exists()
    assert not list(run.glob("*.docx"))
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "codex-analysis"
    assert metadata["failure_class"] == "source-evidence-analysis"
    assert metadata["error"]["type"] == "SourceEvidenceError"


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


def test_apify_provider_dispatches_locally_before_confirmation(
    master_resume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posting = {
        "fetch_status": "success",
        "requested_url": (
            "https://www.linkedin.com/jobs/view/general-ai-role-4123456789/"
        ),
        "final_resolved_url": (
            "https://www.linkedin.com/jobs/view/general-ai-role-4123456789/"
        ),
        "linkedin_job_id": "4123456789",
        "job_title": "Synthetic AI Engineer",
        "company": "Example Systems",
        "location": "Remote",
        "workplace_type": "remote",
        "employment_type": "Full-time",
        "salary": None,
        "normalized_job_description": (
            "Build safe Python services for synthetic evidence workflows. "
            "Collaborate with engineers, validate structured inputs, write tests, "
            "document decisions, maintain APIs, review failures, and improve "
            "reliable automation across a privacy-conscious local application. "
            "This additional synthetic detail ensures the posting is substantive "
            "without containing personal, company-confidential, or résumé data."
        ),
        "responsibilities": [],
        "required_qualifications": [],
        "preferred_qualifications": [],
        "technologies_and_skills": [],
        "ai_focus_areas": [],
        "warnings": ["Synthetic Apify fixture."],
    }
    calls: list[str] = []

    def apify_fetch(**_: object) -> dict[str, object]:
        calls.append("apify")
        return posting

    monkeypatch.setattr(cli_module, "invoke_apify_job_extraction", apify_fetch)
    monkeypatch.setattr(
        cli_module,
        "invoke_linkedin_job_extraction",
        lambda **_: pytest.fail("Antigravity URL retrieval was invoked"),
    )
    monkeypatch.setattr(
        cli_module,
        "_dependency_versions",
        lambda _: {
            "resume_tailor": "0.1.0",
            "codex": "stub",
            "antigravity": "stub",
            "libreoffice": "stub",
        },
    )
    monkeypatch.setattr("builtins.input", lambda _: "reject")
    output_dir = tmp_path / "url-output"

    code = main(
        [
            *_url_arguments(master_resume, output_dir),
            "--linkedin-provider",
            "apify",
        ]
    )

    assert code == ExitCode.APPROVAL
    assert calls == ["apify"]
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["linkedin_retrieval"] == {
        "requested_provider": "apify",
        "resolved_provider": "apify",
        "automatic_fallback": False,
    }
    assert not (run / "codex-analysis.json").exists()


def test_linkedin_envelope_failure_is_stage_specific_and_stops_before_codex(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_LINKEDIN_MODE", "print_response_prose")
    monkeypatch.setattr(
        cli_module,
        "_dependency_versions",
        lambda _cwd: {
            "resume_tailor": "0.1.0",
            "codex": "stub",
            "antigravity": "stub",
        },
    )
    output_dir = tmp_path / "url-output"

    code = main(_url_arguments(master_resume, output_dir))

    assert code == ExitCode.MODEL
    run = next(output_dir.iterdir())
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    assert metadata["stage"] == "linkedin-job-extraction"
    assert metadata["failure_class"] == "linkedin-response-envelope"
    assert metadata["error"]["type"] == "LinkedInResponseEnvelopeError"
    assert metadata["error"]["response_envelope_type"].startswith(
        "stream-json-event-result:"
    )
    assert (run / "linkedin-response-envelope.json").is_file()
    assert not (run / "codex-analysis.json").exists()
    assert not (run / "antigravity-response.json").exists()
    assert not list(run.glob("*.docx"))


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
