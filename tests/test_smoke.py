from __future__ import annotations

import json
from pathlib import Path

import pytest

import resume_tailor.smoke as smoke_module
from resume_tailor.smoke import (
    SYNTHETIC_SAMPLE_SHA256,
    assert_smoke_input_provenance,
    prepare_smoke_inputs,
    run_semantic_smoke,
)
from resume_tailor.utilities import InputError


def _valid_response(repository_root: Path) -> dict:
    return json.loads(
        (
            repository_root
            / "tests"
            / "fixtures"
            / "analysis_source_ids_valid.json"
        ).read_text(encoding="utf-8")
    )


def test_smoke_defaults_to_hash_pinned_synthetic_inputs(
    repository_root: Path,
) -> None:
    inputs = prepare_smoke_inputs(repository_root=repository_root)

    assert inputs.mode == "synthetic-only"
    assert inputs.resume_path == (
        repository_root / "template" / "sample_resume.docx"
    ).resolve()
    assert inputs.resume_sha256 == SYNTHETIC_SAMPLE_SHA256
    assert inputs.job_file is None
    assert inputs.separately_authorized is False
    assert_smoke_input_provenance(inputs)


@pytest.mark.parametrize("kind", ["resume", "job"])
def test_synthetic_mode_refuses_substituted_artifacts_before_reading(
    kind: str,
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = tmp_path / "Tailored" / "preserved-run"
    forbidden.mkdir(parents=True)
    resume = forbidden / "master_resume.docx"
    job = forbidden / "job-description.txt"
    if kind == "resume":
        kwargs = {"resume_path": resume}
    else:
        kwargs = {"job_file": job}
    monkeypatch.setattr(
        smoke_module,
        "sha256_file",
        lambda _: pytest.fail("a forbidden custom artifact was read"),
    )

    with pytest.raises(InputError, match="Synthetic smoke mode refuses custom"):
        prepare_smoke_inputs(repository_root=repository_root, **kwargs)


def test_real_inputs_require_separate_authorization_reference(
    repository_root: Path,
    master_resume: Path,
    job_file: Path,
) -> None:
    with pytest.raises(InputError, match="separate authorization reference"):
        prepare_smoke_inputs(
            repository_root=repository_root,
            resume_path=master_resume,
            job_file=job_file,
            allow_real_inputs=True,
        )


def test_provenance_is_reported_before_stubbed_provider_launch(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = prepare_smoke_inputs(repository_root=repository_root)
    workspace = tmp_path / "private-smoke"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    events: list[str] = []

    def fake_provider(**kwargs):
        events.append("provider")
        assert kwargs["extracted_resume"]["source"]["sha256"] == (
            SYNTHETIC_SAMPLE_SHA256
        )
        return _valid_response(repository_root)

    monkeypatch.setattr(smoke_module, "invoke_codex_analysis", fake_provider)
    result = run_semantic_smoke(
        inputs,
        run_directory=workspace,
        timeout_seconds=30,
        provenance_reporter=lambda _: events.append("provenance"),
    )

    assert events == ["provenance", "provider"]
    assert result["approval_boundary_reached"] is True
    assert result["downstream_invoked"] is False


def test_provider_launch_requires_a_provenance_reporter(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = prepare_smoke_inputs(repository_root=repository_root)
    workspace = tmp_path / "private-smoke"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    monkeypatch.setattr(
        smoke_module,
        "invoke_codex_analysis",
        lambda **_: pytest.fail("provider launched without provenance reporting"),
    )

    with pytest.raises(InputError, match="provenance reporter is required"):
        run_semantic_smoke(
            inputs,
            run_directory=workspace,
            timeout_seconds=30,
        )


def test_changed_input_is_refused_before_stubbed_provider(
    repository_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume_copy = tmp_path / "authorized-synthetic-copy.docx"
    resume_copy.write_bytes(
        (repository_root / "template" / "sample_resume.docx").read_bytes()
    )
    job_copy = tmp_path / "authorized-synthetic-job.txt"
    job_copy.write_text("Synthetic Python role.", encoding="utf-8")
    inputs = prepare_smoke_inputs(
        repository_root=repository_root,
        resume_path=resume_copy,
        job_file=job_copy,
        allow_real_inputs=True,
        authorization_reference="synthetic-test-authorization",
    )
    job_copy.write_text("Changed synthetic role.", encoding="utf-8")
    workspace = tmp_path / "private-smoke"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    monkeypatch.setattr(
        smoke_module,
        "invoke_codex_analysis",
        lambda **_: pytest.fail("provider launched after input hash changed"),
    )

    with pytest.raises(InputError, match="changed after provenance validation"):
        run_semantic_smoke(
            inputs,
            run_directory=workspace,
            timeout_seconds=30,
            provenance_reporter=lambda _: None,
        )
