from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

import resume_tailor.antigravity_transport as transport_module
from resume_tailor.antigravity_transport import (
    MAX_ANTIGRAVITY_PROMPT_BYTES,
    antigravity_print_args,
    run_antigravity_prompt,
)
from resume_tailor.antigravity_writer import (
    build_tailoring_prompt,
    invoke_antigravity,
)
from resume_tailor.docx_extract import extract_resume
from resume_tailor.evidence import resolve_analysis_evidence
from resume_tailor.job_requirements import build_job_requirement_catalog
from resume_tailor.schemas import schema_path
from resume_tailor.utilities import (
    AntigravityLaunchSizeError,
    AntigravityResponseEnvelopeError,
    CancellationError,
    DependencyError,
    ModelError,
    cancellable_commands,
)


def _analysis() -> dict:
    return {
        "role_summary": "Synthetic role",
        "fit_assessment": {"overall": "Fit", "strengths": [], "gaps": []},
        "matched_requirements": [],
        "evidence_map": [],
        "ats_keywords": [],
        "ats_keyword_assessment": [],
        "supported_ats_keywords": [],
        "missing_or_unsupported_requirements": [],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": ["Unsupported synthetic claim"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }


def _resolved_analysis(
    extracted: dict,
    requirements: dict,
) -> dict:
    analysis = _analysis()
    analysis["supported_requirement_mappings"] = []
    analysis["unsupported_requirement_ids"] = [
        item["requirement_id"] for item in requirements["requirements"]
    ]
    resolved, issues = resolve_analysis_evidence(
        analysis,
        extracted,
        requirements,
    )
    assert issues == []
    return resolved


def _wait_for_file(path: Path, *, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for synthetic stub file {path.name}.")
        time.sleep(0.02)


def _wait_for_process_exit(pid: int, *, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while Path(f"/proc/{pid}").exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Synthetic subprocess {pid} was not cleaned up.")
        time.sleep(0.02)


def test_large_utf8_prompt_uses_exact_stdin_and_small_argv(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    synthetic_job = (
        "Synthetic résumé evidence — naïve café; no real person or employer.\n"
        * 3_000
    )
    requirements = build_job_requirement_catalog(
        synthetic_job,
        structured_job={
            "responsibilities": ["Process synthetic evidence safely."]
        },
    )
    analysis = _resolved_analysis(extracted, requirements)
    analysis["forbidden_claims"] = [
        "Synthetic forbidden claim — naïve café; never add it.\n" * 3_000
    ]
    expected_prompt = build_tailoring_prompt(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=synthetic_job,
        job_requirements=requirements,
        approved_analysis=analysis,
        company="Synthetic Systems",
        role="Evidence Engineer",
    )
    expected_bytes = expected_prompt.encode("utf-8")
    assert len(expected_bytes) > 131_072
    with pytest.raises(OSError) as too_large:
        subprocess.run(
            ["/bin/true", expected_prompt],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    assert too_large.value.errno == errno.E2BIG

    log = tmp_path / "transport.json"
    monkeypatch.setenv("STUB_AGY_TRANSPORT_LOG", str(log))
    monkeypatch.setenv(
        "STUB_AGY_EXPECTED_STDIN_BYTES",
        str(len(expected_bytes)),
    )
    monkeypatch.setenv(
        "STUB_AGY_EXPECTED_STDIN_SHA256",
        hashlib.sha256(expected_bytes).hexdigest(),
    )
    content = invoke_antigravity(
        master_content=extracted["content"],
        extracted_resume=extracted,
        job_description=synthetic_job,
        job_requirements=requirements,
        approved_analysis=analysis,
        company="Synthetic Systems",
        role="Evidence Engineer",
        run_directory=tmp_path,
        timeout_seconds=30,
        antigravity_duration="30s",
        executable=str(stubs_on_path / "agy"),
    )

    assert content == extracted["content"]
    observed = json.loads(log.read_text(encoding="utf-8"))
    assert observed["transport"] == "stdin"
    assert observed["payload_bytes"] == len(expected_bytes)
    assert observed["payload_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert observed["argv_total_bytes"] < 4_096
    assert observed["max_argv_item_bytes"] < 4_096


def test_prompt_free_argument_array_contains_only_flags_and_short_paths() -> None:
    marker = "SYNTHETIC_PRIVATE_PROMPT_MARKER"
    args = antigravity_print_args(
        executable="/synthetic/bin/agy",
        schema=Path("/synthetic/schema.json"),
        print_timeout="30s",
    )
    assert "--prompt" in args
    assert "--mode=plan" not in args
    assert marker not in "\0".join(args)
    assert all(len(value.encode("utf-8")) < 4_096 for value in args)


def test_linkedin_transport_selects_documented_stream_json_format() -> None:
    args = antigravity_print_args(
        executable="/synthetic/bin/agy",
        schema=Path("/synthetic/linkedin.schema.json"),
        print_timeout="30s",
        agent_mode="plan",
        output_format="stream-json",
    )

    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--mode=plan" in args


def test_unknown_antigravity_output_format_is_rejected() -> None:
    with pytest.raises(DependencyError, match="output format"):
        antigravity_print_args(
            executable="/synthetic/bin/agy",
            schema=Path("/synthetic/schema.json"),
            print_timeout="30s",
            output_format="synthetic-unknown",
        )


def test_cancellation_hides_prompt_from_cmdline_and_stops_process_group(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    marker = "SYNTHETIC_CMDLINE_SECRET_MARKER"
    requirements = build_job_requirement_catalog(marker)
    analysis = _resolved_analysis(extracted, requirements)
    cancel_event = threading.Event()
    errors: list[BaseException] = []
    pid_log = tmp_path / "agy.pid"
    child_pid_log = tmp_path / "agy-child.pid"
    monkeypatch.setenv("STUB_AGY_MODE", "hang")
    monkeypatch.setenv("STUB_AGY_PID_LOG", str(pid_log))
    monkeypatch.setenv("STUB_AGY_CHILD_PID_LOG", str(child_pid_log))

    def invoke() -> None:
        try:
            with cancellable_commands(cancel_event):
                invoke_antigravity(
                    master_content=extracted["content"],
                    extracted_resume=extracted,
                    job_description=marker,
                    job_requirements=requirements,
                    approved_analysis=analysis,
                    company="Synthetic Systems",
                    role="Evidence Engineer",
                    run_directory=tmp_path,
                    timeout_seconds=30,
                    antigravity_duration="30s",
                    executable=str(stubs_on_path / "agy"),
                )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    _wait_for_file(pid_log)
    _wait_for_file(child_pid_log)
    pid = int(pid_log.read_text(encoding="ascii"))
    child_pid = int(child_pid_log.read_text(encoding="ascii"))
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    assert marker.encode("utf-8") not in cmdline
    assert b"BEGIN_TRUSTED_MASTER_RESUME_CONTENT" not in cmdline

    cancel_event.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], CancellationError)
    _wait_for_process_exit(pid)
    _wait_for_process_exit(child_pid)
    assert not list(tmp_path.glob("*prompt*"))


def test_timeout_stops_antigravity_process_group_without_prompt_file(
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_log = tmp_path / "agy.pid"
    child_pid_log = tmp_path / "agy-child.pid"
    monkeypatch.setenv("STUB_AGY_MODE", "hang")
    monkeypatch.setenv("STUB_AGY_PID_LOG", str(pid_log))
    monkeypatch.setenv("STUB_AGY_CHILD_PID_LOG", str(child_pid_log))

    with pytest.raises(ModelError, match="full subprocess group was stopped"):
        run_antigravity_prompt(
            executable=str(stubs_on_path / "agy"),
            prompt="Synthetic timeout payload — UTF-8.",
            prompt_label="synthetic Antigravity prompt",
            schema=schema_path("tailored_resume.schema.json"),
            print_timeout="30s",
            cwd=tmp_path,
            timeout_seconds=1,
        )
    pid = int(pid_log.read_text(encoding="ascii"))
    child_pid = int(child_pid_log.read_text(encoding="ascii"))
    _wait_for_process_exit(pid)
    _wait_for_process_exit(child_pid)
    assert not list(tmp_path.glob("*prompt*"))


def test_payload_resource_bound_fails_before_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transport_module,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail(
            "Antigravity launched before payload-size validation"
        ),
    )
    with pytest.raises(ModelError, match="local resource safety limit"):
        run_antigravity_prompt(
            executable="/synthetic/bin/agy",
            prompt="x" * (MAX_ANTIGRAVITY_PROMPT_BYTES + 1),
            prompt_label="synthetic Antigravity prompt",
            schema=Path("/synthetic/schema.json"),
            print_timeout="30s",
            cwd=tmp_path,
            timeout_seconds=30,
        )


def test_e2big_is_classified_without_echoing_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        try:
            raise OSError(errno.E2BIG, "Argument list too long")
        except OSError as exc:
            raise DependencyError("Could not run agy.") from exc

    monkeypatch.setattr(transport_module, "run_command", fail)
    with pytest.raises(AntigravityLaunchSizeError) as raised:
        run_antigravity_prompt(
            executable="/synthetic/bin/agy",
            prompt="SYNTHETIC_PRIVATE_PROMPT_MARKER",
            prompt_label="synthetic Antigravity prompt",
            schema=Path("/synthetic/schema.json"),
            print_timeout="30s",
            cwd=tmp_path,
            timeout_seconds=30,
        )
    assert str(raised.value) == (
        "Antigravity could not start because the request exceeded the "
        "operating system's command-line size."
    )
    assert "SYNTHETIC_PRIVATE_PROMPT_MARKER" not in str(raised.value)


def test_malformed_output_diagnostic_omits_provider_text(
    master_resume: Path,
    tmp_path: Path,
    stubs_on_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted, _ = extract_resume(master_resume)
    description = "Synthetic malformed-output test."
    requirements = build_job_requirement_catalog(description)
    analysis = _resolved_analysis(extracted, requirements)
    monkeypatch.setenv("STUB_AGY_MODE", "bad_json")
    with pytest.raises(AntigravityResponseEnvelopeError, match="malformed JSON"):
        invoke_antigravity(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=description,
            job_requirements=requirements,
            approved_analysis=analysis,
            company="Synthetic Systems",
            role="Evidence Engineer",
            run_directory=tmp_path,
            timeout_seconds=30,
            antigravity_duration="30s",
            executable=str(stubs_on_path / "agy"),
        )
    diagnostic = json.loads(
        (tmp_path / "antigravity-response.json").read_text(encoding="utf-8")
    )
    assert diagnostic["provider_output_omitted"] is True
    assert diagnostic["stdout_bytes"] > 0
    assert len(diagnostic["stdout_sha256"]) == 64
    assert "raw_stdout" not in diagnostic
    assert "not-json" not in json.dumps(diagnostic)
    envelope = json.loads(
        (tmp_path / "antigravity-response-envelope.json").read_text(
            encoding="utf-8"
        )
    )
    assert envelope["response_envelope_type"] == "malformed-json-output"
    assert envelope["validation_result"] == "REJECTED"
