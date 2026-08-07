from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.engine.retry import (
    ANALYSIS_APPROVAL_FILENAME,
    analysis_input_manifest,
    build_antigravity_reprocess_context,
    build_antigravity_retry_context,
    build_retry_context,
    load_antigravity_reprocess_inputs,
    load_antigravity_retry_inputs,
    load_retry_inputs,
    record_codex_analysis_approval,
)
from resume_tailor.backend.utils.schemas import (
    load_schema,
    prepare_codex_analysis_transport_schema,
)
from resume_tailor.backend.utils.utilities import InputError, atomic_write_json, sha256_file


def _write_legacy_failure(
    directory: Path,
    master_resume: Path,
) -> tuple[Path, str, str]:
    run = directory / "synthetic-legacy-source-failure"
    run.mkdir()
    description = "Build synthetic Python validation workflows."
    (run / "job-description.txt").write_text(
        description + "\n",
        encoding="utf-8",
    )
    atomic_write_json(
        run / "job-source.json",
        {
            "fetch_status": "success",
            "normalized_job_description": description,
        },
    )
    extracted, _ = extract_resume(master_resume)
    atomic_write_json(run / "extracted-master-resume.json", extracted)
    source_hash = sha256_file(master_resume)
    atomic_write_json(
        run / "run-metadata.json",
        {
            "application": "resume-tailor",
            "status": "FAILED",
            "stage": "codex-analysis",
            "company": "Synthetic Systems",
            "role": "Evidence Engineer",
            "source_resume": {
                "filename": master_resume.name,
                "sha256_before": source_hash,
                "sha256_after": source_hash,
                "unchanged": True,
            },
            "error": {
                "type": "TruthfulnessError",
                "message": (
                    "Codex analysis failed local source-evidence validation:\n"
                    "- synthetic legacy quotation mismatch"
                ),
                "exit_code": 13,
            },
        },
    )
    return run, description, source_hash


def test_legacy_confirmed_inputs_are_cross_checked_without_mutating_source_run(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run, description, source_hash = _write_legacy_failure(
        tmp_path,
        master_resume,
    )
    before = {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }

    context = build_retry_context(run, current_resume=master_resume)
    inputs = load_retry_inputs(context, current_resume=master_resume)

    assert context.legacy_verified is True
    assert inputs.job_description == description
    assert inputs.extracted_resume["source"]["sha256"] == source_hash
    assert inputs.job_requirements["requirements"]
    assert before == {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }


def test_legacy_retry_refuses_extraction_not_derived_from_master(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run, _, _ = _write_legacy_failure(tmp_path, master_resume)
    extraction_path = run / "extracted-master-resume.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction["paragraphs"][1]["text"] = "Synthetic altered source text."
    atomic_write_json(extraction_path, extraction)

    with pytest.raises(InputError, match="does not match the unchanged source"):
        build_retry_context(run, current_resume=master_resume)


def _write_antigravity_launch_failure(
    directory: Path,
    master_resume: Path,
) -> Path:
    run = directory / "synthetic-antigravity-launch-failure"
    run.mkdir()
    description = "Skills: Python and synthetic evidence validation."
    (run / "job-description.txt").write_text(
        description + "\n",
        encoding="utf-8",
    )
    requirements = build_job_requirement_catalog(
        description,
        structured_job={
            "technologies_and_skills": [
                "Python",
                "Synthetic evidence validation",
            ]
        },
    )
    atomic_write_json(run / "job-requirements.json", requirements)
    extracted, _ = extract_resume(master_resume)
    atomic_write_json(run / "extracted-master-resume.json", extracted)
    transport = prepare_codex_analysis_transport_schema(
        extracted,
        requirements,
        run,
    )
    python_id = next(
        item["requirement_id"]
        for item in requirements["requirements"]
        if item["exact_text"] == "Python"
    )
    raw_analysis = {
        "role_summary": "Synthetic evidence role",
        "fit_assessment": {
            "overall": "Synthetic fit",
            "strengths": ["Python"],
            "gaps": ["Unsupported synthetic requirements"],
        },
        "supported_requirement_mappings": [
            {
                "requirement_id": python_id,
                "evidence_source_ids": ["skill_groups.0"],
                "strength": "strong",
            }
        ],
        "unsupported_requirement_ids": [
            item["requirement_id"]
            for item in requirements["requirements"]
            if item["requirement_id"] != python_id
        ],
        "recommended_edits": [],
        "immutable_facts": [],
        "forbidden_claims": ["Unsupported synthetic claim"],
        "content_budget_guidance": [],
        "questions_for_user": [],
    }
    resolved, issues = resolve_analysis_evidence(
        raw_analysis,
        extracted,
        requirements,
    )
    assert issues == []
    atomic_write_json(run / "codex-analysis-resolved.json", resolved)
    source_hash = sha256_file(master_resume)
    approval = record_codex_analysis_approval(
        run,
        source_resume_sha256=source_hash,
        company="Synthetic Systems",
        role="Evidence Engineer",
        approval_mode="interactive",
    )
    atomic_write_json(
        run / "run-metadata.json",
        {
            "application": "resume-tailor",
            "status": "FAILED",
            "stage": "antigravity-tailoring",
            "failure_class": "antigravity-launch-size",
            "company": "Synthetic Systems",
            "role": "Evidence Engineer",
            "job_source": "job-file",
            "source_resume": {
                "filename": master_resume.name,
                "sha256_before": source_hash,
                "sha256_after": source_hash,
                "unchanged": True,
            },
            "analysis_inputs": analysis_input_manifest(
                run,
                source_resume_sha256=source_hash,
            ),
            "job_requirement_catalog": {
                "filename": "job-requirements.json",
                "sha256": sha256_file(run / "job-requirements.json"),
                "requirement_count": len(requirements["requirements"]),
                "source_kind": requirements["source_kind"],
            },
            "codex_analysis_transport_schema": transport.metadata(),
            "codex_analysis_approval": approval,
            "error": {
                "type": "AntigravityLaunchSizeError",
                "message": (
                    "Antigravity could not start because the request exceeded "
                    "the operating system's command-line size."
                ),
                "exit_code": 3,
            },
        },
    )
    return run


def _write_antigravity_waiting_failure(
    directory: Path,
    master_resume: Path,
) -> Path:
    run = _write_antigravity_launch_failure(directory, master_resume)
    atomic_write_json(
        run / "antigravity-response.json",
        {
            "status": "SUCCESS",
            "structured_output": {
                "status": "WAITING",
                "message": (
                    "Plan mode activated. Ready after synthetic requirements "
                    "are provided."
                ),
                "questions_for_user": [
                    "What synthetic task would you like to plan?"
                ],
                "tailored_resume": None,
            },
        },
    )
    metadata_path = run / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("failure_class", None)
    metadata["error"] = {
        "type": "WaitingError",
        "message": (
            "Antigravity needs more information; review the synthetic questions."
        ),
        "exit_code": 3,
    }
    atomic_write_json(metadata_path, metadata)
    return run


def test_antigravity_retry_authenticates_every_approved_input_without_mutation(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_launch_failure(tmp_path, master_resume)
    before = {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }

    context = build_antigravity_retry_context(
        run,
        current_resume=master_resume,
    )
    inputs = load_antigravity_retry_inputs(
        context,
        current_resume=master_resume,
    )

    assert inputs.context == context
    assert inputs.approved_analysis["questions_for_user"] == []
    assert context.approval_record_sha256 == sha256_file(
        run / ANALYSIS_APPROVAL_FILENAME
    )
    assert {
        "job-description.txt",
        "job-requirements.json",
        "extracted-master-resume.json",
        "codex-analysis-transport.schema.json",
        "codex-analysis-resolved.json",
        ANALYSIS_APPROVAL_FILENAME,
    } <= set(inputs.artifact_bytes)
    assert before == {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }


def test_waiting_contract_recovery_authenticates_every_input_without_mutation(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_waiting_failure(tmp_path, master_resume)
    before = {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }

    context = build_antigravity_retry_context(
        run,
        current_resume=master_resume,
    )
    inputs = load_antigravity_retry_inputs(
        context,
        current_resume=master_resume,
    )

    assert context.failure_kind == "legacy_needs_information"
    assert inputs.context == context
    assert inputs.approved_analysis["questions_for_user"] == []
    assert before == {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }


def test_waiting_contract_recovery_refuses_response_with_tailored_content(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_waiting_failure(tmp_path, master_resume)
    response_path = run / "antigravity-response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["structured_output"]["tailored_resume"] = {
        "synthetic": "content must not be classified as waiting"
    }
    atomic_write_json(response_path, response)

    with pytest.raises(InputError, match="does not match"):
        build_antigravity_retry_context(
            run,
            current_resume=master_resume,
        )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "job-description.txt",
        "job-requirements.json",
        "extracted-master-resume.json",
        "codex-analysis-transport.schema.json",
        "codex-analysis-resolved.json",
        ANALYSIS_APPROVAL_FILENAME,
    ],
)
def test_antigravity_retry_refuses_changed_authenticated_artifact(
    artifact_name: str,
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_launch_failure(tmp_path, master_resume)
    path = run / artifact_name
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(InputError, match="changed|invalid|matches"):
        build_antigravity_retry_context(
            run,
            current_resume=master_resume,
        )


def test_antigravity_retry_refuses_changed_master_resume(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_launch_failure(tmp_path, master_resume)
    changed = tmp_path / "changed-synthetic-resume.docx"
    changed.write_bytes(master_resume.read_bytes() + b" ")

    with pytest.raises(InputError, match="source résumé hash changed"):
        build_antigravity_retry_context(run, current_resume=changed)


def test_antigravity_retry_refuses_missing_approval_record(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _write_antigravity_launch_failure(tmp_path, master_resume)
    metadata = json.loads((run / "run-metadata.json").read_text(encoding="utf-8"))
    metadata.pop("codex_analysis_approval")
    atomic_write_json(run / "run-metadata.json", metadata)

    with pytest.raises(InputError, match="predates the authenticated"):
        build_antigravity_retry_context(run, current_resume=master_resume)


def _make_response_envelope_failure(
    directory: Path,
    master_resume: Path,
    *,
    valid_response: bool,
    with_ancestry: bool = False,
) -> Path:
    source = _write_antigravity_launch_failure(directory, master_resume)
    if with_ancestry:
        source.rename(directory / "synthetic-antigravity-ancestor")
        ancestor = directory / "synthetic-antigravity-ancestor"
        source = directory / "synthetic-antigravity-response-failure"
        shutil.copytree(ancestor, source)
    else:
        source.rename(directory / "synthetic-antigravity-response-failure")
        source = directory / "synthetic-antigravity-response-failure"

    extracted = json.loads(
        (source / "extracted-master-resume.json").read_text(encoding="utf-8")
    )
    complete = {
        "status": "complete",
        "message": "Applied the approved synthetic edit plan.",
        "cannot_apply": None,
        "technical_failure": None,
        "tailored_resume": extracted["content"],
    }
    response = {
        "conversation_id": "00000000-0000-4000-8000-000000000000",
        "duration_seconds": 1.25,
        "json_schema": load_schema("tailored_resume.schema.json"),
        "num_turns": 1,
        "response": (
            json.dumps(complete, ensure_ascii=False)
            if valid_response
            else "Synthetic prose followed by {\"status\":\"complete\"}."
        ),
        "status": "SUCCESS",
        "usage": {
            "cache_read_tokens": 10,
            "input_tokens": 20,
            "output_tokens": 5,
            "thinking_tokens": 2,
            "total_tokens": 25,
        },
    }
    atomic_write_json(source / "antigravity-response.json", response)
    metadata_path = source / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["failure_class"] = "antigravity-response-envelope"
    metadata["error"] = {
        "type": "AntigravityResponseEnvelopeError",
        "message": "Antigravity returned JSON in an unsupported response format.",
        "exit_code": 10,
    }
    metadata["tools"] = {"antigravity": "1.1.8-stub"}
    metadata["artifacts"] = sorted(
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name != "run-metadata.json"
    ) + ["run-metadata.json"]
    if with_ancestry:
        metadata["retry_of"] = "synthetic-antigravity-ancestor"
        metadata["retry_kind"] = "antigravity-tailoring"
        metadata["recovery_inputs"] = {
            "source_resume_sha256": metadata["source_resume"]["sha256_before"],
            "extracted_resume_sha256": metadata["analysis_inputs"][
                "extracted_resume_sha256"
            ],
            "job_description_sha256": metadata["analysis_inputs"][
                "job_description_sha256"
            ],
            "job_requirements_sha256": metadata["analysis_inputs"][
                "job_requirements_sha256"
            ],
            "transport_schema_sha256": metadata[
                "codex_analysis_transport_schema"
            ]["sha256"],
            "resolved_analysis_sha256": sha256_file(
                source / "codex-analysis-resolved.json"
            ),
            "approval_record_sha256": metadata["codex_analysis_approval"][
                "sha256"
            ],
        }
    atomic_write_json(metadata_path, metadata)
    return source


def test_authenticated_valid_response_can_be_reprocessed_without_mutating_source(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _make_response_envelope_failure(
        tmp_path,
        master_resume,
        valid_response=True,
        with_ancestry=True,
    )
    before = {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }

    context = build_antigravity_reprocess_context(
        run,
        current_resume=master_resume,
    )
    inputs = load_antigravity_reprocess_inputs(
        context,
        current_resume=master_resume,
    )

    assert context.envelope_type == "json-wrapper-response"
    assert context.ancestry_run == "synthetic-antigravity-ancestor"
    assert inputs.response_metadata["reprocessed_offline"] is True
    assert inputs.tailored_content == inputs.retry_inputs.extracted_resume["content"]
    assert before == {
        path.name: sha256_file(path)
        for path in run.iterdir()
        if path.is_file()
    }


def test_unstructured_response_cannot_be_reprocessed_but_remains_retryable(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    run = _make_response_envelope_failure(
        tmp_path,
        master_resume,
        valid_response=False,
    )

    retry = build_antigravity_retry_context(
        run,
        current_resume=master_resume,
    )
    assert retry.failure_kind == "response_envelope"
    with pytest.raises(InputError, match="cannot be reprocessed|unavailable"):
        build_antigravity_reprocess_context(
            run,
            current_resume=master_resume,
        )
