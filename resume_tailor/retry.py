from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import (
    ANALYSIS_RESOLVED_FILENAME,
    CODEX_ANALYSIS_RESOLVED_FILENAME,
    unwrap_resolved_analysis_document,
)
from .utilities import InputError, atomic_write_json, sha256_file, utc_now_iso


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 2_000_000
_MAX_EXTRACTION_BYTES = 4_000_000
_MAX_JOB_BYTES = 500_001
_MAX_SCHEMA_BYTES = 250_000
_MAX_ANTIGRAVITY_RESPONSE_BYTES = 4_000_000
ANALYSIS_APPROVAL_FILENAME = "codex-analysis-approval.json"


@dataclass(frozen=True)
class RetryContext:
    source_directory: Path
    company: str
    role: str
    source_resume_sha256: str
    extracted_resume_sha256: str
    job_description_sha256: str
    job_requirements_sha256: str
    legacy_verified: bool = False


@dataclass(frozen=True)
class RetryInputs:
    context: RetryContext
    extracted_resume: dict[str, Any]
    job_description: str
    job_requirements: dict[str, Any]


@dataclass(frozen=True)
class AntigravityRetryContext:
    source_directory: Path
    company: str
    role: str
    job_source: str
    source_resume_sha256: str
    extracted_resume_sha256: str
    job_description_sha256: str
    job_requirements_sha256: str
    transport_schema_sha256: str
    resolved_analysis_sha256: str
    approval_record_sha256: str
    failure_kind: str
    job_source_sha256: str | None = None


@dataclass(frozen=True)
class AntigravityRetryInputs:
    context: AntigravityRetryContext
    extracted_resume: dict[str, Any]
    job_description: str
    job_requirements: dict[str, Any]
    approved_analysis: dict[str, Any]
    artifact_bytes: dict[str, bytes]


@dataclass(frozen=True)
class AntigravityReprocessContext:
    retry_context: AntigravityRetryContext
    response_sha256: str
    tailoring_schema_sha256: str
    envelope_type: str
    source_cli_version: str
    ancestry_run: str | None = None
    ancestry_metadata_sha256: str | None = None


@dataclass(frozen=True)
class AntigravityReprocessInputs:
    context: AntigravityReprocessContext
    retry_inputs: AntigravityRetryInputs
    tailored_content: dict[str, Any]
    response_bytes: bytes
    response_metadata: dict[str, Any]


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_child(
    directory: Path,
    name: str,
    *,
    maximum: int,
) -> bytes:
    path = directory / name
    try:
        parent = directory.resolve(strict=True)
        if path.is_symlink() or not path.is_file() or path.resolve().parent != parent:
            raise InputError(f"Retry input {name!r} is not a safe regular artifact.")
        if path.stat().st_size > maximum:
            raise InputError(f"Retry input {name!r} exceeds its safety limit.")
        return path.read_bytes()
    except InputError:
        raise
    except OSError as exc:
        raise InputError(f"Retry input {name!r} could not be read safely.") from exc


def _json_object(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"Stored {label} is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise InputError(f"Stored {label} must be a JSON object.")
    return payload


def _hash_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise InputError(f"Stored retry metadata is missing the {label} hash.")
    return value


def _canonical_json(value: Any) -> str:
    """Normalize JSON-compatible containers before integrity comparison."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_failure(metadata: dict[str, Any]) -> bool:
    if metadata.get("failure_class") == "source-evidence-analysis":
        return True
    error = metadata.get("error")
    if not isinstance(error, dict):
        return False
    error_type = error.get("type")
    message = error.get("message")
    return (
        metadata.get("stage") == "codex-analysis"
        and error_type in {"SourceEvidenceError", "TruthfulnessError"}
        and isinstance(message, str)
        and message.startswith(
            "Codex analysis failed local source-evidence validation:"
        )
    )


def analysis_input_manifest(
    run_directory: Path,
    *,
    source_resume_sha256: str,
) -> dict[str, Any]:
    extraction = _read_child(
        run_directory,
        "extracted-master-resume.json",
        maximum=_MAX_EXTRACTION_BYTES,
    )
    job = _read_child(
        run_directory,
        "job-description.txt",
        maximum=_MAX_JOB_BYTES,
    )
    job_requirements = _read_child(
        run_directory,
        "job-requirements.json",
        maximum=_MAX_METADATA_BYTES,
    )
    return {
        "version": 2,
        "source_resume_sha256": _hash_value(
            source_resume_sha256,
            label="source résumé",
        ),
        "extracted_resume_sha256": _digest(extraction),
        "job_description_sha256": _digest(job),
        "job_requirements_sha256": _digest(job_requirements),
    }


def _legacy_job_requirement_catalog(
    source_directory: Path,
    *,
    job_description: str,
    job_bytes: bytes,
) -> dict[str, Any]:
    from .job_requirements import build_job_requirement_catalog

    structured_job: dict[str, Any] | None = None
    job_source_path = source_directory / "job-source.json"
    if job_source_path.exists():
        candidate = _json_object(
            _read_child(
                source_directory,
                "job-source.json",
                maximum=_MAX_METADATA_BYTES,
            ),
            label="structured job source",
        )
        confirmed_description = candidate.get("normalized_job_description")
        if isinstance(confirmed_description, str):
            confirmed_bytes = (confirmed_description.rstrip() + "\n").encode("utf-8")
            if confirmed_bytes == job_bytes:
                structured_job = candidate
    return build_job_requirement_catalog(
        job_description,
        structured_job=structured_job,
    )


def _validated_retry_payloads(
    source_directory: Path,
    *,
    current_resume: Path,
) -> tuple[RetryContext, dict[str, Any], str, dict[str, Any]]:
    if source_directory.is_symlink() or not source_directory.is_dir():
        raise InputError("The preserved retry directory is not a safe directory.")
    source_directory = source_directory.resolve()

    metadata = _json_object(
        _read_child(
            source_directory,
            "run-metadata.json",
            maximum=_MAX_METADATA_BYTES,
        ),
        label="run metadata",
    )
    if metadata.get("application") != "resume-tailor":
        raise InputError("The preserved directory is not a Resume Tailor run.")
    if metadata.get("status") != "FAILED" or not _source_failure(metadata):
        raise InputError(
            "Retry is available only for a failed Codex source-evidence analysis."
        )

    company = metadata.get("company")
    role = metadata.get("role")
    if not isinstance(company, str) or not company.strip():
        raise InputError("Stored retry metadata is missing the confirmed company.")
    if not isinstance(role, str) or not role.strip():
        raise InputError("Stored retry metadata is missing the confirmed role.")

    extraction_bytes = _read_child(
        source_directory,
        "extracted-master-resume.json",
        maximum=_MAX_EXTRACTION_BYTES,
    )
    extraction = _json_object(extraction_bytes, label="résumé extraction")
    job_bytes = _read_child(
        source_directory,
        "job-description.txt",
        maximum=_MAX_JOB_BYTES,
    )
    try:
        job_description = job_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("Stored job description is not valid UTF-8 text.") from exc
    if not job_description.strip():
        raise InputError("Stored job description is empty.")

    source = metadata.get("source_resume")
    extraction_source = extraction.get("source")
    if not isinstance(source, dict) or not isinstance(extraction_source, dict):
        raise InputError("Stored retry metadata is missing source résumé integrity data.")
    before_hash = _hash_value(source.get("sha256_before"), label="source résumé")
    after_hash = _hash_value(source.get("sha256_after"), label="final source résumé")
    extracted_source_hash = _hash_value(
        extraction_source.get("sha256"),
        label="extracted source résumé",
    )
    if source.get("unchanged") is not True or len(
        {before_hash, after_hash, extracted_source_hash}
    ) != 1:
        raise InputError(
            "The preserved source résumé integrity chain does not match; start a new run."
        )
    if not current_resume.is_file() or sha256_file(current_resume) != before_hash:
        raise InputError(
            "The current source résumé hash changed; start a new run instead of retrying."
        )

    # A stored extraction cannot authenticate itself. Re-extract the unchanged
    # source locally and compare every legacy field before constructing a fresh
    # source catalog. JSON normalization accounts only for tuple/list encoding.
    from .docx_extract import extract_resume

    current_extraction, _ = extract_resume(current_resume)
    comparison_extraction = dict(current_extraction)
    if "source_blocks" not in extraction:
        comparison_extraction.pop("source_blocks", None)
    if _canonical_json(extraction) != _canonical_json(comparison_extraction):
        raise InputError(
            "The stored résumé extraction does not match the unchanged source; "
            "start a new run instead of retrying."
        )

    extraction_hash = _digest(extraction_bytes)
    job_hash = _digest(job_bytes)
    manifest = metadata.get("analysis_inputs")
    legacy_verified = False
    job_requirements: dict[str, Any]
    job_requirements_hash: str
    if isinstance(manifest, dict):
        if manifest.get("version") not in {1, 2}:
            raise InputError("Stored analysis-input hash metadata has an unknown version.")
        if _hash_value(
            manifest.get("source_resume_sha256"),
            label="analysis source résumé",
        ) != before_hash:
            raise InputError("Stored source résumé hashes disagree; start a new run.")
        if _hash_value(
            manifest.get("extracted_resume_sha256"),
            label="résumé extraction",
        ) != extraction_hash:
            raise InputError(
                "The stored résumé extraction changed; start a new run instead of retrying."
            )
        if _hash_value(
            manifest.get("job_description_sha256"),
            label="job description",
        ) != job_hash:
            raise InputError(
                "The stored job description changed; start a new run instead of retrying."
            )
        if manifest.get("version") == 2:
            requirement_bytes = _read_child(
                source_directory,
                "job-requirements.json",
                maximum=_MAX_METADATA_BYTES,
            )
            job_requirements = _json_object(
                requirement_bytes,
                label="job-requirement catalog",
            )
            from .job_requirements import validate_job_requirement_catalog

            validate_job_requirement_catalog(
                job_requirements,
                job_description=job_description.rstrip("\n"),
            )
            job_requirements_hash = _digest(requirement_bytes)
            if _hash_value(
                manifest.get("job_requirements_sha256"),
                label="job-requirement catalog",
            ) != job_requirements_hash:
                raise InputError(
                    "The stored job-requirement catalog changed; start a new run "
                    "instead of retrying."
                )
        else:
            job_requirements = _legacy_job_requirement_catalog(
                source_directory,
                job_description=job_description.rstrip("\n"),
                job_bytes=job_bytes,
            )
            job_requirements_hash = _digest(
                (json.dumps(
                    job_requirements,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n").encode("utf-8")
            )
            legacy_verified = True
    else:
        job_source = _json_object(
            _read_child(
                source_directory,
                "job-source.json",
                maximum=_MAX_METADATA_BYTES,
            ),
            label="structured job source",
        )
        confirmed_description = job_source.get("normalized_job_description")
        if not isinstance(confirmed_description, str):
            raise InputError(
                "This legacy run has no independently stored confirmed job text; "
                "start a new run."
            )
        confirmed_bytes = (confirmed_description.rstrip() + "\n").encode("utf-8")
        if confirmed_bytes != job_bytes:
            raise InputError(
                "The legacy confirmed job inputs disagree; start a new run instead "
                "of retrying."
            )
        legacy_verified = True
        job_requirements = _legacy_job_requirement_catalog(
            source_directory,
            job_description=job_description.rstrip("\n"),
            job_bytes=job_bytes,
        )
        job_requirements_hash = _digest(
            (json.dumps(
                job_requirements,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n").encode("utf-8")
        )

    context = RetryContext(
        source_directory=source_directory,
        company=company.strip(),
        role=role.strip(),
        source_resume_sha256=before_hash,
        extracted_resume_sha256=extraction_hash,
        job_description_sha256=job_hash,
        job_requirements_sha256=job_requirements_hash,
        legacy_verified=legacy_verified,
    )
    return (
        context,
        current_extraction,
        job_description.rstrip("\n"),
        job_requirements,
    )


def build_retry_context(
    source_directory: Path,
    *,
    current_resume: Path,
) -> RetryContext:
    context, _, _, _ = _validated_retry_payloads(
        source_directory,
        current_resume=current_resume,
    )
    return context


def load_retry_inputs(
    context: RetryContext,
    *,
    current_resume: Path,
) -> RetryInputs:
    current, extraction, job_description, job_requirements = _validated_retry_payloads(
        context.source_directory,
        current_resume=current_resume,
    )
    if current != context:
        raise InputError(
            "Stored retry inputs changed after verification; start a new run."
        )
    return RetryInputs(
        context=current,
        extracted_resume=extraction,
        job_description=job_description,
        job_requirements=job_requirements,
    )


_COMMON_APPROVED_ANALYSIS_ARTIFACTS: dict[str, tuple[str, int]] = {
    "job_description": ("job-description.txt", _MAX_JOB_BYTES),
    "job_requirements": ("job-requirements.json", _MAX_METADATA_BYTES),
    "resume_extraction": ("extracted-master-resume.json", _MAX_EXTRACTION_BYTES),
}

# Historical Codex-only key retained for reading older approval records.
_LEGACY_RESOLVED_CODEX_KEY = "resolved_codex_analysis"
_LEGACY_RESOLVED_CODEX_NAME = CODEX_ANALYSIS_RESOLVED_FILENAME

_APPROVED_ANALYSIS_ARTIFACTS: dict[str, tuple[str, int]] = {
    **_COMMON_APPROVED_ANALYSIS_ARTIFACTS,
    "codex_transport_schema": (
        "codex-analysis-transport.schema.json",
        _MAX_SCHEMA_BYTES,
    ),
    "resolved_analysis": (ANALYSIS_RESOLVED_FILENAME, _MAX_METADATA_BYTES),
    # Legacy alias kept only so historical approval records still authenticate.
    _LEGACY_RESOLVED_CODEX_KEY: (
        _LEGACY_RESOLVED_CODEX_NAME,
        _MAX_METADATA_BYTES,
    ),
}


def _artifact_size_limit(filename: str) -> int:
    if filename == "job-description.txt":
        return _MAX_JOB_BYTES
    if filename.endswith(".schema.json") or filename.endswith("-schema.json"):
        return _MAX_SCHEMA_BYTES
    if filename == "extracted-master-resume.json":
        return _MAX_EXTRACTION_BYTES
    return _MAX_METADATA_BYTES


def _approval_artifacts_for_run(run_directory: Path) -> dict[str, tuple[str, int]]:
    """Select approval artifact paths for the current run's analysis provider."""
    artifacts = dict(_COMMON_APPROVED_ANALYSIS_ARTIFACTS)
    # Prefer the provider-neutral resolved document when present.
    if (run_directory / ANALYSIS_RESOLVED_FILENAME).is_file():
        artifacts["resolved_analysis"] = (
            ANALYSIS_RESOLVED_FILENAME,
            _MAX_METADATA_BYTES,
        )
    elif (run_directory / CODEX_ANALYSIS_RESOLVED_FILENAME).is_file():
        # Pre-neutralization Codex runs (or Codex alias-only recoveries).
        artifacts[_LEGACY_RESOLVED_CODEX_KEY] = (
            CODEX_ANALYSIS_RESOLVED_FILENAME,
            _MAX_METADATA_BYTES,
        )
    else:
        raise InputError(
            "Resolved analysis artifact is missing; analysis approval cannot be "
            "recorded."
        )
    if (run_directory / "codex-analysis-transport.schema.json").is_file():
        artifacts["codex_transport_schema"] = (
            "codex-analysis-transport.schema.json",
            _MAX_SCHEMA_BYTES,
        )
    if (run_directory / "grok-analysis-schema.json").is_file():
        artifacts["grok_analysis_schema"] = (
            "grok-analysis-schema.json",
            _MAX_SCHEMA_BYTES,
        )
    return artifacts


def _resolved_analysis_from_parsed(
    parsed: dict[str, dict[str, Any]],
    approval: dict[str, Any],
) -> dict[str, Any]:
    """Load the bare analysis from a new or historical approved artifact."""
    approval_artifacts = approval.get("artifacts")
    if not isinstance(approval_artifacts, dict):
        raise InputError("The stored analysis approval is missing artifact metadata.")
    for key in ("resolved_analysis", _LEGACY_RESOLVED_CODEX_KEY):
        entry = approval_artifacts.get(key)
        if not isinstance(entry, dict):
            continue
        name = entry.get("filename")
        if not isinstance(name, str) or name not in parsed:
            continue
        return unwrap_resolved_analysis_document(parsed[name])
    # Fallback for partially loaded historical fixtures.
    for name in (ANALYSIS_RESOLVED_FILENAME, CODEX_ANALYSIS_RESOLVED_FILENAME):
        if name in parsed:
            return unwrap_resolved_analysis_document(parsed[name])
    raise InputError(
        "The stored analysis approval is missing an authenticated resolved "
        "analysis artifact."
    )


def record_codex_analysis_approval(
    run_directory: Path,
    *,
    source_resume_sha256: str,
    company: str,
    role: str,
    approval_mode: str,
) -> dict[str, Any]:
    """Persist a hash-only record after the analysis gate is approved."""
    if approval_mode not in {"interactive", "assume_yes"}:
        raise InputError("The Codex approval mode is invalid.")
    source_hash = _hash_value(source_resume_sha256, label="source résumé")
    artifacts: dict[str, dict[str, str]] = {}
    for key, (name, maximum) in _approval_artifacts_for_run(run_directory).items():
        value = _read_child(run_directory, name, maximum=maximum)
        artifacts[key] = {"filename": name, "sha256": _digest(value)}
    job_source = run_directory / "job-source.json"
    if job_source.exists():
        value = _read_child(
            run_directory,
            "job-source.json",
            maximum=_MAX_METADATA_BYTES,
        )
        artifacts["job_source"] = {
            "filename": "job-source.json",
            "sha256": _digest(value),
        }
    record = {
        "version": 1,
        "kind": "codex-analysis",
        "decision": "approved",
        "approval_mode": approval_mode,
        "approved_at": utc_now_iso(),
        "confirmed_identity": {
            "company": company,
            "role": role,
        },
        "source_resume_sha256": source_hash,
        "artifacts": artifacts,
    }
    path = run_directory / ANALYSIS_APPROVAL_FILENAME
    atomic_write_json(path, record)
    return {
        "filename": ANALYSIS_APPROVAL_FILENAME,
        "sha256": sha256_file(path),
        "version": 1,
        "decision": "approved",
    }


def is_antigravity_launch_size_failure(metadata: dict[str, Any]) -> bool:
    if metadata.get("failure_class") == "antigravity-launch-size":
        return True
    error = metadata.get("error")
    if not isinstance(error, dict):
        return False
    message = error.get("message")
    return (
        metadata.get("status") == "FAILED"
        and metadata.get("stage") == "antigravity-tailoring"
        and error.get("type")
        in {"AntigravityLaunchSizeError", "DependencyError"}
        and isinstance(message, str)
        and "Argument list too long" in message
        and ("agy" in message or "Antigravity" in message)
    )


def antigravity_retry_failure_kind(metadata: dict[str, Any]) -> str | None:
    if is_antigravity_launch_size_failure(metadata):
        return "launch_size"
    failure_class = metadata.get("failure_class")
    mapped = {
        "antigravity-response-envelope": "response_envelope",
        "antigravity-tailoring-contract": "tailoring_contract",
        "antigravity-cannot-apply": "cannot_apply",
        "antigravity-technical-failure": "technical_failure",
    }.get(failure_class)
    if mapped is not None:
        return mapped
    error = metadata.get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    if (
        metadata.get("status") == "FAILED"
        and metadata.get("stage") == "antigravity-tailoring"
        and error.get("type") == "WaitingError"
        and isinstance(message, str)
        and message.startswith("Antigravity needs more information;")
    ):
        return "legacy_needs_information"
    if (
        metadata.get("status") == "FAILED"
        and metadata.get("stage") == "antigravity-tailoring"
        and error.get("type") in {"ModelError", "AntigravityResponseEnvelopeError"}
        and isinstance(message, str)
        and message
        in {
            "Antigravity JSON did not contain structured_output.",
            "Antigravity returned JSON in an unsupported response format.",
        }
    ):
        return "response_envelope"
    return None


def _stored_antigravity_status(source_directory: Path) -> tuple[str | None, bool]:
    payload = _json_object(
        _read_child(
            source_directory,
            "antigravity-response.json",
            maximum=_MAX_METADATA_BYTES,
        ),
        label="Antigravity response",
    )
    candidate = payload.get("structured_output")
    if isinstance(candidate, str):
        candidate = _json_object(
            candidate.encode("utf-8"),
            label="Antigravity structured response",
        )
    if not isinstance(candidate, dict):
        candidate = payload
    return (
        candidate.get("status")
        if isinstance(candidate.get("status"), str)
        else None,
        candidate.get("tailored_resume") is not None,
    )


def _approved_artifact_bytes(
    source_directory: Path,
    approval: dict[str, Any],
    *,
    key: str,
    expected_name: str,
    maximum: int,
) -> bytes:
    artifacts = approval.get("artifacts")
    entry = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(entry, dict) or entry.get("filename") != expected_name:
        raise InputError(
            f"The stored Codex approval record is missing authenticated {key} data; "
            "start a new run."
        )
    value = _read_child(source_directory, expected_name, maximum=maximum)
    if _hash_value(entry.get("sha256"), label=f"approved {key}") != _digest(value):
        raise InputError(
            f"The authenticated {key} artifact changed after approval; start a new run."
        )
    return value


def verify_tailoring_run_artifacts(
    run_directory: Path,
    *,
    source_resume_sha256: str,
    extracted_resume: dict[str, Any],
    job_description: str,
    job_requirements: dict[str, Any],
    approved_analysis: dict[str, Any],
    company: str,
    role: str,
) -> dict[str, Any]:
    """Authenticate every required post-approval artifact before Antigravity."""
    source_hash = _hash_value(source_resume_sha256, label="source résumé")
    metadata = _json_object(
        _read_child(
            run_directory,
            "run-metadata.json",
            maximum=_MAX_METADATA_BYTES,
        ),
        label="run metadata",
    )
    approval_metadata = metadata.get("codex_analysis_approval")
    if (
        not isinstance(approval_metadata, dict)
        or approval_metadata.get("filename") != ANALYSIS_APPROVAL_FILENAME
        or approval_metadata.get("decision") != "approved"
        or approval_metadata.get("version") != 1
    ):
        raise InputError(
            "Authenticated Codex approval is missing; no Antigravity provider "
            "request was launched."
        )
    approval_bytes = _read_child(
        run_directory,
        ANALYSIS_APPROVAL_FILENAME,
        maximum=_MAX_METADATA_BYTES,
    )
    approval_hash = _digest(approval_bytes)
    if _hash_value(
        approval_metadata.get("sha256"),
        label="Codex approval record",
    ) != approval_hash:
        raise InputError(
            "The Codex approval record hash changed; no Antigravity provider "
            "request was launched."
        )
    approval = _json_object(approval_bytes, label="Codex approval record")
    identity = approval.get("confirmed_identity")
    if (
        approval.get("version") != 1
        or approval.get("kind") != "codex-analysis"
        or approval.get("decision") != "approved"
        or not isinstance(identity, dict)
        or identity.get("company") != company
        or identity.get("role") != role
        or _hash_value(
            approval.get("source_resume_sha256"),
            label="approved source résumé",
        )
        != source_hash
    ):
        raise InputError(
            "The Codex approval record does not match the confirmed tailoring "
            "inputs; no Antigravity provider request was launched."
        )

    artifacts: dict[str, bytes] = {}
    approval_artifacts = approval.get("artifacts")
    if not isinstance(approval_artifacts, dict):
        raise InputError(
            "The stored Codex approval record is missing artifact metadata; "
            "no Antigravity provider request was launched."
        )
    for key, entry in approval_artifacts.items():
        if key == "job_source":
            continue
        if not isinstance(entry, dict):
            continue
        name = entry.get("filename")
        if not isinstance(name, str) or not name:
            raise InputError(
                f"The stored Codex approval record is missing authenticated {key} "
                "data; start a new run."
            )
        artifacts[name] = _approved_artifact_bytes(
            run_directory,
            approval,
            key=key,
            expected_name=name,
            maximum=_artifact_size_limit(name),
        )
    if approval_artifacts.get("job_source") is not None:
        _approved_artifact_bytes(
            run_directory,
            approval,
            key="job_source",
            expected_name="job-source.json",
            maximum=_MAX_METADATA_BYTES,
        )

    try:
        stored_job = artifacts["job-description.txt"].decode("utf-8").rstrip("\n")
    except UnicodeDecodeError as exc:
        raise InputError(
            "The approved job description is not valid UTF-8; no Antigravity "
            "provider request was launched."
        ) from exc
    stored_resolved = _resolved_analysis_from_parsed(
        {
            name: _json_object(value, label=name)
            for name, value in artifacts.items()
            if name.endswith(".json")
        },
        approval,
    )
    comparisons = {
        "confirmed job description": stored_job == job_description.rstrip("\n"),
        "résumé extraction": _canonical_json(
            _json_object(
                artifacts["extracted-master-resume.json"],
                label="résumé extraction",
            )
        )
        == _canonical_json(extracted_resume),
        "job-requirement catalog": _canonical_json(
            _json_object(
                artifacts["job-requirements.json"],
                label="job-requirement catalog",
            )
        )
        == _canonical_json(job_requirements),
        "resolved analysis": _canonical_json(stored_resolved)
        == _canonical_json(approved_analysis),
    }
    if not all(comparisons.values()):
        raise InputError(
            "An in-memory tailoring input no longer matches its approved artifact; "
            "no Antigravity provider request was launched."
        )

    manifest = metadata.get("analysis_inputs")
    expected_manifest = {
        "source_resume_sha256": source_hash,
        "extracted_resume_sha256": _digest(
            artifacts["extracted-master-resume.json"]
        ),
        "job_description_sha256": _digest(artifacts["job-description.txt"]),
        "job_requirements_sha256": _digest(artifacts["job-requirements.json"]),
    }
    if not isinstance(manifest, dict) or manifest.get("version") != 2:
        raise InputError(
            "Version-2 authenticated analysis inputs are missing; no Antigravity "
            "provider request was launched."
        )
    for key, expected in expected_manifest.items():
        if _hash_value(manifest.get(key), label=key.replace("_", " ")) != expected:
            raise InputError(
                "An authenticated analysis input hash changed; no Antigravity "
                "provider request was launched."
            )
    return {
        "status": "PASS",
        "approval_record_sha256": approval_hash,
        "artifact_count": len(artifacts),
    }


def _validated_antigravity_retry_payloads(
    source_directory: Path,
    *,
    current_resume: Path,
) -> tuple[
    AntigravityRetryContext,
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, bytes],
]:
    if source_directory.is_symlink() or not source_directory.is_dir():
        raise InputError("The preserved recovery directory is not a safe directory.")
    source_directory = source_directory.resolve()
    metadata_bytes = _read_child(
        source_directory,
        "run-metadata.json",
        maximum=_MAX_METADATA_BYTES,
    )
    metadata = _json_object(metadata_bytes, label="run metadata")
    if metadata.get("application") != "resume-tailor":
        raise InputError("The preserved directory is not a Resume Tailor run.")
    failure_kind = antigravity_retry_failure_kind(metadata)
    if failure_kind is None:
        raise InputError(
            "Antigravity recovery is unavailable for this failure stage."
        )
    if failure_kind == "legacy_needs_information":
        status, has_content = _stored_antigravity_status(source_directory)
        if status != "WAITING" or has_content:
            raise InputError(
                "The preserved legacy Antigravity response does not match the "
                "recoverable tailoring-contract failure shape."
            )
    elif failure_kind != "launch_size":
        _stored_antigravity_status(source_directory)

    company = metadata.get("company")
    role = metadata.get("role")
    job_source = metadata.get("job_source")
    if not isinstance(company, str) or not company.strip():
        raise InputError("Stored recovery metadata is missing the confirmed company.")
    if not isinstance(role, str) or not role.strip():
        raise InputError("Stored recovery metadata is missing the confirmed role.")
    if not isinstance(job_source, str) or not job_source:
        raise InputError("Stored recovery metadata is missing the confirmed job source.")

    approval_metadata = metadata.get("codex_analysis_approval")
    if (
        not isinstance(approval_metadata, dict)
        or approval_metadata.get("filename") != ANALYSIS_APPROVAL_FILENAME
        or approval_metadata.get("decision") != "approved"
        or approval_metadata.get("version") != 1
    ):
        raise InputError(
            "This run predates the authenticated Codex approval record; start a new "
            "run rather than retrying Antigravity."
        )
    approval_bytes = _read_child(
        source_directory,
        ANALYSIS_APPROVAL_FILENAME,
        maximum=_MAX_METADATA_BYTES,
    )
    approval_hash = _digest(approval_bytes)
    if _hash_value(
        approval_metadata.get("sha256"),
        label="Codex approval record",
    ) != approval_hash:
        raise InputError(
            "The Codex approval record changed after approval; start a new run."
        )
    approval = _json_object(approval_bytes, label="Codex approval record")
    if (
        approval.get("version") != 1
        or approval.get("kind") != "codex-analysis"
        or approval.get("decision") != "approved"
        or approval.get("approval_mode") not in {"interactive", "assume_yes"}
    ):
        raise InputError(
            "The stored Codex approval record is invalid; start a new run."
        )
    identity = approval.get("confirmed_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("company") != company
        or identity.get("role") != role
    ):
        raise InputError(
            "The confirmed company or role changed after analysis approval; start a "
            "new run."
        )

    artifact_bytes: dict[str, bytes] = {}
    parsed: dict[str, dict[str, Any]] = {}
    approval_artifacts = approval.get("artifacts")
    if not isinstance(approval_artifacts, dict):
        raise InputError(
            "The stored Codex approval record is missing artifact metadata; "
            "start a new run."
        )
    for key, entry in approval_artifacts.items():
        if key == "job_source":
            continue
        if not isinstance(entry, dict):
            continue
        name = entry.get("filename")
        if not isinstance(name, str) or not name:
            raise InputError(
                f"The stored Codex approval record is missing authenticated {key} "
                "data; start a new run."
            )
        value = _approved_artifact_bytes(
            source_directory,
            approval,
            key=key,
            expected_name=name,
            maximum=_artifact_size_limit(name),
        )
        artifact_bytes[name] = value
        if name.endswith(".json"):
            parsed[name] = _json_object(value, label=key.replace("_", " "))
    artifact_bytes[ANALYSIS_APPROVAL_FILENAME] = approval_bytes

    job_source_entry = approval_artifacts.get("job_source")
    job_source_hash: str | None = None
    if job_source_entry is not None:
        value = _approved_artifact_bytes(
            source_directory,
            approval,
            key="job_source",
            expected_name="job-source.json",
            maximum=_MAX_METADATA_BYTES,
        )
        artifact_bytes["job-source.json"] = value
        job_source_hash = _digest(value)
    elif job_source == "linkedin-url":
        raise InputError(
            "The confirmed LinkedIn source is not authenticated by the approval "
            "record; start a new run."
        )

    job_bytes = artifact_bytes["job-description.txt"]
    try:
        job_description = job_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("Stored job description is not valid UTF-8 text.") from exc
    if not job_description.strip():
        raise InputError("Stored job description is empty.")
    extraction = parsed["extracted-master-resume.json"]
    job_requirements = parsed["job-requirements.json"]
    approved_analysis = _resolved_analysis_from_parsed(parsed, approval)

    source = metadata.get("source_resume")
    extraction_source = extraction.get("source")
    if not isinstance(source, dict) or not isinstance(extraction_source, dict):
        raise InputError("Stored recovery metadata is missing résumé integrity data.")
    before_hash = _hash_value(source.get("sha256_before"), label="source résumé")
    after_hash = _hash_value(source.get("sha256_after"), label="final source résumé")
    extracted_source_hash = _hash_value(
        extraction_source.get("sha256"),
        label="extracted source résumé",
    )
    approved_source_hash = _hash_value(
        approval.get("source_resume_sha256"),
        label="approved source résumé",
    )
    if source.get("unchanged") is not True or len(
        {before_hash, after_hash, extracted_source_hash, approved_source_hash}
    ) != 1:
        raise InputError(
            "The preserved source résumé integrity chain does not match; start a "
            "new run."
        )
    if not current_resume.is_file() or sha256_file(current_resume) != before_hash:
        raise InputError(
            "The current source résumé hash changed; start a new run instead of "
            "retrying."
        )
    from .docx_extract import extract_resume

    current_extraction, _ = extract_resume(current_resume)
    if _canonical_json(extraction) != _canonical_json(current_extraction):
        raise InputError(
            "The stored résumé extraction does not match the unchanged source; "
            "start a new run."
        )

    extraction_hash = _digest(artifact_bytes["extracted-master-resume.json"])
    job_hash = _digest(job_bytes)
    requirements_hash = _digest(artifact_bytes["job-requirements.json"])
    manifest = metadata.get("analysis_inputs")
    if not isinstance(manifest, dict) or manifest.get("version") != 2:
        raise InputError(
            "Antigravity recovery requires version-2 authenticated analysis inputs; "
            "start a new run."
        )
    expected_manifest = {
        "source_resume_sha256": before_hash,
        "extracted_resume_sha256": extraction_hash,
        "job_description_sha256": job_hash,
        "job_requirements_sha256": requirements_hash,
    }
    for key, expected in expected_manifest.items():
        if _hash_value(manifest.get(key), label=key.replace("_", " ")) != expected:
            raise InputError(
                "An authenticated analysis input changed after approval; start a "
                "new run."
            )

    from .job_requirements import validate_job_requirement_catalog

    validate_job_requirement_catalog(
        job_requirements,
        job_description=job_description.rstrip("\n"),
    )

    transport_hash = _digest(
        artifact_bytes["codex-analysis-transport.schema.json"]
    )
    transport_metadata = metadata.get("codex_analysis_transport_schema")
    if (
        not isinstance(transport_metadata, dict)
        or transport_metadata.get("filename")
        != "codex-analysis-transport.schema.json"
        or _hash_value(
            transport_metadata.get("sha256"),
            label="Codex transport schema",
        )
        != transport_hash
    ):
        raise InputError(
            "The authenticated Codex transport schema metadata is invalid; start a "
            "new run."
        )
    try:
        from .schemas import (
            CodexAnalysisTransportArtifact,
            validate_codex_analysis_transport_artifact,
        )

        validate_codex_analysis_transport_artifact(
            CodexAnalysisTransportArtifact(
                path=(
                    source_directory
                    / "codex-analysis-transport.schema.json"
                ).resolve(),
                sha256=transport_hash,
                size_bytes=int(transport_metadata["size_bytes"]),
                evidence_source_id_count=int(
                    transport_metadata["evidence_source_id_count"]
                ),
                editable_source_id_count=int(
                    transport_metadata["editable_source_id_count"]
                ),
                job_requirement_id_count=int(
                    transport_metadata["job_requirement_id_count"]
                ),
            ),
            extraction,
            job_requirements,
            source_directory,
        )
    except Exception as exc:
        raise InputError(
            "The authenticated Codex transport schema no longer matches its local "
            "catalogs; start a new run."
        ) from exc

    from .evidence import resolve_analysis_evidence

    re_resolved, issues = resolve_analysis_evidence(
        approved_analysis,
        extraction,
        job_requirements,
    )
    if issues or _canonical_json(re_resolved) != _canonical_json(approved_analysis):
        raise InputError(
            "The approved Codex analysis no longer resolves exactly against its "
            "authenticated catalogs; start a new run."
        )
    if approved_analysis.get("questions_for_user"):
        raise InputError(
            "The approved Codex analysis contains unanswered questions; start a "
            "new run."
        )

    resolved_filename = next(
        (
            name
            for name in (ANALYSIS_RESOLVED_FILENAME, CODEX_ANALYSIS_RESOLVED_FILENAME)
            if name in artifact_bytes
        ),
        None,
    )
    if resolved_filename is None:
        raise InputError(
            "The stored analysis approval is missing an authenticated resolved "
            "analysis artifact."
        )
    resolved_analysis_hash = _digest(artifact_bytes[resolved_filename])
    context = AntigravityRetryContext(
        source_directory=source_directory,
        company=company.strip(),
        role=role.strip(),
        job_source=job_source,
        source_resume_sha256=before_hash,
        extracted_resume_sha256=extraction_hash,
        job_description_sha256=job_hash,
        job_requirements_sha256=requirements_hash,
        transport_schema_sha256=transport_hash,
        resolved_analysis_sha256=resolved_analysis_hash,
        approval_record_sha256=approval_hash,
        failure_kind=failure_kind,
        job_source_sha256=job_source_hash,
    )
    return (
        context,
        current_extraction,
        job_description.rstrip("\n"),
        job_requirements,
        approved_analysis,
        artifact_bytes,
    )


def build_antigravity_retry_context(
    source_directory: Path,
    *,
    current_resume: Path,
) -> AntigravityRetryContext:
    context, _, _, _, _, _ = _validated_antigravity_retry_payloads(
        source_directory,
        current_resume=current_resume,
    )
    return context


def load_antigravity_retry_inputs(
    context: AntigravityRetryContext,
    *,
    current_resume: Path,
) -> AntigravityRetryInputs:
    current, extraction, job_description, requirements, analysis, artifacts = (
        _validated_antigravity_retry_payloads(
            context.source_directory,
            current_resume=current_resume,
        )
    )
    if current != context:
        raise InputError(
            "Stored Antigravity recovery inputs changed after verification; start "
            "a new run."
        )
    return AntigravityRetryInputs(
        context=current,
        extracted_resume=extraction,
        job_description=job_description,
        job_requirements=requirements,
        approved_analysis=analysis,
        artifact_bytes=artifacts,
    )


def _verified_recovery_ancestry(
    source_directory: Path,
    *,
    metadata: dict[str, Any],
    retry_context: AntigravityRetryContext,
) -> tuple[str | None, str | None]:
    retry_of = metadata.get("retry_of")
    if retry_of is None:
        return None, None
    if (
        not isinstance(retry_of, str)
        or not retry_of
        or Path(retry_of).name != retry_of
        or retry_of.startswith(".")
    ):
        raise InputError("The stored Antigravity recovery ancestry is invalid.")
    ancestor = source_directory.parent / retry_of
    if (
        ancestor.is_symlink()
        or not ancestor.is_dir()
        or ancestor.resolve().parent != source_directory.parent.resolve()
    ):
        raise InputError("The stored Antigravity recovery ancestor is unavailable.")

    recovery_inputs = metadata.get("recovery_inputs")
    expected_inputs = {
        "source_resume_sha256": retry_context.source_resume_sha256,
        "extracted_resume_sha256": retry_context.extracted_resume_sha256,
        "job_description_sha256": retry_context.job_description_sha256,
        "job_requirements_sha256": retry_context.job_requirements_sha256,
        "transport_schema_sha256": retry_context.transport_schema_sha256,
        "resolved_analysis_sha256": retry_context.resolved_analysis_sha256,
        "approval_record_sha256": retry_context.approval_record_sha256,
    }
    if not isinstance(recovery_inputs, dict) or any(
        recovery_inputs.get(key) != value
        for key, value in expected_inputs.items()
    ):
        raise InputError(
            "The Antigravity recovery-input hash chain no longer authenticates."
        )

    ancestor_metadata_bytes = _read_child(
        ancestor,
        "run-metadata.json",
        maximum=_MAX_METADATA_BYTES,
    )
    ancestor_metadata = _json_object(
        ancestor_metadata_bytes,
        label="Antigravity recovery-ancestor metadata",
    )
    ancestor_approval = ancestor_metadata.get("codex_analysis_approval")
    if (
        ancestor_metadata.get("application") != "resume-tailor"
        or not isinstance(ancestor_approval, dict)
        or ancestor_approval.get("sha256")
        != retry_context.approval_record_sha256
    ):
        raise InputError(
            "The Antigravity recovery ancestor does not authenticate the approval "
            "record."
        )
    ancestor_approval_bytes = _read_child(
        ancestor,
        ANALYSIS_APPROVAL_FILENAME,
        maximum=_MAX_METADATA_BYTES,
    )
    if _digest(ancestor_approval_bytes) != retry_context.approval_record_sha256:
        raise InputError(
            "The Antigravity recovery ancestor approval artifact changed."
        )
    return retry_of, _digest(ancestor_metadata_bytes)


def _validated_antigravity_reprocess_payloads(
    source_directory: Path,
    *,
    current_resume: Path,
) -> tuple[AntigravityReprocessContext, AntigravityReprocessInputs]:
    (
        retry_context,
        extraction,
        job_description,
        requirements,
        analysis,
        artifacts,
    ) = _validated_antigravity_retry_payloads(
        source_directory,
        current_resume=current_resume,
    )
    if retry_context.failure_kind != "response_envelope":
        raise InputError(
            "Offline reprocessing is available only for an authenticated "
            "Antigravity response-envelope failure."
        )

    metadata_bytes = _read_child(
        source_directory,
        "run-metadata.json",
        maximum=_MAX_METADATA_BYTES,
    )
    metadata = _json_object(metadata_bytes, label="run metadata")
    listed_artifacts = metadata.get("artifacts")
    if (
        not isinstance(listed_artifacts, list)
        or "antigravity-response.json" not in listed_artifacts
    ):
        raise InputError(
            "The preserved Antigravity response is not authenticated by the run "
            "manifest."
        )

    response_bytes = _read_child(
        source_directory,
        "antigravity-response.json",
        maximum=_MAX_ANTIGRAVITY_RESPONSE_BYTES,
    )
    response_hash = _digest(response_bytes)
    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(
            "Stored Antigravity response is not valid UTF-8 text."
        ) from exc
    stored_response_metadata = metadata.get("antigravity_response")
    output_format = "json"
    if stored_response_metadata is not None:
        if not isinstance(stored_response_metadata, dict):
            raise InputError("Stored Antigravity response metadata is invalid.")
        output_format_value = stored_response_metadata.get("output_format")
        if output_format_value not in {"json", "stream-json"}:
            raise InputError("Stored Antigravity output format is unsupported.")
        output_format = output_format_value
    from .antigravity_writer import resolve_tailoring_response_text_with_envelope
    from .schemas import schema_path
    from .utilities import AntigravityResponseEnvelopeError

    tailoring_schema = schema_path("tailored_resume.schema.json")
    tailoring_schema_hash = sha256_file(tailoring_schema)
    try:
        tailored_content, envelope_type = resolve_tailoring_response_text_with_envelope(
            response_text,
            output_format=output_format,
            approved_analysis=analysis,
        )
    except Exception as exc:
        if isinstance(exc, AntigravityResponseEnvelopeError):
            raise InputError(
                "The preserved provider response does not contain one documented, "
                "schema-valid tailoring result; offline reprocessing is unavailable."
            ) from exc
        raise InputError(
            "The preserved provider response fails the strict tailoring contract; "
            "offline reprocessing is unavailable."
        ) from exc

    if stored_response_metadata is not None:
        stored_response = stored_response_metadata.get("response")
        stored_schema = stored_response_metadata.get("schema")
        if (
            not isinstance(stored_response, dict)
            or stored_response.get("filename") != "antigravity-response.json"
            or stored_response.get("sha256") != response_hash
            or not isinstance(stored_schema, dict)
            or stored_schema.get("filename") != tailoring_schema.name
            or stored_schema.get("sha256") != tailoring_schema_hash
        ):
            raise InputError(
                "The stored Antigravity response or expected-schema hash changed."
            )

    from .evidence import validate_tailored_content

    evidence = validate_tailored_content(
        original=extraction["content"],
        tailored=tailored_content,
        extracted_resume=extraction,
        analysis=analysis,
        target_role=retry_context.role,
    )
    if not evidence.passed:
        raise InputError(
            "The preserved response failed local factual-integrity validation; "
            "offline reprocessing is unavailable."
        )

    ancestry_run, ancestry_metadata_hash = _verified_recovery_ancestry(
        source_directory,
        metadata=metadata,
        retry_context=retry_context,
    )
    tools = metadata.get("tools")
    source_cli_version = (
        tools.get("antigravity")
        if isinstance(tools, dict)
        and isinstance(tools.get("antigravity"), str)
        else "unavailable"
    )
    context = AntigravityReprocessContext(
        retry_context=retry_context,
        response_sha256=response_hash,
        tailoring_schema_sha256=tailoring_schema_hash,
        envelope_type=envelope_type,
        source_cli_version=source_cli_version[:200],
        ancestry_run=ancestry_run,
        ancestry_metadata_sha256=ancestry_metadata_hash,
    )
    retry_inputs = AntigravityRetryInputs(
        context=retry_context,
        extracted_resume=extraction,
        job_description=job_description,
        job_requirements=requirements,
        approved_analysis=analysis,
        artifact_bytes=artifacts,
    )
    response_metadata = {
        "version": 1,
        "provider": "antigravity",
        "execution_mode": "print",
        "agent_mode": "default",
        "output_format": output_format,
        "sandboxed": True,
        "response_envelope_type": envelope_type,
        "validation_result": "PASS",
        "cli_version": source_cli_version[:200],
        "reprocessed_offline": True,
        "schema": {
            "filename": tailoring_schema.name,
            "sha256": tailoring_schema_hash,
        },
        "response": {
            "filename": "antigravity-response.json",
            "sha256": response_hash,
        },
    }
    inputs = AntigravityReprocessInputs(
        context=context,
        retry_inputs=retry_inputs,
        tailored_content=tailored_content,
        response_bytes=response_bytes,
        response_metadata=response_metadata,
    )
    return context, inputs


def build_antigravity_reprocess_context(
    source_directory: Path,
    *,
    current_resume: Path,
) -> AntigravityReprocessContext:
    context, _ = _validated_antigravity_reprocess_payloads(
        source_directory,
        current_resume=current_resume,
    )
    return context


def load_antigravity_reprocess_inputs(
    context: AntigravityReprocessContext,
    *,
    current_resume: Path,
) -> AntigravityReprocessInputs:
    current, inputs = _validated_antigravity_reprocess_payloads(
        context.retry_context.source_directory,
        current_resume=current_resume,
    )
    if current != context:
        raise InputError(
            "Stored Antigravity reprocessing inputs changed after verification; "
            "start a new run."
        )
    return inputs
