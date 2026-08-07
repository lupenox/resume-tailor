from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from resume_tailor.backend.providers.codex_analysis import invoke_codex_analysis
from resume_tailor.backend.documents.docx_extract import extract_resume
from resume_tailor.backend.engine.evidence import resolve_analysis_evidence
from resume_tailor.backend.jobs.job_requirements import build_job_requirement_catalog
from resume_tailor.backend.utils.utilities import InputError, SourceEvidenceError, read_text_file, sha256_file


SYNTHETIC_SAMPLE_SHA256 = (
    "cb525cdbc9445dcdc6912dfea909646e5f63fb72cdee7eca51e0543d9d3b2e05"
)
SYNTHETIC_JOB_DESCRIPTION = """Synthetic AI validation engineer role.
Skills:
- Python
- JSON Schema
- evidence-gated local validation
- RAG
- Kubernetes
- rule-based link scoring
Responsibilities:
- Classify absent requirements as unsupported without asking for unlisted experience.
"""
REQUIRED_UNSUPPORTED_SMOKE_TERMS = (
    "RAG",
    "Kubernetes",
    "rule-based link scoring",
)


@dataclass(frozen=True)
class SmokeInputs:
    mode: str
    resume_path: Path
    resume_sha256: str
    job_description: str
    job_description_sha256: str
    job_file: Path | None
    separately_authorized: bool

    def provenance(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "resume": {
                "classification": (
                    "bundled-synthetic-fixture"
                    if self.mode == "synthetic-only"
                    else "explicitly-authorized-local-input"
                ),
                "sha256": self.resume_sha256,
            },
            "job_description": {
                "classification": (
                    "built-in-synthetic-text"
                    if self.mode == "synthetic-only"
                    else "explicitly-authorized-local-input"
                ),
                "sha256": self.job_description_sha256,
            },
            "separate_real_input_authorization": self.separately_authorized,
            "content_logged": False,
        }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _regular_docx(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".docx":
        raise InputError("The smoke-test résumé must be a non-symlink DOCX file.")
    return path.resolve()


def prepare_smoke_inputs(
    *,
    repository_root: Path,
    resume_path: Path | None = None,
    job_file: Path | None = None,
    allow_real_inputs: bool = False,
    authorization_reference: str | None = None,
) -> SmokeInputs:
    """Prepare inputs without exposing content; synthetic-only is fail-closed."""
    repository_root = repository_root.resolve()
    sample_path = repository_root / "template" / "sample_resume.docx"
    custom_inputs_requested = resume_path is not None or job_file is not None

    if custom_inputs_requested and not allow_real_inputs:
        raise InputError(
            "Synthetic smoke mode refuses custom résumé and job artifacts. "
            "A separate explicit real-input authorization is required."
        )

    if custom_inputs_requested:
        if not authorization_reference or not authorization_reference.strip():
            raise InputError(
                "Real-input smoke mode requires a separate authorization reference."
            )
        if resume_path is None or job_file is None:
            raise InputError(
                "Real-input smoke mode requires both an authorized résumé and job file."
            )
        resolved_resume = _regular_docx(resume_path)
        if job_file.is_symlink() or not job_file.is_file():
            raise InputError("The authorized smoke-test job input is not a safe file.")
        resolved_job_file = job_file.resolve()
        job_description = read_text_file(
            resolved_job_file,
            label="authorized smoke-test job description",
        )
        return SmokeInputs(
            mode="explicitly-authorized-real-inputs",
            resume_path=resolved_resume,
            resume_sha256=sha256_file(resolved_resume),
            job_description=job_description,
            job_description_sha256=_text_sha256(job_description),
            job_file=resolved_job_file,
            separately_authorized=True,
        )

    resolved_sample = _regular_docx(sample_path)
    if resolved_sample != sample_path.resolve():
        raise InputError("The bundled synthetic résumé path failed validation.")
    sample_hash = sha256_file(resolved_sample)
    if sample_hash != SYNTHETIC_SAMPLE_SHA256:
        raise InputError(
            "The bundled synthetic résumé failed its hash pin; provider launch is refused."
        )
    return SmokeInputs(
        mode="synthetic-only",
        resume_path=resolved_sample,
        resume_sha256=sample_hash,
        job_description=SYNTHETIC_JOB_DESCRIPTION.strip(),
        job_description_sha256=_text_sha256(SYNTHETIC_JOB_DESCRIPTION.strip()),
        job_file=None,
        separately_authorized=False,
    )


def assert_smoke_input_provenance(inputs: SmokeInputs) -> dict[str, Any]:
    """Recheck hashes immediately before a provider process may be launched."""
    if inputs.mode == "synthetic-only":
        if inputs.separately_authorized or inputs.job_file is not None:
            raise InputError("Synthetic smoke provenance contains a forbidden input.")
        if inputs.resume_sha256 != SYNTHETIC_SAMPLE_SHA256:
            raise InputError("Synthetic smoke provenance does not match the hash pin.")
        expected_job = SYNTHETIC_JOB_DESCRIPTION.strip()
        if (
            inputs.job_description != expected_job
            or inputs.job_description_sha256 != _text_sha256(expected_job)
        ):
            raise InputError(
                "Synthetic smoke provenance does not match the built-in job fixture."
            )
    elif not inputs.separately_authorized:
        raise InputError("Non-synthetic smoke inputs lack separate authorization.")

    if sha256_file(inputs.resume_path) != inputs.resume_sha256:
        raise InputError("The smoke-test résumé changed after provenance validation.")
    if inputs.job_file is not None:
        current_job = read_text_file(
            inputs.job_file,
            label="authorized smoke-test job description",
        )
        if _text_sha256(current_job) != inputs.job_description_sha256:
            raise InputError(
                "The smoke-test job input changed after provenance validation."
            )
    elif _text_sha256(inputs.job_description) != inputs.job_description_sha256:
        raise InputError("The built-in synthetic job input failed its hash check.")
    return inputs.provenance()


def _require_private_run_directory(run_directory: Path) -> None:
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise InputError("The smoke-test workspace must be a private directory.")
    mode = stat.S_IMODE(run_directory.stat().st_mode)
    if mode != 0o700:
        raise InputError("The smoke-test workspace must have mode 0700.")


def run_semantic_smoke(
    inputs: SmokeInputs,
    *,
    run_directory: Path,
    timeout_seconds: int,
    executable: str | None = None,
    provenance_reporter: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one authorized analysis and stop at the analysis-approval boundary."""
    _require_private_run_directory(run_directory)
    provenance = assert_smoke_input_provenance(inputs)
    if provenance_reporter is None:
        raise InputError(
            "A content-free provenance reporter is required before provider launch."
        )
    provenance_reporter(provenance)

    extracted, _ = extract_resume(inputs.resume_path)
    if extracted.get("source", {}).get("sha256") != inputs.resume_sha256:
        raise InputError("The extracted smoke-test résumé hash is inconsistent.")
    job_requirements = build_job_requirement_catalog(inputs.job_description)
    raw_analysis = invoke_codex_analysis(
        extracted_resume=extracted,
        job_description=inputs.job_description,
        job_requirements=job_requirements,
        company="Synthetic Systems",
        role="Evidence Validation Engineer",
        run_directory=run_directory,
        timeout_seconds=timeout_seconds,
        executable=executable,
    )
    resolved, issues = resolve_analysis_evidence(
        raw_analysis,
        extracted,
        job_requirements,
    )
    if issues:
        locations = ", ".join(
            f"{issue.location}:{issue.code}" for issue in issues
        )
        raise SourceEvidenceError(
            "Synthetic semantic smoke failed local evidence validation at "
            f"{locations}."
        )

    unsupported_requirements = {
        str(item).casefold()
        for item in resolved.get("missing_or_unsupported_requirements", [])
    }
    unsupported_ats = {
        str(item).casefold()
        for item in resolved.get("unsupported_ats_keywords", [])
    }
    evidence_requirements = {
        str(item.get("requirement", "")).casefold()
        for item in resolved.get("evidence_map", [])
    }
    required = {item.casefold() for item in REQUIRED_UNSUPPORTED_SMOKE_TERMS}
    if not required.issubset(unsupported_requirements):
        raise SourceEvidenceError(
            "Synthetic semantic smoke did not classify every absent requirement as unsupported."
        )
    if not required.issubset(unsupported_ats):
        raise SourceEvidenceError(
            "Synthetic semantic smoke did not preserve every unsupported ATS term."
        )
    if required & evidence_requirements:
        raise SourceEvidenceError(
            "Synthetic semantic smoke assigned evidence to an unsupported requirement."
        )
    questions = resolved.get("questions_for_user", [])
    if questions:
        raise SourceEvidenceError(
            "Synthetic semantic smoke generated a forbidden factual question."
        )

    return {
        "input_provenance": provenance,
        "source_id_validation": "pass",
        "evidence_issue_count": 0,
        "unsupported_requirement_count": len(unsupported_requirements),
        "unsupported_ats_count": len(unsupported_ats),
        "question_count": 0,
        "approval_boundary_reached": True,
        "downstream_invoked": False,
        "provider_usage": "unavailable unless reported by supported structured output",
    }
