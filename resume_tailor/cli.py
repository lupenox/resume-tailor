from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .apify_job import invoke_apify_job_extraction, resolve_linkedin_provider
from .antigravity_writer import (
    ANTIGRAVITY_RESPONSE_METADATA_FILENAME,
    invoke_antigravity,
    load_antigravity_response_metadata,
    preflight_tailoring_inputs,
)
from .clipboard import read_clipboard
from .codex_analysis import invoke_codex_analysis, readable_analysis
from .evidence import (
    build_content_diff,
    resolve_analysis_evidence,
    validate_tailored_content,
)
from .linkedin_job import (
    invoke_linkedin_job_extraction,
    posting_confirmation_text,
    validate_linkedin_url,
)
from .job_requirements import build_job_requirement_catalog
from .orchestration import PipelineHooks
from .qa import invoke_final_qa
from .retry import (
    ANALYSIS_APPROVAL_FILENAME,
    AntigravityReprocessContext,
    AntigravityRetryContext,
    RetryContext,
    analysis_input_manifest,
    load_antigravity_reprocess_inputs,
    load_antigravity_retry_inputs,
    load_retry_inputs,
    record_codex_analysis_approval,
    verify_tailoring_run_artifacts,
)
from .schemas import (
    CodexAnalysisTransportArtifact,
    prepare_codex_analysis_transport_schema,
    validate_codex_analysis_transport_artifact,
)
from .utilities import (
    ApifyConfigurationError,
    ApifyProviderError,
    ApprovalError,
    AntigravityCannotApplyError,
    AntigravityLaunchSizeError,
    AntigravityResponseEnvelopeError,
    AntigravityTailoringContractError,
    AntigravityTailoringPreflightError,
    AntigravityTechnicalFailureError,
    CancellationError,
    ExitCode,
    InputError,
    IntegrityError,
    LinkedInResponseEnvelopeError,
    QAError,
    ResumeTailorError,
    SourceEvidenceError,
    TruthfulnessError,
    WaitingError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    cancellable_commands,
    create_unique_run_dir,
    filename_component,
    parse_duration,
    read_text_file,
    relative_artifacts,
    rename_run_directory,
    require_executable,
    run_command,
    sha256_file,
    utc_now_iso,
)


def _duration_argument(value: str) -> tuple[int, str]:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailor-resume",
        description=(
            "Create a truthful, human-gated tailored DOCX/PDF from a structured "
            "master resume."
        ),
    )
    parser.add_argument("--resume", required=True, type=Path, help="master .docx path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--clipboard",
        action="store_true",
        help="read the job description from the Linux clipboard",
    )
    source.add_argument("--job-file", type=Path, help="UTF-8 job-description file")
    source.add_argument(
        "--job-url",
        help="public HTTPS LinkedIn /jobs/view/ URL to retrieve and validate",
    )
    parser.add_argument(
        "--linkedin-provider",
        choices=("auto", "apify", "antigravity"),
        default="auto",
        help=(
            "URL retrieval provider: auto prefers Apify when APIFY_API_TOKEN is "
            "configured, otherwise Antigravity (default: auto)"
        ),
    )
    parser.add_argument(
        "--company",
        help="target company (required with --clipboard or --job-file)",
    )
    parser.add_argument(
        "--role",
        help="target role (required with --clipboard or --job-file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/Documents/Resumes/Tailored"),
        help="artifact parent directory (default: ~/Documents/Resumes/Tailored)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip human prompts (truthfulness and safety checks remain enforced)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep internal LibreOffice/model QA working files",
    )
    parser.add_argument(
        "--timeout",
        type=_duration_argument,
        default=_duration_argument("15m"),
        metavar="DURATION",
        help="model timeout such as 90s, 15m, or 1h (default: 15m)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _validate_mode_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.job_url is not None:
        if args.company is not None or args.role is not None:
            parser.error(
                "--company and --role must be omitted with --job-url; both are "
                "derived from the fetched posting"
            )
        return
    if args.linkedin_provider != "auto":
        parser.error("--linkedin-provider is only valid with --job-url")
    if args.company is None or args.role is None:
        parser.error(
            "--company and --role are required with --clipboard or --job-file"
        )


def _validate_label(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise InputError(f"{label} must not be empty.")
    if len(value) > 200:
        raise InputError(f"{label} must be 200 characters or fewer.")
    if any(ord(character) < 32 for character in value):
        raise InputError(f"{label} contains a control character.")
    return value


def _validate_resume_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.exists():
        raise InputError(f"Resume does not exist: {expanded}")
    if not expanded.is_file():
        raise InputError(f"Resume is not a regular file: {expanded}")
    if expanded.suffix.casefold() != ".docx":
        raise InputError(f"Resume must have a .docx extension: {expanded}")
    return expanded.resolve()


def _tool_version(executable: str, arguments: list[str], *, cwd: Path) -> str:
    result = run_command(
        [executable, *arguments],
        cwd=cwd,
        timeout_seconds=15,
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0][:200] if result.returncode == 0 and output else "unavailable"


def _runtime_dependency_versions() -> dict[str, str]:
    try:
        python_docx_version = importlib.metadata.version("python-docx")
        jsonschema_version = importlib.metadata.version("jsonschema")
    except importlib.metadata.PackageNotFoundError as exc:
        from .utilities import DependencyError

        raise DependencyError(
            f"Missing Python dependency {exc.name!r}. Install project dependencies "
            "as documented in README.md."
        ) from exc
    return {
        "resume_tailor": __version__,
        "python": sys.version.split()[0],
        "python_docx": python_docx_version,
        "jsonschema": jsonschema_version,
    }


def _analysis_dependency_versions(cwd: Path) -> dict[str, str]:
    codex = require_executable("codex")
    return {
        **_runtime_dependency_versions(),
        "codex": _tool_version(codex, ["--version"], cwd=cwd),
    }


def _tailoring_dependency_versions(cwd: Path) -> dict[str, str]:
    agy = require_executable("agy")
    libreoffice = require_executable("libreoffice")
    require_executable("pdfinfo")
    require_executable("pdftotext")
    require_executable("pdftoppm")
    return {
        "antigravity": _tool_version(agy, ["--version"], cwd=cwd),
        "libreoffice": _tool_version(libreoffice, ["--version"], cwd=cwd),
    }


def _dependency_versions(cwd: Path) -> dict[str, str]:
    return {
        **_analysis_dependency_versions(cwd),
        **_tailoring_dependency_versions(cwd),
    }


def _header_name(extracted: dict[str, Any]) -> str:
    for paragraph in extracted["paragraphs"]:
        if paragraph["content_id"] == "header.name":
            return paragraph["text"].strip()
    raise InputError("Could not determine the resume owner's name from the template.")


def _required_pdf_text(extracted: dict[str, Any]) -> list[str]:
    from .docx_extract import SECTION_HEADINGS

    required = [_header_name(extracted), *SECTION_HEADINGS]
    required.extend(
        link["text"]
        for link in extracted["document"]["hyperlinks"]
        if link["paragraph_index"] == 1
    )
    return required


def _update_metadata(
    metadata: dict[str, Any],
    path: Path,
    *,
    run_directory: Path,
) -> None:
    metadata["updated_at"] = utc_now_iso()
    metadata["artifacts"] = relative_artifacts(run_directory)
    atomic_write_json(path, metadata)


def _elapsed_label(elapsed_seconds: float) -> str:
    total = max(0, int(elapsed_seconds))
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def run_pipeline(
    args: argparse.Namespace,
    *,
    hooks: PipelineHooks | None = None,
) -> Path:
    active_hooks = hooks or PipelineHooks()
    with cancellable_commands(active_hooks.cancel_event):
        return _run_pipeline(args, active_hooks)


def _run_pipeline(args: argparse.Namespace, hooks: PipelineHooks) -> Path:
    hooks.progress("validating_input", "Validating the résumé and job input.")
    resume_path = _validate_resume_path(args.resume)
    timeout_seconds, antigravity_duration = args.timeout
    retry_context = getattr(args, "retry_context", None)
    retry_inputs = None
    antigravity_retry_context = getattr(
        args,
        "antigravity_retry_context",
        None,
    )
    antigravity_reprocess_context = getattr(
        args,
        "antigravity_reprocess_context",
        None,
    )
    antigravity_retry_inputs = None
    antigravity_reprocess_inputs = None
    recovery_modes = sum(
        item is not None
        for item in (
            retry_context,
            antigravity_retry_context,
            antigravity_reprocess_context,
        )
    )
    if recovery_modes > 1:
        raise InputError("Only one internal recovery mode may be active.")
    if retry_context is not None:
        if not isinstance(retry_context, RetryContext):
            raise InputError("Invalid internal source-evidence retry context.")
        if args.job_url is not None or args.clipboard or args.job_file is not None:
            raise InputError("A source-evidence retry cannot accept new job input.")
        retry_inputs = load_retry_inputs(
            retry_context,
            current_resume=resume_path,
        )
    if antigravity_retry_context is not None:
        if not isinstance(
            antigravity_retry_context,
            AntigravityRetryContext,
        ):
            raise InputError("Invalid internal Antigravity recovery context.")
        if args.job_url is not None or args.clipboard or args.job_file is not None:
            raise InputError(
                "An Antigravity recovery cannot accept new job input."
            )
        antigravity_retry_inputs = load_antigravity_retry_inputs(
            antigravity_retry_context,
            current_resume=resume_path,
        )
    if antigravity_reprocess_context is not None:
        if not isinstance(
            antigravity_reprocess_context,
            AntigravityReprocessContext,
        ):
            raise InputError("Invalid internal Antigravity reprocessing context.")
        if args.job_url is not None or args.clipboard or args.job_file is not None:
            raise InputError(
                "Antigravity reprocessing cannot accept new job input."
            )
        antigravity_reprocess_inputs = load_antigravity_reprocess_inputs(
            antigravity_reprocess_context,
            current_resume=resume_path,
        )
        antigravity_retry_inputs = antigravity_reprocess_inputs.retry_inputs
    requested_linkedin_url = None
    linkedin_provider_requested = getattr(args, "linkedin_provider", "auto")
    linkedin_provider_resolved: str | None = None
    fetched_job: dict[str, Any] | None = None
    if antigravity_retry_inputs is not None:
        company = _validate_label(
            antigravity_retry_inputs.context.company,
            "Stored company",
        )
        role = _validate_label(
            antigravity_retry_inputs.context.role,
            "Stored role",
        )
        job_description = antigravity_retry_inputs.job_description
        job_source = (
            "antigravity-response-reprocess"
            if antigravity_reprocess_inputs is not None
            else "antigravity-tailoring-retry"
        )
    elif retry_inputs is not None:
        company = _validate_label(retry_inputs.context.company, "Stored company")
        role = _validate_label(retry_inputs.context.role, "Stored role")
        job_description = retry_inputs.job_description
        job_source = "source-evidence-retry"
    elif args.job_url is not None:
        requested_linkedin_url = validate_linkedin_url(args.job_url)
        linkedin_provider_resolved = resolve_linkedin_provider(
            linkedin_provider_requested
        )
        company: str | None = None
        role: str | None = None
        job_description: str | None = None
        job_source = "linkedin-url"
    else:
        assert args.company is not None and args.role is not None
        company = _validate_label(args.company, "Company")
        role = _validate_label(args.role, "Role")
        if args.clipboard:
            job_description, job_source = read_clipboard()
        else:
            assert args.job_file is not None
            job_description = read_text_file(
                args.job_file.expanduser(),
                label="job description",
            )
            job_source = "job-file"

    source_hash = sha256_file(resume_path)
    if (
        retry_inputs is not None
        and source_hash != retry_inputs.context.source_resume_sha256
    ):
        raise IntegrityError(
            "The source résumé changed after retry verification; start a new run."
        )
    if (
        antigravity_retry_inputs is not None
        and source_hash
        != antigravity_retry_inputs.context.source_resume_sha256
    ):
        raise IntegrityError(
            "The source résumé changed after Antigravity recovery verification; "
            "start a new run."
        )
    output_dir = args.output_dir.expanduser()
    if requested_linkedin_url is not None:
        run_directory = create_unique_run_dir(
            output_dir,
            "linkedin",
            "job-fetch",
        )
    else:
        assert company is not None and role is not None
        run_directory = create_unique_run_dir(output_dir, company, role)
    metadata_path = run_directory / "run-metadata.json"
    work_directory = run_directory / "work"
    metadata: dict[str, Any] = {
        "application": "resume-tailor",
        "application_version": __version__,
        "status": "RUNNING",
        "stage": "initializing",
        "created_at": utc_now_iso(),
        "company": company,
        "role": role,
        "job_source": job_source,
        "source_resume": {
            "filename": resume_path.name,
            "sha256_before": source_hash,
            "sha256_after": None,
            "unchanged": None,
        },
        "tools": {},
        "artifacts": [],
    }
    if retry_inputs is not None:
        metadata["retry_of"] = retry_inputs.context.source_directory.name
        metadata["retry_kind"] = "codex-source-evidence-analysis"
        metadata["legacy_retry_inputs_verified"] = (
            retry_inputs.context.legacy_verified
        )
    if antigravity_retry_inputs is not None:
        metadata["retry_of"] = (
            antigravity_retry_inputs.context.source_directory.name
        )
        metadata["retry_kind"] = (
            "antigravity-response-reprocess"
            if antigravity_reprocess_inputs is not None
            else "antigravity-tailoring"
        )
        metadata["approved_analysis_reused"] = True
    if antigravity_reprocess_inputs is not None:
        metadata["provider_calls_reused"] = {
            "linkedin": False,
            "codex_analysis": False,
            "antigravity_tailoring": False,
        }
    _update_metadata(metadata, metadata_path, run_directory=run_directory)
    hooks.progress(
        "validating_input",
        "Created an isolated run-artifact directory.",
        run_directory=str(run_directory),
    )

    caught_error: ResumeTailorError | None = None
    try:
        if job_description is not None:
            if antigravity_retry_inputs is not None:
                atomic_write_bytes(
                    run_directory / "job-description.txt",
                    antigravity_retry_inputs.artifact_bytes[
                        "job-description.txt"
                    ],
                )
                job_source_bytes = antigravity_retry_inputs.artifact_bytes.get(
                    "job-source.json"
                )
                if job_source_bytes is not None:
                    atomic_write_bytes(
                        run_directory / "job-source.json",
                        job_source_bytes,
                    )
            else:
                atomic_write_text(
                    run_directory / "job-description.txt",
                    job_description.rstrip() + "\n",
                )
        metadata["stage"] = "dependency-check"
        hooks.progress(
            "validating_input",
            "Checking local pipeline dependencies and verified CLI adapters.",
        )
        if antigravity_retry_inputs is not None:
            metadata["tools"] = _tailoring_dependency_versions(run_directory)
        elif retry_inputs is not None:
            metadata["tools"] = _analysis_dependency_versions(run_directory)
        else:
            metadata["tools"] = _dependency_versions(run_directory)
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if requested_linkedin_url is not None:
            metadata["stage"] = "linkedin-job-extraction"
            assert linkedin_provider_resolved is not None
            provider_label = (
                "Apify"
                if linkedin_provider_resolved == "apify"
                else "Antigravity"
            )
            hooks.progress(
                "fetching_job",
                f"{provider_label} is retrieving the public LinkedIn posting.",
            )
            metadata["linkedin_retrieval"] = {
                "requested_provider": linkedin_provider_requested,
                "resolved_provider": linkedin_provider_resolved,
                "automatic_fallback": False,
            }
            metadata["linkedin_job"] = {
                "requested_url": requested_linkedin_url.normalized,
                "final_resolved_url": None,
                "linkedin_job_id": requested_linkedin_url.job_id,
            }
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            if linkedin_provider_resolved == "apify":
                fetched_job = invoke_apify_job_extraction(
                    requested_url=requested_linkedin_url,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    progress_handler=lambda elapsed, status: hooks.progress(
                        "fetching_job",
                        (
                            "Apify job-detail retrieval is still running "
                            f"({_elapsed_label(elapsed)}; status {status})."
                        ),
                    ),
                )
            else:
                fetched_job = invoke_linkedin_job_extraction(
                    requested_url=requested_linkedin_url,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    antigravity_duration=antigravity_duration,
                )
            job_description = fetched_job["normalized_job_description"]
            atomic_write_text(
                run_directory / "job-description.txt",
                job_description.rstrip() + "\n",
            )
            metadata["linkedin_job"] = {
                "requested_url": fetched_job["requested_url"],
                "final_resolved_url": fetched_job["final_resolved_url"],
                "linkedin_job_id": fetched_job["linkedin_job_id"],
            }
            metadata["stage"] = "linkedin-posting-confirmation"
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            hooks.progress(
                "confirming_posting",
                "The posting was extracted and is waiting for explicit confirmation.",
            )
            if hooks.approval_handler is None:
                print(posting_confirmation_text(fetched_job))
            posting_approval = hooks.approve(
                kind="linkedin_posting",
                title="LinkedIn posting",
                payload=fetched_job,
                assume_yes=args.yes,
            )
            if posting_approval.action == "use_pasted":
                pasted_description = str(
                    posting_approval.data.get("job_description", "")
                ).strip()
                if not pasted_description:
                    raise InputError(
                        "The pasted fallback description is empty; no résumé work "
                        "was started."
                    )
                if len(pasted_description.encode("utf-8")) > 500_000:
                    raise InputError(
                        "The pasted fallback description exceeds the 500,000-byte "
                        "safety limit."
                    )
                job_description = pasted_description
                job_source = "pasted-fallback"
                metadata["job_source"] = job_source
                metadata["linkedin_job"]["used_pasted_fallback"] = True
                atomic_write_text(
                    run_directory / "job-description.txt",
                    job_description.rstrip() + "\n",
                )
                hooks.progress(
                    "confirming_posting",
                    "Using the explicitly supplied pasted description for this run.",
                )

            company = _validate_label(fetched_job["company"], "Extracted company")
            role = _validate_label(fetched_job["job_title"], "Extracted job title")
            metadata["company"] = company
            metadata["role"] = role
            run_directory = rename_run_directory(run_directory, company, role)
            metadata_path = run_directory / "run-metadata.json"
            work_directory = run_directory / "work"
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            hooks.progress(
                "confirming_posting",
                "Posting confirmed. The run directory now uses the extracted identity.",
                run_directory=str(run_directory),
                company=company,
                role=role,
            )

        assert company is not None and role is not None and job_description is not None
        if antigravity_retry_inputs is not None:
            job_requirements = antigravity_retry_inputs.job_requirements
        elif retry_inputs is not None:
            job_requirements = retry_inputs.job_requirements
        else:
            job_requirements = build_job_requirement_catalog(
                job_description,
                structured_job=(
                    fetched_job
                    if fetched_job is not None and job_source == "linkedin-url"
                    else None
                ),
            )
        if antigravity_retry_inputs is not None:
            atomic_write_bytes(
                run_directory / "job-requirements.json",
                antigravity_retry_inputs.artifact_bytes[
                    "job-requirements.json"
                ],
            )
        else:
            atomic_write_json(
                run_directory / "job-requirements.json",
                job_requirements,
            )
        if (
            retry_inputs is not None
            and sha256_file(run_directory / "job-requirements.json")
            != retry_inputs.context.job_requirements_sha256
        ):
            raise IntegrityError(
                "The authenticated job-requirement catalog changed during retry setup; "
                "start a new run."
            )
        if (
            antigravity_retry_inputs is not None
            and sha256_file(run_directory / "job-requirements.json")
            != antigravity_retry_inputs.context.job_requirements_sha256
        ):
            raise IntegrityError(
                "The authenticated job-requirement catalog changed during "
                "Antigravity recovery setup; start a new run."
            )
        metadata["job_requirement_catalog"] = {
            "filename": "job-requirements.json",
            "sha256": sha256_file(run_directory / "job-requirements.json"),
            "requirement_count": len(job_requirements["requirements"]),
            "source_kind": job_requirements["source_kind"],
        }
        from .docx_extract import extract_resume, source_blocks_from_paragraphs

        metadata["stage"] = "extracting-master"
        hooks.progress(
            (
                "codex_analysis"
                if requested_linkedin_url is not None
                else "validating_input"
            ),
            "Structurally validating and extracting the master résumé.",
        )
        if antigravity_retry_inputs is not None:
            extracted = antigravity_retry_inputs.extracted_resume
            extracted_source = extracted.get("source")
            if (
                not isinstance(extracted_source, dict)
                or extracted_source.get("sha256") != source_hash
            ):
                raise IntegrityError(
                    "The authenticated recovery extraction no longer matches the "
                    "source résumé."
                )
        elif retry_inputs is None:
            extracted, _ = extract_resume(resume_path)
        else:
            extracted = retry_inputs.extracted_resume
            extracted_source = extracted.get("source")
            if (
                not isinstance(extracted_source, dict)
                or extracted_source.get("sha256") != source_hash
            ):
                raise IntegrityError(
                    "The preserved extraction no longer matches the source résumé."
                )
            if not isinstance(extracted.get("source_blocks"), list):
                extracted["source_blocks"] = source_blocks_from_paragraphs(
                    extracted["paragraphs"]
                )
        if antigravity_retry_inputs is not None:
            atomic_write_bytes(
                run_directory / "extracted-master-resume.json",
                antigravity_retry_inputs.artifact_bytes[
                    "extracted-master-resume.json"
                ],
            )
        else:
            atomic_write_json(
                run_directory / "extracted-master-resume.json",
                extracted,
            )
        metadata["analysis_inputs"] = analysis_input_manifest(
            run_directory,
            source_resume_sha256=source_hash,
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if antigravity_retry_inputs is not None:
            metadata["stage"] = "antigravity-recovery-verification"
            hooks.progress(
                "antigravity_tailoring",
                "Verifying and reusing the previously approved Codex analysis. "
                "LinkedIn and Codex are not being invoked.",
            )
            transport_bytes = antigravity_retry_inputs.artifact_bytes[
                "codex-analysis-transport.schema.json"
            ]
            transport_path = (
                run_directory / "codex-analysis-transport.schema.json"
            )
            atomic_write_bytes(transport_path, transport_bytes)
            source_blocks = extracted.get("source_blocks", [])
            transport_artifact = CodexAnalysisTransportArtifact(
                path=transport_path.resolve(),
                sha256=antigravity_retry_inputs.context.transport_schema_sha256,
                size_bytes=len(transport_bytes),
                evidence_source_id_count=sum(
                    1
                    for block in source_blocks
                    if isinstance(block, dict)
                    and block.get("evidence_allowed") is True
                ),
                editable_source_id_count=sum(
                    1
                    for block in source_blocks
                    if isinstance(block, dict)
                    and block.get("editable") is True
                ),
                job_requirement_id_count=len(
                    job_requirements["requirements"]
                ),
            )
            validate_codex_analysis_transport_artifact(
                transport_artifact,
                extracted,
                job_requirements,
                run_directory,
            )
            metadata["codex_analysis_transport_schema"] = (
                transport_artifact.metadata()
            )
            analysis_bytes = antigravity_retry_inputs.artifact_bytes[
                "codex-analysis-resolved.json"
            ]
            atomic_write_bytes(
                run_directory / "codex-analysis-resolved.json",
                analysis_bytes,
            )
            approval_bytes = antigravity_retry_inputs.artifact_bytes[
                ANALYSIS_APPROVAL_FILENAME
            ]
            atomic_write_bytes(
                run_directory / ANALYSIS_APPROVAL_FILENAME,
                approval_bytes,
            )
            if (
                sha256_file(run_directory / "codex-analysis-resolved.json")
                != antigravity_retry_inputs.context.resolved_analysis_sha256
                or sha256_file(run_directory / ANALYSIS_APPROVAL_FILENAME)
                != antigravity_retry_inputs.context.approval_record_sha256
            ):
                raise IntegrityError(
                    "Authenticated analysis recovery artifacts changed while "
                    "creating the isolated run."
                )
            metadata["codex_analysis_approval"] = {
                "filename": ANALYSIS_APPROVAL_FILENAME,
                "sha256": (
                    antigravity_retry_inputs.context.approval_record_sha256
                ),
                "version": 1,
                "decision": "approved",
            }
            metadata["recovery_inputs"] = {
                "source_resume_sha256": source_hash,
                "extracted_resume_sha256": (
                    antigravity_retry_inputs.context.extracted_resume_sha256
                ),
                "job_description_sha256": (
                    antigravity_retry_inputs.context.job_description_sha256
                ),
                "job_requirements_sha256": (
                    antigravity_retry_inputs.context.job_requirements_sha256
                ),
                "transport_schema_sha256": (
                    antigravity_retry_inputs.context.transport_schema_sha256
                ),
                "resolved_analysis_sha256": (
                    antigravity_retry_inputs.context.resolved_analysis_sha256
                ),
                "approval_record_sha256": (
                    antigravity_retry_inputs.context.approval_record_sha256
                ),
            }
            if antigravity_reprocess_inputs is not None:
                atomic_write_bytes(
                    run_directory / "antigravity-response.json",
                    antigravity_reprocess_inputs.response_bytes,
                )
                atomic_write_json(
                    run_directory / ANTIGRAVITY_RESPONSE_METADATA_FILENAME,
                    antigravity_reprocess_inputs.response_metadata,
                )
                if (
                    sha256_file(run_directory / "antigravity-response.json")
                    != antigravity_reprocess_inputs.context.response_sha256
                ):
                    raise IntegrityError(
                        "The preserved Antigravity response changed while creating "
                        "the isolated reprocessing run."
                    )
                metadata["reprocess_inputs"] = {
                    "response_sha256": (
                        antigravity_reprocess_inputs.context.response_sha256
                    ),
                    "tailoring_schema_sha256": (
                        antigravity_reprocess_inputs.context.tailoring_schema_sha256
                    ),
                    "response_envelope_type": (
                        antigravity_reprocess_inputs.context.envelope_type
                    ),
                    "ancestry_run": (
                        antigravity_reprocess_inputs.context.ancestry_run
                    ),
                    "ancestry_metadata_sha256": (
                        antigravity_reprocess_inputs.context.ancestry_metadata_sha256
                    ),
                }
                metadata["antigravity_response"] = dict(
                    antigravity_reprocess_inputs.response_metadata
                )
            analysis = antigravity_retry_inputs.approved_analysis
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
        else:
            metadata["stage"] = "codex-analysis-schema-preflight"
            hooks.progress(
                "codex_analysis",
                "Generating and validating the source-bound Codex output schema.",
            )
            transport_artifact = prepare_codex_analysis_transport_schema(
                extracted,
                job_requirements,
                run_directory,
            )
            metadata["codex_analysis_transport_schema"] = (
                transport_artifact.metadata()
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)

            metadata["stage"] = "codex-analysis"
            hooks.progress(
                "codex_analysis",
                "Codex analysis started. Strong reasoning may take several minutes; "
                "no unreliable ETA is shown.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            raw_analysis = invoke_codex_analysis(
                extracted_resume=extracted,
                job_description=job_description,
                job_requirements=job_requirements,
                company=company,
                role=role,
                run_directory=run_directory,
                timeout_seconds=timeout_seconds,
                transport_artifact=transport_artifact,
                progress_handler=lambda elapsed, alive: hooks.progress(
                    "codex_analysis",
                    (
                        "Codex analysis is still running"
                        if alive
                        else (
                            "No Codex process detected; the process exited and local "
                            "structured-output validation is continuing"
                        )
                    )
                    + f" — elapsed {_elapsed_label(elapsed)}.",
                    elapsed_seconds=max(0, int(elapsed)),
                    provider_process_alive=alive,
                ),
            )
            analysis, analysis_issues = resolve_analysis_evidence(
                raw_analysis,
                extracted,
                job_requirements,
            )
            if analysis_issues:
                raise SourceEvidenceError(
                    "Codex analysis failed local source-evidence validation:\n"
                    + "\n".join(
                        f"- {issue.describe()}" for issue in analysis_issues
                    )
                )
            atomic_write_json(
                run_directory / "codex-analysis-resolved.json",
                analysis,
            )
            if hooks.approval_handler is None:
                print(readable_analysis(analysis))
            if analysis["questions_for_user"]:
                questions = "\n".join(
                    f"- {question}" for question in analysis["questions_for_user"]
                )
                raise WaitingError(
                    "Codex has unanswered factual questions. The pipeline stopped "
                    f"instead of guessing:\n{questions}"
                )
            hooks.progress(
                "reviewing_changes",
                "Codex analysis passed local evidence checks and awaits approval.",
            )
            analysis_approval = hooks.approve(
                kind="codex_analysis",
                title="Codex analysis",
                payload=analysis,
                assume_yes=args.yes,
            )
            if analysis_approval.action != "approve":
                raise ApprovalError(
                    "Codex analysis was not approved; artifacts were preserved."
                )
            metadata["codex_analysis_approval"] = (
                record_codex_analysis_approval(
                    run_directory,
                    source_resume_sha256=source_hash,
                    company=company,
                    role=role,
                    approval_mode=(
                        "assume_yes" if args.yes else "interactive"
                    ),
                )
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)

            if retry_inputs is not None:
                metadata["stage"] = "tailoring-dependency-check"
                hooks.progress(
                    "antigravity_tailoring",
                    "Checking downstream tools after the renewed analysis approval.",
                )
                metadata["tools"].update(
                    _tailoring_dependency_versions(run_directory)
                )
                _update_metadata(
                    metadata,
                    metadata_path,
                    run_directory=run_directory,
                )

        metadata["stage"] = "antigravity-tailoring-preflight"
        hooks.progress(
            "antigravity_tailoring",
            "Authenticating the approved tailoring inputs before Antigravity.",
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        try:
            preflight_tailoring_inputs(
                master_content=extracted["content"],
                extracted_resume=extracted,
                job_description=job_description,
                job_requirements=job_requirements,
                approved_analysis=analysis,
                company=company,
                role=role,
            )
            metadata["antigravity_tailoring_preflight"] = (
                verify_tailoring_run_artifacts(
                    run_directory,
                    source_resume_sha256=source_hash,
                    extracted_resume=extracted,
                    job_description=job_description,
                    job_requirements=job_requirements,
                    approved_analysis=analysis,
                    company=company,
                    role=role,
                )
            )
        except AntigravityTailoringPreflightError:
            raise
        except InputError as exc:
            raise AntigravityTailoringPreflightError(str(exc)) from exc
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if antigravity_reprocess_inputs is not None:
            metadata["stage"] = "antigravity-response-reprocessing"
            hooks.progress(
                "antigravity_tailoring",
                "Reprocessing the authenticated preserved Antigravity response "
                "entirely offline. No provider is being invoked.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            tailored_content = antigravity_reprocess_inputs.tailored_content
        else:
            metadata["stage"] = "antigravity-tailoring"
            hooks.progress(
                "antigravity_tailoring",
                "Antigravity is applying the approved schema-constrained edits.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            tailored_content = invoke_antigravity(
                master_content=extracted["content"],
                extracted_resume=extracted,
                job_description=job_description,
                job_requirements=job_requirements,
                approved_analysis=analysis,
                company=company,
                role=role,
                run_directory=run_directory,
                timeout_seconds=timeout_seconds,
                antigravity_duration=antigravity_duration,
            )
        atomic_write_json(
            run_directory / "tailored-content.json",
            tailored_content,
        )

        metadata["stage"] = "local-evidence-check"
        hooks.progress(
            "evidence_validation",
            "Running deterministic factual-integrity and content-budget checks.",
        )
        report = validate_tailored_content(
            original=extracted["content"],
            tailored=tailored_content,
            extracted_resume=extracted,
            analysis=analysis,
            target_role=role,
        )
        content_diff = build_content_diff(
            extracted["content"],
            tailored_content,
            report,
        )
        atomic_write_text(run_directory / "content-diff.md", content_diff)
        if hooks.approval_handler is None:
            print("\n" + content_diff)
        if not report.passed:
            detail = "\n".join(f"- {issue}" for issue in report.issues)
            raise TruthfulnessError(
                "Local evidence checks blocked rendering. No claims were silently "
                f"rewritten:\n{detail}"
            )
        diff_approval = hooks.approve(
            kind="tailored_content",
            title="Tailored content diff",
            payload={
                "content_diff": content_diff,
                "tailored_content": tailored_content,
                "evidence": {
                    "passed": report.passed,
                    "issues": report.issues,
                    "introduced_technologies": report.introduced_technologies,
                    "introduced_metrics": report.introduced_metrics,
                    "introduced_role_labels": report.introduced_role_labels,
                    "introduced_availability": report.introduced_availability,
                },
            },
            assume_yes=args.yes,
        )
        if diff_approval.action != "approve":
            raise ApprovalError(
                "Tailored content diff was not approved; artifacts were preserved."
            )
        metadata["factual_integrity"] = {
            "status": "PASS",
            "issues": [],
        }

        metadata["stage"] = "docx-render"
        hooks.progress(
            "rendering",
            "Rendering the approved content into a protected copy of the DOCX.",
        )
        work_directory.mkdir(parents=True, exist_ok=True)
        person = filename_component(_header_name(extracted), fallback="Resume")
        basename = "-".join(
            (
                person,
                filename_component(company, fallback="Company"),
                filename_component(role, fallback="Role"),
            )
        )
        docx_path = run_directory / f"{basename}.docx"
        pdf_path = run_directory / f"{basename}.pdf"
        preview_path = run_directory / "preview.png"

        from .docx_render import export_and_validate_pdf, render_tailored_docx

        render_tailored_docx(
            source_path=resume_path,
            destination_path=docx_path,
            tailored_content=tailored_content,
            expected_source_hash=source_hash,
        )
        metadata["stage"] = "pdf-export-validation"
        hooks.progress(
            "rendering",
            "Exporting and validating the one-page PDF and PNG preview.",
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        pdf_text = export_and_validate_pdf(
            docx_path=docx_path,
            pdf_path=pdf_path,
            preview_path=preview_path,
            working_directory=work_directory,
            required_text=_required_pdf_text(extracted),
        )
        metadata["layout_validation"] = {
            "status": "PASS",
            "pages": 1,
            "required_text_present": True,
            "bounding_boxes_valid": True,
        }

        metadata["stage"] = "final-codex-qa"
        hooks.progress(
            "final_qa",
            "Codex is performing the final read-only content and visual QA review.",
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        qa_result = invoke_final_qa(
            original_extraction=extracted,
            job_description=job_description,
            analysis=analysis,
            tailored_pdf_text=pdf_text,
            content_diff=content_diff,
            preview_path=preview_path,
            run_directory=run_directory,
            work_directory=work_directory,
            timeout_seconds=timeout_seconds,
        )
        if qa_result["status"] != "PASS" or qa_result["material_issues"]:
            issues = qa_result["material_issues"] or [
                "Codex marked the final QA as requiring review."
            ]
            raise QAError(
                "Final read-only QA found a material issue. Outputs were preserved:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        metadata["final_qa"] = {
            "status": qa_result["status"],
            "summary": qa_result["summary"],
            "material_issues": qa_result["material_issues"],
            "improvement_assessment": qa_result["improvement_assessment"],
        }

        metadata["status"] = "COMPLETE"
        metadata["stage"] = "complete"
        metadata["completed_at"] = utc_now_iso()
        hooks.progress(
            "complete",
            "Tailoring completed successfully; local artifacts are ready.",
            run_directory=str(run_directory),
            company=company,
            role=role,
        )
        return run_directory
    except CancellationError as exc:
        caught_error = exc
        metadata["status"] = "CANCELLED"
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": int(exc.exit_code),
        }
        raise
    except ResumeTailorError as exc:
        caught_error = exc
        metadata["status"] = "FAILED"
        if isinstance(exc, SourceEvidenceError):
            metadata["failure_class"] = "source-evidence-analysis"
        if isinstance(exc, ApifyConfigurationError):
            metadata["failure_class"] = "apify-configuration"
        if isinstance(exc, ApifyProviderError):
            metadata["failure_class"] = "apify-retrieval"
        if isinstance(exc, AntigravityLaunchSizeError):
            metadata["failure_class"] = "antigravity-launch-size"
        if isinstance(exc, LinkedInResponseEnvelopeError):
            metadata["failure_class"] = "linkedin-response-envelope"
        elif isinstance(exc, AntigravityResponseEnvelopeError):
            metadata["failure_class"] = "antigravity-response-envelope"
        if isinstance(exc, AntigravityTailoringContractError):
            metadata["failure_class"] = "antigravity-tailoring-contract"
        if isinstance(exc, AntigravityCannotApplyError):
            metadata["failure_class"] = "antigravity-cannot-apply"
        if isinstance(exc, AntigravityTechnicalFailureError):
            metadata["failure_class"] = "antigravity-technical-failure"
        if isinstance(exc, AntigravityTailoringPreflightError):
            metadata["failure_class"] = "antigravity-tailoring-preflight"
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": int(exc.exit_code),
        }
        if isinstance(exc, AntigravityResponseEnvelopeError):
            metadata["error"]["response_envelope_type"] = exc.envelope_type
        raise
    except Exception as exc:
        caught_error = InputError(f"Unexpected internal error: {type(exc).__name__}: {exc}")
        metadata["status"] = "FAILED"
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": 1,
        }
        raise caught_error from exc
    finally:
        if work_directory.exists() and not args.keep_workdir:
            shutil.rmtree(work_directory)
        response_metadata = load_antigravity_response_metadata(run_directory)
        if response_metadata is not None:
            response_metadata.setdefault(
                "cli_version",
                metadata.get("tools", {}).get(
                    "antigravity",
                    "unavailable",
                ),
            )
            metadata["antigravity_response"] = response_metadata
        actual_hash = sha256_file(resume_path) if resume_path.is_file() else None
        unchanged = actual_hash == source_hash
        metadata["source_resume"]["sha256_after"] = actual_hash
        metadata["source_resume"]["unchanged"] = unchanged
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        if not unchanged:
            raise IntegrityError(
                "CRITICAL: the master resume hash changed during the run. Stop and "
                "restore it from a trusted copy."
            ) from caught_error


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mode_arguments(parser, args)
    try:
        run_directory = run_pipeline(args)
    except ResumeTailorError as exc:
        print(f"tailor-resume: {exc}", file=sys.stderr)
        return int(exc.exit_code)
    print(f"\nCompleted successfully. Artifacts: {run_directory}")
    return int(ExitCode.OK)


if __name__ == "__main__":
    raise SystemExit(main())
