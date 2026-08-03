from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .analytics import (
    ANALYTICS_DATABASE_FILENAME,
    ANALYTICS_SCHEMA_VERSION,
    AnalyticsStore,
    default_analytics_database_path,
    observation_from_canonical_job,
    observation_from_local_job,
)
from .apify_job import invoke_apify_linkedin_retrieval
from .antigravity_writer import (
    ANTIGRAVITY_RESPONSE_METADATA_FILENAME,
    invoke_antigravity,
    load_antigravity_response_metadata,
    preflight_tailoring_inputs,
)
from .clipboard import read_clipboard
from .analysis import (
    ANALYSIS_PROVIDERS,
    ANALYSIS_RESOLVED_FILENAME,
    CODEX_ANALYSIS_RESOLVED_FILENAME,
    DEFAULT_ANALYSIS_PROVIDER,
    analysis_provider_label,
    invoke_analysis,
    normalize_analysis_provider,
    readable_analysis,
    write_resolved_analysis_artifact,
)
# Re-export for tests and callers that still patch/import the Codex entrypoint.
from .codex_analysis import invoke_codex_analysis
from .evidence import (
    build_content_diff,
    resolve_analysis_evidence,
    validate_tailored_content,
)
from .linkedin_job import (
    posting_confirmation_text,
    validate_linkedin_url,
)
from .job_requirements import build_job_requirement_catalog, job_description_sha256
from .job_text import validate_confirmed_job_description
from .orchestration import PipelineHooks
from .ollama_transport import ollama_dependency_versions
from .ollama_writer import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_REVISION_RESPONSE_METADATA_FILENAME,
    invoke_ollama,
    invoke_ollama_revision,
    load_ollama_response_metadata,
    validate_ollama_model_name,
)
from .qa import invoke_final_qa
from .revision import (
    REVISION_RESPONSE_METADATA_FILENAME,
    REVISION_SCHEMA_NAME,
    approved_revision_targets,
    build_revision_diff,
    invoke_antigravity_revision,
    validate_revision_scope,
)
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
    schema_path,
    validate_codex_analysis_transport_artifact,
)
from .utilities import (
    ApifyConfigurationError,
    ApifyLinkedInRetrievalError,
    ApprovalError,
    AnalysisProviderError,
    AntigravityCannotApplyError,
    AntigravityLaunchSizeError,
    AntigravityResponseEnvelopeError,
    AntigravityRevisionCannotApplyError,
    AntigravityRevisionContractError,
    AntigravityRevisionTechnicalFailureError,
    AntigravityTailoringContractError,
    AntigravityTailoringPreflightError,
    AntigravityTechnicalFailureError,
    CancellationError,
    CodexUsageLimitError,
    ExitCode,
    GemmaAnalysisError,
    GemmaAnalysisTimeoutError,
    GemmaConnectionError,
    GemmaInnerAnalysisError,
    GemmaModelUnavailableError,
    GemmaOllamaInternalError,
    GemmaOllamaUnavailableError,
    GemmaOutputLimitError,
    GemmaResponseTooLargeError,
    GemmaStructuredOutputError,
    GemmaTransportEnvelopeError,
    GrokAnalysisError,
    GrokAuthenticationError,
    GrokExecutableError,
    GrokInnerAnalysisError,
    GrokProcessError,
    GrokPromptTooLargeError,
    GrokTimeoutError,
    GrokTransportEnvelopeError,
    GrokUsageLimitError,
    InputError,
    RequirementExtractionError,
    IntegrityError,
    ModelError,
    OllamaBudgetError,
    OllamaCanonicalSchemaError,
    OllamaCannotApplyError,
    OllamaConnectionError,
    OllamaEvidenceRejectionError,
    OllamaMalformedJSONError,
    OllamaOutputTruncationError,
    OllamaResponseEnvelopeError,
    OllamaRevisionCannotApplyError,
    OllamaRevisionContractError,
    OllamaRevisionTechnicalFailureError,
    OllamaTailoringContractError,
    OllamaTechnicalFailureError,
    OllamaTransportSchemaError,
    QAError,
    RevisionValidationError,
    ResumeTailorError,
    SourceEvidenceError,
    TailoringPreflightError,
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
        "--analytics-db",
        type=Path,
        default=default_analytics_database_path(),
        help="private local SQLite analytics database (default: XDG application data)",
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
    parser.add_argument(
        "--writer-provider",
        choices=("ollama", "antigravity"),
        default="ollama",
        help=(
            "résumé-writing provider (default: ollama; antigravity remains a "
            "compatibility option)"
        ),
    )
    parser.add_argument(
        "--analysis-provider",
        choices=ANALYSIS_PROVIDERS,
        default=DEFAULT_ANALYSIS_PROVIDER,
        help=(
            "résumé-analysis provider (default: gemma_local; codex and grok_cli "
            "are explicit alternatives and are never selected automatically)"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"local Ollama model/profile (default: {DEFAULT_OLLAMA_MODEL})",
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


def _analysis_dependency_versions(
    cwd: Path,
    analysis_provider: str = DEFAULT_ANALYSIS_PROVIDER,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> dict[str, str]:
    provider = normalize_analysis_provider(analysis_provider)
    versions = {
        **_runtime_dependency_versions(),
        "analysis_provider": provider,
    }
    if provider == "gemma_local":
        from .gemma_analysis import resolve_gemma_analysis_model
        from .ollama_transport import ollama_dependency_versions

        model = resolve_gemma_analysis_model(ollama_model)
        versions.update(
            ollama_dependency_versions(
                model=model,
                cwd=cwd,
                timeout_seconds=15,
            )
        )
        versions["gemma_analysis_model"] = model
        return versions
    if provider == "grok_cli":
        from .grok_analysis import resolve_grok_executable

        grok = resolve_grok_executable()
        versions["grok"] = _tool_version(
            grok,
            ["--no-auto-update", "--version"],
            cwd=cwd,
        )
        return versions
    codex = require_executable("codex")
    versions["codex"] = _tool_version(codex, ["--version"], cwd=cwd)
    return versions


def _tailoring_dependency_versions(
    cwd: Path,
    provider: str = "antigravity",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
) -> dict[str, str]:
    libreoffice = require_executable("libreoffice")
    require_executable("pdfinfo")
    require_executable("pdftotext")
    require_executable("pdftoppm")
    versions = {
        "libreoffice": _tool_version(libreoffice, ["--version"], cwd=cwd),
    }
    if provider == "antigravity":
        agy = require_executable("agy")
        versions["antigravity"] = _tool_version(agy, ["--version"], cwd=cwd)
        return versions
    if provider == "ollama":
        versions.update(
            ollama_dependency_versions(
                model=validate_ollama_model_name(ollama_model),
                cwd=cwd,
            )
        )
        return versions
    raise InputError(f"Unsupported résumé writer provider: {provider!r}.")


def _header_name(extracted: dict[str, Any]) -> str:
    for paragraph in extracted["paragraphs"]:
        if paragraph["content_id"] == "header.name":
            return paragraph["text"].strip()
    raise InputError("Could not determine the resume owner's name from the template.")


def _required_pdf_text(
    extracted: dict[str, Any],
    *,
    document_format: str = "master-template",
) -> list[str]:
    if document_format == "headless":
        from .headless_render import HEADLESS_SECTION_HEADINGS as headings
    else:
        from .docx_extract import SECTION_HEADINGS as headings

    required = [_header_name(extracted), *headings]
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


def _publish_authenticated_artifact(
    source_path: Path,
    destination_path: Path,
) -> dict[str, str]:
    if not source_path.is_file():
        raise IntegrityError(
            f"Cannot publish missing generated artifact {source_path.name}."
        )
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise IntegrityError(
            f"Cannot read generated artifact {source_path.name} for publication."
        ) from exc
    if not payload:
        raise IntegrityError(
            f"Cannot publish empty generated artifact {source_path.name}."
        )
    source_sha256 = sha256_file(source_path)
    atomic_write_bytes(destination_path, payload)
    published_sha256 = sha256_file(destination_path)
    if published_sha256 != source_sha256:
        raise IntegrityError(
            f"Published artifact {destination_path.name} failed hash verification."
        )
    return {
        "source_filename": source_path.name,
        "filename": destination_path.name,
        "sha256": published_sha256,
    }


def _publish_final_generation(
    *,
    run_directory: Path,
    basename: str,
    generation: str,
) -> dict[str, Any]:
    if generation not in {"initial", "revision-1"}:
        raise IntegrityError("The final artifact generation is invalid.")
    sources = {
        "tailored_content": run_directory / f"tailored-content.{generation}.json",
        "content_diff": run_directory / f"content-diff.{generation}.md",
        "docx": run_directory / f"{basename}.{generation}.docx",
        "pdf": run_directory / f"{basename}.{generation}.pdf",
        "preview": run_directory / f"preview.{generation}.png",
        "final_qa": run_directory / f"final-qa.{generation}.md",
    }
    destinations = {
        "tailored_content": run_directory / "tailored-content.json",
        "content_diff": run_directory / "content-diff.md",
        "docx": run_directory / f"{basename}.docx",
        "pdf": run_directory / f"{basename}.pdf",
        "preview": run_directory / "preview.png",
        "final_qa": run_directory / "final-qa.md",
    }
    published: dict[str, Any] = {"generation": generation}
    for label, source_path in sources.items():
        published[label] = _publish_authenticated_artifact(
            source_path,
            destinations[label],
        )
    warnings_path = (
        run_directory / f"final-qa.{generation}.normalization-warnings.json"
    )
    if warnings_path.is_file():
        published["final_qa_normalization_warnings"] = (
            _publish_authenticated_artifact(
                warnings_path,
                run_directory / "final-qa-normalization-warnings.json",
            )
        )
    return published


def _elapsed_label(elapsed_seconds: float) -> str:
    total = max(0, int(elapsed_seconds))
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _initial_generation_metadata(
    *,
    response_metadata: dict[str, Any],
    writer_provider: str,
    ollama_model: str,
    tailored_content_path: Path,
) -> tuple[dict[str, Any], bool]:
    """Build run metadata for provider-backed or deterministic generation.

    A deterministic-only Ollama entrypoint deliberately has no provider response
    artifact.  Keep that absence honest while retaining the local execution
    envelope and the authenticated tailored-content reference.
    """
    execution = response_metadata.get("execution")
    if not isinstance(execution, dict):
        # Compatibility with the initial hybrid metadata key used while the
        # deterministic compiler was introduced.
        execution = response_metadata.get("hybrid")
    if not isinstance(execution, dict):
        execution = {}

    invoked_value = execution.get("ollama_invoked")
    if not isinstance(invoked_value, bool):
        invoked_value = response_metadata.get("ollama_invoked")
    ollama_invoked = (
        invoked_value
        if isinstance(invoked_value, bool)
        else writer_provider == "ollama"
    )
    deterministic_only = writer_provider == "ollama" and not ollama_invoked

    provider_value = response_metadata.get("provider")
    provider = (
        provider_value
        if isinstance(provider_value, str) and provider_value
        else writer_provider
    )
    model_value = response_metadata.get("model")
    model = (
        None
        if deterministic_only
        else (
            model_value
            if isinstance(model_value, str)
            else ollama_model if writer_provider == "ollama" else None
        )
    )
    envelope_type = response_metadata.get("response_envelope_type")
    if not isinstance(envelope_type, str):
        envelope_type = (
            "deterministic-local-patches"
            if deterministic_only
            else "provider-response-metadata-unavailable"
        )
    output_format = response_metadata.get("output_format")
    if not isinstance(output_format, str):
        output_format = (
            "deterministic-json"
            if deterministic_only
            else "structured-output"
        )
    execution_mode = response_metadata.get("execution_mode")
    if not isinstance(execution_mode, str):
        hybrid_execution_mode = execution.get("execution_mode")
        execution_mode = (
            hybrid_execution_mode
            if isinstance(hybrid_execution_mode, str)
            else "deterministic_only" if deterministic_only else "provider"
        )

    initial: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "response_envelope_type": envelope_type,
        "output_format": output_format,
        "execution_mode": execution_mode,
        "tailored_content": {
            "filename": tailored_content_path.name,
            "sha256": sha256_file(tailored_content_path),
        },
    }
    if writer_provider == "ollama":
        initial["ollama_invoked"] = ollama_invoked
    runtime = response_metadata.get("runtime")
    if isinstance(runtime, str):
        initial["runtime"] = runtime
    response = response_metadata.get("response")
    if isinstance(response, dict):
        initial["response"] = dict(response)
    if execution:
        initial["execution"] = dict(execution)
    budget_repair = response_metadata.get("budget_repair")
    if isinstance(budget_repair, dict):
        initial["budget_repair"] = dict(budget_repair)
    return initial, deterministic_only


def _apify_progress_message(
    phase: str,
    elapsed_seconds: float,
    status: str | None,
) -> str:
    if phase == "starting_actor":
        return "Starting Apify retrieval."
    if phase == "waiting_for_actor":
        status_text = f"; status {status}" if status else ""
        return (
            "Waiting for the Apify Actor "
            f"— elapsed {_elapsed_label(elapsed_seconds)}{status_text}."
        )
    if phase == "reading_job_result":
        return "Reading the matching job result from the Apify dataset."
    if phase == "normalizing_job_posting":
        return "Normalizing the Apify result into the canonical job posting."
    if phase == "ready_for_review":
        return "Canonical validation passed. The job posting is ready for review."
    return "Apify LinkedIn retrieval is continuing."


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
    writer_provider = getattr(args, "writer_provider", "antigravity")
    if writer_provider not in {"ollama", "antigravity"}:
        raise InputError(
            f"Unsupported résumé writer provider: {writer_provider!r}."
        )
    analysis_provider = normalize_analysis_provider(
        getattr(args, "analysis_provider", DEFAULT_ANALYSIS_PROVIDER)
    )
    analysis_label = analysis_provider_label(analysis_provider)
    ollama_model = validate_ollama_model_name(
        getattr(args, "ollama_model", DEFAULT_OLLAMA_MODEL)
    )
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
    if antigravity_retry_inputs is not None:
        writer_provider = "antigravity"
    writer_name = "Gemma 4 12B" if writer_provider == "ollama" else "Antigravity"
    document_format = (
        "headless" if writer_provider == "ollama" else "master-template"
    )
    requested_linkedin_url = None
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
            job_source = getattr(args, "job_source_override", "job-file")
            if job_source not in {"job-file", "file", "pasted", "pasted_text"}:
                raise InputError("Invalid internal local job-source classification.")

    if job_description is not None:
        validate_confirmed_job_description(job_description)

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
            "apify-linkedin",
            "retrieval",
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
        "writer": {
            "provider": writer_provider,
            "name": writer_name,
            "model": ollama_model if writer_provider == "ollama" else None,
            "document_format": document_format,
        },
        "analysis": {
            "provider": analysis_provider,
            "name": analysis_label,
            "automatic_fallback": False,
        },
        "source_resume": {
            "filename": resume_path.name,
            "sha256_before": source_hash,
            "sha256_after": None,
            "unchanged": None,
        },
        "tools": {},
        "artifacts": [],
        "revision_cycle": {
            "state": "initial_generation",
            "maximum_attempts": 1,
            "attempt_count": 0,
            "authorization": None,
            "initial": {},
            "revision_1": None,
            "final_generation": None,
        },
    }
    analytics_job_id: int | None = None
    analytics_application_id: int | None = None
    analytics_event_run_identifier = run_directory.name
    metadata["analytics"] = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "database_filename": ANALYTICS_DATABASE_FILENAME,
        "job_id": None,
        "application_id": None,
        "resume_version_id": None,
        "warnings": [],
    }
    analytics_setup_error: Exception | None = None
    try:
        analytics_path = getattr(args, "analytics_db", None)
        if analytics_path is None:
            analytics_path = default_analytics_database_path()
        analytics_store: AnalyticsStore | None = AnalyticsStore(
            analytics_path
        )
    except Exception as exc:  # analytics configuration is failure-isolated too
        analytics_store = None
        analytics_setup_error = exc

    def analytics_write(operation: str, callback: Any) -> Any:
        try:
            if analytics_store is None:
                assert analytics_setup_error is not None
                raise analytics_setup_error
            result = callback()
        except Exception as exc:  # analytics must never invalidate a résumé run
            warning = {
                "operation": operation,
                "error_type": type(exc).__name__,
                "retryable_from_preserved_artifacts": True,
            }
            warnings = metadata["analytics"]["warnings"]
            if warning not in warnings:
                warnings.append(warning)
            hooks.warning(
                "Local analytics could not record "
                f"{operation}; the résumé pipeline will continue. Preserved run "
                "artifacts support a safe local retry.",
                analytics_operation=operation,
                analytics_error_type=type(exc).__name__,
            )
            result = None
        try:
            _update_metadata(
                metadata,
                metadata_path,
                run_directory=run_directory,
            )
        except Exception as exc:  # this extra analytics checkpoint is non-critical
            hooks.warning(
                "The local analytics checkpoint could not be added to run metadata; "
                "the résumé pipeline will continue.",
                analytics_operation=operation,
                analytics_checkpoint_error_type=type(exc).__name__,
            )
        return result
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
            "apify_linkedin_retrieval": False,
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
            metadata["tools"] = _analysis_dependency_versions(
                run_directory,
                analysis_provider,
                ollama_model,
            )
        else:
            metadata["tools"] = _analysis_dependency_versions(
                run_directory,
                analysis_provider,
                ollama_model,
            )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if requested_linkedin_url is not None:
            metadata["stage"] = "apify-linkedin-retrieval"
            hooks.progress(
                "fetching_job",
                "Validating the LinkedIn URL and locally extracted job ID.",
            )
            metadata["apify_linkedin_retrieval"] = {
                "provider": "apify",
                "interface": "Apify API v2",
                "actor_configuration": "APIFY_ACTOR_ID",
                "actor_input_format": "searchUrls",
                "authentication_transport": "bearer-header",
                "retrieval_only": True,
                "automatic_fallback": False,
            }
            metadata["linkedin_job"] = {
                "requested_url": requested_linkedin_url.normalized,
                "final_resolved_url": None,
                "linkedin_job_id": requested_linkedin_url.job_id,
            }
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            fetched_job = invoke_apify_linkedin_retrieval(
                requested_url=requested_linkedin_url,
                run_directory=run_directory,
                timeout_seconds=timeout_seconds,
                progress_handler=lambda phase, elapsed, status: hooks.progress(
                    "fetching_job",
                    _apify_progress_message(phase, elapsed, status),
                    elapsed_seconds=max(0, int(elapsed)),
                    apify_phase=phase,
                    actor_status=status,
                ),
            )
            job_description = validate_confirmed_job_description(
                fetched_job["normalized_job_description"]
            )
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
                "The posting was retrieved and is waiting for explicit confirmation.",
            )
            if hooks.approval_handler is None:
                print(posting_confirmation_text(fetched_job))

            def record_canonical_view() -> None:
                nonlocal analytics_job_id, analytics_application_id
                result = analytics_write(
                    "the validated viewed posting",
                    lambda: analytics_store.record_job_viewed(
                        observation_from_canonical_job(fetched_job),
                        run_identifier=analytics_event_run_identifier,
                    ),
                )
                if result is not None:
                    analytics_job_id = result.job_id
                    analytics_application_id = result.application_id
                    metadata["analytics"]["job_id"] = analytics_job_id
                    metadata["analytics"]["application_id"] = (
                        analytics_application_id
                    )

            posting_approval = hooks.approve(
                kind="linkedin_posting",
                title="LinkedIn posting",
                payload=fetched_job,
                assume_yes=args.yes,
                on_presented=record_canonical_view,
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
                job_description = validate_confirmed_job_description(
                    pasted_description
                )
                job_source = "pasted-fallback"
                metadata["job_source"] = job_source
                metadata["linkedin_job"]["used_pasted_fallback"] = True
                atomic_write_text(
                    run_directory / "job-description.txt",
                    job_description.rstrip() + "\n",
                )
                if analytics_job_id is not None:
                    pasted_observation = replace(
                        observation_from_canonical_job(fetched_job),
                        source="pasted_text",
                        description_sha256=job_description_sha256(job_description),
                    )
                    pasted_result = analytics_write(
                        "the explicitly pasted posting observation",
                        lambda: analytics_store.record_job_viewed(
                            pasted_observation,
                            run_identifier=analytics_event_run_identifier,
                        ),
                    )
                    if pasted_result is not None:
                        analytics_job_id = pasted_result.job_id
                        analytics_application_id = pasted_result.application_id
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
                "Posting confirmed. The run directory now uses the retrieved identity.",
                run_directory=str(run_directory),
                company=company,
                role=role,
            )

        assert company is not None and role is not None and job_description is not None
        if (
            requested_linkedin_url is None
            and retry_inputs is None
            and antigravity_retry_inputs is None
        ):
            local_result = analytics_write(
                "the validated local viewed posting",
                lambda: analytics_store.record_job_viewed(
                    observation_from_local_job(
                        company=company,
                        title=role,
                        description=job_description,
                        source=job_source,
                    ),
                    run_identifier=analytics_event_run_identifier,
                ),
            )
            if local_result is not None:
                analytics_job_id = local_result.job_id
                analytics_application_id = local_result.application_id
                metadata["analytics"]["job_id"] = analytics_job_id
                metadata["analytics"]["application_id"] = (
                    analytics_application_id
                )
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
                run_directory=run_directory,
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
        if analytics_job_id is not None:
            analytics_write(
                "the validated job-requirement catalog",
                lambda: analytics_store.record_requirements(
                    analytics_job_id,
                    job_requirements,
                    job_description=job_description,
                ),
            )
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
                "Apify retrieval and Codex analysis are not being invoked.",
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
            resolved_name = next(
                (
                    name
                    for name in (
                        ANALYSIS_RESOLVED_FILENAME,
                        CODEX_ANALYSIS_RESOLVED_FILENAME,
                    )
                    if name in antigravity_retry_inputs.artifact_bytes
                ),
                CODEX_ANALYSIS_RESOLVED_FILENAME,
            )
            analysis_bytes = antigravity_retry_inputs.artifact_bytes[resolved_name]
            atomic_write_bytes(
                run_directory / resolved_name,
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
                sha256_file(run_directory / resolved_name)
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
            transport_artifact = None
            if analysis_provider == "codex":
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
            elif analysis_provider == "grok_cli":
                metadata["stage"] = "analysis-schema-preflight"
                hooks.progress(
                    "codex_analysis",
                    f"Generating the source-bound {analysis_label} analysis schema.",
                )
                from .grok_analysis import prepare_grok_analysis_schema

                grok_schema = prepare_grok_analysis_schema(
                    extracted,
                    job_requirements,
                    run_directory,
                )
                metadata["grok_analysis_schema"] = {
                    "filename": Path(grok_schema["path"]).name,
                    "sha256": grok_schema["sha256"],
                    "size_bytes": grok_schema["size_bytes"],
                    "evidence_source_id_count": grok_schema[
                        "evidence_source_id_count"
                    ],
                    "editable_source_id_count": grok_schema[
                        "editable_source_id_count"
                    ],
                    "job_requirement_id_count": grok_schema[
                        "job_requirement_id_count"
                    ],
                    "generated_from_source_and_requirement_catalogs": True,
                }
                _update_metadata(metadata, metadata_path, run_directory=run_directory)
            else:
                metadata["stage"] = "analysis-schema-preflight"
                hooks.progress(
                    "codex_analysis",
                    f"Generating the source-bound {analysis_label} analysis schema.",
                )
                from .gemma_analysis import prepare_gemma_analysis_schema

                gemma_schema = prepare_gemma_analysis_schema(
                    extracted,
                    job_requirements,
                    run_directory,
                )
                metadata["gemma_analysis_schema"] = {
                    "filename": Path(gemma_schema["path"]).name,
                    "sha256": gemma_schema["sha256"],
                    "size_bytes": gemma_schema["size_bytes"],
                    "evidence_source_id_count": gemma_schema[
                        "evidence_source_id_count"
                    ],
                    "editable_source_id_count": gemma_schema[
                        "editable_source_id_count"
                    ],
                    "job_requirement_id_count": gemma_schema[
                        "job_requirement_id_count"
                    ],
                    "generated_from_source_and_requirement_catalogs": True,
                    "architecture": "two_phase_coverage_and_edits",
                }
                _update_metadata(metadata, metadata_path, run_directory=run_directory)

            metadata["stage"] = "codex-analysis"
            hooks.progress(
                "codex_analysis",
                f"{analysis_label} analysis started. Strong reasoning may take "
                "several minutes; no unreliable ETA is shown.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            def _analysis_progress(elapsed: float, alive: bool) -> None:
                hooks.progress(
                    "codex_analysis",
                    (
                        f"{analysis_label} analysis is still running"
                        if alive
                        else (
                            f"No {analysis_label} process detected; the process "
                            "exited and local structured-output validation is "
                            "continuing"
                        )
                    )
                    + f" — elapsed {_elapsed_label(elapsed)}.",
                    elapsed_seconds=max(0, int(elapsed)),
                    provider_process_alive=alive,
                )

            if analysis_provider == "gemma_local":
                from .gemma_analysis import invoke_gemma_analysis

                raw_analysis = invoke_gemma_analysis(
                    extracted_resume=extracted,
                    job_description=job_description,
                    job_requirements=job_requirements,
                    company=company,
                    role=role,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    model=ollama_model,
                    progress_handler=_analysis_progress,
                    status_handler=lambda status: hooks.progress(
                        "codex_analysis",
                        f"{analysis_label}: {status}.",
                    ),
                )
            else:
                raw_analysis = invoke_analysis(
                    provider=analysis_provider,
                    extracted_resume=extracted,
                    job_description=job_description,
                    job_requirements=job_requirements,
                    company=company,
                    role=role,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    transport_artifact=transport_artifact,
                    model=ollama_model,
                    progress_handler=_analysis_progress,
                )
            analysis, analysis_issues = resolve_analysis_evidence(
                raw_analysis,
                extracted,
                job_requirements,
            )
            if analysis_issues:
                raise SourceEvidenceError(
                    f"{analysis_label} analysis failed local source-evidence "
                    "validation:\n"
                    + "\n".join(
                        f"- {issue.describe()}" for issue in analysis_issues
                    )
                )
            resolved_meta = write_resolved_analysis_artifact(
                run_directory,
                analysis,
                provider=analysis_provider,
            )
            metadata["analysis_resolved"] = {
                "filename": resolved_meta["filename"],
                "provider": resolved_meta["provider"],
                "provider_label": resolved_meta["provider_label"],
                "legacy_codex_alias_written": resolved_meta[
                    "legacy_codex_alias_written"
                ],
                "sha256": sha256_file(
                    run_directory / resolved_meta["filename"]
                ),
            }
            if hooks.approval_handler is None:
                print(
                    readable_analysis(
                        analysis,
                        provider_label=analysis_label,
                    )
                )
            if analysis["questions_for_user"]:
                questions = "\n".join(
                    f"- {question}" for question in analysis["questions_for_user"]
                )
                raise WaitingError(
                    f"{analysis_label} has unanswered factual questions. The "
                    "pipeline stopped instead of guessing:\n"
                    f"{questions}"
                )
            hooks.progress(
                "reviewing_changes",
                f"{analysis_label} analysis passed local evidence checks and "
                "awaits approval.",
            )
            analysis_approval = hooks.approve(
                kind="codex_analysis",
                title=f"{analysis_label} analysis",
                payload=analysis,
                assume_yes=args.yes,
            )
            if analysis_approval.action != "approve":
                raise ApprovalError(
                    f"{analysis_label} analysis was not approved; artifacts were "
                    "preserved."
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
            if analytics_job_id is not None:
                analytics_write(
                    "validated requirement gap outcomes",
                    lambda: analytics_store.record_gap_assessments(
                        analytics_job_id,
                        analysis,
                    ),
                )
                planned_application_id = analytics_write(
                    "the tailoring approval",
                    lambda: analytics_store.record_tailoring_approval(
                        analytics_job_id,
                        run_identifier=run_directory.name,
                        source="pipeline",
                    ),
                )
                if planned_application_id is not None:
                    analytics_application_id = planned_application_id
                    metadata["analytics"]["application_id"] = (
                        analytics_application_id
                    )

            metadata["stage"] = "tailoring-dependency-check"
            hooks.progress(
                "antigravity_tailoring",
                "Checking downstream tools after the Codex analysis approval.",
            )
            metadata["tools"].update(
                _tailoring_dependency_versions(
                    run_directory,
                    writer_provider,
                    ollama_model,
                )
            )
            _update_metadata(
                metadata,
                metadata_path,
                run_directory=run_directory,
            )

        metadata["stage"] = f"{writer_provider}-tailoring-preflight"
        hooks.progress(
            "antigravity_tailoring",
            (
                "Authenticating the approved inputs before local tailoring."
                if writer_provider == "ollama"
                else f"Authenticating the approved tailoring inputs before {writer_name}."
            ),
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        try:
            # The Ollama writer entrypoint owns its authenticated preflight so
            # deterministic-only calls cannot bypass it and prose calls do not
            # run it twice. Antigravity retains its existing CLI preflight.
            if writer_provider == "antigravity":
                preflight_tailoring_inputs(
                    master_content=extracted["content"],
                    extracted_resume=extracted,
                    job_description=job_description,
                    job_requirements=job_requirements,
                    approved_analysis=analysis,
                    company=company,
                    role=role,
                )
            metadata[f"{writer_provider}_tailoring_preflight"] = (
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
        except AntigravityTailoringPreflightError as exc:
            if writer_provider == "ollama":
                raise TailoringPreflightError(
                    "Local Ollama tailoring preflight failed. No writer request "
                    "was launched."
                ) from exc
            raise
        except RequirementExtractionError as exc:
            diagnostic_path = run_directory / "requirement-extraction-diagnostic.json"
            if exc.diagnostic:
                atomic_write_json(diagnostic_path, exc.diagnostic)

            error_message = "The job posting could not be divided into reliable individual requirements. No analysis provider was called."
            if writer_provider == "ollama":
                raise TailoringPreflightError(error_message) from exc
            raise AntigravityTailoringPreflightError(error_message) from exc
        except InputError as exc:
            if writer_provider == "ollama":
                raise TailoringPreflightError(str(exc)) from exc
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
            metadata["stage"] = f"{writer_provider}-tailoring"
            if writer_provider == "ollama":
                hooks.progress(
                    "antigravity_tailoring",
                    "Python is compiling approved edits locally. Gemma 4 12B "
                    "will be invoked only if prose authoring is required.",
                )
            else:
                hooks.progress(
                    "antigravity_tailoring",
                    f"{writer_name} is writing the complete tailored résumé content "
                    "from the approved schema-constrained edits.",
                )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            if writer_provider == "ollama":
                tailored_content = invoke_ollama(
                    master_content=extracted["content"],
                    extracted_resume=extracted,
                    job_description=job_description,
                    job_requirements=job_requirements,
                    approved_analysis=analysis,
                    company=company,
                    role=role,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    model=ollama_model,
                    heartbeat_handler=lambda elapsed, alive: hooks.progress(
                        "antigravity_tailoring",
                        (
                            "Gemma 4 12B is still writing locally"
                            if alive
                            else "Gemma 4 12B completed; local validation is continuing"
                        )
                        + f" — elapsed {_elapsed_label(elapsed)}.",
                        elapsed_seconds=max(0, int(elapsed)),
                    ),
                )
            else:
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
        initial_content_path = run_directory / "tailored-content.initial.json"
        atomic_write_json(initial_content_path, tailored_content)
        initial_response_metadata = (
            load_ollama_response_metadata(run_directory)
            if writer_provider == "ollama"
            else load_antigravity_response_metadata(run_directory)
        )
        if initial_response_metadata is None:
            raise IntegrityError(
                f"The initial {writer_name} response metadata is unavailable."
            )
        initial_generation, deterministic_only = _initial_generation_metadata(
            response_metadata=initial_response_metadata,
            writer_provider=writer_provider,
            ollama_model=ollama_model,
            tailored_content_path=initial_content_path,
        )
        metadata["revision_cycle"]["initial"] = initial_generation
        if deterministic_only:
            metadata["writer"].update(
                {
                    "provider": "deterministic",
                    "name": "Deterministic local compiler",
                    "model": None,
                    "runtime": "local",
                    "ollama_invoked": False,
                }
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
        initial_diff_path = run_directory / "content-diff.initial.md"
        atomic_write_text(initial_diff_path, content_diff)
        approval_artifacts = {
            "tailored_content": _publish_authenticated_artifact(
                initial_content_path,
                run_directory / "tailored-content.json",
            ),
            "content_diff": _publish_authenticated_artifact(
                initial_diff_path,
                run_directory / "content-diff.md",
            ),
        }
        metadata["revision_cycle"]["initial"]["validation"] = {
            "status": "PASS" if report.passed else "BLOCKED",
            "issues": report.issues,
            "diff": {
                "filename": initial_diff_path.name,
                "sha256": sha256_file(initial_diff_path),
            },
            "approval_artifacts": approval_artifacts,
        }
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
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
        metadata["revision_cycle"]["initial"]["content_approval"] = {
            "decision": "approved",
            "mode": "assume_yes" if args.yes else "explicit",
            "timestamp": utc_now_iso(),
            "diff": {
                "filename": initial_diff_path.name,
                "sha256": sha256_file(initial_diff_path),
            },
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
        docx_path = run_directory / f"{basename}.initial.docx"
        pdf_path = run_directory / f"{basename}.initial.pdf"
        preview_path = run_directory / "preview.initial.png"

        from .docx_render import export_and_validate_pdf, render_tailored_docx

        if document_format == "headless":
            from .headless_render import render_headless_docx

            render_headless_docx(
                source_path=resume_path,
                destination_path=docx_path,
                tailored_content=tailored_content,
                extracted_resume=extracted,
                expected_source_hash=source_hash,
            )
        else:
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
            working_directory=work_directory / "initial",
            required_text=_required_pdf_text(
                extracted,
                document_format=document_format,
            ),
        )
        initial_layout = {
            "status": "PASS",
            "pages": 1,
            "required_text_present": True,
            "bounding_boxes_valid": True,
            "docx": {"filename": docx_path.name, "sha256": sha256_file(docx_path)},
            "pdf": {"filename": pdf_path.name, "sha256": sha256_file(pdf_path)},
            "preview": {
                "filename": preview_path.name,
                "sha256": sha256_file(preview_path),
            },
        }
        metadata["layout_validation"] = initial_layout
        metadata["revision_cycle"]["initial"]["layout_validation"] = initial_layout

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
            work_directory=work_directory / "initial-qa",
            timeout_seconds=timeout_seconds,
            generation="initial",
        )
        initial_qa_path = run_directory / "final-qa.initial.json"
        initial_qa_metadata = {
            "generation": "initial",
            "provider": "codex",
            "session": "fresh_ephemeral_read_only",
            "status": qa_result["status"],
            "summary": qa_result["summary"],
            "issues": qa_result["issues"],
            "technical_failure": qa_result["technical_failure"],
            "result": {
                "filename": initial_qa_path.name,
                "sha256": sha256_file(initial_qa_path),
            },
        }
        metadata["final_qa"] = initial_qa_metadata
        metadata["revision_cycle"]["initial"]["qa"] = initial_qa_metadata
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if qa_result["status"] == "technical_failure":
            metadata["revision_cycle"]["state"] = "initial_qa_technical_failure"
            raise QAError(
                "The first fresh read-only Codex QA could not complete reliably. "
                "All initial artifacts were preserved; no revision was invoked."
            )
        if qa_result["status"] == "pass":
            metadata["revision_cycle"]["state"] = "initial_passed_qa"
            metadata["revision_cycle"]["final_generation"] = "initial"
        else:
            metadata["revision_cycle"]["state"] = "awaiting_revision_authorization"
            metadata["stage"] = "revision-authorization"
            hooks.progress(
                "revision_phase",
                f"Codex found material issues. One optional {writer_name} revision "
                "requires explicit authorization.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            if hooks.approval_handler is None:
                print("\nCodex material findings:")
                for issue in qa_result["issues"]:
                    print(
                        f"- {issue['issue_id']} ({issue['category']}): "
                        f"{issue['description']}"
                    )
            revision_authorization = hooks.authorize_revision(
                payload={
                    "qa_result": qa_result,
                    "maximum_attempts": 1,
                    "initial_preview": preview_path.name,
                    "writer_provider": writer_provider,
                    "writer_name": writer_name,
                },
                provider_name=writer_name,
            )
            authorization_record = {
                "decision": revision_authorization.action,
                "mode": (
                    "web" if hooks.approval_handler is not None else "interactive"
                ),
                "timestamp": utc_now_iso(),
                "yes_flag_did_not_authorize": bool(args.yes),
            }
            metadata["revision_cycle"]["authorization"] = authorization_record
            if revision_authorization.action != "revise_once":
                metadata["revision_cycle"]["state"] = "stopped_after_initial_qa"
                _update_metadata(metadata, metadata_path, run_directory=run_directory)
                raise QAError(
                    "Codex found material issues and the optional revision was not "
                    "authorized. Initial artifacts were preserved."
                )

            if metadata["revision_cycle"]["attempt_count"] != 0:
                raise RevisionValidationError(
                    "The one-revision limit was already consumed."
                )
            metadata["revision_cycle"]["attempt_count"] = 1
            metadata["revision_cycle"]["state"] = "revision_1_authorized"
            allowed_targets = approved_revision_targets(
                qa_result=qa_result,
                approved_analysis=analysis,
            )
            revision_request_path = run_directory / "revision-request.json"
            revision_input_manifest = {
                "version": 1,
                "attempt": 1,
                "maximum_attempts": 1,
                "authorization": authorization_record,
                "qa_issues": qa_result["issues"],
                "allowed_target_issue_map": allowed_targets,
                "inputs": {
                    "source_resume_sha256": source_hash,
                    "extracted_resume_sha256": sha256_file(
                        run_directory / "extracted-master-resume.json"
                    ),
                    "approved_analysis_sha256": sha256_file(
                        run_directory
                        / (
                            ANALYSIS_RESOLVED_FILENAME
                            if (
                                run_directory / ANALYSIS_RESOLVED_FILENAME
                            ).is_file()
                            else CODEX_ANALYSIS_RESOLVED_FILENAME
                        )
                    ),
                    "initial_tailored_content_sha256": sha256_file(
                        initial_content_path
                    ),
                    "initial_qa_sha256": sha256_file(initial_qa_path),
                    "revision_schema_sha256": sha256_file(
                        schema_path(REVISION_SCHEMA_NAME)
                    ),
                },
            }
            atomic_write_json(revision_request_path, revision_input_manifest)
            revision_input_manifest["request"] = {
                "filename": revision_request_path.name,
                "sha256": sha256_file(revision_request_path),
            }
            metadata["revision_cycle"]["revision_1"] = {
                "state": "provider_in_progress",
                "input_manifest": revision_input_manifest,
            }
            metadata["stage"] = f"{writer_provider}-revision-1"
            hooks.progress(
                "revision_phase",
                f"{writer_name} is applying the one authorized QA revision.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            if writer_provider == "ollama":
                revised_content = invoke_ollama_revision(
                    current_tailored_content=tailored_content,
                    extracted_resume=extracted,
                    approved_analysis=analysis,
                    qa_result=qa_result,
                    company=company,
                    role=role,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    attempt_number=1,
                    model=ollama_model,
                )
                revision_response_metadata = load_ollama_response_metadata(
                    run_directory,
                    filename=OLLAMA_REVISION_RESPONSE_METADATA_FILENAME,
                )
            else:
                revised_content = invoke_antigravity_revision(
                    current_tailored_content=tailored_content,
                    extracted_resume=extracted,
                    approved_analysis=analysis,
                    qa_result=qa_result,
                    company=company,
                    role=role,
                    run_directory=run_directory,
                    timeout_seconds=timeout_seconds,
                    antigravity_duration=antigravity_duration,
                    attempt_number=1,
                )
                revision_response_metadata = load_antigravity_response_metadata(
                    run_directory,
                    filename=REVISION_RESPONSE_METADATA_FILENAME,
                )
            if revision_response_metadata is None:
                raise IntegrityError(
                    f"The {writer_name} revision response metadata is unavailable."
                )
            revised_content_path = (
                run_directory / "tailored-content.revision-1.json"
            )
            atomic_write_json(revised_content_path, revised_content)
            metadata["revision_cycle"]["revision_1"].update(
                {
                    "state": "local_validation",
                    "provider": writer_provider,
                    "model": ollama_model if writer_provider == "ollama" else None,
                    "response": dict(revision_response_metadata["response"]),
                    "response_envelope_type": revision_response_metadata[
                        "response_envelope_type"
                    ],
                    "output_format": revision_response_metadata["output_format"],
                    "tailored_content": {
                        "filename": revised_content_path.name,
                        "sha256": sha256_file(revised_content_path),
                    },
                }
            )
            metadata["stage"] = "revision-1-local-evidence-check"
            hooks.progress(
                "revision_phase",
                "Python is validating revision 1 against every original evidence "
                "and QA authorization boundary.",
            )
            revision_report = validate_tailored_content(
                original=extracted["content"],
                tailored=revised_content,
                extracted_resume=extracted,
                analysis=analysis,
                target_role=role,
            )
            issue_map = validate_revision_scope(
                initial_content=tailored_content,
                revised_content=revised_content,
                qa_result=qa_result,
                approved_analysis=analysis,
            )
            master_revision_diff = build_content_diff(
                extracted["content"],
                revised_content,
                revision_report,
            )
            revision_diff = build_revision_diff(
                initial_content=tailored_content,
                revised_content=revised_content,
                issue_map=issue_map,
                master_to_revision_diff=master_revision_diff,
            )
            revision_diff_path = run_directory / "content-diff.revision-1.md"
            atomic_write_text(revision_diff_path, revision_diff)
            if not revision_report.passed:
                raise RevisionValidationError(
                    "Revision 1 failed local factual, structural, or content-budget "
                    "validation. No provider content was promoted."
                )
            metadata["revision_cycle"]["revision_1"]["validation"] = {
                "status": "PASS",
                "changed_target_issue_map": issue_map,
                "diff": {
                    "filename": revision_diff_path.name,
                    "sha256": sha256_file(revision_diff_path),
                },
            }
            metadata["revision_cycle"]["revision_1"]["state"] = (
                "awaiting_content_approval"
            )
            metadata["stage"] = "revision-1-content-approval"
            hooks.progress(
                "revision_phase",
                "Revision 1 passed local validation and awaits explicit approval.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            if hooks.approval_handler is None:
                print("\n" + revision_diff)
            revised_approval = hooks.approve_revised_content(
                payload={
                    "content_diff": revision_diff,
                    "tailored_content": revised_content,
                    "issue_map": issue_map,
                    "evidence": {
                        "passed": revision_report.passed,
                        "issues": revision_report.issues,
                        "introduced_technologies": (
                            revision_report.introduced_technologies
                        ),
                        "introduced_metrics": revision_report.introduced_metrics,
                        "introduced_role_labels": (
                            revision_report.introduced_role_labels
                        ),
                        "introduced_availability": (
                            revision_report.introduced_availability
                        ),
                    },
                }
            )
            revised_approval_record = {
                "decision": revised_approval.action,
                "mode": (
                    "web" if hooks.approval_handler is not None else "interactive"
                ),
                "timestamp": utc_now_iso(),
                "yes_flag_did_not_authorize": bool(args.yes),
            }
            metadata["revision_cycle"]["revision_1"]["content_approval"] = (
                revised_approval_record
            )
            if revised_approval.action != "approve":
                metadata["revision_cycle"]["revision_1"]["state"] = (
                    "content_rejected"
                )
                metadata["revision_cycle"]["state"] = "revision_1_rejected"
                _update_metadata(metadata, metadata_path, run_directory=run_directory)
                raise ApprovalError(
                    "Revision 1 was rejected. Initial and revised artifacts were "
                    "preserved; no revised document was rendered."
                )

            revision_docx_path = run_directory / f"{basename}.revision-1.docx"
            revision_pdf_path = run_directory / f"{basename}.revision-1.pdf"
            revision_preview_path = run_directory / "preview.revision-1.png"
            metadata["stage"] = "revision-1-docx-render"
            hooks.progress(
                "revision_phase",
                "Python is rendering the approved revision 1 without overwriting "
                "the initial generation.",
            )
            if document_format == "headless":
                render_headless_docx(
                    source_path=resume_path,
                    destination_path=revision_docx_path,
                    tailored_content=revised_content,
                    extracted_resume=extracted,
                    expected_source_hash=source_hash,
                )
            else:
                render_tailored_docx(
                    source_path=resume_path,
                    destination_path=revision_docx_path,
                    tailored_content=revised_content,
                    expected_source_hash=source_hash,
                )
            revision_pdf_text = export_and_validate_pdf(
                docx_path=revision_docx_path,
                pdf_path=revision_pdf_path,
                preview_path=revision_preview_path,
                working_directory=work_directory / "revision-1",
                required_text=_required_pdf_text(
                    extracted,
                    document_format=document_format,
                ),
            )
            revision_layout = {
                "status": "PASS",
                "pages": 1,
                "required_text_present": True,
                "bounding_boxes_valid": True,
                "docx": {
                    "filename": revision_docx_path.name,
                    "sha256": sha256_file(revision_docx_path),
                },
                "pdf": {
                    "filename": revision_pdf_path.name,
                    "sha256": sha256_file(revision_pdf_path),
                },
                "preview": {
                    "filename": revision_preview_path.name,
                    "sha256": sha256_file(revision_preview_path),
                },
            }
            metadata["revision_cycle"]["revision_1"]["layout_validation"] = (
                revision_layout
            )
            metadata["stage"] = "revision-1-final-codex-qa"
            hooks.progress(
                "revision_phase",
                "A second fresh Codex session is reviewing revision 1. No further "
                "revision is permitted.",
            )
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
            second_qa = invoke_final_qa(
                original_extraction=extracted,
                job_description=job_description,
                analysis=analysis,
                tailored_pdf_text=revision_pdf_text,
                content_diff=revision_diff,
                preview_path=revision_preview_path,
                run_directory=run_directory,
                work_directory=work_directory / "revision-1-qa",
                timeout_seconds=timeout_seconds,
                generation="revision-1",
            )
            second_qa_path = run_directory / "final-qa.revision-1.json"
            second_qa_metadata = {
                "generation": "revision-1",
                "provider": "codex",
                "session": "fresh_ephemeral_read_only",
                "status": second_qa["status"],
                "summary": second_qa["summary"],
                "issues": second_qa["issues"],
                "technical_failure": second_qa["technical_failure"],
                "result": {
                    "filename": second_qa_path.name,
                    "sha256": sha256_file(second_qa_path),
                },
            }
            metadata["revision_cycle"]["revision_1"]["qa"] = second_qa_metadata
            metadata["final_qa"] = second_qa_metadata
            metadata["layout_validation"] = revision_layout
            if second_qa["status"] != "pass":
                metadata["revision_cycle"]["revision_1"]["state"] = (
                    "qa_not_passed"
                )
                metadata["revision_cycle"]["state"] = "one_revision_limit_reached"
                _update_metadata(metadata, metadata_path, run_directory=run_directory)
                raise QAError(
                    "Revision 1 did not pass the second fresh Codex QA. The "
                    "one-revision limit was reached; all artifacts were preserved."
                )
            metadata["revision_cycle"]["revision_1"]["state"] = "passed_qa"
            metadata["revision_cycle"]["state"] = "revision_1_passed_qa"
            metadata["revision_cycle"]["final_generation"] = "revision-1"

        final_generation = metadata["revision_cycle"]["final_generation"]
        metadata["revision_cycle"]["published_artifacts"] = (
            _publish_final_generation(
                run_directory=run_directory,
                basename=basename,
                generation=final_generation,
            )
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        if analytics_application_id is not None:
            published_docx = metadata["revision_cycle"]["published_artifacts"].get(
                "docx",
                {},
            )
            artifact_reference = published_docx.get("filename")
            qa_outcome = metadata.get("final_qa", {}).get("status")
            if isinstance(artifact_reference, str) and isinstance(qa_outcome, str):
                resume_version_id = analytics_write(
                    "the successfully published résumé version",
                    lambda: analytics_store.record_resume_version(
                        analytics_application_id,
                        run_identifier=run_directory.name,
                        artifact_reference=artifact_reference,
                        writer_provider=str(
                            metadata.get("writer", {}).get(
                                "provider",
                                writer_provider,
                            )
                        ),
                        qa_outcome=qa_outcome,
                    ),
                )
                if resume_version_id is not None:
                    metadata["analytics"]["resume_version_id"] = resume_version_id
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
        revision_stage = "revision" in str(metadata.get("stage", ""))
        if isinstance(exc, SourceEvidenceError):
            metadata["failure_class"] = "source-evidence-analysis"
        if isinstance(exc, CodexUsageLimitError):
            metadata["failure_class"] = "codex-usage-limit"
            metadata["analysis_failure_classification"] = exc.classification
            metadata["analysis_provider_suggestion"] = "gemma_local"
        if isinstance(exc, GrokExecutableError):
            metadata["failure_class"] = "grok-executable-unavailable"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokPromptTooLargeError):
            metadata["failure_class"] = "grok-prompt-too-large"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokUsageLimitError):
            metadata["failure_class"] = "grok-usage-limit"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokAuthenticationError):
            metadata["failure_class"] = "grok-authentication-failure"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokTimeoutError):
            metadata["failure_class"] = "grok-timeout"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokProcessError):
            metadata["failure_class"] = "grok-nonzero-exit"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokTransportEnvelopeError):
            metadata["failure_class"] = "grok-transport-envelope"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GrokInnerAnalysisError):
            metadata["failure_class"] = "grok-inner-analysis"
            metadata["analysis_failure_classification"] = exc.classification
        if (
            isinstance(exc, GrokAnalysisError)
            and "failure_class" not in metadata
        ):
            metadata["failure_class"] = "grok-analysis-failure"
            metadata["analysis_failure_classification"] = getattr(
                exc,
                "classification",
                "generic_provider_failure",
            )
        if isinstance(exc, GemmaOllamaUnavailableError):
            metadata["failure_class"] = "gemma-ollama-unavailable"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaModelUnavailableError):
            metadata["failure_class"] = "gemma-model-unavailable"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaAnalysisTimeoutError):
            metadata["failure_class"] = "gemma-analysis-timeout"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaOllamaInternalError):
            metadata["failure_class"] = "gemma-ollama-internal-error"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaOutputLimitError):
            metadata["failure_class"] = "gemma-output-limit"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaConnectionError):
            metadata["failure_class"] = "gemma-connection-failure"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaResponseTooLargeError):
            metadata["failure_class"] = "gemma-response-too-large"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaTransportEnvelopeError):
            metadata["failure_class"] = "gemma-transport-envelope"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaInnerAnalysisError):
            metadata["failure_class"] = "gemma-inner-analysis"
            metadata["analysis_failure_classification"] = exc.classification
        if isinstance(exc, GemmaStructuredOutputError):
            metadata["failure_class"] = "gemma-structured-output"
            metadata["analysis_failure_classification"] = exc.classification
        if (
            isinstance(exc, GemmaAnalysisError)
            and "failure_class" not in metadata
        ):
            metadata["failure_class"] = "gemma-analysis-failure"
            metadata["analysis_failure_classification"] = getattr(
                exc,
                "classification",
                "generic_provider_failure",
            )
        if (
            isinstance(exc, AnalysisProviderError)
            and "failure_class" not in metadata
        ):
            metadata["failure_class"] = "analysis-provider-failure"
            metadata["analysis_failure_classification"] = getattr(
                exc,
                "classification",
                "generic_provider_failure",
            )
        if isinstance(exc, TruthfulnessError) and "failure_class" not in metadata:
            # Schema-valid writer output rejected by the deterministic evidence
            # and content-budget gate. Attribute it to the writer provider so a
            # preserved historical Ollama failure is distinguishable from a
            # schema failure. The exception type and exit code are unchanged.
            stage_name = str(metadata.get("stage", ""))
            if writer_provider == "ollama" and "evidence" in stage_name:
                metadata["failure_class"] = "ollama-downstream-evidence"
        if isinstance(exc, ApifyConfigurationError):
            metadata["failure_class"] = "apify-configuration"
            metadata["retrieval_classification"] = exc.classification
        if isinstance(exc, ApifyLinkedInRetrievalError):
            metadata["failure_class"] = "apify-linkedin-retrieval"
            metadata["retrieval_classification"] = exc.classification
        if isinstance(exc, AntigravityLaunchSizeError):
            metadata["failure_class"] = "antigravity-launch-size"
        if isinstance(exc, AntigravityResponseEnvelopeError):
            metadata["failure_class"] = (
                "antigravity-revision-response-envelope"
                if revision_stage
                else "antigravity-response-envelope"
            )
        if isinstance(exc, AntigravityTailoringContractError):
            metadata["failure_class"] = "antigravity-tailoring-contract"
        if isinstance(exc, AntigravityCannotApplyError):
            metadata["failure_class"] = "antigravity-cannot-apply"
        if isinstance(exc, AntigravityTechnicalFailureError):
            metadata["failure_class"] = "antigravity-technical-failure"
        if isinstance(exc, AntigravityRevisionContractError):
            metadata["failure_class"] = "antigravity-revision-contract"
        if isinstance(exc, AntigravityRevisionCannotApplyError):
            metadata["failure_class"] = "antigravity-revision-cannot-apply"
        if isinstance(exc, AntigravityRevisionTechnicalFailureError):
            metadata["failure_class"] = "antigravity-revision-technical-failure"
        if isinstance(exc, OllamaConnectionError):
            metadata["failure_class"] = "ollama-connection"
        if isinstance(exc, OllamaBudgetError):
            metadata["failure_class"] = "ollama-budget-preflight"
        if isinstance(exc, OllamaTailoringContractError):
            metadata["failure_class"] = "ollama-tailoring-contract"
        # Specific sanitized classifications override the generic contract
        # class so a preserved failure names one validation path.
        if isinstance(exc, OllamaMalformedJSONError):
            metadata["failure_class"] = "ollama-malformed-json"
        if isinstance(exc, OllamaResponseEnvelopeError):
            metadata["failure_class"] = "ollama-response-envelope"
        if isinstance(exc, OllamaTransportSchemaError):
            metadata["failure_class"] = "ollama-transport-schema"
        if isinstance(exc, OllamaCanonicalSchemaError):
            metadata["failure_class"] = "ollama-canonical-schema"
        if isinstance(exc, OllamaOutputTruncationError):
            metadata["failure_class"] = "ollama-output-truncation"
        if isinstance(exc, OllamaEvidenceRejectionError):
            metadata["failure_class"] = "ollama-downstream-evidence"
        if isinstance(exc, OllamaCannotApplyError):
            metadata["failure_class"] = "ollama-cannot-apply"
        if isinstance(exc, OllamaTechnicalFailureError):
            metadata["failure_class"] = "ollama-technical-failure"
        if isinstance(exc, OllamaRevisionContractError):
            metadata["failure_class"] = "ollama-revision-contract"
        if isinstance(exc, OllamaRevisionCannotApplyError):
            metadata["failure_class"] = "ollama-revision-cannot-apply"
        if isinstance(exc, OllamaRevisionTechnicalFailureError):
            metadata["failure_class"] = "ollama-revision-technical-failure"
        if isinstance(exc, RevisionValidationError):
            metadata["failure_class"] = "revision-local-validation"
        if isinstance(exc, QAError):
            metadata["failure_class"] = "codex-final-qa"
        if isinstance(exc, AntigravityTailoringPreflightError):
            metadata["failure_class"] = "antigravity-tailoring-preflight"
        if (
            isinstance(exc, TailoringPreflightError)
            and not isinstance(exc, AntigravityTailoringPreflightError)
        ):
            metadata["failure_class"] = "ollama-tailoring-preflight"
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
        revision_response_metadata = load_antigravity_response_metadata(
            run_directory,
            filename=REVISION_RESPONSE_METADATA_FILENAME,
        )
        if revision_response_metadata is not None:
            revision_state = metadata.get("revision_cycle", {}).get("revision_1")
            if isinstance(revision_state, dict):
                revision_state["response_metadata"] = revision_response_metadata
        ollama_response_metadata = load_ollama_response_metadata(run_directory)
        if ollama_response_metadata is not None:
            metadata["ollama_response"] = ollama_response_metadata
        ollama_revision_metadata = load_ollama_response_metadata(
            run_directory,
            filename=OLLAMA_REVISION_RESPONSE_METADATA_FILENAME,
        )
        if ollama_revision_metadata is not None:
            revision_state = metadata.get("revision_cycle", {}).get("revision_1")
            if isinstance(revision_state, dict):
                revision_state["response_metadata"] = ollama_revision_metadata
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
