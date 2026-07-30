from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .antigravity_writer import invoke_antigravity
from .clipboard import read_clipboard
from .codex_analysis import invoke_codex_analysis, readable_analysis
from .evidence import (
    build_content_diff,
    validate_analysis_evidence,
    validate_tailored_content,
)
from .linkedin_job import (
    invoke_linkedin_job_extraction,
    posting_confirmation_text,
    validate_linkedin_url,
)
from .orchestration import PipelineHooks
from .qa import invoke_final_qa
from .utilities import (
    ApprovalError,
    CancellationError,
    ExitCode,
    InputError,
    IntegrityError,
    QAError,
    ResumeTailorError,
    TruthfulnessError,
    WaitingError,
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
        help="public HTTPS LinkedIn /jobs/view/ URL to extract with Antigravity",
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


def _dependency_versions(cwd: Path) -> dict[str, str]:
    codex = require_executable("codex")
    agy = require_executable("agy")
    libreoffice = require_executable("libreoffice")
    require_executable("pdfinfo")
    require_executable("pdftotext")
    require_executable("pdftoppm")
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
        "codex": _tool_version(codex, ["--version"], cwd=cwd),
        "antigravity": _tool_version(agy, ["--version"], cwd=cwd),
        "libreoffice": _tool_version(libreoffice, ["--version"], cwd=cwd),
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
    requested_linkedin_url = None
    if args.job_url is not None:
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
            job_source = "job-file"

    source_hash = sha256_file(resume_path)
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
    _update_metadata(metadata, metadata_path, run_directory=run_directory)
    hooks.progress(
        "validating_input",
        "Created an isolated run-artifact directory.",
        run_directory=str(run_directory),
    )

    caught_error: ResumeTailorError | None = None
    try:
        if job_description is not None:
            atomic_write_text(
                run_directory / "job-description.txt",
                job_description.rstrip() + "\n",
            )
        metadata["stage"] = "dependency-check"
        hooks.progress(
            "validating_input",
            "Checking local pipeline dependencies and verified CLI adapters.",
        )
        metadata["tools"] = _dependency_versions(run_directory)
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        if requested_linkedin_url is not None:
            metadata["stage"] = "linkedin-job-extraction"
            hooks.progress(
                "fetching_job",
                "Antigravity is passively extracting the public LinkedIn posting.",
            )
            metadata["linkedin_job"] = {
                "requested_url": requested_linkedin_url.normalized,
                "final_resolved_url": None,
                "linkedin_job_id": requested_linkedin_url.job_id,
            }
            _update_metadata(metadata, metadata_path, run_directory=run_directory)
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
        from .docx_extract import extract_resume

        metadata["stage"] = "extracting-master"
        hooks.progress(
            (
                "codex_analysis"
                if requested_linkedin_url is not None
                else "validating_input"
            ),
            "Structurally validating and extracting the master résumé.",
        )
        extracted, _ = extract_resume(resume_path)
        atomic_write_json(
            run_directory / "extracted-master-resume.json",
            extracted,
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)

        metadata["stage"] = "codex-analysis"
        hooks.progress(
            "codex_analysis",
            "Codex is performing read-only résumé-to-job evidence analysis.",
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        analysis = invoke_codex_analysis(
            extracted_resume=extracted,
            job_description=job_description,
            company=company,
            role=role,
            run_directory=run_directory,
            timeout_seconds=timeout_seconds,
        )
        analysis_issues = validate_analysis_evidence(analysis, extracted)
        if analysis_issues:
            raise TruthfulnessError(
                "Codex analysis failed local source-evidence validation:\n"
                + "\n".join(f"- {issue}" for issue in analysis_issues)
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

        metadata["stage"] = "antigravity-tailoring"
        hooks.progress(
            "antigravity_tailoring",
            "Antigravity is producing schema-constrained résumé content.",
        )
        _update_metadata(metadata, metadata_path, run_directory=run_directory)
        tailored_content = invoke_antigravity(
            master_content=extracted["content"],
            extracted_resume=extracted,
            job_description=job_description,
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
        metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": int(exc.exit_code),
        }
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
