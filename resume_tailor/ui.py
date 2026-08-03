from __future__ import annotations

import argparse
import hmac
import json
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

from docx import Document
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.datastructures import FormData, UploadFile

from . import __version__
from .analytics import (
    APPLICATION_STATUSES,
    INTERVIEW_TYPES,
    AnalyticsError,
    AnalyticsStore,
    default_analytics_database_path,
)
from .cli import _validate_label, run_pipeline
from .docx_extract import validate_template
from .linkedin_job import validate_linkedin_url
from .job_text import (
    MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS,
    validate_confirmed_job_description,
)
from .orchestration import (
    ApprovalRequest,
    ApprovalResponse,
    PipelineHooks,
)
from .analysis import (
    ANALYSIS_PROVIDERS,
    DEFAULT_ANALYSIS_PROVIDER,
    normalize_analysis_provider,
    workflow_stages_for_provider,
)
from .ollama_writer import DEFAULT_OLLAMA_MODEL
from .retry import (
    antigravity_retry_failure_kind,
    build_antigravity_reprocess_context,
    build_antigravity_retry_context,
    build_retry_context,
)
from .utilities import (
    ApifyConfigurationError,
    ApifyLinkedInRetrievalError,
    ApprovalError,
    AnalysisProviderError,
    AntigravityCannotApplyError,
    AntigravityLaunchSizeError,
    AntigravityResponseEnvelopeError,
    AntigravityTailoringContractError,
    AntigravityTailoringPreflightError,
    AntigravityTechnicalFailureError,
    CancellationError,
    CodexSchemaCompatibilityError,
    CodexUsageLimitError,
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
    ModelError,
    OllamaBudgetError,
    OllamaCannotApplyError,
    OllamaConnectionError,
    OllamaRevisionCannotApplyError,
    OllamaRevisionContractError,
    OllamaRevisionTechnicalFailureError,
    OllamaTailoringContractError,
    OllamaTechnicalFailureError,
    ResumeTailorError,
    SourceEvidenceError,
    TailoringPreflightError,
    atomic_write_text,
    parse_duration,
    utc_now_iso,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_RESUME_BYTES = 5 * 1024 * 1024
MAX_JOB_BYTES = 500_000
MAX_REQUEST_BYTES = MAX_RESUME_BYTES + MAX_JOB_BYTES + 256_000
COOKIE_NAME = "resume_tailor_session"

# Stage keys are stable; labels for the analysis step are provider-specific and
# are resolved per run via workflow_stages_for_provider().
WORKFLOW_STAGES: tuple[tuple[str, str], ...] = workflow_stages_for_provider(None)
_STAGE_INDEX = {name: index for index, (name, _) in enumerate(WORKFLOW_STAGES)}
_TERMINAL_STATUSES = {"COMPLETE", "FAILED", "CANCELLED"}
_DOWNLOAD_EXACT = {
    "job-source.json",
    "apify-linkedin-retrieval-diagnostic.json",
    "job-description.txt",
    "job-requirements.json",
    "extracted-master-resume.json",
    "codex-analysis.json",
    "analysis-resolved.json",
    "codex-analysis-resolved.json",
    "codex-analysis-normalization-warnings.json",
    "codex-analysis-transport.schema.json",
    "grok-analysis-prompt.sanitized.txt",
    "grok-analysis-transport.json",
    "grok-analysis-response.sanitized.json",
    "grok-analysis-schema.json",
    "grok-analysis-diagnostic.json",
    "gemma-analysis-prompt.sanitized.txt",
    "gemma-analysis-schema.json",
    "gemma-analysis-response.sanitized.json",
    "gemma-analysis-diagnostic.json",
    "gemma-analysis-coverage-prompt.sanitized.txt",
    "gemma-analysis-coverage-schema.json",
    "gemma-analysis-coverage-response.sanitized.json",
    "gemma-analysis-coverage-diagnostic.json",
    "gemma-analysis-edits-prompt.sanitized.txt",
    "gemma-analysis-edits-schema.json",
    "gemma-analysis-edits-response.sanitized.json",
    "gemma-analysis-edits-diagnostic.json",
    "antigravity-response.json",
    "antigravity-response-envelope.json",
    "antigravity-revision-response.json",
    "antigravity-revision-response-envelope.json",
    "ollama-response.json",
    "ollama-response-envelope.json",
    "ollama-tailoring-transport.schema.json",
    "ollama-budget-repair-response.json",
    "ollama-budget-repair-response-envelope.json",
    "ollama-budget-repair-transport.schema.json",
    "ollama-revision-response.json",
    "ollama-revision-response-envelope.json",
    "ollama-revision-transport.schema.json",
    "revision-request.json",
    "tailored-content.json",
    "tailored-content.initial.json",
    "tailored-content.revision-1.json",
    "content-diff.md",
    "content-diff.initial.md",
    "content-diff.revision-1.md",
    "preview.png",
    "preview.initial.png",
    "preview.revision-1.png",
    "final-qa.md",
    "final-qa.initial.json",
    "final-qa.initial.md",
    "final-qa.revision-1.json",
    "final-qa.revision-1.md",
    "final-qa-normalization-warnings.json",
    "final-qa.initial.normalization-warnings.json",
    "final-qa.revision-1.normalization-warnings.json",
    "run-metadata.json",
}


PipelineRunner = Callable[..., Path]


class ActiveRunError(InputError):
    pass


@dataclass(frozen=True)
class UISettings:
    host: str
    port: int
    output_directory: Path
    master_resume: Path
    launch_token: str
    timeout: tuple[int, str]
    analytics_database: Path


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    source_mode: str
    company: str | None
    role: str | None
    namespace: argparse.Namespace
    staging_directory: Path
    status: str = "QUEUED"
    stage: str = "validating_input"
    message: str = "Run queued."
    artifact_directory: Path | None = None
    error: str | None = None
    failure_kind: str | None = None
    retrieval_classification: str | None = None
    events: list[dict[str, str]] = field(default_factory=list)
    approval: ApprovalRequest | None = None
    approval_response: ApprovalResponse | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    revision: int = 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_master_resume() -> Path:
    return _project_root() / "template" / "master_resume.docx"


def default_output_directory() -> Path:
    return Path("~/Documents/Resumes/Tailored").expanduser()


def _is_direct_child(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().parent == parent.resolve()
    except OSError:
        return False


def _safe_remove_staging(path: Path, staging_root: Path) -> None:
    try:
        resolved = path.resolve()
        root = staging_root.resolve()
    except OSError:
        return
    if resolved.parent == root and resolved.name.startswith("run-") and resolved.is_dir():
        shutil.rmtree(resolved)


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 2_000_000:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_kind(name: str) -> str:
    suffix = Path(name).suffix.casefold()
    if suffix == ".docx":
        return "DOCX"
    if suffix == ".pdf":
        return "PDF"
    if suffix == ".png":
        return "Preview"
    if suffix == ".json":
        return "JSON"
    if suffix == ".md":
        return "Report"
    return "Text"


def _is_downloadable_name(name: str) -> bool:
    if (
        not name
        or Path(name).name != name
        or name in {".", ".."}
        or any(ord(character) < 32 for character in name)
    ):
        return False
    return name in _DOWNLOAD_EXACT or Path(name).suffix.casefold() in {
        ".docx",
        ".pdf",
    }


def _artifact_entries(run_directory: Path | None) -> list[dict[str, str]]:
    if run_directory is None or not run_directory.is_dir():
        return []
    entries: list[dict[str, str]] = []
    for path in sorted(run_directory.iterdir(), key=lambda item: item.name.casefold()):
        if (
            path.is_file()
            and not path.is_symlink()
            and _is_downloadable_name(path.name)
        ):
            entries.append({"name": path.name, "kind": _artifact_kind(path.name)})
    return entries


def _analysis_view(analysis: Mapping[str, Any]) -> dict[str, Any]:
    major_sections = (
        "Professional Summary",
        "Education & Certifications",
        "Technical Skills",
        "AI Engineering Projects",
        "Open Source Contribution",
        "Experience",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {
        "summary": [],
        "experience": [],
        "projects": [],
        "other": [],
    }
    touched: set[str] = set()
    for edit in analysis.get("recommended_edits", []):
        section = str(edit.get("resume_section", ""))
        normalized = section.casefold()
        if "summary" in normalized or "objective" in normalized:
            grouped["summary"].append(edit)
            touched.add("Professional Summary")
        elif "experience" in normalized or "employment" in normalized:
            grouped["experience"].append(edit)
            touched.add("Experience")
        elif "project" in normalized:
            grouped["projects"].append(edit)
            touched.add("AI Engineering Projects")
        else:
            grouped["other"].append(edit)
            for candidate in major_sections:
                if candidate.casefold().split()[0] in normalized:
                    touched.add(candidate)
                    break
    return {
        "analysis": analysis,
        "grouped_edits": grouped,
        "unchanged_sections": [item for item in major_sections if item not in touched],
    }


class RunManager:
    def __init__(
        self,
        *,
        settings: UISettings,
        pipeline_runner: PipelineRunner,
    ) -> None:
        self.settings = settings
        self.pipeline_runner = pipeline_runner
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._records: dict[str, RunRecord] = {}
        self._active_run_id: str | None = None
        self._staging_root = settings.output_directory / ".ui-staging"
        self._staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def create_staging_directory(self) -> Path:
        path = self._staging_root / f"run-{uuid.uuid4().hex}"
        path.mkdir(mode=0o700)
        return path

    def start(
        self,
        *,
        namespace: argparse.Namespace,
        staging_directory: Path,
        source_mode: str,
        company: str | None,
        role: str | None,
    ) -> RunRecord:
        with self._condition:
            if self._active_run_id is not None:
                active = self._records.get(self._active_run_id)
                if active is not None and active.status not in _TERMINAL_STATUSES:
                    raise ActiveRunError(
                        "A tailoring run is already active. Finish or cancel it "
                        "before starting another."
                    )
            record = RunRecord(
                run_id=secrets.token_urlsafe(12),
                created_at=utc_now_iso(),
                source_mode=source_mode,
                company=company,
                role=role,
                namespace=namespace,
                staging_directory=staging_directory,
            )
            record.events.append(
                {
                    "time": _clock_text(),
                    "stage": record.stage,
                    "message": "Run accepted and queued on this localhost server.",
                }
            )
            self._records[record.run_id] = record
            self._active_run_id = record.run_id
            thread = threading.Thread(
                target=self._execute,
                args=(record.run_id,),
                name=f"resume-tailor-{record.run_id}",
                daemon=False,
            )
            record.thread = thread
            thread.start()
            return record

    def _execute(self, run_id: str) -> None:
        with self._condition:
            record = self._records[run_id]
            record.status = "RUNNING"
            record.revision += 1
        hooks = PipelineHooks(
            progress_handler=lambda stage, message, payload: self._progress(
                run_id,
                stage,
                message,
                payload,
            ),
            approval_handler=lambda request: self._await_approval(run_id, request),
            approval_handler_presents=True,
            warning_handler=lambda message, payload: self._warning(
                run_id,
                message,
                payload,
            ),
            cancel_event=record.cancel_event,
        )
        try:
            run_directory = self.pipeline_runner(record.namespace, hooks=hooks)
        except (CancellationError, ApprovalError) as exc:
            with self._condition:
                record.status = "CANCELLED"
                record.error = _sanitized_technical_details(exc)
                record.message = "Run cancelled. Useful diagnostics were preserved."
                record.approval = None
                record.events.append(
                    {
                        "time": _clock_text(),
                        "stage": record.stage,
                        "message": record.message,
                    }
                )
                record.revision += 1
                self._condition.notify_all()
        except ResumeTailorError as exc:
            with self._condition:
                record.status = "FAILED"
                record.error = _sanitized_technical_details(exc)
                record.failure_kind = _failure_kind_for_error(exc, record.stage)
                if isinstance(
                    exc,
                    (ApifyConfigurationError, ApifyLinkedInRetrievalError),
                ):
                    record.retrieval_classification = exc.classification
                record.message = _safe_error_message(exc)
                record.approval = None
                record.events.append(
                    {
                        "time": _clock_text(),
                        "stage": record.stage,
                        "message": record.message,
                    }
                )
                record.revision += 1
                self._condition.notify_all()
        except Exception as exc:  # defensive boundary for a localhost UI thread
            with self._condition:
                record.status = "FAILED"
                record.failure_kind = "internal"
                record.error = (
                    _sanitized_technical_details(
                        RuntimeError(
                            f"Unexpected internal error: {type(exc).__name__}: {exc}"
                        )
                    )
                )
                record.message = (
                    "The local pipeline stopped unexpectedly. Existing artifacts "
                    "were preserved for review."
                )
                record.approval = None
                record.events.append(
                    {
                        "time": _clock_text(),
                        "stage": record.stage,
                        "message": record.message,
                    }
                )
                record.revision += 1
                self._condition.notify_all()
        else:
            with self._condition:
                record.artifact_directory = run_directory.resolve()
                record.status = "COMPLETE"
                record.stage = "complete"
                record.message = "Run complete. DOCX, PDF, and reports are ready."
                record.revision += 1
                self._condition.notify_all()
        finally:
            with self._condition:
                if self._active_run_id == run_id:
                    self._active_run_id = None
                self._condition.notify_all()
            _safe_remove_staging(record.staging_directory, self._staging_root)

    def _progress(
        self,
        run_id: str,
        stage: str,
        message: str,
        payload: Mapping[str, Any],
    ) -> None:
        if stage not in _STAGE_INDEX:
            return
        with self._condition:
            record = self._records[run_id]
            record.stage = stage
            record.message = message[:500]
            if stage != "complete" and record.status != "AWAITING_APPROVAL":
                record.status = "RUNNING"
            run_directory = payload.get("run_directory")
            if isinstance(run_directory, str):
                candidate = Path(run_directory)
                if _is_direct_child(candidate, self.settings.output_directory):
                    record.artifact_directory = candidate.resolve()
            company = payload.get("company")
            role = payload.get("role")
            if isinstance(company, str):
                record.company = company[:300]
            if isinstance(role, str):
                record.role = role[:300]
            event = {
                "time": _clock_text(),
                "stage": stage,
                "message": message[:500],
            }
            if not record.events or record.events[-1]["message"] != event["message"]:
                record.events.append(event)
                record.events[:] = record.events[-100:]
            record.revision += 1
            self._condition.notify_all()

    def _warning(
        self,
        run_id: str,
        message: str,
        _payload: Mapping[str, Any],
    ) -> None:
        with self._condition:
            record = self._records[run_id]
            safe_message = message[:500]
            record.message = safe_message
            record.events.append(
                {
                    "time": _clock_text(),
                    "stage": record.stage,
                    "message": safe_message,
                }
            )
            record.events[:] = record.events[-100:]
            record.revision += 1
            self._condition.notify_all()

    def _await_approval(
        self,
        run_id: str,
        request: ApprovalRequest,
    ) -> ApprovalResponse:
        with self._condition:
            record = self._records[run_id]
            if request.kind == "linkedin_posting":
                company = request.payload.get("company")
                role = request.payload.get("job_title")
                if isinstance(company, str):
                    record.company = company[:300]
                if isinstance(role, str):
                    record.role = role[:300]
            record.approval = request
            record.approval_response = None
            record.status = "AWAITING_APPROVAL"
            record.message = f"{request.title} needs your explicit approval."
            record.events.append(
                {
                    "time": _clock_text(),
                    "stage": record.stage,
                    "message": record.message,
                }
            )
            record.revision += 1
            if request.on_presented is not None:
                request.on_presented()
            self._condition.notify_all()
            while record.approval_response is None:
                if record.cancel_event.is_set():
                    record.approval = None
                    raise CancellationError(
                        "The run was cancelled while waiting for approval."
                    )
                self._condition.wait(timeout=0.25)
            response = record.approval_response
            record.approval = None
            record.approval_response = None
            if response.action in {
                "approve",
                "use_pasted",
                "revise_once",
                "stop",
                "reject",
            }:
                record.status = "RUNNING"
                record.message = f"{request.title} approved."
                record.events.append(
                    {
                        "time": _clock_text(),
                        "stage": record.stage,
                        "message": record.message,
                    }
                )
                record.revision += 1
                self._condition.notify_all()
            return response

    def respond_to_approval(
        self,
        run_id: str,
        *,
        action: str,
        job_description: str = "",
    ) -> None:
        with self._condition:
            record = self._require_record(run_id)
            if record.status != "AWAITING_APPROVAL" or record.approval is None:
                raise InputError("This run is not currently waiting for approval.")
            kind = record.approval.kind
            if kind == "linkedin_posting":
                allowed = {"approve", "cancel", "use_pasted"}
            elif kind == "qa_revision":
                allowed = {"revise_once", "stop", "cancel"}
            elif kind == "revised_content":
                allowed = {"approve", "reject", "cancel"}
            else:
                allowed = {"approve", "cancel"}
            if action not in allowed:
                raise InputError("That approval action is not valid for this gate.")
            data: dict[str, Any] = {}
            if action == "use_pasted":
                description = job_description.strip()
                if not description:
                    raise InputError("Paste a complete job description first.")
                validate_confirmed_job_description(description)
                data["job_description"] = description
            if action == "cancel":
                record.cancel_event.set()
            record.approval_response = ApprovalResponse(action, data)
            record.revision += 1
            self._condition.notify_all()

    def cancel(self, run_id: str) -> None:
        with self._condition:
            record = self._require_record(run_id)
            if record.status in _TERMINAL_STATUSES:
                return
            record.cancel_event.set()
            if record.approval is not None and record.approval_response is None:
                record.approval_response = ApprovalResponse("cancel")
            record.message = "Cancellation requested; stopping active work safely."
            record.events.append(
                {
                    "time": _clock_text(),
                    "stage": record.stage,
                    "message": record.message,
                }
            )
            record.revision += 1
            self._condition.notify_all()

    def snapshot(self, run_id: str) -> dict[str, Any]:
        if run_id.startswith("history-"):
            return self._historical_snapshot(run_id)
        with self._condition:
            record = self._require_record(run_id)
            return self._snapshot_locked(record)

    def _snapshot_locked(self, record: RunRecord) -> dict[str, Any]:
        approval: dict[str, Any] | None = None
        if record.approval is not None:
            payload = dict(record.approval.payload)
            approval = {
                "kind": record.approval.kind,
                "title": record.approval.title,
                "payload": payload,
            }
            if record.approval.kind == "codex_analysis":
                approval.update(_analysis_view(payload))
        metadata = (
            _safe_json(record.artifact_directory / "run-metadata.json")
            if record.artifact_directory is not None
            else {}
        )
        company = record.company or metadata.get("company")
        role = record.role or metadata.get("role")
        artifacts = _artifact_entries(record.artifact_directory)
        final_generation = (
            metadata.get("revision_cycle", {}).get("final_generation")
            if isinstance(metadata.get("revision_cycle"), dict)
            else None
        )
        generation_marker = (
            ".revision-1." if final_generation == "revision-1" else ".initial."
        )
        pdf = next(
            (
                item["name"]
                for item in artifacts
                if item["kind"] == "PDF" and generation_marker in item["name"]
            ),
            next(
                (item["name"] for item in artifacts if item["kind"] == "PDF"),
                None,
            ),
        )
        failure_kind = record.failure_kind or _failure_kind_from_metadata(metadata)
        retrieval_classification = (
            record.retrieval_classification
            or _retrieval_classification_from_metadata(metadata)
        )
        retry_eligible, retry_reason = self._retry_state(
            record.artifact_directory,
            failure_kind=failure_kind,
        )
        (
            antigravity_retry_eligible,
            antigravity_retry_reason,
        ) = self._antigravity_retry_state(
            record.artifact_directory,
            failure_kind=failure_kind,
        )
        (
            antigravity_reprocess_eligible,
            antigravity_reprocess_reason,
        ) = self._antigravity_reprocess_state(
            record.artifact_directory,
            failure_kind=failure_kind,
        )
        analysis_meta = metadata.get("analysis")
        analysis_provider = (
            analysis_meta.get("provider")
            if isinstance(analysis_meta, dict)
            else None
        )
        return {
            "run_id": record.run_id,
            "created_at": record.created_at,
            "source_mode": record.source_mode,
            "company": company,
            "role": role,
            "status": record.status,
            "stage": record.stage,
            "stage_index": _STAGE_INDEX.get(record.stage, 0),
            "analysis_provider": analysis_provider,
            "workflow_stages": workflow_stages_for_provider(
                analysis_provider if isinstance(analysis_provider, str) else None
            ),
            "message": record.message,
            "artifact_directory": (
                str(record.artifact_directory)
                if record.artifact_directory is not None
                else None
            ),
            "error": record.error,
            "failure_kind": failure_kind,
            "retrieval_classification": retrieval_classification,
            "retry_eligible": retry_eligible,
            "retry_reason": retry_reason,
            "antigravity_retry_eligible": antigravity_retry_eligible,
            "antigravity_retry_reason": antigravity_retry_reason,
            "antigravity_reprocess_eligible": antigravity_reprocess_eligible,
            "antigravity_reprocess_reason": antigravity_reprocess_reason,
            "events": [dict(item) for item in record.events],
            "approval": approval,
            "revision": record.revision,
            "artifacts": artifacts,
            "pdf_name": pdf,
            "final_generation": final_generation,
            "metadata": metadata,
        }

    def _history_directory(self, run_id: str) -> Path:
        directory_name = run_id.removeprefix("history-")
        if (
            not run_id.startswith("history-")
            or Path(directory_name).name != directory_name
            or directory_name.startswith(".")
        ):
            raise InputError("Invalid historical run identifier.")
        run_directory = self.settings.output_directory / directory_name
        if (
            run_directory.is_symlink()
            or not run_directory.is_dir()
            or not _is_direct_child(run_directory, self.settings.output_directory)
        ):
            raise InputError("The requested historical run is not available.")
        metadata = _safe_json(run_directory / "run-metadata.json")
        if metadata.get("application") != "resume-tailor":
            raise InputError("The requested directory is not a validated run.")
        return run_directory.resolve()

    def _retry_state(
        self,
        run_directory: Path | None,
        *,
        failure_kind: str | None,
    ) -> tuple[bool, str]:
        if failure_kind != "source_evidence" or run_directory is None:
            return False, ""
        try:
            context = build_retry_context(
                run_directory,
                current_resume=self.settings.master_resume,
            )
        except InputError as exc:
            return False, str(exc)
        legacy = " Legacy inputs were cross-checked." if context.legacy_verified else ""
        return True, f"Stored input hashes match.{legacy}"

    def _antigravity_retry_state(
        self,
        run_directory: Path | None,
        *,
        failure_kind: str | None,
    ) -> tuple[bool, str]:
        if failure_kind not in {
            "antigravity_launch_size",
            "antigravity_response_envelope",
            "antigravity_tailoring_contract",
            "antigravity_cannot_apply",
            "antigravity_technical_failure",
        } or run_directory is None:
            return False, ""
        try:
            build_antigravity_retry_context(
                run_directory,
                current_resume=self.settings.master_resume,
            )
        except InputError as exc:
            if "predates the authenticated Codex approval record" in str(exc):
                return (
                    False,
                    "This run predates the authenticated Codex approval record; "
                    "a new run is required.",
                )
            return (
                False,
                "At least one approved input or its authenticated hash no longer "
                "matches; a new run is required.",
            )
        return (
            True,
            "The source résumé, confirmed job input, requirement catalog, "
            "transport schema, resolved analysis, and approval record all match "
            "their authenticated hashes.",
        )

    def _antigravity_reprocess_state(
        self,
        run_directory: Path | None,
        *,
        failure_kind: str | None,
    ) -> tuple[bool, str]:
        if failure_kind != "antigravity_response_envelope" or run_directory is None:
            return False, ""
        try:
            build_antigravity_reprocess_context(
                run_directory,
                current_resume=self.settings.master_resume,
            )
        except InputError:
            return (
                False,
                "The preserved response is not one complete, schema-valid "
                "tailoring result, so it cannot be reprocessed offline.",
            )
        return (
            True,
            "The preserved response, expected schema, approved edits, source "
            "résumé, requirement catalog, resolved analysis, approval record, "
            "and recovery ancestry all match their authenticated hashes.",
        )

    def _historical_snapshot(self, run_id: str) -> dict[str, Any]:
        run_directory = self._history_directory(run_id)
        metadata = _safe_json(run_directory / "run-metadata.json")
        status = str(metadata.get("status", "UNKNOWN"))
        stage = _ui_stage_from_metadata(str(metadata.get("stage", "")))
        failure_kind = _failure_kind_from_metadata(metadata)
        retrieval_classification = _retrieval_classification_from_metadata(metadata)
        artifacts = _artifact_entries(run_directory)
        pdf = next((item["name"] for item in artifacts if item["kind"] == "PDF"), None)
        retry_eligible, retry_reason = self._retry_state(
            run_directory,
            failure_kind=failure_kind,
        )
        (
            antigravity_retry_eligible,
            antigravity_retry_reason,
        ) = self._antigravity_retry_state(
            run_directory,
            failure_kind=failure_kind,
        )
        (
            antigravity_reprocess_eligible,
            antigravity_reprocess_reason,
        ) = self._antigravity_reprocess_state(
            run_directory,
            failure_kind=failure_kind,
        )
        error_payload = metadata.get("error")
        error_message = (
            error_payload.get("message")
            if isinstance(error_payload, dict)
            and isinstance(error_payload.get("message"), str)
            else ""
        )
        if failure_kind == "source_evidence":
            technical = (
                "Local evidence-contract validation rejected model requirement or "
                "source references. "
                "Exact résumé and job text are omitted from this view."
            )
            message = _SOURCE_EVIDENCE_UI_MESSAGE
        elif failure_kind == "antigravity_launch_size":
            technical = (
                "The operating system rejected the legacy Antigravity process "
                "argument vector before provider startup. Prompt content is omitted."
            )
            message = _ANTIGRAVITY_LAUNCH_SIZE_UI_MESSAGE
        elif failure_kind == "apify_linkedin_retrieval":
            technical = (
                "Apify LinkedIn retrieval was rejected with the bounded local "
                f"classification {retrieval_classification or 'provider_failure'}. "
                "The API token, authorization header, raw Actor result, and posting "
                "content are omitted."
            )
            message = _APIFY_LINKEDIN_UI_MESSAGES.get(
                retrieval_classification or "provider_failure",
                _APIFY_LINKEDIN_UI_MESSAGES["provider_failure"],
            )
        elif failure_kind == "antigravity_response_envelope":
            technical = (
                "Antigravity returned a documented print-mode JSON wrapper, but "
                "no single supported structured-output candidate passed strict "
                "schema validation. Provider prose is omitted."
            )
            message = _ANTIGRAVITY_RESPONSE_ENVELOPE_UI_MESSAGE
        elif failure_kind == "antigravity_tailoring_contract":
            technical = (
                "Antigravity returned a post-approval status that did not apply "
                "the authenticated edit plan. Provider prose and prompt content "
                "are omitted."
            )
            message = _ANTIGRAVITY_TAILORING_CONTRACT_UI_MESSAGE
        elif failure_kind == "antigravity_cannot_apply":
            technical = (
                "Antigravity returned a bounded cannot_apply result for one "
                "approved edit. Provider prose is omitted."
            )
            message = _ANTIGRAVITY_CANNOT_APPLY_UI_MESSAGE
        elif failure_kind == "antigravity_technical_failure":
            technical = (
                "Antigravity returned a structured technical_failure result. "
                "Provider prose is omitted."
            )
            message = _ANTIGRAVITY_TECHNICAL_FAILURE_UI_MESSAGE
        elif failure_kind == "ollama_preflight":
            technical = (
                "Local authentication of the approved inputs failed before any "
                "Gemma 4 12B or Ollama request. Résumé and job content are omitted."
            )
            message = _OLLAMA_PREFLIGHT_UI_MESSAGE
        elif failure_kind == "ollama_connection":
            technical = (
                "The fixed localhost Ollama endpoint or configured model did not "
                "complete the bounded request. Prompt and provider content are omitted."
            )
            message = _OLLAMA_CONNECTION_UI_MESSAGE
        elif failure_kind == "ollama_contract":
            technical = (
                "The Gemma 4 12B response failed strict JSON, canonical schema, edit-ID, "
                "or revision validation. Provider content is omitted."
            )
            message = _OLLAMA_CONTRACT_UI_MESSAGE
        elif failure_kind == "ollama_cannot_apply":
            technical = (
                "Gemma 4 12B returned a bounded cannot_apply result for one authenticated "
                "edit or QA issue. Provider prose is omitted."
            )
            message = _OLLAMA_CANNOT_APPLY_UI_MESSAGE
        elif failure_kind == "ollama_technical_failure":
            technical = (
                "Gemma 4 12B returned a structured technical_failure result. Provider "
                "prose is omitted."
            )
            message = _OLLAMA_TECHNICAL_FAILURE_UI_MESSAGE
        else:
            technical = (
                _sanitized_technical_details(RuntimeError(error_message))
                if error_message
                else None
            )
            message = (
                "Run complete. DOCX, PDF, and reports are ready."
                if status == "COMPLETE"
                else "The preserved run stopped safely."
            )
        analysis_meta = metadata.get("analysis")
        analysis_provider = (
            analysis_meta.get("provider")
            if isinstance(analysis_meta, dict)
            else None
        )
        return {
            "run_id": run_id,
            "created_at": metadata.get("created_at", ""),
            "source_mode": metadata.get("job_source", ""),
            "company": metadata.get("company"),
            "role": metadata.get("role"),
            "status": status,
            "stage": stage,
            "stage_index": _STAGE_INDEX.get(stage, 0),
            "analysis_provider": analysis_provider,
            "workflow_stages": workflow_stages_for_provider(
                analysis_provider if isinstance(analysis_provider, str) else None
            ),
            "message": message,
            "artifact_directory": str(run_directory),
            "error": technical,
            "failure_kind": failure_kind,
            "retrieval_classification": retrieval_classification,
            "retry_eligible": retry_eligible,
            "retry_reason": retry_reason,
            "antigravity_retry_eligible": antigravity_retry_eligible,
            "antigravity_retry_reason": antigravity_retry_reason,
            "antigravity_reprocess_eligible": antigravity_reprocess_eligible,
            "antigravity_reprocess_reason": antigravity_reprocess_reason,
            "events": [
                {
                    "time": "—",
                    "stage": stage,
                    "message": message,
                }
            ],
            "approval": None,
            "revision": 0,
            "artifacts": artifacts,
            "pdf_name": pdf,
            "metadata": metadata,
            "is_history": True,
        }

    def retry_codex_analysis(self, run_id: str) -> RunRecord:
        if run_id.startswith("history-"):
            source_directory = self._history_directory(run_id)
        else:
            with self._condition:
                source = self._require_record(run_id)
                source_directory = source.artifact_directory
            if source_directory is None:
                raise InputError("The failed run has no preserved input directory.")
        context = build_retry_context(
            source_directory,
            current_resume=self.settings.master_resume,
        )
        staging = self.create_staging_directory()
        namespace = argparse.Namespace(
            resume=self.settings.master_resume,
            clipboard=False,
            job_file=None,
            job_url=None,
            company=context.company,
            role=context.role,
            output_dir=self.settings.output_directory,
            yes=False,
            keep_workdir=False,
            timeout=self.settings.timeout,
            retry_context=context,
            writer_provider="ollama",
            ollama_model=DEFAULT_OLLAMA_MODEL,
            analysis_provider=DEFAULT_ANALYSIS_PROVIDER,
        )
        # Preserve an explicit prior analysis provider when authenticated metadata
        # records one; never invent a switch from Codex to Grok on retry.
        if source_directory is not None:
            prior_metadata = _safe_json(source_directory / "run-metadata.json")
            if isinstance(prior_metadata, dict):
                prior_analysis = prior_metadata.get("analysis")
                if isinstance(prior_analysis, dict):
                    prior_provider = prior_analysis.get("provider")
                    if isinstance(prior_provider, str):
                        try:
                            namespace.analysis_provider = (
                                normalize_analysis_provider(prior_provider)
                            )
                        except InputError:
                            pass
        try:
            return self.start(
                namespace=namespace,
                staging_directory=staging,
                source_mode="retry",
                company=context.company,
                role=context.role,
            )
        except Exception:
            _safe_remove_staging(staging, self._staging_root)
            raise

    def retry_antigravity_tailoring(self, run_id: str) -> RunRecord:
        if run_id.startswith("history-"):
            source_directory = self._history_directory(run_id)
        else:
            with self._condition:
                source = self._require_record(run_id)
                source_directory = source.artifact_directory
            if source_directory is None:
                raise InputError("The failed run has no preserved input directory.")
        context = build_antigravity_retry_context(
            source_directory,
            current_resume=self.settings.master_resume,
        )
        staging = self.create_staging_directory()
        namespace = argparse.Namespace(
            resume=self.settings.master_resume,
            clipboard=False,
            job_file=None,
            job_url=None,
            company=context.company,
            role=context.role,
            output_dir=self.settings.output_directory,
            yes=False,
            keep_workdir=False,
            timeout=self.settings.timeout,
            antigravity_retry_context=context,
            writer_provider="antigravity",
            ollama_model=DEFAULT_OLLAMA_MODEL,
            analysis_provider=DEFAULT_ANALYSIS_PROVIDER,
        )
        try:
            return self.start(
                namespace=namespace,
                staging_directory=staging,
                source_mode="antigravity-retry",
                company=context.company,
                role=context.role,
            )
        except Exception:
            _safe_remove_staging(staging, self._staging_root)
            raise

    def reprocess_antigravity_response(self, run_id: str) -> RunRecord:
        if run_id.startswith("history-"):
            source_directory = self._history_directory(run_id)
        else:
            with self._condition:
                source = self._require_record(run_id)
                source_directory = source.artifact_directory
            if source_directory is None:
                raise InputError("The failed run has no preserved response directory.")
        context = build_antigravity_reprocess_context(
            source_directory,
            current_resume=self.settings.master_resume,
        )
        staging = self.create_staging_directory()
        namespace = argparse.Namespace(
            resume=self.settings.master_resume,
            clipboard=False,
            job_file=None,
            job_url=None,
            company=context.retry_context.company,
            role=context.retry_context.role,
            output_dir=self.settings.output_directory,
            yes=False,
            keep_workdir=False,
            timeout=self.settings.timeout,
            antigravity_reprocess_context=context,
            writer_provider="antigravity",
            ollama_model=DEFAULT_OLLAMA_MODEL,
            analysis_provider=DEFAULT_ANALYSIS_PROVIDER,
        )
        try:
            return self.start(
                namespace=namespace,
                staging_directory=staging,
                source_mode="antigravity-response-reprocess",
                company=context.retry_context.company,
                role=context.retry_context.role,
            )
        except Exception:
            _safe_remove_staging(staging, self._staging_root)
            raise

    def history(self) -> list[dict[str, Any]]:
        with self._condition:
            in_memory = {
                str(record.artifact_directory.resolve()): self._snapshot_locked(record)
                for record in self._records.values()
                if record.artifact_directory is not None
            }
            pending = [
                self._snapshot_locked(record)
                for record in self._records.values()
                if record.artifact_directory is None
            ]
        disk: list[dict[str, Any]] = []
        try:
            children = list(self.settings.output_directory.iterdir())
        except OSError:
            children = []
        for child in children:
            if (
                child.name.startswith(".")
                or child.is_symlink()
                or not child.is_dir()
                or not _is_direct_child(child, self.settings.output_directory)
            ):
                continue
            key = str(child.resolve())
            if key in in_memory:
                continue
            metadata = _safe_json(child / "run-metadata.json")
            if metadata.get("application") != "resume-tailor":
                continue
            artifacts = _artifact_entries(child)
            disk.append(
                {
                    "run_id": f"history-{child.name}",
                    "created_at": metadata.get("created_at", ""),
                    "company": metadata.get("company"),
                    "role": metadata.get("role"),
                    "status": metadata.get("status", "UNKNOWN"),
                    "stage": metadata.get("stage", ""),
                    "artifact_directory": str(child.resolve()),
                    "artifacts": artifacts,
                    "is_history": True,
                }
            )
        values = [*pending, *in_memory.values(), *disk]
        return sorted(values, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def resolve_artifact(self, run_id: str, artifact_name: str) -> Path:
        if not _is_downloadable_name(artifact_name):
            raise InputError("That artifact name is not allowed.")
        with self._condition:
            if run_id.startswith("history-"):
                directory_name = run_id.removeprefix("history-")
                if (
                    Path(directory_name).name != directory_name
                    or directory_name.startswith(".")
                ):
                    raise InputError("Invalid historical run identifier.")
                run_directory = self.settings.output_directory / directory_name
                metadata = _safe_json(run_directory / "run-metadata.json")
                if metadata.get("application") != "resume-tailor":
                    raise InputError(
                        "The requested directory is not a validated resume-tailor run."
                    )
            else:
                record = self._require_record(run_id)
                run_directory = record.artifact_directory
        if run_directory is None or not _is_direct_child(
            run_directory, self.settings.output_directory
        ):
            raise InputError("The requested run directory is not available.")
        candidate = run_directory / artifact_name
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve().parent != run_directory.resolve()
        ):
            raise InputError("The requested artifact is not available.")
        return candidate.resolve()

    def _require_record(self, run_id: str) -> RunRecord:
        record = self._records.get(run_id)
        if record is None:
            raise InputError("Unknown run identifier.")
        return record

    def wait_for(
        self,
        run_id: str,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 10,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                snapshot = self._snapshot_locked(self._require_record(run_id))
                if predicate(snapshot):
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for run {run_id}.")
                self._condition.wait(timeout=min(0.25, remaining))

    def shutdown(self) -> None:
        with self._condition:
            records = list(self._records.values())
            for record in records:
                if record.status not in _TERMINAL_STATUSES:
                    record.cancel_event.set()
                    if record.approval is not None:
                        record.approval_response = ApprovalResponse("cancel")
            self._condition.notify_all()
        for record in records:
            if record.thread is not None and record.thread.is_alive():
                record.thread.join(timeout=5)


def _clock_text() -> str:
    return datetime.now().astimezone().strftime("%H:%M:%S")


_SOURCE_EVIDENCE_UI_MESSAGE = (
    "Codex returned job-requirement or résumé-source references that violated "
    "the authoritative local evidence contract. This is a model evidence-contract "
    "failure; changing or refetching the confirmed job input will not correct it."
)
_ANTIGRAVITY_LAUNCH_SIZE_UI_MESSAGE = (
    "Antigravity could not start because the request exceeded the operating "
    "system’s command-line size."
)
_ANTIGRAVITY_RESPONSE_ENVELOPE_UI_MESSAGE = (
    "Antigravity returned JSON in an unsupported response format."
)
_APIFY_LINKEDIN_UI_MESSAGES = {
    "missing_token": (
        "Apify is not configured. Set APIFY_API_TOKEN to the complete token, "
        "including its apify_api_ prefix."
    ),
    "missing_actor_id": (
        "Apify is not configured. Set APIFY_ACTOR_ID to the Actor ID or "
        "username/actor-name used for LinkedIn job retrieval."
    ),
    "invalid_token": (
        "APIFY_API_TOKEN is malformed. Preserve the complete token exactly as issued."
    ),
    "invalid_actor_id": (
        "APIFY_ACTOR_ID is malformed. Use an Actor ID or username/actor-name."
    ),
    "authentication_failure": (
        "Apify rejected authentication. Verify the locally configured API token."
    ),
    "actor_not_found": (
        "Apify could not find the configured Actor. Verify APIFY_ACTOR_ID and access."
    ),
    "actor_timeout": "The Apify Actor exceeded the bounded retrieval timeout.",
    "actor_failure": "The Apify Actor run stopped before retrieval succeeded.",
    "empty_dataset": "The Apify Actor completed but returned no dataset items.",
    "no_matching_result": (
        "No unique Apify result matched the requested LinkedIn URL or job ID."
    ),
    "insufficient_content": (
        "The Apify result lacked a meaningful title or complete job description."
    ),
    "network_error": (
        "Resume Tailor could not complete the bounded Apify HTTPS request."
    ),
    "rate_limited": (
        "Apify rate-limited this retrieval. Wait for the provider limit to reset."
    ),
    "provider_failure": "Apify LinkedIn retrieval stopped with a provider failure.",
    "malformed_output": (
        "The Apify result failed the local canonical job-posting contract."
    ),
}
_ANTIGRAVITY_TAILORING_CONTRACT_UI_MESSAGE = (
    "Antigravity did not apply the approved tailoring plan and returned a "
    "non-actionable request for another task. All authenticated inputs were "
    "already present; no factual information is requested."
)
_ANTIGRAVITY_CANNOT_APPLY_UI_MESSAGE = (
    "Antigravity could not safely apply one approved edit. The authenticated "
    "inputs and approved plan were preserved for a manual step-6 retry."
)
_ANTIGRAVITY_TECHNICAL_FAILURE_UI_MESSAGE = (
    "Antigravity reported a bounded technical tailoring failure. The authenticated "
    "inputs and approved plan were preserved."
)
_OLLAMA_CONNECTION_UI_MESSAGE = (
    "Resume Tailor could not reach the local Ollama service or load the configured "
    "Gemma 4 12B model at 127.0.0.1:11434. No remote fallback was attempted."
)
_OLLAMA_CONTRACT_UI_MESSAGE = (
    "Gemma 4 12B returned content that did not satisfy the local structured-output and "
    "evidence contract. The response and sanitized validation envelope were preserved."
)
_OLLAMA_CANNOT_APPLY_UI_MESSAGE = (
    "Gemma 4 12B could not safely apply one authenticated approved edit. No unsupported "
    "claim was substituted."
)
_OLLAMA_TECHNICAL_FAILURE_UI_MESSAGE = (
    "Gemma 4 12B reported a bounded local writing failure. Authenticated inputs and "
    "diagnostics were preserved."
)
_OLLAMA_PREFLIGHT_UI_MESSAGE = (
    "The authenticated tailoring inputs failed local completeness preflight. "
    "No Gemma 4 12B or Ollama request was launched."
)
_OLLAMA_BUDGET_UI_MESSAGE = (
    "The approved tailoring prompt does not fit the configured local model "
    "context window with room for a complete response. No Ollama request was "
    "launched and no approved content was trimmed."
)


def _failure_kind_for_error(
    error: ResumeTailorError,
    stage: str,
) -> str:
    if isinstance(error, SourceEvidenceError):
        return "source_evidence"
    if isinstance(error, (ApifyConfigurationError, ApifyLinkedInRetrievalError)):
        return "apify_linkedin_retrieval"
    if isinstance(error, AntigravityLaunchSizeError) or (
        stage == "antigravity_tailoring"
        and "Argument list too long" in str(error)
    ):
        return "antigravity_launch_size"
    if isinstance(error, AntigravityResponseEnvelopeError):
        return "antigravity_response_envelope"
    if isinstance(error, AntigravityTailoringContractError):
        return "antigravity_tailoring_contract"
    if isinstance(error, AntigravityCannotApplyError):
        return "antigravity_cannot_apply"
    if isinstance(error, AntigravityTechnicalFailureError):
        return "antigravity_technical_failure"
    if isinstance(error, AntigravityTailoringPreflightError):
        return "antigravity_tailoring_preflight"
    if isinstance(error, TailoringPreflightError):
        return "ollama_preflight"
    # Must precede OllamaConnectionError: budget refusals subclass it.
    if isinstance(error, OllamaBudgetError):
        return "ollama_preflight"
    if isinstance(error, OllamaConnectionError):
        return "ollama_connection"
    if isinstance(error, (OllamaTailoringContractError, OllamaRevisionContractError)):
        return "ollama_contract"
    if isinstance(error, (OllamaCannotApplyError, OllamaRevisionCannotApplyError)):
        return "ollama_cannot_apply"
    if isinstance(error, (OllamaTechnicalFailureError, OllamaRevisionTechnicalFailureError)):
        return "ollama_technical_failure"
    if isinstance(error, CodexSchemaCompatibilityError):
        return "schema"
    if isinstance(error, CodexUsageLimitError):
        return "codex_usage_limit"
    if isinstance(error, GrokExecutableError):
        return "grok_executable"
    if isinstance(error, GrokPromptTooLargeError):
        return "grok_prompt_too_large"
    if isinstance(error, GrokUsageLimitError):
        return "grok_usage_limit"
    if isinstance(error, GrokAuthenticationError):
        return "grok_authentication"
    if isinstance(error, GrokTimeoutError):
        return "grok_timeout"
    if isinstance(error, GrokTransportEnvelopeError):
        return "grok_transport"
    if isinstance(error, GrokInnerAnalysisError):
        return "grok_inner_analysis"
    if isinstance(error, GemmaOllamaUnavailableError):
        return "gemma_ollama_unavailable"
    if isinstance(error, GemmaModelUnavailableError):
        return "gemma_model_unavailable"
    if isinstance(error, GemmaAnalysisTimeoutError):
        return "gemma_timeout"
    if isinstance(error, GemmaOllamaInternalError):
        return "gemma_ollama_internal"
    if isinstance(error, GemmaOutputLimitError):
        return "gemma_output_limit"
    if isinstance(error, GemmaConnectionError):
        return "gemma_connection"
    if isinstance(error, GemmaResponseTooLargeError):
        return "gemma_response_too_large"
    if isinstance(error, GemmaTransportEnvelopeError):
        return "gemma_transport"
    if isinstance(error, GemmaInnerAnalysisError):
        return "gemma_inner_analysis"
    if isinstance(error, GemmaStructuredOutputError):
        return "gemma_structured_output"
    if isinstance(
        error,
        (GrokProcessError, GrokAnalysisError, GemmaAnalysisError, AnalysisProviderError),
    ):
        return "analysis_provider"
    if stage in {"fetching_job", "confirming_posting"}:
        return "retrieval"
    if isinstance(error, ModelError):
        return "model"
    return "pipeline"


def _failure_kind_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    if metadata.get("failure_class") == "source-evidence-analysis":
        return "source_evidence"
    failure_class = str(metadata.get("failure_class", ""))
    if failure_class == "codex-usage-limit":
        return "codex_usage_limit"
    if failure_class == "grok-executable-unavailable":
        return "grok_executable"
    if failure_class == "grok-prompt-too-large":
        return "grok_prompt_too_large"
    if failure_class == "grok-usage-limit":
        return "grok_usage_limit"
    if failure_class == "grok-authentication-failure":
        return "grok_authentication"
    if failure_class == "grok-timeout":
        return "grok_timeout"
    if failure_class == "grok-transport-envelope":
        return "grok_transport"
    if failure_class == "grok-inner-analysis":
        return "grok_inner_analysis"
    if failure_class == "gemma-ollama-unavailable":
        return "gemma_ollama_unavailable"
    if failure_class == "gemma-model-unavailable":
        return "gemma_model_unavailable"
    if failure_class == "gemma-analysis-timeout":
        return "gemma_timeout"
    if failure_class == "gemma-ollama-internal-error":
        return "gemma_ollama_internal"
    if failure_class == "gemma-output-limit":
        return "gemma_output_limit"
    if failure_class == "gemma-connection-failure":
        return "gemma_connection"
    if failure_class == "gemma-response-too-large":
        return "gemma_response_too_large"
    if failure_class == "gemma-transport-envelope":
        return "gemma_transport"
    if failure_class == "gemma-inner-analysis":
        return "gemma_inner_analysis"
    if failure_class == "gemma-structured-output":
        return "gemma_structured_output"
    if failure_class in {
        "grok-nonzero-exit",
        "grok-analysis-failure",
        "gemma-analysis-failure",
        "analysis-provider-failure",
    }:
        return "analysis_provider"
    error = metadata.get("error")
    error_type = error.get("type") if isinstance(error, Mapping) else None
    error_message = error.get("message") if isinstance(error, Mapping) else None
    stage = str(metadata.get("stage", ""))
    if (
        metadata.get("failure_class")
        in {"apify-configuration", "apify-linkedin-retrieval"}
        or error_type
        in {"ApifyConfigurationError", "ApifyLinkedInRetrievalError"}
    ):
        return "apify_linkedin_retrieval"
    antigravity_failure = antigravity_retry_failure_kind(dict(metadata))
    if antigravity_failure == "launch_size":
        return "antigravity_launch_size"
    if antigravity_failure == "response_envelope":
        return "antigravity_response_envelope"
    if antigravity_failure in {
        "tailoring_contract",
        "legacy_needs_information",
    }:
        return "antigravity_tailoring_contract"
    if antigravity_failure == "cannot_apply":
        return "antigravity_cannot_apply"
    if antigravity_failure == "technical_failure":
        return "antigravity_technical_failure"
    ollama_failure = str(metadata.get("failure_class", ""))
    if ollama_failure == "ollama-connection":
        return "ollama_connection"
    if ollama_failure in {"ollama-tailoring-preflight", "ollama-budget-preflight"}:
        return "ollama_preflight"
    if ollama_failure in {
        "ollama-tailoring-contract",
        "ollama-revision-contract",
        # Phase 1 sanitized sub-classifications. These stay on the existing
        # contract guidance branch; they narrow diagnosis in run metadata
        # without changing recovery eligibility.
        "ollama-malformed-json",
        "ollama-response-envelope",
        "ollama-transport-schema",
        "ollama-canonical-schema",
        "ollama-output-truncation",
        "ollama-downstream-evidence",
    }:
        return "ollama_contract"
    if ollama_failure in {"ollama-cannot-apply", "ollama-revision-cannot-apply"}:
        return "ollama_cannot_apply"
    if ollama_failure in {"ollama-technical-failure", "ollama-revision-technical-failure"}:
        return "ollama_technical_failure"
    if (
        stage == "codex-analysis"
        and error_type in {"SourceEvidenceError", "TruthfulnessError"}
        and isinstance(error_message, str)
        and (
            error_message.startswith(
                "Codex analysis failed local source-evidence validation:"
            )
            or error_message.startswith(
                "Grok analysis failed local source-evidence validation:"
            )
        )
    ):
        return "source_evidence"
    if stage in {"apify-linkedin-retrieval", "linkedin-posting-confirmation"}:
        return "retrieval"
    return None


def _retrieval_classification_from_metadata(
    metadata: Mapping[str, Any],
) -> str | None:
    value = metadata.get("retrieval_classification")
    if isinstance(value, str) and value in _APIFY_LINKEDIN_UI_MESSAGES:
        return value
    return None


def _ui_stage_from_metadata(stage: str) -> str:
    return {
        "initializing": "validating_input",
        "dependency-check": "validating_input",
        "apify-linkedin-retrieval": "fetching_job",
        "linkedin-posting-confirmation": "confirming_posting",
        "extracting-master": "codex_analysis",
        "codex-analysis-schema-preflight": "codex_analysis",
        "codex-analysis": "codex_analysis",
        "tailoring-dependency-check": "antigravity_tailoring",
        "antigravity-recovery-verification": "antigravity_tailoring",
        "antigravity-tailoring-preflight": "antigravity_tailoring",
        "antigravity-tailoring": "antigravity_tailoring",
        "antigravity-response-reprocessing": "antigravity_tailoring",
        "ollama-tailoring-preflight": "antigravity_tailoring",
        "ollama-tailoring": "antigravity_tailoring",
        "local-evidence-check": "evidence_validation",
        "docx-render": "rendering",
        "pdf-export-validation": "rendering",
        "final-codex-qa": "final_qa",
        "revision-authorization": "revision_phase",
        "antigravity-revision-1": "revision_phase",
        "ollama-revision-1": "revision_phase",
        "revision-1-local-evidence-check": "revision_phase",
        "revision-1-content-approval": "revision_phase",
        "revision-1-docx-render": "revision_phase",
        "revision-1-final-codex-qa": "revision_phase",
        "complete": "complete",
    }.get(stage, "validating_input")


def _safe_error_message(error: ResumeTailorError) -> str:
    if isinstance(error, SourceEvidenceError):
        return _SOURCE_EVIDENCE_UI_MESSAGE
    if isinstance(error, (ApifyConfigurationError, ApifyLinkedInRetrievalError)):
        return _APIFY_LINKEDIN_UI_MESSAGES[error.classification]
    if isinstance(error, AntigravityLaunchSizeError) or (
        "Argument list too long" in str(error)
        and "agy" in str(error)
    ):
        return _ANTIGRAVITY_LAUNCH_SIZE_UI_MESSAGE
    if isinstance(error, AntigravityResponseEnvelopeError):
        return _ANTIGRAVITY_RESPONSE_ENVELOPE_UI_MESSAGE
    if isinstance(error, AntigravityTailoringContractError):
        return _ANTIGRAVITY_TAILORING_CONTRACT_UI_MESSAGE
    if isinstance(error, AntigravityCannotApplyError):
        return _ANTIGRAVITY_CANNOT_APPLY_UI_MESSAGE
    if isinstance(error, AntigravityTechnicalFailureError):
        return _ANTIGRAVITY_TECHNICAL_FAILURE_UI_MESSAGE
    if isinstance(error, AntigravityTailoringPreflightError):
        return (
            "The authenticated tailoring inputs failed local completeness "
            "preflight. No résumé-writer request was launched."
        )
    if isinstance(error, TailoringPreflightError):
        return _OLLAMA_PREFLIGHT_UI_MESSAGE
    # OllamaBudgetError subclasses OllamaConnectionError, so it must be matched
    # first: no request was launched, so connection guidance would mislead.
    if isinstance(error, OllamaBudgetError):
        return _OLLAMA_BUDGET_UI_MESSAGE
    if isinstance(error, OllamaConnectionError):
        return _OLLAMA_CONNECTION_UI_MESSAGE
    if isinstance(error, (OllamaTailoringContractError, OllamaRevisionContractError)):
        return _OLLAMA_CONTRACT_UI_MESSAGE
    if isinstance(error, (OllamaCannotApplyError, OllamaRevisionCannotApplyError)):
        return _OLLAMA_CANNOT_APPLY_UI_MESSAGE
    if isinstance(error, (OllamaTechnicalFailureError, OllamaRevisionTechnicalFailureError)):
        return _OLLAMA_TECHNICAL_FAILURE_UI_MESSAGE
    if isinstance(error, CodexSchemaCompatibilityError):
        return "Codex could not start because its output schema was incompatible."
    if isinstance(error, CodexUsageLimitError):
        return (
            "Codex hit its usage limit. Resume Tailor did not switch providers "
            "automatically. Wait for the limit to reset, or start a new run and "
            "explicitly select Grok as the analysis provider."
        )
    if isinstance(error, GrokExecutableError):
        return (
            "The Grok Build CLI executable was not found or could not be started. "
            "Install Grok Build and ensure ~/.grok/bin/grok is available."
        )
    if isinstance(error, GrokPromptTooLargeError):
        return (
            "Grok analysis could not start because the analysis prompt exceeded "
            "the local process argument-size limit. Reduce the confirmed posting "
            "or résumé size and start a new run. Prompt contents are not shown."
        )
    if isinstance(error, GrokAuthenticationError):
        return (
            "Grok Build CLI authentication failed. Log in through grok.com, then "
            "start a new run with analysis provider set to Grok CLI."
        )
    if isinstance(error, GrokTimeoutError):
        return (
            "Grok analysis exceeded the bounded timeout. The process group was "
            "stopped; sanitized diagnostics were preserved."
        )
    if isinstance(error, GrokTransportEnvelopeError):
        return (
            "Grok returned a malformed transport envelope. Thought content and "
            "provider body text were omitted; sanitized diagnostics were preserved."
        )
    if isinstance(error, GrokInnerAnalysisError):
        return (
            "Grok envelope text was not exactly one canonical analysis JSON "
            "document. Markdown fences, trailing commentary, and multiple "
            "documents are rejected."
        )
    if isinstance(error, (GrokUsageLimitError, GrokProcessError, GrokAnalysisError)):
        return (
            "Grok analysis stopped with a classified provider failure. Sanitized "
            "technical details are available below; useful artifacts were preserved."
        )
    if isinstance(error, GemmaOllamaUnavailableError):
        return (
            "Local Ollama is unavailable for Gemma analysis. Confirm Ollama is "
            "running on 127.0.0.1:11434. No automatic provider fallback was attempted."
        )
    if isinstance(error, GemmaModelUnavailableError):
        return (
            "The configured Gemma analysis model is not available in local Ollama. "
            "Pull or create resume-tailor-gemma (or GEMMA_ANALYSIS_MODEL), then "
            "start a new run."
        )
    if isinstance(error, GemmaAnalysisTimeoutError):
        return (
            "Gemma Local analysis exceeded its generation time limit. Ollama was "
            "running and no automatic provider fallback occurred. Start a new run "
            "or explicitly select another analysis provider."
        )
    if isinstance(error, GemmaOutputLimitError):
        return (
            "Gemma Local analysis reached its output-token limit before completing "
            "a valid response. Truncated JSON was not accepted. Start a new run "
            "or raise GEMMA_ANALYSIS_MAX_OUTPUT_TOKENS only after reviewing "
            "diagnostics."
        )
    if isinstance(error, GemmaOllamaInternalError):
        return (
            "Local Ollama returned an internal error during Gemma analysis. "
            "Provider content was omitted; no automatic provider fallback occurred."
        )
    if isinstance(error, (GemmaConnectionError, GemmaResponseTooLargeError)):
        return (
            "Gemma analysis could not complete the bounded localhost Ollama "
            "request. Sanitized diagnostics were preserved."
        )
    if isinstance(error, GemmaTransportEnvelopeError):
        return (
            "Gemma analysis returned a malformed Ollama response envelope. "
            "Provider body content was omitted."
        )
    if isinstance(error, GemmaInnerAnalysisError):
        return (
            "Gemma analysis content was not exactly one canonical analysis JSON "
            "document. Markdown fences, trailing commentary, and multiple "
            "documents are rejected."
        )
    if isinstance(error, (GemmaStructuredOutputError, GemmaAnalysisError)):
        return (
            "Gemma Local analysis stopped with a classified provider failure. "
            "Sanitized technical details are available below; useful artifacts "
            "were preserved."
        )
    if isinstance(error, ModelError):
        provider = (
            "Codex"
            if "codex" in str(error).casefold()
            else "Grok CLI"
            if "grok" in str(error).casefold()
            else "Antigravity"
            if "antigravity" in str(error).casefold()
            else "Gemma Local"
            if "gemma" in str(error).casefold() or "ollama" in str(error).casefold()
            else "The model stage"
        )
        return (
            f"{provider} stopped with an error. Sanitized technical details are "
            "available below; useful artifacts were preserved."
        )
    message = str(error)
    normalized = message.casefold()
    if any(
        term in normalized
        for term in (
            "login",
            "expired",
            "unavailable",
            "permission",
            "substantive",
            "linkedin",
        )
    ):
        return (
            f"{message} Use a UTF-8 job file or pasted description as a safe fallback."
        )
    return message


_TECHNICAL_BLOCK = re.compile(
    r"BEGIN_[A-Z0-9_]+\s.*?END_[A-Z0-9_]+",
    flags=re.DOTALL,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(?:OPENAI|CODEX|ANTIGRAVITY|AGY|APIFY|GROK|XAI)_[A-Z0-9_]*"
    r"(?:KEY|TOKEN|COOKIE|SECRET|PASSWORD)"
    r"\s*=\s*[^\s]+"
)


def _sanitized_technical_details(error: BaseException) -> str:
    detail = _TECHNICAL_BLOCK.sub(
        "[delimited prompt content omitted]",
        str(error).strip(),
    )
    detail = _SENSITIVE_ASSIGNMENT.sub("[credential omitted]", detail)
    lines: list[str] = []
    for line in detail.splitlines()[:40]:
        lines.append(line if len(line) <= 500 else line[:497] + "...")
    sanitized = "\n".join(lines).strip()
    if len(sanitized) > 3_000:
        sanitized = sanitized[:2_997] + "..."
    return sanitized or type(error).__name__


def _render(
    app: FastAPI,
    request: Request,
    template_name: str,
    context: Mapping[str, Any],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    environment: Environment = app.state.templates
    template = environment.get_template(template_name)
    html = template.render(
        request=request,
        csrf_token=app.state.settings.launch_token,
        version=__version__,
        workflow_stages=(
            context["run"].get("workflow_stages")
            if isinstance(context.get("run"), Mapping)
            and isinstance(context["run"].get("workflow_stages"), (list, tuple))
            else WORKFLOW_STAGES
        ),
        max_job_description_characters=(
            MAX_CONFIRMED_JOB_DESCRIPTION_CHARACTERS
        ),
        **context,
    )
    response = HTMLResponse(html, status_code=status_code)
    response.set_cookie(
        COOKIE_NAME,
        app.state.settings.launch_token,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )
    return response


def _session_valid(request: Request) -> bool:
    expected = request.app.state.settings.launch_token
    supplied = request.cookies.get(COOKIE_NAME, "")
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def _require_session(request: Request) -> None:
    if not _session_valid(request):
        raise HTTPException(status_code=403, detail="Invalid localhost UI session.")


def _require_csrf(request: Request, form: FormData) -> None:
    _require_session(request)
    supplied = str(form.get("csrf_token", ""))
    expected = request.app.state.settings.launch_token
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Invalid CSRF token.")


async def _limited_form(request: Request, *, max_files: int) -> FormData:
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The submitted form exceeds the {MAX_REQUEST_BYTES:,}-byte "
                    "request limit."
                ),
            )
    return await request.form(
        max_files=max_files,
        max_fields=20,
        max_part_size=MAX_JOB_BYTES,
    )


async def _upload_bytes(
    upload: UploadFile,
    *,
    maximum: int,
    label: str,
) -> bytes:
    value = await upload.read(maximum + 1)
    if len(value) > maximum:
        raise InputError(f"{label} exceeds the {maximum:,}-byte upload limit.")
    if not value:
        raise InputError(f"{label} is empty.")
    return value


def _validate_docx_archive(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise InputError("The uploaded résumé is not a valid DOCX archive.")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            expanded_size = sum(item.file_size for item in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise InputError("The uploaded résumé could not be read as DOCX.") from exc
    if "[Content_Types].xml" not in names or "word/document.xml" not in names:
        raise InputError("The uploaded file does not contain a Word document.")
    if expanded_size > 25 * 1024 * 1024:
        raise InputError("The uploaded DOCX expands beyond the 25 MiB safety limit.")
    validate_template(Document(path))


def _form_text(form: FormData, name: str) -> str:
    value = form.get(name, "")
    return value.strip() if isinstance(value, str) else ""


def _form_local_timestamp(form: FormData, name: str) -> datetime | None:
    value = _form_text(form, name)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{name.replace('_', ' ').title()} is not a valid date/time.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        local_timezone = datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=local_timezone)
    return parsed


async def _prepare_namespace(
    *,
    request: Request,
    form: FormData,
    manager: RunManager,
) -> tuple[argparse.Namespace, Path, str, str | None, str | None]:
    source_mode = _form_text(form, "job_mode")
    if source_mode not in {"url", "pasted", "file"}:
        raise InputError("Choose exactly one job-description input mode.")
    resume_mode = _form_text(form, "resume_mode") or "master"
    if resume_mode not in {"master", "upload"}:
        raise InputError("Choose the bundled master résumé or a DOCX upload.")

    staging = manager.create_staging_directory()
    try:
        if resume_mode == "master":
            resume_path = manager.settings.master_resume
            if not resume_path.is_file():
                raise InputError(
                    "The bundled master résumé is missing from this installation."
                )
        else:
            upload = form.get("resume_upload")
            if not isinstance(upload, UploadFile) or not upload.filename:
                raise InputError("Choose a .docx résumé to upload.")
            if Path(upload.filename).suffix.casefold() != ".docx":
                raise InputError("Uploaded résumés must use the .docx extension.")
            value = await _upload_bytes(
                upload,
                maximum=MAX_RESUME_BYTES,
                label="Résumé upload",
            )
            resume_path = staging / "uploaded-resume.docx"
            resume_path.write_bytes(value)
            _validate_docx_archive(resume_path)

        company: str | None = None
        role: str | None = None
        job_url: str | None = None
        job_file: Path | None = None
        if source_mode == "url":
            job_url = _form_text(form, "job_url")
            if not job_url:
                raise InputError("Enter a LinkedIn job URL.")
            job_url = validate_linkedin_url(job_url).normalized
        else:
            company = _validate_label(_form_text(form, "company"), "Company")
            role = _validate_label(_form_text(form, "role"), "Role")
            if source_mode == "pasted":
                description = _form_text(form, "pasted_description")
                if not description:
                    raise InputError("Paste a complete job description.")
                validate_confirmed_job_description(description)
            else:
                upload = form.get("job_file")
                if not isinstance(upload, UploadFile) or not upload.filename:
                    raise InputError("Choose a UTF-8 .txt job-description file.")
                if Path(upload.filename).suffix.casefold() != ".txt":
                    raise InputError("Job-description uploads must use .txt.")
                value = await _upload_bytes(
                    upload,
                    maximum=MAX_JOB_BYTES,
                    label="Job-description upload",
                )
                try:
                    description = value.decode("utf-8-sig").strip()
                except UnicodeDecodeError as exc:
                    raise InputError(
                        "The job-description upload must be UTF-8 text."
                    ) from exc
                if not description:
                    raise InputError("The job-description upload is empty.")
                validate_confirmed_job_description(description)
            job_file = staging / "job-description.txt"
            atomic_write_text(job_file, description.rstrip() + "\n")

        namespace = argparse.Namespace(
            resume=resume_path,
            clipboard=False,
            job_file=job_file,
            job_source_override=source_mode,
            job_url=job_url,
            company=company,
            role=role,
            output_dir=manager.settings.output_directory,
            analytics_db=manager.settings.analytics_database,
            yes=False,
            keep_workdir=False,
            timeout=manager.settings.timeout,
            writer_provider="ollama",
            ollama_model=DEFAULT_OLLAMA_MODEL,
            analysis_provider=normalize_analysis_provider(
                _form_text(form, "analysis_provider") or DEFAULT_ANALYSIS_PROVIDER
            ),
        )
        return namespace, staging, source_mode, company, role
    except Exception:
        _safe_remove_staging(staging, manager._staging_root)
        raise


def create_app(
    *,
    output_directory: Path | None = None,
    master_resume: Path | None = None,
    launch_token: str | None = None,
    timeout: tuple[int, str] | None = None,
    analytics_database_path: Path | None = None,
    port: int = DEFAULT_PORT,
    pipeline_runner: PipelineRunner = run_pipeline,
) -> FastAPI:
    root = _project_root()
    static_directory = root / "resume_tailor" / "static"
    template_directory = root / "resume_tailor" / "templates"
    settings = UISettings(
        host=DEFAULT_HOST,
        port=port,
        output_directory=(output_directory or default_output_directory())
        .expanduser()
        .resolve(),
        master_resume=(master_resume or default_master_resume()).expanduser().resolve(),
        launch_token=launch_token or secrets.token_urlsafe(32),
        timeout=timeout or parse_duration("15m"),
        analytics_database=(
            analytics_database_path or default_analytics_database_path()
        ).expanduser().resolve(),
    )
    settings.output_directory.mkdir(parents=True, exist_ok=True)
    manager = RunManager(settings=settings, pipeline_runner=pipeline_runner)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="resume-tailor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.manager = manager
    app.state.analytics_store = AnalyticsStore(settings.analytics_database)
    app.state.templates = Environment(
        loader=FileSystemLoader(str(template_directory)),
        autoescape=select_autoescape(("html", "xml")),
        enable_async=False,
    )
    app.mount("/static", StaticFiles(directory=static_directory), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]):
        response = await call_next(request)
        inline_pdf = (
            request.url.path.casefold().endswith(".pdf")
            and request.query_params.get("inline", "").casefold() in {"1", "true"}
            and response.headers.get("content-type", "").casefold().startswith(
                "application/pdf"
            )
        )
        frame_ancestors = "'self'" if inline_pdf else "'none'"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; frame-src 'self'; object-src 'none'; "
            f"base-uri 'none'; form-action 'self'; frame-ancestors {frame_ancestors}"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = (
            "SAMEORIGIN" if inline_pdf else "DENY"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "application": "resume-tailor",
            "version": __version__,
            "bind_host": settings.host,
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return _render(
            app,
            request,
            "dashboard.html",
            {
                "history": manager.history(),
                "master_resume_name": settings.master_resume.name,
                "form_error": None,
                "form_values": {},
            },
        )

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_dashboard(request: Request) -> HTMLResponse:
        _require_session(request)
        notice_messages = {
            "status": "Application status recorded locally.",
            "correction": "Correction event appended; prior history was preserved.",
            "interview": "Manually confirmed interview recorded locally.",
            "note": "Application note recorded locally.",
        }
        analytics_error: str | None = None
        summary: dict[str, Any] | None
        try:
            summary = app.state.analytics_store.summary()
        except (AnalyticsError, InputError):
            summary = None
            analytics_error = (
                "The local analytics database is unavailable. Tailoring runs remain "
                "usable; check the private application-data directory and retry."
            )
        return _render(
            app,
            request,
            "analytics.html",
            {
                "summary": summary,
                "analytics_error": analytics_error,
                "notice": notice_messages.get(request.query_params.get("notice", "")),
                "application_statuses": APPLICATION_STATUSES,
                "interview_types": INTERVIEW_TYPES,
                "analytics_database_name": settings.analytics_database.name,
            },
            status_code=503 if analytics_error else 200,
        )

    @app.post("/analytics/applications/{application_id}/status")
    async def analytics_status(
        request: Request,
        application_id: int,
    ) -> RedirectResponse:
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        actions = {
            "save": "saved",
            "apply": "applied",
            "screening": "screening",
            "reject": "rejected",
            "offer": "offer",
            "withdraw": "withdrawn",
        }
        new_status = actions.get(_form_text(form, "action"))
        if new_status is None:
            raise HTTPException(status_code=422, detail="Choose a supported status action.")
        try:
            app.state.analytics_store.set_application_status(
                application_id,
                new_status,
                source="manual_ui",
                note=_form_text(form, "note") or None,
            )
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalyticsError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local analytics database could not record this status.",
            ) from exc
        return RedirectResponse("/analytics?notice=status", status_code=303)

    @app.post("/analytics/applications/{application_id}/correction")
    async def analytics_correction(
        request: Request,
        application_id: int,
    ) -> RedirectResponse:
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            app.state.analytics_store.correct_application_status(
                application_id,
                _form_text(form, "new_status"),
                confirmed=_form_text(form, "confirm_correction") == "yes",
                source="manual_ui",
                note=_form_text(form, "note") or None,
            )
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalyticsError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local analytics database could not append this correction.",
            ) from exc
        return RedirectResponse("/analytics?notice=correction", status_code=303)

    @app.post("/analytics/applications/{application_id}/interviews")
    async def analytics_interview(
        request: Request,
        application_id: int,
    ) -> RedirectResponse:
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            app.state.analytics_store.record_interview(
                application_id,
                _form_text(form, "interview_type"),
                confirmed=_form_text(form, "confirm_interview") == "yes",
                scheduled_at=_form_local_timestamp(form, "scheduled_at"),
                completed_at=_form_local_timestamp(form, "completed_at"),
                contact_label=_form_text(form, "contact_label") or None,
                result=_form_text(form, "result") or None,
                notes=_form_text(form, "notes") or None,
                source="manual_ui",
            )
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalyticsError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local analytics database could not record this interview.",
            ) from exc
        return RedirectResponse("/analytics?notice=interview", status_code=303)

    @app.post("/analytics/applications/{application_id}/notes")
    async def analytics_note(
        request: Request,
        application_id: int,
    ) -> RedirectResponse:
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            app.state.analytics_store.add_note(
                application_id,
                _form_text(form, "note"),
                source="manual_ui",
            )
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalyticsError as exc:
            raise HTTPException(
                status_code=503,
                detail="The local analytics database could not record this note.",
            ) from exc
        return RedirectResponse("/analytics?notice=note", status_code=303)

    @app.post("/runs", response_class=HTMLResponse)
    async def start_run(request: Request):
        form = await _limited_form(request, max_files=2)
        _require_csrf(request, form)
        staging: Path | None = None
        try:
            namespace, staging, source_mode, company, role = await _prepare_namespace(
                request=request,
                form=form,
                manager=manager,
            )
            record = manager.start(
                namespace=namespace,
                staging_directory=staging,
                source_mode=source_mode,
                company=company,
                role=role,
            )
        except ActiveRunError as exc:
            if staging is not None:
                _safe_remove_staging(staging, manager._staging_root)
            return _render(
                app,
                request,
                "dashboard.html",
                {
                    "history": manager.history(),
                    "master_resume_name": settings.master_resume.name,
                    "form_error": str(exc),
                    "form_values": _safe_form_values(form),
                },
                status_code=409,
            )
        except ResumeTailorError as exc:
            return _render(
                app,
                request,
                "dashboard.html",
                {
                    "history": manager.history(),
                    "master_resume_name": settings.master_resume.name,
                    "form_error": str(exc),
                    "form_values": _safe_form_values(form),
                },
                status_code=422,
            )
        return RedirectResponse(f"/runs/{record.run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        _require_session(request)
        try:
            run = manager.snapshot(run_id)
        except InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _render(app, request, "run.html", {"run": run})

    @app.get("/api/runs/{run_id}", response_class=JSONResponse)
    async def run_status(request: Request, run_id: str) -> dict[str, Any]:
        _require_session(request)
        try:
            snapshot = manager.snapshot(run_id)
        except InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            key: snapshot[key]
            for key in (
                "run_id",
                "company",
                "role",
                "status",
                "stage",
                "stage_index",
                "message",
                "events",
                "revision",
            )
        } | {"approval_kind": (snapshot["approval"] or {}).get("kind")}

    @app.post("/runs/{run_id}/approval")
    async def approval(request: Request, run_id: str):
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            manager.respond_to_approval(
                run_id,
                action=_form_text(form, "action"),
                job_description=_form_text(form, "fallback_description"),
            )
        except InputError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/cancel")
    async def cancel(request: Request, run_id: str):
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            manager.cancel(run_id)
        except InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/retry-codex-analysis")
    async def retry_codex_analysis(request: Request, run_id: str):
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            record = manager.retry_codex_analysis(run_id)
        except InputError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{record.run_id}", status_code=303)

    @app.post("/runs/{run_id}/retry-antigravity-tailoring")
    async def retry_antigravity_tailoring(request: Request, run_id: str):
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            record = manager.retry_antigravity_tailoring(run_id)
        except InputError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{record.run_id}", status_code=303)

    @app.post("/runs/{run_id}/reprocess-antigravity-response")
    async def reprocess_antigravity_response(request: Request, run_id: str):
        form = await _limited_form(request, max_files=0)
        _require_csrf(request, form)
        try:
            record = manager.reprocess_antigravity_response(run_id)
        except InputError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/runs/{record.run_id}", status_code=303)

    @app.get("/runs/{run_id}/artifacts/{artifact_name:path}")
    async def artifact(
        request: Request,
        run_id: str,
        artifact_name: str,
        inline: bool = False,
    ):
        _require_session(request)
        try:
            path = manager.resolve_artifact(run_id, artifact_name)
        except InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type = {
            ".docx": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".json": "application/json",
            ".md": "text/markdown; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
        }.get(path.suffix.casefold(), "application/octet-stream")
        disposition = (
            "inline"
            if inline and path.suffix.casefold() == ".pdf"
            else "attachment"
        )

        async def file_chunks():
            with path.open("rb") as handle:
                while chunk := handle.read(64 * 1024):
                    yield chunk

        encoded_name = quote(path.name, safe="")
        return StreamingResponse(
            file_chunks(),
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f"{disposition}; filename*=UTF-8''{encoded_name}"
                )
            },
        )

    return app


def _safe_form_values(form: FormData) -> dict[str, str]:
    allowed = {
        "resume_mode",
        "job_mode",
        "job_url",
        "company",
        "role",
        "pasted_description",
        "analysis_provider",
    }
    return {
        name: value
        for name in allowed
        if isinstance((value := form.get(name)), str)
    }
