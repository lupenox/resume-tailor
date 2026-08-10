from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


class ExitCode(IntEnum):
    OK = 0
    DEPENDENCY = 3
    INPUT = 4
    MODEL = 10
    WAITING = 11
    APPROVAL = 12
    TRUTHFULNESS = 13
    TEMPLATE = 14
    RENDER = 15
    QA = 16
    INTEGRITY = 17
    CANCELLED = 18


class ResumeTailorError(Exception):
    exit_code = ExitCode.INPUT


class DependencyError(ResumeTailorError):
    exit_code = ExitCode.DEPENDENCY


class AntigravityLaunchSizeError(DependencyError):
    """Antigravity failed to exec because its argument vector was too large."""


class CodexSchemaCompatibilityError(DependencyError):
    """A bundled Codex transport schema cannot be sent safely."""


class InputError(ResumeTailorError):
    exit_code = ExitCode.INPUT


class RequirementExtractionError(InputError):
    """Job requirement extraction failed due to pathological input."""
    def __init__(self, message: str, diagnostic: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class ModelError(ResumeTailorError):
    exit_code = ExitCode.MODEL


class AnalysisProviderError(ModelError):
    """A résumé-analysis provider failed with a classified, sanitized reason.

    ``classification`` is a stable machine-readable token used by UI guidance
    and run metadata. Provider prose is omitted from subclasses that represent
    transport or authentication failures.
    """

    classification = "generic_provider_failure"
    provider = "analysis"

    def __init__(self, message: str, *, classification: str | None = None) -> None:
        if classification is not None:
            self.classification = classification
        super().__init__(message)


class CodexUsageLimitError(AnalysisProviderError):
    """Codex reported the known usage-limit message; no automatic fallback."""

    classification = "provider_usage_limit"
    provider = "codex"

    def __init__(self) -> None:
        super().__init__(
            "Codex reported a usage limit. No automatic provider fallback was "
            "attempted. Wait for the limit to reset, or explicitly select Grok "
            "as the analysis provider and start a new run."
        )


class GrokAnalysisError(AnalysisProviderError):
    """Grok Build analysis failed; subclass names one sanitized path."""

    classification = "generic_provider_failure"
    provider = "grok"


class GrokExecutableError(DependencyError):
    """The Grok Build CLI executable could not be resolved or started."""

    classification = "executable_unavailable"
    provider = "grok"


class GrokPromptTooLargeError(DependencyError):
    """The Grok analysis prompt exceeded the OS process argument-size limit."""

    classification = "prompt_too_large"
    provider = "grok"

    def __init__(self) -> None:
        super().__init__(
            "Grok analysis could not start because the analysis prompt exceeded "
            "the local process argument-size limit (OS E2BIG). Reduce the "
            "confirmed posting or résumé size and start a new run. Prompt "
            "contents are omitted from this error."
        )


class GrokUsageLimitError(GrokAnalysisError):
    """Grok reported a provider usage or quota limit."""

    classification = "provider_usage_limit"


class GrokAuthenticationError(GrokAnalysisError):
    """Grok CLI authentication is missing or rejected."""

    classification = "authentication_failure"

    def __init__(self) -> None:
        super().__init__(
            "Grok Build CLI authentication failed. Log in through grok.com, then "
            "retry with the analysis provider explicitly set to grok. No "
            "credential values are stored or displayed."
        )


class GrokTimeoutError(GrokAnalysisError):
    """The Grok analysis subprocess exceeded its bounded timeout."""

    classification = "timeout"

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            f"Grok analysis timed out after {timeout_seconds}s. The full "
            "subprocess group was stopped. Retry with a larger bounded "
            "--timeout only after confirming the local Grok CLI is responsive."
        )


class GrokProcessError(GrokAnalysisError):
    """Grok exited nonzero without a more specific classified reason."""

    classification = "nonzero_exit"


class GrokTransportEnvelopeError(GrokAnalysisError):
    """The Grok CLI transport envelope was missing, malformed, or incomplete."""

    classification = "malformed_transport_envelope"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            "Grok returned a malformed transport envelope. Provider thought and "
            f"response body content were omitted ({detail})."
        )


class GrokInnerAnalysisError(GrokAnalysisError):
    """The envelope text field was not exactly one canonical analysis document."""

    classification = "malformed_inner_analysis"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            "Grok envelope text was not exactly one canonical résumé-analysis "
            f"JSON document ({detail}). Markdown fences, trailing commentary, "
            "and multiple JSON documents are rejected."
        )


class GemmaAnalysisError(AnalysisProviderError):
    """Local Gemma analysis failed; subclass names one sanitized path."""

    classification = "generic_provider_failure"
    provider = "gemma_local"


class GemmaOllamaUnavailableError(GemmaAnalysisError):
    """The localhost Ollama server is not reachable for analysis."""

    classification = "ollama_unavailable"

    def __init__(self) -> None:
        super().__init__(
            "Local Ollama is unavailable for Gemma analysis. Confirm Ollama is "
            "running on 127.0.0.1:11434. No automatic provider fallback was "
            "attempted."
        )


class GemmaModelUnavailableError(GemmaAnalysisError):
    """The configured Gemma analysis model is not available in Ollama."""

    classification = "model_unavailable"

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"The configured Gemma analysis model {model!r} is not available in "
            "local Ollama. Pull or create the model, then start a new run with "
            "analysis provider gemma_local."
        )


class GemmaAnalysisTimeoutError(GemmaAnalysisError):
    """The Gemma analysis request exceeded its generation time limit."""

    classification = "analysis_timeout"

    def __init__(self, timeout_seconds: int) -> None:
        super().__init__(
            "Gemma Local analysis exceeded its generation time limit. Ollama was "
            f"running and the request was stopped after {timeout_seconds}s. No "
            "automatic provider fallback was attempted."
        )


class GemmaConnectionError(GemmaAnalysisError):
    """A bounded localhost Ollama connection failure during analysis."""

    classification = "connection_failure"


class GemmaOllamaInternalError(GemmaAnalysisError):
    """Ollama returned an HTTP 500 (or similar) not caused by a local timeout."""

    classification = "ollama_internal_error"

    def __init__(self, *, http_status: int | None = None) -> None:
        self.http_status = http_status
        status = f" (HTTP {http_status})" if http_status is not None else ""
        super().__init__(
            f"Local Ollama returned an internal error{status} during Gemma "
            "analysis. Provider body content was omitted. No automatic provider "
            "fallback was attempted."
        )


class GemmaOutputLimitError(GemmaAnalysisError):
    """Generation stopped at the configured analysis output-token ceiling."""

    classification = "output_limit_reached"

    def __init__(
        self,
        max_output_tokens: int,
        phase: str | None = None,
        content_bytes: int = 0,
        thinking_present: bool = False
    ) -> None:
        self.max_output_tokens = max_output_tokens
        self.phase = phase
        self.content_bytes = content_bytes
        self.thinking_present = thinking_present
        super().__init__(
            "Gemma Local analysis reached its configured output-token limit "
            f"({max_output_tokens}) before producing a complete valid response. "
            "Truncated JSON was not accepted. No automatic provider fallback was "
            "attempted."
        )


class GemmaResponseTooLargeError(GemmaAnalysisError):
    """The Ollama analysis response exceeded the local size bound."""

    classification = "response_too_large"

    def __init__(self) -> None:
        super().__init__(
            "Gemma analysis returned a response larger than the local safety "
            "limit. Provider content was omitted; reduce the posting or résumé "
            "size and start a new run."
        )


class GemmaTransportEnvelopeError(GemmaAnalysisError):
    """The Ollama chat envelope was missing, incomplete, or not structured."""

    classification = "malformed_transport_envelope"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            "Gemma analysis returned a malformed Ollama transport envelope. "
            f"Provider body content was omitted ({detail})."
        )


class GemmaInnerAnalysisError(GemmaAnalysisError):
    """message.content was not exactly one canonical analysis document."""

    classification = "malformed_inner_analysis"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            "Gemma analysis content was not exactly one canonical résumé-analysis "
            f"JSON document ({detail}). Markdown fences, trailing commentary, "
            "and multiple JSON documents are rejected."
        )


class GemmaStructuredOutputError(GemmaAnalysisError):
    """Structured-output grammar was ignored or incomplete."""

    classification = "structured_output_failure"


class OllamaConnectionError(ModelError):
    """The localhost-only Ollama API could not complete a bounded request."""

    classification = "connection_failure"
    http_status: int | None = None
    transport_error: str | None = None


class OllamaRequestError(OllamaConnectionError):
    """Structured localhost Ollama transport failure with a sanitized class.

    ``classification`` is one of:
    ``connection_refused``, ``timeout``, ``http_error``, ``response_too_large``,
    ``malformed_envelope``, or ``transport_failure``. Provider body content is
    never included in the message.
    """

    def __init__(
        self,
        message: str,
        *,
        classification: str,
        http_status: int | None = None,
        transport_error: str | None = None,
    ) -> None:
        self.classification = classification
        self.http_status = http_status
        self.transport_error = transport_error
        super().__init__(message)


class OllamaTailoringContractError(ModelError):
    """Gemma 4 12B did not satisfy the post-approval tailoring contract.

    The subclasses below name one specific, sanitized validation path so a
    preserved failure can be diagnosed without reading résumé content. Every
    subclass remains an ``OllamaTailoringContractError`` so existing handlers,
    exit codes, and recovery gates keep their current behaviour.
    """

    #: Stable, sanitized identifier persisted in the response envelope.
    validation_path = "tailoring_contract"


class OllamaMalformedJSONError(OllamaTailoringContractError):
    """Gemma 4 12B returned message content that is not one parseable JSON object."""

    validation_path = "malformed_json"


class OllamaResponseEnvelopeError(OllamaTailoringContractError):
    """The Ollama chat envelope was missing, incomplete, or not structured."""

    validation_path = "response_envelope"


class OllamaTransportSchemaError(OllamaTailoringContractError):
    """Parseable JSON that does not satisfy the derived transport schema.

    This is the signature of a model that ignored the structured-output
    grammar, for example by returning a bare résumé object instead of the
    required status envelope.
    """

    validation_path = "transport_schema"


class OllamaCanonicalSchemaError(OllamaTailoringContractError):
    """Transport-valid JSON that fails the full canonical Draft 2020-12 schema."""

    validation_path = "canonical_schema"


class OllamaOutputTruncationError(OllamaTailoringContractError):
    """Generation stopped at a length or context limit before completing."""

    validation_path = "output_truncation"


class OllamaEvidenceRejectionError(OllamaTailoringContractError):
    """Schema-valid output rejected by a downstream evidence or edit rule."""

    validation_path = "downstream_evidence"


class OllamaBudgetError(OllamaConnectionError):
    """The deterministic prompt/context/output budget refuses to launch."""

    validation_path = "budget_preflight"


class OllamaCannotApplyError(ModelError):
    """Gemma 4 12B could not apply one authenticated approved edit."""


class OllamaTechnicalFailureError(ModelError):
    """Gemma 4 12B reported a bounded technical tailoring failure."""


class OllamaRevisionContractError(ModelError):
    """Gemma 4 12B violated the bounded one-shot revision contract."""


class OllamaRevisionCannotApplyError(ModelError):
    """Gemma 4 12B could not apply one authenticated QA correction."""


class OllamaRevisionTechnicalFailureError(ModelError):
    """Gemma 4 12B reported a bounded revision execution failure."""


_APIFY_CONFIGURATION_MESSAGES = {
    "missing_token": (
        "LinkedIn URL retrieval requires APIFY_API_TOKEN. Preserve the complete "
        "token, including its apify_api_ prefix, in local configuration."
    ),
    "missing_actor_id": (
        "LinkedIn URL retrieval requires APIFY_ACTOR_ID set to the Actor ID or "
        "username/actor-name used in Apify."
    ),
    "invalid_token": (
        "APIFY_API_TOKEN is malformed. Store the complete token exactly as issued, "
        "including its apify_api_ prefix and without surrounding whitespace."
    ),
    "invalid_actor_id": (
        "APIFY_ACTOR_ID must be an Apify Actor ID or username/actor-name value."
    ),
}


class ApifyConfigurationError(InputError):
    """Apify retrieval configuration is absent or locally invalid."""

    def __init__(self, classification: str) -> None:
        if classification not in _APIFY_CONFIGURATION_MESSAGES:
            classification = "invalid_actor_id"
        self.classification = classification
        super().__init__(_APIFY_CONFIGURATION_MESSAGES[classification])


_APIFY_RETRIEVAL_MESSAGES = {
    "authentication_failure": (
        "Apify rejected authentication. Verify APIFY_API_TOKEN and its full "
        "apify_api_ prefix."
    ),
    "actor_not_found": (
        "The configured Apify Actor was not found. Verify APIFY_ACTOR_ID and "
        "account access."
    ),
    "actor_timeout": "The Apify Actor did not finish before the bounded timeout.",
    "actor_failure": "The Apify Actor run stopped without succeeding.",
    "empty_dataset": "The Apify Actor completed but returned an empty dataset.",
    "no_matching_result": (
        "Apify returned no unique result matching the requested LinkedIn job URL "
        "or job ID."
    ),
    "malformed_output": (
        "The Apify result could not be normalized into the canonical job-posting "
        "contract."
    ),
    "insufficient_content": (
        "The Apify result did not contain a meaningful job title and complete "
        "job description."
    ),
    "network_error": "Resume Tailor could not complete the bounded Apify HTTPS request.",
    "rate_limited": (
        "Apify rate-limited this retrieval. Wait for the provider limit to reset "
        "before retrying."
    ),
    "provider_failure": "Apify LinkedIn retrieval stopped with a provider failure.",
}


class ApifyLinkedInRetrievalError(ModelError):
    """A structured, token-free failure from Apify LinkedIn retrieval."""

    def __init__(
        self,
        classification: str,
        *,
        http_status: int | None = None,
        provider_message: str | None = None,
        run_id: str | None = None,
        run_status: str | None = None,
        dataset_id: str | None = None,
        item_count: int | None = None,
    ) -> None:
        if classification not in _APIFY_RETRIEVAL_MESSAGES:
            classification = "provider_failure"
        self.classification = classification
        self.http_status = http_status
        self.provider_message = provider_message
        self.run_id = run_id
        self.run_status = run_status
        self.dataset_id = dataset_id
        self.item_count = item_count
        super().__init__(
            _APIFY_RETRIEVAL_MESSAGES[classification]
            + " Use pasted text, --job-file, or --clipboard as a bounded fallback."
        )


class AntigravityTailoringContractError(ModelError):
    """Antigravity did not satisfy the post-approval tailoring contract."""


class AntigravityResponseEnvelopeError(ModelError):
    """Antigravity returned no unique documented structured-output envelope."""

    def __init__(self, message: str, *, envelope_type: str) -> None:
        super().__init__(message)
        self.envelope_type = envelope_type


class AntigravityCannotApplyError(ModelError):
    """Antigravity could not apply one authenticated approved edit."""


class AntigravityTechnicalFailureError(ModelError):
    """Antigravity reported a bounded technical tailoring failure."""


class AntigravityRevisionContractError(ModelError):
    """Antigravity violated the bounded one-shot revision contract."""


class AntigravityRevisionCannotApplyError(ModelError):
    """Antigravity could not apply one authenticated QA correction."""


class AntigravityRevisionTechnicalFailureError(ModelError):
    """Antigravity reported a bounded revision execution failure."""


class RevisionValidationError(ModelError):
    """A revision crossed a local authorization or integrity boundary."""


class TailoringPreflightError(InputError):
    """Local authenticated writer inputs are incomplete or inconsistent."""


class AntigravityTailoringPreflightError(TailoringPreflightError):
    """Local authenticated tailoring inputs are incomplete or inconsistent."""


class WaitingError(ResumeTailorError):
    exit_code = ExitCode.WAITING


class ApprovalError(ResumeTailorError):
    exit_code = ExitCode.APPROVAL


class TruthfulnessError(ResumeTailorError):
    exit_code = ExitCode.TRUTHFULNESS


class SourceEvidenceError(TruthfulnessError):
    """Codex returned invalid or inappropriate local source references."""


class TemplateError(ResumeTailorError):
    exit_code = ExitCode.TEMPLATE


class RenderError(ResumeTailorError):
    exit_code = ExitCode.RENDER


class QAError(ResumeTailorError):
    exit_code = ExitCode.QA


class IntegrityError(ResumeTailorError):
    exit_code = ExitCode.INTEGRITY


class CancellationError(ResumeTailorError):
    exit_code = ExitCode.CANCELLED


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


_DURATION_RE = re.compile(r"^(?P<number>[1-9][0-9]*)(?P<unit>s|m|h)?$")
_CANCELLATION_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "resume_tailor_cancellation_event",
    default=None,
)


def parse_duration(value: str) -> tuple[int, str]:
    """Return a subprocess timeout in seconds and an Antigravity duration."""
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("use a positive duration such as 90s, 15m, or 1h")
    number = int(match.group("number"))
    unit = match.group("unit") or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    seconds = number * multiplier
    if seconds > 24 * 60 * 60:
        raise ValueError("duration must not exceed 24h")
    return seconds, f"{number}{unit}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_text_file(path: Path, *, label: str, max_bytes: int = 500_000) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"Cannot read {label} {path}: {exc}") from exc
    if size > max_bytes:
        raise InputError(
            f"{label.capitalize()} is {size:,} bytes; the safety limit is "
            f"{max_bytes:,} bytes."
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{label.capitalize()} must be UTF-8 text: {path}") from exc
    except OSError as exc:
        raise InputError(f"Cannot read {label} {path}: {exc}") from exc
    if not text.strip():
        raise InputError(f"{label.capitalize()} is empty: {path}")
    return text.strip()


def slugify(value: str, *, fallback: str = "resume") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:80].rstrip("-") or fallback).strip(".")


def filename_component(value: str, *, fallback: str = "Resume") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", ascii_value) if part]
    if not parts:
        parts = [fallback]
    formatted = [
        part if part.isupper() and len(part) <= 3 else part.capitalize()
        for part in parts
    ]
    return "-".join(formatted)[:100].rstrip("-")


def create_unique_run_dir(
    output_dir: Path,
    company: str,
    role: str,
    *,
    now: datetime | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(company, fallback='company')}-{slugify(role, fallback='role')}-{stamp}"
    for suffix in ("", *(f"-{number}" for number in range(2, 1000))):
        candidate = output_dir / f"{base}{suffix}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise InputError(f"Could not create a unique run directory under {output_dir}")


def rename_run_directory(
    run_directory: Path,
    company: str,
    role: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Atomically rename a provisional run after URL-derived identity is approved."""
    resolved = run_directory.resolve()
    parent = resolved.parent
    if not resolved.is_dir() or resolved == parent:
        raise InputError(f"Cannot rename invalid run directory: {run_directory}")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(company, fallback='company')}-{slugify(role, fallback='role')}-{stamp}"
    for suffix in ("", *(f"-{number}" for number in range(2, 1000))):
        candidate = parent / f"{base}{suffix}"
        if candidate.exists():
            continue
        try:
            resolved.rename(candidate)
        except FileExistsError:
            continue
        except OSError as exc:
            raise InputError(
                f"Could not rename URL run directory for the retrieved posting: {exc}"
            ) from exc
        return candidate
    raise InputError(f"Could not create a unique run directory under {parent}")


def require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise DependencyError(
            f"Required executable '{name}' was not found in PATH. See README.md "
            "for dependency installation instructions."
        )
    return resolved


@contextmanager
def cancellable_commands(cancel_event: threading.Event | None) -> Iterator[None]:
    """Make shared subprocess calls cooperatively cancellable in this context."""
    token = _CANCELLATION_EVENT.set(cancel_event)
    try:
        yield
    finally:
        _CANCELLATION_EVENT.reset(token)


def check_cancelled() -> None:
    event = _CANCELLATION_EVENT.get()
    if event is not None and event.is_set():
        raise CancellationError("The run was cancelled; useful artifacts were preserved.")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    heartbeat_handler: Callable[[float, bool], None] | None = None,
    heartbeat_interval_seconds: float = 15.0,
) -> CommandResult:
    """Run an argument array. This project intentionally never invokes a shell."""
    check_cancelled()
    if heartbeat_interval_seconds <= 0:
        raise InputError("Subprocess heartbeat interval must be positive.")
    # GitHub access is owned exclusively by the read-only REST adapter. Provider
    # CLIs, document tools, and every other child process must never inherit the
    # credential, even if untrusted repository text attempts to inspect its
    # environment. Explicit environments remain supported, but this one secret
    # is always removed at the process boundary.
    process_environment = dict(os.environ if env is None else env)
    github_credential_names = {
        "github_token",
        "gh_token",
        "github_enterprise_token",
        "gh_enterprise_token",
    }
    for key in tuple(process_environment):
        if key.casefold() in github_credential_names:
            process_environment.pop(key, None)
    try:
        process = subprocess.Popen(
            [str(arg) for arg in args],
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=process_environment,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise DependencyError(f"Could not run {Path(args[0]).name}: {exc}") from exc

    started = time.monotonic()
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_interval_seconds
    if heartbeat_handler is not None:
        try:
            heartbeat_handler(0.0, True)
        except BaseException:
            _stop_process(process)
            raise
    pending_input = input_text
    while True:
        event = _CANCELLATION_EVENT.get()
        if event is not None and event.is_set():
            _stop_process(process)
            raise CancellationError(
                "The run was cancelled; the active subprocess was stopped and "
                "useful artifacts were preserved."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise ModelError(
                f"Command timed out after {timeout_seconds}s: {Path(args[0]).name}. "
                "The full subprocess group was stopped; retry with a larger bounded "
                "--timeout only after confirming the local provider is responsive."
            )
        now = time.monotonic()
        if heartbeat_handler is not None and now >= next_heartbeat:
            try:
                heartbeat_handler(now - started, process.poll() is None)
            except BaseException:
                _stop_process(process)
                raise
            next_heartbeat = now + heartbeat_interval_seconds
        try:
            poll_timeout = min(0.25, remaining)
            if heartbeat_handler is not None:
                poll_timeout = min(
                    poll_timeout,
                    max(0.001, next_heartbeat - time.monotonic()),
                )
            stdout, stderr = process.communicate(
                input=pending_input,
                timeout=poll_timeout,
            )
            break
        except subprocess.TimeoutExpired:
            pending_input = None

    if heartbeat_handler is not None:
        heartbeat_handler(time.monotonic() - started, False)

    return CommandResult(
        tuple(str(arg) for arg in args),
        stdout,
        stderr,
        process.returncode,
    )


def concise_process_error(result: CommandResult, label: str) -> str:
    detail = (result.stderr or result.stdout).strip()
    if len(detail) > 2_000:
        detail = detail[-2_000:]
    return f"{label} exited with status {result.returncode}" + (
        f":\n{detail}" if detail else "."
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ask_for_approval(stage: str, *, assume_yes: bool) -> None:
    if assume_yes:
        print(f"{stage}: approved by --yes.")
        return
    try:
        response = input(f'{stage}: type "approve" to continue: ').strip()
    except EOFError as exc:
        raise ApprovalError(f"{stage} was not approved (input closed).") from exc
    if response != "approve":
        raise ApprovalError(f"{stage} was not approved; artifacts were preserved.")


def relative_artifacts(run_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from flatten_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from flatten_strings(nested)
